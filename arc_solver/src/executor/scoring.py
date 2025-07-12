from __future__ import annotations

"""Rule scoring and strategy utilities for the executor.

This module evaluates symbolic rules by comparing their predictions against
expected grids.  Older revisions heavily penalised long composite programs
using raw structural cost.  The new logic reduces this bias by weighting the
penalty per unique transformation type.  A small bonus is still granted to
perfect composites to encourage valid chains.
"""

from typing import Any, Dict, List, Tuple
import math
from collections import Counter

from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.executor.failure_logger import log_failure
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.symbolic.vocabulary import SymbolicRule, TransformationType


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

# Mapping from shape delta to preferred rule types
STRATEGY_REGISTRY: Dict[Tuple[str, ...], List[str]] = {
    ("shrink",): ["FILTER", "REPLACE"],
    ("grow",): ["REPEAT", "REPEAT→REPLACE"],
    ("equal",): ["TRANSLATE", "ROTATE"],
}

SCORE_FAILURE_THRESHOLD = 0.2

# Toggle integration of zone-aware heuristics during rule scoring
ENABLE_ZONE_SCORING = False

# Relative weights for each transformation type when computing structural cost.
OP_WEIGHTS: Dict[Any, float] = {
    TransformationType.REPLACE: 1.0,
    TransformationType.TRANSLATE: 1.0,
    TransformationType.MERGE: 1.1,
    TransformationType.FILTER: 1.0,
    TransformationType.ROTATE: 1.2,
    TransformationType.ROTATE90: 1.2,
    TransformationType.REFLECT: 1.1,
    TransformationType.REPEAT: 1.3,
    TransformationType.SHAPE_ABSTRACT: 1.3,
    TransformationType.CONDITIONAL: 1.2,
    TransformationType.REGION: 1.1,
    # Slightly penalise heavy functional operators such as pattern_fill or
    # morphology-based zone expansion.  These tend to produce large diffs and
    # should have a higher cost than basic logical transformations.
    TransformationType.FUNCTIONAL: 1.4,
    TransformationType.COMPOSITE: 1.0,
    # Functional operator specifics
    "mirror_tile": 4.0,
    "draw_line": 3.0,
    "dilate_zone": 2.5,
    "erode_zone": 2.5,
    "rotate_about_point": 4.0,
    "zone_remap": 1.5,
}


def grid_color_entropy(grid: Grid) -> float:
    """Return normalized color entropy of ``grid``."""

    counts: Counter[int] = Counter()
    total = 0
    for row in grid.data:
        for val in row:
            if val <= 0:
                continue
            counts[val] += 1
            total += 1

    if total == 0:
        return 0.0

    n_colors = len(counts)
    if n_colors <= 1:
        return 0.0

    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)

    max_ent = math.log2(n_colors)
    return ent / max_ent if max_ent else 0.0


def _shape_delta(input_grid: Grid, output_grid: Grid) -> str:
    """Return simple size delta type between ``input_grid`` and ``output_grid``."""
    ih, iw = input_grid.shape()
    oh, ow = output_grid.shape()
    in_area = ih * iw
    out_area = oh * ow
    if out_area > in_area:
        return "grow"
    if out_area < in_area:
        return "shrink"
    return "equal"


def preferred_rule_types(input_grid: Grid, output_grid: Grid) -> List[str]:
    """Return preferred rule categories for the given shape delta."""
    delta = _shape_delta(input_grid, output_grid)
    return STRATEGY_REGISTRY.get((delta,), [])


# ---------------------------------------------------------------------------
# Rule scoring
# ---------------------------------------------------------------------------


def _unique_ops(rule: SymbolicRule | CompositeRule) -> int:
    """Return count of unique transformation types used by ``rule``."""

    if isinstance(rule, CompositeRule):
        return len({step.transformation.ttype for step in rule.steps}) or 1
    if rule.transformation.ttype is TransformationType.COMPOSITE:
        steps = rule.transformation.params.get("steps", [])
        return len(set(steps)) or 1
    return 1


def _op_name(rule: SymbolicRule) -> str | TransformationType:
    """Return canonical operator identifier for ``rule``."""

    t = rule.transformation.ttype
    if t is TransformationType.FUNCTIONAL:
        return rule.transformation.params.get("op", t)
    if t is TransformationType.ROTATE and {
        "cx",
        "cy",
        "angle",
    }.issubset(rule.transformation.params):
        return "rotate_about_point"
    return t


def _op_cost(rule: SymbolicRule | CompositeRule) -> float:
    """Return weighted cost of unique operations used by ``rule``."""

    if isinstance(rule, CompositeRule):
        ops = {_op_name(step) for step in rule.steps}
    else:
        ops = {_op_name(rule)}

    if not ops:
        return 0.0

    cost = 0.0
    for op in ops:
        if isinstance(op, TransformationType):
            cost += OP_WEIGHTS.get(op, 1.0)
        else:
            cost += OP_WEIGHTS.get(op, OP_WEIGHTS.get(TransformationType.FUNCTIONAL, 1.0))
    return float(cost)


def rule_cost(rule: SymbolicRule | CompositeRule) -> float:
    """Return weighted complexity cost for ``rule``."""

    if isinstance(rule, CompositeRule):
        cost = sum(_op_cost(step) for step in rule.steps)
        cost += 0.5 * len(rule.steps)
        return float(cost)
    return float(_op_cost(rule))


def _extract_zones(rule: SymbolicRule | CompositeRule) -> List[str]:
    """Return sorted unique zones referenced by ``rule``."""

    zones: set[str] = set()

    def _gather(meta: Dict[str, Any] | None) -> None:
        if not meta:
            return
        for key in ("zone", "input_zones", "output_zones"):
            val = meta.get(key)
            if not val:
                continue
            if isinstance(val, str):
                zones.add(val)
            else:
                zones.update(val)

    if isinstance(rule, CompositeRule):
        for step in rule.steps:
            _gather(getattr(step, "condition", None) or {})
            _gather(getattr(step, "meta", None) or {})
    else:
        _gather(getattr(rule, "condition", None) or {})
        _gather(getattr(rule, "meta", None) or {})

    return sorted(zones)


def op_penalty(rule: SymbolicRule | CompositeRule) -> float:
    """Return nonlinear penalty for ``rule`` based on its operation cost."""

    cost = _op_cost(rule)
    # Saturating penalty curve to avoid excessive punishment for long yet
    # precise programs.  Scales with ``tanh`` so that heavy chains converge
    # towards a constant penalty.
    return 0.03 * math.tanh(cost / 5.0)


def score_rule(
    input_grid: Grid,
    output_grid: Grid,
    rule: SymbolicRule | CompositeRule,
    *,
    prefer_composites: bool = False,
    details: bool = False,
    return_trace: bool = False,
) -> float | Dict[str, float]:
    """Return heuristic score of ``rule`` for transforming ``input_grid`` to ``output_grid``.

    When ``details`` is ``True`` a dictionary containing individual score
    components is returned instead of just the final score.  The
    ``prefer_composites`` flag is kept for compatibility but no longer affects
    scoring.
    """

    try:
        pred = rule.simulate(input_grid) if isinstance(rule, CompositeRule) else simulate_rules(input_grid, [rule])
    except Exception:
        return 0.0

    before_pixel = input_grid.compare_to(output_grid)
    after_pixel = pred.compare_to(output_grid)
    diff = pred.diff_summary(output_grid)
    zone_match = diff.get("zone_coverage_match", 0.0)
    shape_bonus = 1.0 if pred.shape() == output_grid.shape() else 0.0

    # Basic similarity score
    base = 0.55 * after_pixel + 0.35 * zone_match + 0.1 * shape_bonus

    # Reward improvement over the input similarity
    improvement = after_pixel - before_pixel
    if improvement > 0:
        base += 0.25 * improvement

    # Complexity penalty uses a nonlinear saturation curve
    penalty = op_penalty(rule)

    # Entropy penalty discourages rules that create high chaos without
    # improving coverage
    ent_pred = grid_color_entropy(pred)
    ent_target = grid_color_entropy(output_grid)
    if ent_pred > ent_target and improvement <= 0:
        penalty += 0.05 * (ent_pred - ent_target)

    # Composite bonus only when the rule perfectly matches the output.  Additional
    # small boost for each unique functional operator used.
    func_ops: List[str] = []
    if isinstance(rule, CompositeRule):
        func_ops = [
            step.transformation.params.get("op")
            for step in rule.steps
            if step.transformation.ttype is TransformationType.FUNCTIONAL
            and step.transformation.params.get("op")
        ]
    if isinstance(rule, CompositeRule) and base >= 0.95:
        bonus = 0.2 + 0.02 * len(set(func_ops))
        bonus = min(bonus, 0.3)
    else:
        bonus = 0.0

    final = base - penalty + bonus

    # === Zone-Aware Scoring Hooks  ===
    if ENABLE_ZONE_SCORING:
        from arc_solver.src.scoring.zone_adjustments import (
            zone_alignment_bonus,
            zone_coverage_weight,
            zone_entropy_penalty,
        )
        zones = _extract_zones(rule)
        z_pen = zone_entropy_penalty(input_grid, zones)
        z_bonus = zone_alignment_bonus(pred, output_grid, zones)
        z_weight = zone_coverage_weight(pred, zones)
        final = (final - z_pen + z_bonus) * z_weight

    trace = {
        "similarity": float(base),
        "unique_ops": int(_unique_ops(rule)),
        "op_cost": float(_op_cost(rule)),
        "penalty": float(penalty),
        "bonus": float(bonus),
        "final_score": float(final),
        "functional_ops": func_ops,
        "rule_steps": [
            step.transformation.ttype.value for step in rule.steps
        ]
        if isinstance(rule, CompositeRule)
        else [rule.transformation.ttype.value],
    }

    if ENABLE_ZONE_SCORING:
        trace["zone_entropy_penalty"] = float(z_pen)
        trace["zone_alignment_bonus"] = float(z_bonus)
        trace["zone_coverage_weight"] = float(z_weight)

    if func_ops:
        note = f"[score] {'/'.join(func_ops)} penalty={penalty:.3f}"
        trace.setdefault("notes", []).append(note)

    if final < SCORE_FAILURE_THRESHOLD:
        log_failure(
            task_id=None,
            rule_id=str(rule),
            rule_type="composite" if isinstance(rule, CompositeRule) else "atomic",
            rule_steps=[str(s) for s in rule.steps] if isinstance(rule, CompositeRule) else [str(rule)],
            rejection_stage="scoring",
            failed_step_index=len(rule.steps) - 1 if isinstance(rule, CompositeRule) else 0,
            reason="score_below_threshold",
            color_lineage=[],
            intermediate_grids=[],
            score_trace=trace,
        )

    if return_trace:
        return trace

    if details:
        return {
            "similarity": float(base),
            "op_cost": float(_op_cost(rule)),
            "penalty": float(penalty),
            "bonus": float(bonus),
            "final_score": float(final),
        }

    return final


__all__ = [
    "score_rule",
    "preferred_rule_types",
    "STRATEGY_REGISTRY",
    "op_penalty",
    "rule_cost",
]
