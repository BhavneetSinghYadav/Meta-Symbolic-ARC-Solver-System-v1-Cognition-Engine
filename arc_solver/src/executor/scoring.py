from __future__ import annotations

"""Rule scoring and strategy utilities for the executor.

This module evaluates symbolic rules by comparing their predictions against
expected grids.  Older revisions heavily penalised long composite programs
using raw structural cost.  The new logic reduces this bias by weighting the
penalty per unique transformation type.  A small bonus is still granted to
perfect composites to encourage valid chains.
"""

from typing import Dict, List, Tuple

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

# Relative weights for each transformation type when computing structural cost.
OP_WEIGHTS: Dict[TransformationType, float] = {
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
    TransformationType.FUNCTIONAL: 1.3,
    TransformationType.COMPOSITE: 1.0,
}


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


def _op_cost(rule: SymbolicRule | CompositeRule) -> float:
    """Return weighted cost of unique operations used by ``rule``."""

    if isinstance(rule, CompositeRule):
        ops = {step.transformation.ttype for step in rule.steps}
    elif rule.transformation.ttype is TransformationType.COMPOSITE:
        step_names = rule.transformation.params.get("steps", [])
        ops = {TransformationType[s] if isinstance(s, str) else s for s in step_names}
    else:
        ops = {rule.transformation.ttype}

    if not ops:
        return 0.0

    return float(sum(OP_WEIGHTS.get(op, 1.0) for op in ops))


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

    # Complexity penalty weighted by unique operation types
    penalty = 0.006 * _op_cost(rule)

    # Composite bonus only when the rule perfectly matches the output
    bonus = 0.2 if isinstance(rule, CompositeRule) and base >= 0.95 else 0.0

    final = base - penalty + bonus

    trace = {
        "similarity": float(base),
        "unique_ops": int(_unique_ops(rule)),
        "op_cost": float(_op_cost(rule)),
        "penalty": float(penalty),
        "bonus": float(bonus),
        "final_score": float(final),
        "rule_steps": [
            step.transformation.ttype.value for step in rule.steps
        ]
        if isinstance(rule, CompositeRule)
        else [rule.transformation.ttype.value],
    }

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


__all__ = ["score_rule", "preferred_rule_types", "STRATEGY_REGISTRY"]
