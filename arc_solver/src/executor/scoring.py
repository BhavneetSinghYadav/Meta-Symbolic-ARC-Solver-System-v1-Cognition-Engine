from __future__ import annotations

"""Rule scoring and strategy utilities for the executor.

This module evaluates symbolic rules by comparing their predictions against
expected grids.  The previous scoring heavily penalised long composite programs
based on their structural cost which resulted in perfect multi-step solutions
being discarded.  The scoring logic has been simplified so that only the number
of *unique* transformation types contributes to the complexity penalty.  A
small bonus is granted to perfect composites to encourage valid chains.
"""

from typing import Dict, List, Tuple

from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.executor.failure_logger import log_failure
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.symbolic.vocabulary import SymbolicRule


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
    return 1


def score_rule(
    input_grid: Grid,
    output_grid: Grid,
    rule: SymbolicRule | CompositeRule,
    *,
    prefer_composites: bool = False,
    details: bool = False,
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
    base = 0.6 * after_pixel + 0.3 * zone_match + 0.1 * shape_bonus

    # Reward improvement over the input similarity
    improvement = after_pixel - before_pixel
    if improvement > 0:
        base += 0.2 * improvement

    # Complexity penalty based on unique operation types
    penalty = 0.005 * _unique_ops(rule)

    # Composite bonus only when the rule perfectly matches the output
    bonus = 0.2 if isinstance(rule, CompositeRule) and base == 1.0 else 0.0

    final = base - penalty + bonus

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
        )

    if details:
        return {
            "similarity": float(base),
            "penalty": float(penalty),
            "bonus": float(bonus),
            "final_score": float(final),
        }

    return final


__all__ = ["score_rule", "preferred_rule_types", "STRATEGY_REGISTRY"]
