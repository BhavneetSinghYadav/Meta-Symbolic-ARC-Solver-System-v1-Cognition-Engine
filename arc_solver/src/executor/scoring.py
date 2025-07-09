from __future__ import annotations

"""Rule scoring and strategy utilities for the executor."""

from typing import Dict, List, Tuple

from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
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


def score_rule(
    input_grid: Grid,
    output_grid: Grid,
    rule: SymbolicRule | CompositeRule,
    *,
    prefer_composites: bool = False,
) -> float:
    """Return heuristic score of ``rule`` for transforming ``input_grid`` to ``output_grid``.

    ``prefer_composites`` can be toggled to soften complexity penalties for multi step
    rules when experimenting with ranking strategies.
    """

    try:
        if isinstance(rule, CompositeRule):
            pred = rule.simulate(input_grid)
            steps = len(rule.steps)
        else:
            pred = simulate_rules(input_grid, [rule])
            steps = 1
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

    # Penalize complex rules (softer for composites)
    from arc_solver.src.abstractions.rule_generator import rule_cost
    complexity = rule_cost(rule)
    penalty = 0.05 * complexity
    if isinstance(rule, CompositeRule):
        penalty /= steps if prefer_composites else 1.0

    final = base - penalty
    if final < 0.0:
        final = 0.0
    if final > 1.0:
        final = 1.0
    return final


__all__ = ["score_rule", "preferred_rule_types", "STRATEGY_REGISTRY"]
