from __future__ import annotations

"""Helper scoring utilities for structural cost and composite bonuses."""

from typing import Iterable

from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.symbolic.vocabulary import SymbolicRule
from arc_solver.src.core.grid import Grid


def structural_cost(rule: SymbolicRule | CompositeRule) -> float:
    """Return cost weighted by unique transformation types."""
    if isinstance(rule, CompositeRule):
        seen = {}
        for step in rule.steps:
            ttype = step.transformation.ttype
            if ttype not in seen:
                seen[ttype] = structural_cost(step)
        return float(sum(seen.values()))
    zone = rule.condition.get("zone", "") if rule.condition else ""
    zlen = len(str(zone))
    transform_complexity = len(str(rule.transformation.ttype.value))
    return 0.5 * zlen + transform_complexity


def composite_bonus(rule: CompositeRule, input_grid: Grid, output_grid: Grid) -> float:
    """Return a small bonus if the composite fully explains the output."""
    try:
        pred = rule.simulate(input_grid)
    except Exception:
        return 0.0
    if pred.compare_to(output_grid) == 1.0:
        return 0.05 * len(rule.steps)
    return 0.0


__all__ = ["structural_cost", "composite_bonus"]
