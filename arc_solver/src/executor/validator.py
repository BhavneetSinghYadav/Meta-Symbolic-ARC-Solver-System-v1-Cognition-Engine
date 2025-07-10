from __future__ import annotations

"""Validation helpers for composite rule colour dependencies."""

from typing import List, Set

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import SymbolicRule, SymbolType
from arc_solver.src.symbolic.rule_language import CompositeRule
from .failure_logger import log_failure


def simulate_step(rule_step: SymbolicRule | CompositeRule, grid: Grid) -> Grid:
    """Return ``grid`` after applying ``rule_step`` without checks."""
    from .simulator import safe_apply_rule  # local import to avoid circularity

    if isinstance(rule_step, CompositeRule):
        out = Grid([row[:] for row in grid.data])
        for st in rule_step.steps:
            out = simulate_step(st, out)
        return out
    return safe_apply_rule(rule_step, grid, perform_checks=False)


def get_color_set(grid: Grid) -> Set[int]:
    """Return set of colors present in ``grid`` excluding background ``0``."""
    return {v for row in grid.data for v in row if v != 0}


def validate_color_dependencies(
    rule_chain: List[SymbolicRule | CompositeRule],
    input_grid: Grid,
    *,
    debug: bool = False,
    rule_id: str | None = None,
) -> bool:
    """Validate that ``rule_chain`` preserves all required colours.

    The chain is simulated step by step and the colour set after each step is
    recorded. Only the final colour set is compared against the colours required
    by the chain's source symbols. Intermediate colour removals are allowed.
    """
    working = Grid([row[:] for row in input_grid.data])
    color_lineage: List[Set[int]] = []
    required: Set[int] = set()

    for step in rule_chain:
        color_lineage.append(get_color_set(working))
        if isinstance(step, CompositeRule):
            sub_steps = step.steps
        else:
            sub_steps = [step]
        for st in sub_steps:
            for sym in st.source:
                if sym.type is SymbolType.COLOR:
                    try:
                        val = int(sym.value)
                        if val != 0:
                            required.add(val)
                    except ValueError:
                        pass
            working = simulate_step(st, working)
    color_lineage.append(get_color_set(working))

    final_colors = color_lineage[-1]
    missing = {c for c in required if c not in final_colors}
    if missing:
        divergence = None
        for i in range(len(color_lineage) - 1):
            before = color_lineage[i]
            after = color_lineage[i + 1]
            if any(c in before and c not in after for c in missing):
                divergence = i
                break
        log_failure(
            {
                "rule": rule_id or "chain",
                "reason": "missing_final_colors",
                "missing": sorted(missing),
                "divergence_step": divergence,
                "color_lineage": [sorted(list(s)) for s in color_lineage],
            }
        )
        return False

    if debug:
        log_failure(
            {
                "rule": rule_id or "chain",
                "reason": "debug_lineage",
                "color_lineage": [sorted(list(s)) for s in color_lineage],
            }
        )
    return True


__all__ = ["validate_color_dependencies", "simulate_step", "get_color_set"]
