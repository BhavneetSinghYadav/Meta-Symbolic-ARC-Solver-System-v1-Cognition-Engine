from __future__ import annotations

"""Rule coverage utilities."""

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import SymbolicRule


def rule_coverage(rule: SymbolicRule, grid: Grid) -> int:
    """Return the number of cells that would change if ``rule`` is applied."""
    from arc_solver.src.executor import simulator as _sim

    if not _sim.validate_rule_application(rule, grid):
        return 0
    try:
        tentative = _sim.check_symmetry_break(rule, grid)
    except _sim.ReflexOverrideException:
        return 0
    h, w = grid.shape()
    th, tw = tentative.shape()
    max_h = max(h, th)
    max_w = max(w, tw)
    count = 0
    for r in range(max_h):
        for c in range(max_w):
            if grid.get(r, c) != tentative.get(r, c):
                count += 1
    return count


__all__ = ["rule_coverage"]
