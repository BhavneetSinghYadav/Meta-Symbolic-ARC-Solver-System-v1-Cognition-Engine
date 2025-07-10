from __future__ import annotations

"""Utilities for tracking colour states across composite steps."""

from typing import List, Set

from arc_solver.src.core.grid import Grid


def _colors(grid: Grid) -> Set[int]:
    return {v for row in grid.data for v in row}


class ColorLineage:
    """Track colour sets after each simulation step."""

    def __init__(self, grid: Grid) -> None:
        self.states: List[Set[int]] = [_colors(grid)]

    def record(self, grid: Grid) -> None:
        self.states.append(_colors(grid))

    def final(self) -> Set[int]:
        return self.states[-1] if self.states else set()

    def state(self, idx: int) -> Set[int]:
        if 0 <= idx < len(self.states):
            return self.states[idx]
        return set()


__all__ = ["ColorLineage"]
