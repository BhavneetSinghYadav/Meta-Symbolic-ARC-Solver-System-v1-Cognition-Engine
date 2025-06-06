from __future__ import annotations

"""Low-level grid utilities."""

from typing import List

from .grid import Grid


def compute_conflict_map(before: Grid, after: Grid) -> List[List[int]]:
    """Return map of cell conflicts between ``before`` and ``after`` grids."""
    h1, w1 = before.shape()
    h2, w2 = after.shape()
    h = max(h1, h2)
    w = max(w1, w2)
    conflict: List[List[int]] = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if before.get(r, c, None) != after.get(r, c, None):
                conflict[r][c] = 1
    return conflict


__all__ = ["compute_conflict_map"]

