"""Grid manipulation and validation helpers."""

from __future__ import annotations

from typing import Tuple
import math

from arc_solver.src.core.grid import Grid


def rotate(grid: Grid) -> Grid:
    """Return rotated grid (placeholder)."""
    return grid


def validate_grid(grid: Grid, expected_shape: Tuple[int, int] | None = None) -> bool:
    """Return ``True`` if ``grid`` is well formed and matches ``expected_shape``."""

    if not isinstance(grid, Grid):
        return False

    shape = grid.shape()
    if expected_shape and shape != expected_shape:
        return False

    h, w = shape
    if h == 0 or w == 0:
        return False

    all_zero = True
    for row in grid.data:
        if len(row) != w:
            return False
        for val in row:
            if not isinstance(val, int) or val < 0 or val > 9:
                return False
            if val != 0:
                all_zero = False

    if all_zero:
        return False

    try:
        data_list = grid.to_list()
        if not isinstance(data_list, list) or any(not isinstance(r, list) for r in data_list):
            return False
    except Exception:
        return False

    return True


def compute_grid_entropy(grid: Grid) -> float:
    """Return the Shannon entropy of the color distribution in ``grid``."""
    counts = grid.count_colors()
    total = sum(counts.values())
    ent = 0.0
    for v in counts.values():
        if v == 0:
            continue
        p = v / total
        ent -= p * math.log2(p)
    return ent


__all__ = ["rotate", "validate_grid", "compute_grid_entropy"]
