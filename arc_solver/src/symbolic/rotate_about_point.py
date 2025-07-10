"""Grid rotation around an arbitrary pivot."""

from __future__ import annotations

from typing import Tuple

from arc_solver.src.core.grid import Grid

__all__ = ["rotate_about_point"]


def rotate_about_point(grid: Grid, center: Tuple[int, int], angle: int) -> Grid:
    """Return ``grid`` rotated ``angle`` degrees about ``center``.

    Parameters
    ----------
    grid:
        Input :class:`Grid` to rotate.
    center:
        (row, col) coordinate about which the grid is rotated.
    angle:
        Rotation angle in degrees. Must be one of ``90``, ``180`` or ``270``.
    """

    if angle not in {90, 180, 270}:
        raise ValueError("angle must be 90, 180, or 270 degrees")

    h, w = grid.shape()
    cx, cy = center  # center row and column
    new_data = [[0 for _ in range(w)] for _ in range(h)]

    for r in range(h):
        for c in range(w):
            val = grid.get(r, c)
            if angle == 90:
                nr = cx - (c - cy)
                nc = cy + (r - cx)
            elif angle == 180:
                nr = 2 * cx - r
                nc = 2 * cy - c
            else:  # angle == 270
                nr = cx + (c - cy)
                nc = cy - (r - cx)

            if 0 <= nr < h and 0 <= nc < w:
                new_data[nr][nc] = val

    return Grid(new_data)
