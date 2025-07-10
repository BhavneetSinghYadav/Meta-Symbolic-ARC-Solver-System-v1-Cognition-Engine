"""Standalone symbolic transformation operators."""

from __future__ import annotations

from arc_solver.src.core.grid import Grid

__all__ = ["mirror_tile"]


def mirror_tile(grid: Grid, axis: str, count: int) -> Grid:
    """Return grid tiled ``count`` times while mirroring every other tile.

    Parameters
    ----------
    grid:
        Input :class:`Grid` to tile.
    axis:
        ``"horizontal"`` or ``"vertical"``. Determines both the tiling
        direction and the axis of mirroring.
    count:
        Number of tiles to produce. Must be >= 2.
    """

    if count <= 1:
        return grid

    axis = axis.lower()
    if axis not in {"horizontal", "vertical"}:
        raise ValueError("axis must be 'horizontal' or 'vertical'")

    h, w = grid.shape()

    if axis == "horizontal":
        new_w = w * count
        new_data = [[0 for _ in range(new_w)] for _ in range(h)]
        for idx in range(count):
            tile = grid.flip_horizontal() if idx % 2 == 1 else grid
            for r in range(h):
                for c in range(w):
                    new_data[r][idx * w + c] = tile.get(r, c)
    else:  # axis == "vertical"
        new_h = h * count
        new_data = [[0 for _ in range(w)] for _ in range(new_h)]
        for idx in range(count):
            tile = grid.flip_vertical() if idx % 2 == 1 else grid
            for r in range(h):
                for c in range(w):
                    new_data[idx * h + r][c] = tile.get(r, c)

    return Grid(new_data)
