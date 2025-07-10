"""Line drawing utilities for symbolic grid manipulation.

This module defines :func:`draw_line`, a helper used to connect points on a
2D grid with a 4-connected straight line. The operator is intended for use
within the ARC solver's symbolic DSL when tasks require linking shapes or
constructing edges programmatically.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - numpy not available
    np = None  # type: ignore
    _HAS_NUMPY = False

__all__ = ["draw_line"]

Point = Tuple[int, int]


def _bresenham_path(p1: Point, p2: Point) -> List[Point]:
    """Return list of coordinates along an 8-connected Bresenham line."""
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x2 >= x1 else -1
    sy = 1 if y2 >= y1 else -1
    err = dx - dy
    x, y = x1, y1
    points = []
    while True:
        points.append((x, y))
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points


def _make_4_connected(path: Iterable[Point]) -> List[Point]:
    """Expand an 8-connected path so consecutive points share an edge."""
    path = list(path)
    if not path:
        return []
    result = [path[0]]
    for target in path[1:]:
        curr = result[-1]
        dr = 1 if target[0] > curr[0] else -1
        dc = 1 if target[1] > curr[1] else -1
        while curr[0] != target[0] or curr[1] != target[1]:
            if curr[0] != target[0]:
                curr = (curr[0] + dr, curr[1])
                result.append(curr)
            if curr[1] != target[1]:
                curr = (curr[0], curr[1] + dc)
                result.append(curr)
    return result


def _validate_grid(grid: Sequence[Sequence[int]]) -> Tuple[int, int]:
    """Validate grid shape and return (height, width)."""
    if _HAS_NUMPY and isinstance(grid, np.ndarray):
        if grid.ndim != 2:
            raise ValueError("grid must be 2-dimensional")
        h, w = grid.shape
    elif isinstance(grid, Sequence) and grid and isinstance(grid[0], Sequence):
        h = len(grid)
        w = len(grid[0])
        for row in grid:
            if len(row) != w:
                raise ValueError("all grid rows must have the same length")
    else:
        raise ValueError("grid must be a 2D list or numpy array")
    return h, w


def draw_line(grid: Sequence[Sequence[int]], point1: Point, point2: Point, color: int):
    """Draw a 4-connected line on ``grid`` between ``point1`` and ``point2``.

    Parameters
    ----------
    grid:
        2D list or ``numpy.ndarray`` representing ARC cell colours.
    point1:
        ``(row, col)`` start coordinate within grid bounds.
    point2:
        ``(row, col)`` end coordinate within grid bounds.
    color:
        Integer colour index ``0``-``9`` from the ARC palette.

    Returns
    -------
    Same type as ``grid`` with the line drawn. A copy is modified so the
    input grid remains unchanged.

    Notes
    -----
    Drawing lines programmatically helps connect shapes, reconstruct edges or
    build paths during ARC task solving.
    """

    if not isinstance(point1, tuple) or not isinstance(point2, tuple) or len(point1) != 2 or len(point2) != 2:
        raise ValueError("point1 and point2 must be 2-tuples of integers")
    if not all(isinstance(v, int) for v in point1 + point2):
        raise ValueError("point1 and point2 must contain integers")
    if not isinstance(color, int) or not (0 <= color <= 9):
        raise ValueError("color must be an integer between 0 and 9")

    h, w = _validate_grid(grid)

    r1, c1 = point1
    r2, c2 = point2
    if not (0 <= r1 < h and 0 <= c1 < w and 0 <= r2 < h and 0 <= c2 < w):
        raise ValueError("points must lie within grid bounds")

    if _HAS_NUMPY and isinstance(grid, np.ndarray):
        out = grid.copy()
    else:
        out = [list(row) for row in grid]

    path8 = _bresenham_path(point1, point2)
    path4 = _make_4_connected(path8)

    for r, c in path4:
        if _HAS_NUMPY and isinstance(out, np.ndarray):
            out[r, c] = color
        else:
            out[r][c] = color

    return out


if __name__ == "__main__":
    demo_grid = [[0 for _ in range(10)] for _ in range(10)]
    result = draw_line(demo_grid, (1, 1), (8, 6), 2)
    if _HAS_NUMPY and isinstance(result, np.ndarray):
        print(result)
    else:
        for row in result:
            print(" ".join(str(v) for v in row))
