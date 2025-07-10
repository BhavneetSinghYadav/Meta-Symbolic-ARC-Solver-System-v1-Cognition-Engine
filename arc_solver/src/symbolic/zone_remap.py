from __future__ import annotations

"""Zone-based recoloring operator for ARC symbolic reasoning.

This module defines :func:`zone_remap`, a helper that replaces zone IDs
in a segmentation overlay with specified colours. It is useful when a
puzzle specifies fixed recolouring of segmented regions.

Parameters
----------
grid : 2D list or ``numpy.ndarray``
    Base colour grid to transform.
zone_overlay : 2D list or ``numpy.ndarray``
    Integer zone identifiers matching ``grid`` shape.
zone_to_color : dict
    Mapping from zone IDs to target ARC palette values (0-9).

Returns
-------
New grid of the same type with zones recoloured accordingly.

Notes
-----
``zone_remap`` should be preferred over the standard REPLACE operator
when transformations depend on pre-computed segment IDs rather than raw
colour values.
"""

from typing import Sequence

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - numpy optional
    np = None  # type: ignore
    _HAS_NUMPY = False

__all__ = ["zone_remap"]


def _validate_grids(
    grid: Sequence[Sequence[int]] | "np.ndarray",
    overlay: Sequence[Sequence[int]] | "np.ndarray",
) -> tuple[int, int]:
    """Validate ``grid`` and ``overlay`` shapes and return dimensions."""
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

    if _HAS_NUMPY and isinstance(overlay, np.ndarray):
        if overlay.ndim != 2:
            raise ValueError("zone_overlay must be 2-dimensional")
        if overlay.shape != (h, w):
            raise ValueError("grid and zone_overlay must have the same shape")
    elif isinstance(overlay, Sequence) and overlay and isinstance(overlay[0], Sequence):
        if len(overlay) != h or any(len(row) != w for row in overlay):
            raise ValueError("grid and zone_overlay must have the same shape")
    else:
        raise ValueError("zone_overlay must be a 2D list or numpy array")

    return h, w


def zone_remap(
    grid: Sequence[Sequence[int]] | "np.ndarray",
    zone_overlay: Sequence[Sequence[int]] | "np.ndarray",
    zone_to_color: dict,
):
    """Return a new grid with zones recoloured via ``zone_to_color``."""

    h, w = _validate_grids(grid, zone_overlay)

    # Validate mapping
    if not isinstance(zone_to_color, dict):
        raise ValueError("zone_to_color must be a dict")
    for zone_id, color in zone_to_color.items():
        if not isinstance(color, int) or not (0 <= color <= 9):
            raise ValueError("colour values must be integers 0-9")

    if _HAS_NUMPY:
        g_arr = grid if isinstance(grid, np.ndarray) else np.array(grid)
        o_arr = zone_overlay if isinstance(zone_overlay, np.ndarray) else np.array(zone_overlay)
        out = g_arr.copy()
        for zone_id, color in zone_to_color.items():
            mask = o_arr == zone_id
            if not mask.any():
                raise ValueError(f"zone id {zone_id} not present in overlay")
            out[mask] = color
        return out if isinstance(grid, np.ndarray) else out.tolist()

    # Fallback without numpy
    out = [list(row) for row in grid]
    found_zones = {z: False for z in zone_to_color}
    for r in range(h):
        for c in range(w):
            z = zone_overlay[r][c]
            if z in zone_to_color:
                out[r][c] = zone_to_color[z]
                found_zones[z] = True
    missing = [z for z, seen in found_zones.items() if not seen]
    if missing:
        raise ValueError(f"zone id(s) {missing} not present in overlay")
    return out


if __name__ == "__main__":
    base = [[0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]]

    overlay = [[1, 1, 1, 2, 2, 2],
               [1, 1, 1, 2, 2, 2],
               [1, 1, 1, 2, 2, 2],
               [3, 3, 3, 4, 4, 4],
               [3, 3, 3, 4, 4, 4],
               [3, 3, 3, 4, 4, 4]]

    mapping = {1: 3, 2: 5, 3: 7, 4: 9}

    result = zone_remap(base, overlay, mapping)

    print("Original grid:")
    for row in base:
        print(" ".join(str(v) for v in row))
    print("\nRemapped grid:")
    if _HAS_NUMPY and isinstance(result, np.ndarray):
        for row in result.tolist():
            print(" ".join(str(v) for v in row))
    else:
        for row in result:
            print(" ".join(str(v) for v in row))
