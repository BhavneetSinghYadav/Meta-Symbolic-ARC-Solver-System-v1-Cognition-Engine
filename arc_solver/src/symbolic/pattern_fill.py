"""Pattern fill operator using zone overlays.

This module implements :func:`pattern_fill`, a symbolic transformation for
Advanced Reasoning Corpus (ARC) grids. Many ARC tasks require copying a
texture or motif detected in one region of the puzzle to other regions
that share a structural role. While the core DSL contains repeat and
conditional fill primitives, ``pattern_fill`` generalises these by cloning
an arbitrary subgrid pattern from a *source zone* and broadcasting it to a
*target zone*.

By detecting the smallest bounding box containing all coloured pixels of
the source zone, the operator can mirror textures, clone decorations or
propagate inferred shapes. It behaves like a combination of repeat tiling
and zone-constrained painting and is helpful when an object's interior
appears multiple times across the board.
"""

from __future__ import annotations

from typing import Sequence, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - numpy optional
    np = None  # type: ignore
    _HAS_NUMPY = False

__all__ = ["pattern_fill"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_grids(grid: Sequence[Sequence[int]], overlay: Sequence[Sequence[int]]) -> Tuple[int, int]:
    """Validate ``grid`` and ``overlay`` are 2D arrays of equal shape."""
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
            raise ValueError("overlay must be 2-dimensional")
        if overlay.shape != (h, w):
            raise ValueError("grid and overlay must have the same shape")
    elif isinstance(overlay, Sequence) and overlay and isinstance(overlay[0], Sequence):
        if len(overlay) != h or any(len(row) != w for row in overlay):
            raise ValueError("grid and overlay must have the same shape")
    else:
        raise ValueError("overlay must be a 2D list or numpy array")

    return h, w


# ---------------------------------------------------------------------------
# Main operator
# ---------------------------------------------------------------------------

def pattern_fill(grid: Sequence[Sequence[int]], source_zone_id: int, target_zone_id: int, zone_overlay: Sequence[Sequence[int]]):
    """Return ``grid`` with a pattern copied from ``source_zone_id`` to ``target_zone_id``.

    Parameters
    ----------
    grid:
        Input grid as nested list or ``numpy.ndarray`` of colour indices.
    source_zone_id:
        Integer label identifying the zone to extract the pattern from.
    target_zone_id:
        Integer label identifying the zone to fill with the pattern.
    zone_overlay:
        2D array of zone identifiers matching ``grid`` shape.

    Returns
    -------
    Same type as ``grid`` with the pattern replicated into the target zone.

    Raises
    ------
    ValueError
        If shapes mismatch, either zone ID is absent or no coloured pixels
        exist in the source zone.
    """

    h, w = _validate_grids(grid, zone_overlay)

    if _HAS_NUMPY:
        g_arr = grid if isinstance(grid, np.ndarray) else np.array(grid)
        ov_arr = zone_overlay if isinstance(zone_overlay, np.ndarray) else np.array(zone_overlay)

        if not np.any(ov_arr == source_zone_id):
            raise ValueError("source_zone_id not present in overlay")
        if not np.any(ov_arr == target_zone_id):
            raise ValueError("target_zone_id not present in overlay")

        mask = (ov_arr == source_zone_id) & (g_arr != 0)
        if not np.any(mask):
            raise ValueError("source zone contains no coloured pixels")
        rows, cols = np.where(mask)
        top, left = rows.min(), cols.min()
        bottom, right = rows.max() + 1, cols.max() + 1
        pattern = g_arr[top:bottom, left:right]
        ph, pw = pattern.shape

        target_mask = ov_arr == target_zone_id
        if not np.any(target_mask):
            raise ValueError("target zone is empty")
        tr, tc = np.where(target_mask)
        t_top, t_left = tr.min(), tc.min()

        out = g_arr.copy()
        for r, c in zip(tr, tc):
            pr = (r - t_top) % ph
            pc = (c - t_left) % pw
            out[r, c] = pattern[pr, pc]

        return out if isinstance(grid, np.ndarray) else out.tolist()

    # Manual implementation without numpy
    out = [list(row) for row in grid]
    source_cells = []
    target_cells = []
    for r in range(h):
        for c in range(w):
            if zone_overlay[r][c] == source_zone_id and grid[r][c] != 0:
                source_cells.append((r, c))
            if zone_overlay[r][c] == target_zone_id:
                target_cells.append((r, c))
    if not source_cells:
        raise ValueError("source zone contains no coloured pixels")
    if not target_cells:
        raise ValueError("target zone is empty")

    rows = [r for r, _ in source_cells]
    cols = [c for _, c in source_cells]
    top, left = min(rows), min(cols)
    bottom, right = max(rows) + 1, max(cols) + 1
    pattern = [grid[r][left:right] for r in range(top, bottom)]
    ph = bottom - top
    pw = right - left

    t_rows = [r for r, _ in target_cells]
    t_cols = [c for _, c in target_cells]
    t_top, t_left = min(t_rows), min(t_cols)

    for r, c in target_cells:
        pr = (r - t_top) % ph
        pc = (c - t_left) % pw
        out[r][c] = pattern[pr][pc]

    return out


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_grid = [[0 for _ in range(6)] for _ in range(6)]
    overlay = [
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
    ]

    base_grid[0][0] = 3
    base_grid[1][0] = 4
    base_grid[0][1] = 5

    print("Before:")
    for row in base_grid:
        print(" ".join(str(v) for v in row))

    result = pattern_fill(base_grid, 1, 2, overlay)

    print("\nAfter:")
    if _HAS_NUMPY and isinstance(result, np.ndarray):
        for row in result.tolist():
            print(" ".join(str(v) for v in row))
    else:
        for row in result:
            print(" ".join(str(v) for v in row))
