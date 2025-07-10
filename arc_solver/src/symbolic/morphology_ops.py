"""Morphological operations constrained by segmentation zones."""

from __future__ import annotations

from typing import Sequence, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - numpy optional
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from scipy.ndimage import binary_dilation, binary_erosion
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy optional
    binary_dilation = binary_erosion = None  # type: ignore
    _HAS_SCIPY = False


__all__ = ["dilate_zone", "erode_zone"]

Neighbour = Tuple[int, int]
_DIRECTIONS: Tuple[Neighbour, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


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


def _overlay_to_mask(overlay: Sequence[Sequence[int]], zone_id: int) -> "np.ndarray":
    if not _HAS_NUMPY:
        raise RuntimeError("numpy required for mask conversion")
    if isinstance(overlay, np.ndarray):
        return overlay == zone_id
    arr = np.array(overlay)
    return arr == zone_id


def dilate_zone(grid: Sequence[Sequence[int]], zone_id: int, zone_overlay: Sequence[Sequence[int]]):
    """Dilate the pixels of ``zone_id`` by one cell inside ``zone_overlay``.

    Parameters
    ----------
    grid:
        2D list or ``numpy.ndarray`` of colour indices.
    zone_id:
        Integer zone label to dilate.
    zone_overlay:
        Overlay labeling each cell with a zone identifier. Must match ``grid`` shape.

    Returns
    -------
    Modified copy of ``grid`` with pixels of the specified zone expanded by one cell.

    Notes
    -----
    Dilation is restricted to cells with ``zone_id`` in ``zone_overlay`` so
    neighbouring zones remain untouched. Useful for symbolic reasoning where
    transformations apply only within designated ARC segments.
    """

    h, w = _validate_grids(grid, zone_overlay)

    if not any(zone_id == (zone_overlay[r][c] if not (_HAS_NUMPY and isinstance(zone_overlay, np.ndarray)) else zone_overlay[r, c]) for r in range(h) for c in range(w)):
        raise ValueError("zone_id not present in overlay")

    if _HAS_NUMPY:
        g_arr = grid if isinstance(grid, np.ndarray) else np.array(grid)
        mask = _overlay_to_mask(zone_overlay, zone_id)
        if _HAS_SCIPY:
            struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
            new_mask = binary_dilation(mask, structure=struct)
        else:
            new_mask = mask.copy()
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        for dr, dc in _DIRECTIONS:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                new_mask[nr, nc] = True
        out = g_arr.copy()
        for r in range(h):
            for c in range(w):
                if new_mask[r, c] and not mask[r, c]:
                    for dr, dc in _DIRECTIONS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                            out[r, c] = g_arr[nr, nc]
                            break
        return out if isinstance(grid, np.ndarray) else out.tolist()

    # Fallback manual implementation without numpy
    out = [list(row) for row in grid]
    mask = [[zone_overlay[r][c] == zone_id for c in range(w)] for r in range(h)]
    new_mask = [row[:] for row in mask]
    for r in range(h):
        for c in range(w):
            if mask[r][c]:
                for dr, dc in _DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        new_mask[nr][nc] = True
    for r in range(h):
        for c in range(w):
            if new_mask[r][c] and not mask[r][c]:
                for dr, dc in _DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr][nc]:
                        out[r][c] = grid[nr][nc]
                        break
    return out


def erode_zone(grid: Sequence[Sequence[int]], zone_id: int, zone_overlay: Sequence[Sequence[int]]):
    """Erode ``zone_id`` by removing boundary pixels within ``zone_overlay``.

    Parameters
    ----------
    grid:
        2D list or ``numpy.ndarray`` representing colours.
    zone_id:
        Integer zone label to erode.
    zone_overlay:
        Overlay with same dimensions as ``grid`` specifying zone IDs.

    Returns
    -------
    Modified copy of ``grid`` with a 1-pixel outer layer removed from the zone.

    Notes
    -----
    Only cells labelled ``zone_id`` are affected. Pixels eroded away become 0
    (background) so other regions remain untouched.
    """

    h, w = _validate_grids(grid, zone_overlay)

    if not any(zone_id == (zone_overlay[r][c] if not (_HAS_NUMPY and isinstance(zone_overlay, np.ndarray)) else zone_overlay[r, c]) for r in range(h) for c in range(w)):
        raise ValueError("zone_id not present in overlay")

    if _HAS_NUMPY:
        g_arr = grid if isinstance(grid, np.ndarray) else np.array(grid)
        mask = _overlay_to_mask(zone_overlay, zone_id)
        if _HAS_SCIPY:
            struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
            new_mask = binary_erosion(mask, structure=struct)
        else:
            new_mask = mask.copy()
            temp = [[False] * w for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    if mask[r, c] and all(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc] for dr, dc in _DIRECTIONS):
                        temp[r][c] = True
            new_mask = np.array(temp, dtype=bool)
        out = g_arr.copy()
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not new_mask[r, c]:
                    out[r, c] = 0
        return out if isinstance(grid, np.ndarray) else out.tolist()

    # Fallback without numpy
    out = [list(row) for row in grid]
    mask = [[zone_overlay[r][c] == zone_id for c in range(w)] for r in range(h)]
    new_mask = [[False] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if mask[r][c] and all(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr][c + dc] for dr, dc in _DIRECTIONS):
                new_mask[r][c] = True
    for r in range(h):
        for c in range(w):
            if mask[r][c] and not new_mask[r][c]:
                out[r][c] = 0
    return out


if __name__ == "__main__":
    base = [[0 for _ in range(10)] for _ in range(10)]
    for r in range(3, 7):
        for c in range(3, 7):
            base[r][c] = 1
    overlay = [[0 for _ in range(10)] for _ in range(10)]
    for r in range(2, 8):
        for c in range(2, 8):
            overlay[r][c] = 1
    print("Original:")
    for row in base:
        print(" ".join(str(v) for v in row))
    dilated = dilate_zone(base, 1, overlay)
    print("\nDilated:")
    for row in dilated:
        print(" ".join(str(v) for v in row))
    eroded = erode_zone(dilated, 1, overlay)
    print("\nEroded:")
    for row in eroded:
        print(" ".join(str(v) for v in row))
