from __future__ import annotations

"""Pattern fill operator for symbolic grid transformations."""

from arc_solver.src.core.grid import Grid

__all__ = ["pattern_fill"]


def pattern_fill(grid: Grid, mask: Grid, pattern: Grid) -> Grid:
    """Return ``grid`` with ``pattern`` tiled over non-zero ``mask`` cells.

    Parameters
    ----------
    grid:
        Base :class:`Grid` to copy before filling.
    mask:
        Mask with same shape as ``grid``. Any non-zero cell triggers
        pattern placement centred on that coordinate.
    pattern:
        Tile pattern pasted at each mask location. If placement exceeds
        grid bounds, it is cropped.
    """

    if grid.shape() != mask.shape():
        raise ValueError("grid and mask must have the same shape")

    out = Grid(grid.to_list())

    ph, pw = pattern.shape()
    h, w = grid.shape()
    off_r = ph // 2
    off_c = pw // 2

    for r in range(h):
        for c in range(w):
            if mask.get(r, c) == 0:
                continue
            start_r = r - off_r
            start_c = c - off_c
            for pr in range(ph):
                for pc in range(pw):
                    gr = start_r + pr
                    gc = start_c + pc
                    if 0 <= gr < h and 0 <= gc < w:
                        out.set(gr, gc, pattern.get(pr, pc))

    return out
