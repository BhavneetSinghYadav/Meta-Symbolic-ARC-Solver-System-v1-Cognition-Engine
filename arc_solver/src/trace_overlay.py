from __future__ import annotations

"""Build diagnostic overlays showing prediction uncertainty."""

from typing import List, Tuple

import numpy as np

from arc_solver.src.core.grid import Grid


def build_uncertainty_trace(weighted_preds: List[Tuple[Grid, float]]) -> Grid:
    """Return grid of uncertainty scores per cell."""
    grid_shape = weighted_preds[0][0].shape()
    accumulator = np.zeros(grid_shape + (10,))

    for grid, weight in weighted_preds:
        h, w = grid.shape()
        for i in range(h):
            for j in range(w):
                val = grid.get(i, j)
                accumulator[i, j, val] += weight

    conf = np.max(accumulator, axis=-1)
    total = np.sum(accumulator, axis=-1)
    confidence = np.divide(conf, total, out=np.zeros_like(conf), where=total>0)
    uncertainty = 1.0 - confidence
    return Grid(uncertainty.tolist())


__all__ = ["build_uncertainty_trace"]
