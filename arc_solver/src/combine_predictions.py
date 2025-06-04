from __future__ import annotations

"""Utilities for combining weighted grid predictions."""

from typing import List, Tuple

import numpy as np

from arc_solver.src.core.grid import Grid


def combine_predictions(weighted_preds: List[Tuple[Grid, float]]) -> Grid:
    """Combine predictions by weighted voting over cell values."""
    grid_shape = weighted_preds[0][0].shape()
    accumulator = np.zeros(grid_shape + (10,))

    for grid, weight in weighted_preds:
        h, w = grid.shape()
        for i in range(h):
            for j in range(w):
                val = grid.get(i, j)
                accumulator[i, j, val] += weight

    final = np.argmax(accumulator, axis=-1)
    return Grid(final.tolist())


__all__ = ["combine_predictions"]
