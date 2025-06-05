from __future__ import annotations

"""Basic grid feature extractor used for optional conditioning."""

from typing import List

import numpy as np

from arc_solver.src.core.grid import Grid


def color_histogram(grid: Grid, dim: int = 16) -> List[float]:
    """Return normalized color histogram of ``grid``."""

    counts = grid.count_colors()
    total = grid.shape()[0] * grid.shape()[1]
    vec = np.zeros(dim, dtype=float)
    for color, count in counts.items():
        vec[color % dim] += count / float(total)
    return vec.tolist()


__all__ = ["color_histogram"]
