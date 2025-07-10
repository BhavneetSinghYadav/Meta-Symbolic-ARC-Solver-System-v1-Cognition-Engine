from __future__ import annotations

"""Entropy helpers for scoring heuristics."""

from collections import Counter
from math import log2

from arc_solver.src.core.grid import Grid


def grid_color_entropy(grid: Grid) -> float:
    """Return normalized Shannon entropy of the color distribution in ``grid``.

    Cells with value ``0`` or negative numbers are ignored. The result is
    scaled by the maximum possible entropy given the number of colors present,
    yielding a value in the ``[0.0, 1.0]`` range.
    """

    counts: Counter[int] = Counter()
    total = 0
    for row in grid.data:
        for val in row:
            if val <= 0:
                continue
            counts[val] += 1
            total += 1

    if total == 0:
        return 0.0

    n_colors = len(counts)
    if n_colors <= 1:
        return 0.0

    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * log2(p)

    max_entropy = log2(n_colors)
    return entropy / max_entropy if max_entropy else 0.0


__all__ = ["grid_color_entropy"]
