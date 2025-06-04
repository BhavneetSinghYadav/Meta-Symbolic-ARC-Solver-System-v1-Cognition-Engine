"""Visualization utilities."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt


def visualize(grid: Any) -> None:
    """Display ``grid`` using ``matplotlib`` if available."""
    try:
        data = grid.data if hasattr(grid, "data") else grid
        plt.imshow(data, interpolation="nearest")
        plt.axis("off")
        plt.show()
    except Exception:  # pragma: no cover - fallback
        print(grid)
