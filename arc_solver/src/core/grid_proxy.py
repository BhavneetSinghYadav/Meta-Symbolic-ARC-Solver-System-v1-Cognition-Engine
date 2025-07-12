import numpy as np
from typing import List, Optional, Dict, Any
from collections import Counter

from .grid import Grid
from ..segment.segmenter import zone_overlay, segment_connected_regions
from ..utils.grid_utils import compute_grid_entropy


def segment_grid_into_zones(arr: np.ndarray) -> List[List[Optional[Any]]]:
    """Return zone overlay for ``arr`` using existing segmentation utils."""
    grid = Grid(arr.tolist())
    return zone_overlay(grid)


def extract_shape_metadata(arr: np.ndarray) -> List[Dict[str, Any]]:
    """Return basic shape descriptors for connected regions of ``arr``."""
    grid = Grid(arr.tolist())
    regions = segment_connected_regions(grid)
    shapes: List[Dict[str, Any]] = []
    for reg_id, cells in regions.items():
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if not rows or not cols:
            continue
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        hist = Counter(grid.get(r, c) for r, c in cells)
        shapes.append(
            {
                "zone_id": reg_id,
                "bbox": (c0, r0, c1, r1),
                "center": center,
                "dimensions": (r1 - r0 + 1, c1 - c0 + 1),
                "color_histogram": dict(hist),
            }
        )
    return shapes


class GridProxy:
    """Lightweight wrapper around ``np.ndarray`` with cached metadata."""

    def __init__(self, grid: np.ndarray):
        self.grid = np.array(grid, dtype=int).copy()
        self._zone_overlay: Optional[List[List[Optional[Any]]]] = None
        self._entropy: Optional[float] = None
        self._shapes: Optional[List[Dict[str, Any]]] = None
        self._metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def get(self, row: int, col: int, default: Any | None = None) -> Any:
        if 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]:
            return int(self.grid[row, col])
        return default

    def to_grid(self) -> Grid:
        return Grid(self.grid.tolist())

    # ------------------------------------------------------------------
    def get_zone_overlay(self) -> List[List[Optional[Any]]]:
        if self._zone_overlay is None:
            self._zone_overlay = segment_grid_into_zones(self.grid)
        return self._zone_overlay

    def get_entropy(self) -> float:
        if self._entropy is None:
            self._entropy = compute_grid_entropy(self.to_grid())
        return self._entropy

    def get_shapes(self) -> List[Dict[str, Any]]:
        if self._shapes is None:
            self._shapes = extract_shape_metadata(self.grid)
        return self._shapes

    def __getitem__(self, key):
        return self.grid[key]


__all__ = ["GridProxy", "segment_grid_into_zones", "extract_shape_metadata"]
