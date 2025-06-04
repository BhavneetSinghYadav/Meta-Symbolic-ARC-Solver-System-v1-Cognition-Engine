"""Grid segmentation utilities using symbolic representations."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from ..core.grid import Grid
from ..symbolic.vocabulary import Symbol, SymbolType


# ---------------------------------------------------------------------------
# Fixed zone segmentation
# ---------------------------------------------------------------------------

def segment_fixed_zones(grid: Grid) -> Dict[Tuple[int, int], Symbol]:
    """Return a mapping of grid coordinates to symbolic zone labels.

    The grid is divided into predefined regions such as the top-left corner,
    center square, and central bands. Only cells belonging to one of these
    regions are included in the returned dictionary.
    """

    height, width = grid.shape()
    zones: Dict[Tuple[int, int], Symbol] = {}

    top_th = height // 3
    left_th = width // 3
    bottom_th = 2 * height // 3
    right_th = 2 * width // 3

    center_row = height // 2
    center_col = width // 2

    for r in range(height):
        for c in range(width):
            if r < top_th and c < left_th:
                zones[(r, c)] = Symbol(SymbolType.ZONE, "TopLeft")
            elif r >= bottom_th and c >= right_th:
                zones[(r, c)] = Symbol(SymbolType.ZONE, "BottomRight")
            elif top_th <= r < bottom_th and left_th <= c < right_th:
                zones[(r, c)] = Symbol(SymbolType.ZONE, "Center")
            elif abs(c - center_col) <= 1:
                zones[(r, c)] = Symbol(SymbolType.ZONE, "VerticalMidBand")
            elif abs(r - center_row) <= 1:
                zones[(r, c)] = Symbol(SymbolType.ZONE, "HorizontalMidBand")

    return zones


# ---------------------------------------------------------------------------
# Connected region segmentation
# ---------------------------------------------------------------------------

def segment_connected_regions(grid: Grid) -> Dict[int, List[Tuple[int, int]]]:
    """Label connected regions of equal color.

    A simple flood fill (BFS) is used to traverse neighbouring cells with the
    same color. Cells are considered connected if they share an edge.
    Each region receives an incrementing integer identifier.
    """

    height, width = grid.shape()
    visited = [[False for _ in range(width)] for _ in range(height)]
    regions: Dict[int, List[Tuple[int, int]]] = {}
    region_id = 0

    for r in range(height):
        for c in range(width):
            if visited[r][c]:
                continue
            color = grid.get(r, c)
            queue: deque[Tuple[int, int]] = deque([(r, c)])
            visited[r][c] = True
            cells: List[Tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and not visited[nr][nc] and grid.get(nr, nc) == color:
                        visited[nr][nc] = True
                        queue.append((nr, nc))

            regions[region_id] = cells
            region_id += 1

    return regions


# ---------------------------------------------------------------------------
# Overlay utilities
# ---------------------------------------------------------------------------

def zone_overlay(grid: Grid) -> List[List[Optional[Symbol]]]:
    """Return zone label overlay matrix for ``grid``."""
    zones = segment_fixed_zones(grid)
    return assign_zone_labels(grid, zones)


def label_connected_regions(grid: Grid) -> List[List[Optional[int]]]:
    """Return integer region labels overlay for connected components."""
    regions = segment_connected_regions(grid)
    height, width = grid.shape()
    overlay: List[List[Optional[int]]] = [[None for _ in range(width)] for _ in range(height)]
    for reg_id, cells in regions.items():
        for r, c in cells:
            overlay[r][c] = reg_id
    return overlay


# ---------------------------------------------------------------------------
# Zone overlay assignment
# ---------------------------------------------------------------------------

def assign_zone_labels(grid: Grid, zones: Dict[Tuple[int, int], Symbol]) -> List[List[Optional[Symbol]]]:
    """Create a matrix overlay from zone assignments."""

    height, width = grid.shape()
    overlay: List[List[Optional[Symbol]]] = [[None for _ in range(width)] for _ in range(height)]

    for (r, c), sym in zones.items():
        if 0 <= r < height and 0 <= c < width:
            overlay[r][c] = sym

    return overlay


__all__ = [
    "segment_fixed_zones",
    "segment_connected_regions",
    "assign_zone_labels",
    "zone_overlay",
    "label_connected_regions",
]
