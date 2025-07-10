"""Grid segmentation utilities using symbolic representations."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage.morphology import skeletonize

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
# Morphological segmentation
# ---------------------------------------------------------------------------

def segment_morphological_regions(grid: Grid) -> Dict[Tuple[int, int], Symbol]:
    """Return skeleton-based zones for each non-zero color."""

    data = np.array(grid.data)
    zones: Dict[Tuple[int, int], Symbol] = {}
    for color in np.unique(data):
        if color == 0:
            continue
        mask = data == color
        if not np.any(mask):
            continue
        skel = skeletonize(mask)
        label = Symbol(SymbolType.ZONE, f"Skeleton{int(color)}")
        for r, c in zip(*np.nonzero(skel)):
            zones[(int(r), int(c))] = label
    return zones


# ---------------------------------------------------------------------------
# Overlay utilities
# ---------------------------------------------------------------------------

def zone_overlay(
    grid: Grid, *, use_morphology: bool | None = False
) -> List[List[Optional[Symbol]]]:
    """Return zone label overlay matrix for ``grid``.

    If ``use_morphology`` is True, skeleton-based zones are merged with the
    fixed zone layout.
    """

    zones = segment_fixed_zones(grid)
    overlay = assign_zone_labels(grid, zones)
    if use_morphology:
        morph = compute_skeleton_overlay(grid)
        overlay = merge_overlays(overlay, morph)
    return overlay


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


def expand_zone_overlay(
    overlay: List[List[Optional[Symbol]]], label: str
) -> List[List[Optional[Symbol]]]:
    """Return ``overlay`` with ``label`` dilated by one cell."""
    height = len(overlay)
    width = len(overlay[0]) if height else 0
    expanded = [row[:] for row in overlay]
    for r in range(height):
        for c in range(width):
            if overlay[r][c] is not None and overlay[r][c].value == label:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and expanded[nr][nc] is None:
                        expanded[nr][nc] = Symbol(SymbolType.ZONE, label)
    return expanded


def compute_skeleton_overlay(grid: Grid) -> List[List[Optional[Symbol]]]:
    """Return overlay highlighting skeleton cells for each color."""

    zones = segment_morphological_regions(grid)
    return assign_zone_labels(grid, zones)


def merge_overlays(
    base: List[List[Optional[Symbol]]],
    other: List[List[Optional[Symbol]]],
) -> List[List[Optional[Symbol]]]:
    """Merge ``other`` into ``base`` preferring labels from ``other``."""

    height = len(base)
    width = len(base[0]) if height else 0
    merged = [row[:] for row in base]
    for r in range(height):
        for c in range(width):
            if other[r][c] is not None:
                merged[r][c] = other[r][c]
    return merged


__all__ = [
    "segment_fixed_zones",
    "segment_connected_regions",
    "assign_zone_labels",
    "zone_overlay",
    "segment_morphological_regions",
    "compute_skeleton_overlay",
    "merge_overlays",
    "label_connected_regions",
    "expand_zone_overlay",
]
