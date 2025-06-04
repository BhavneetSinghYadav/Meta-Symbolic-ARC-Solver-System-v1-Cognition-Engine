from __future__ import annotations

from typing import List, Tuple

from arc_solver.src.core.grid import Grid
from arc_solver.src.segment.segmenter import zone_overlay


class AttentionMask:
    """Binary mask specifying active grid cells."""

    def __init__(self, shape: Tuple[int, int]):
        h, w = shape
        self.mask: List[List[bool]] = [[True for _ in range(w)] for _ in range(h)]

    def focus_zone(self, grid: Grid, zone_label: str) -> None:
        """Activate only cells belonging to ``zone_label``."""
        overlay = zone_overlay(grid)
        h, w = grid.shape()
        self.mask = [
            [overlay[r][c] is not None and overlay[r][c].value == zone_label for c in range(w)]
            for r in range(h)
        ]

    def as_list(self) -> List[List[bool]]:
        return [row[:] for row in self.mask]


def zone_to_mask(grid: Grid, zone_label: str) -> List[List[bool]]:
    mask = AttentionMask(grid.shape())
    mask.focus_zone(grid, zone_label)
    return mask.as_list()


__all__ = ["AttentionMask", "zone_to_mask"]
