"""Grid utilities for ARC-style problems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Grid:
    """2D grid of integer color values used by the ARC solver."""

    data: List[List[int]]
    overlay: Optional[List[List[Any]]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("Grid cannot be empty")
        row_len = len(self.data[0])
        for row in self.data:
            if len(row) != row_len:
                raise ValueError("All rows must have the same length")

    def get(self, row: int, col: int) -> int:
        """Return the color value at the specified cell."""
        return self.data[row][col]

    def set(self, row: int, col: int, value: int) -> None:
        """Set the color value at the specified cell."""
        self.data[row][col] = value

    def crop(self, top: int, left: int, height: int, width: int) -> "Grid":
        """Return a subgrid defined by the rectangle (top, left, height, width)."""
        cropped = [r[left : left + width] for r in self.data[top : top + height]]
        return Grid(cropped)

    def shape(self) -> Tuple[int, int]:
        """Return the grid shape as (height, width)."""
        return len(self.data), len(self.data[0])

    def count_colors(self) -> Dict[int, int]:
        """Return a mapping from color value to number of occurrences."""
        counts: Dict[int, int] = {}
        for row in self.data:
            for value in row:
                counts[value] = counts.get(value, 0) + 1
        return counts

    def visualize(self) -> None:
        """Pretty-print the grid values."""
        for row in self.data:
            print(" ".join(str(v) for v in row))

    # Advanced operations -------------------------------------------------

    def rotate90(self, times: int = 1) -> "Grid":
        """Return a new grid rotated 90 degrees clockwise ``times`` times."""
        times = times % 4
        result = self.data
        for _ in range(times):
            result = [list(row) for row in zip(*result[::-1])]
        return Grid(result)

    def flip_horizontal(self) -> "Grid":
        """Return a new grid flipped horizontally."""
        flipped = [list(reversed(row)) for row in self.data]
        return Grid(flipped)

    def to_list(self) -> List[List[int]]:
        """Return a deep list copy of the grid data."""
        return [row[:] for row in self.data]

    def attach_overlay(self, overlay: List[List[Any]]) -> None:
        """Attach a symbolic overlay to the grid."""
        h, w = self.shape()
        if len(overlay) != h or any(len(r) != w for r in overlay):
            raise ValueError("Overlay must match grid dimensions")
        self.overlay = overlay

    def compare_to(self, other: "Grid") -> float:
        """Return ratio of matching cells with ``other`` (1.0 equals perfect match)."""
        if self.shape() != other.shape():
            return 0.0
        matches = 0
        total = 0
        for r, r_other in zip(self.data, other.data):
            for a, b in zip(r, r_other):
                total += 1
                if a == b:
                    matches += 1
        return matches / total if total else 1.0

    def __repr__(self) -> str:
        return f"Grid(shape={self.shape()})"

