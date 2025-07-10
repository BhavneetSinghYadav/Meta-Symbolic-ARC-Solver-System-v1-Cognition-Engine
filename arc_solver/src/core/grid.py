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

    def get(self, row: int, col: int, default: Any | None = None) -> Any:
        """Return the color value at ``row``, ``col`` or ``default`` if out of bounds."""
        if row < 0 or col < 0:
            return default
        if row >= len(self.data):
            return default
        if not self.data or col >= len(self.data[0]):
            return default
        if col >= len(self.data[row]):  # defensive for malformed rows
            return default
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

    def flip_vertical(self) -> "Grid":
        """Return a new grid flipped vertically."""
        flipped = [row[:] for row in self.data[::-1]]
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

    def diff_summary(self, other: "Grid") -> Dict[str, Any]:
        """Return a structured diff between this grid and ``other``."""
        if self.shape() != other.shape():
            return {
                "cell_match_ratio": 0.0,
                "zone_coverage_match": 0.0,
                "symbol_mismatch_count": 0,
                "rotation_discrepancy": False,
            }

        h, w = self.shape()
        matches = 0
        zone_total = 0
        zone_matches = 0
        symbol_mismatches = 0

        def _zone(sym: Any) -> Optional[str]:
            if sym is None:
                return None
            if isinstance(sym, list):
                for s in sym:
                    if getattr(s, "type", None).__str__() == "ZONE":
                        return str(s.value)
                return None
            if getattr(sym, "type", None).__str__() == "ZONE":
                return str(getattr(sym, "value", None))
            return None

        for r in range(h):
            for c in range(w):
                a = self.get(r, c)
                b = other.get(r, c)
                if a == b:
                    matches += 1

                z_a = _zone(self.overlay[r][c]) if self.overlay else None
                z_b = _zone(other.overlay[r][c]) if other.overlay else None
                if z_a is not None or z_b is not None:
                    zone_total += 1
                    if z_a == z_b:
                        zone_matches += 1

                if self.overlay and other.overlay:
                    sym_a = self.overlay[r][c]
                    sym_b = other.overlay[r][c]
                    if sym_a is not None and sym_b is not None and type(sym_a) != type(sym_b):
                        symbol_mismatches += 1

        cell_ratio = matches / (h * w) if h * w else 1.0
        zone_ratio = zone_matches / zone_total if zone_total else 1.0

        rotation_discrepancy = False
        try:
            if any(self.rotate90(k).data == other.data for k in range(1, 4)):
                rotation_discrepancy = True
        except Exception:
            rotation_discrepancy = False

        return {
            "cell_match_ratio": cell_ratio,
            "zone_coverage_match": zone_ratio,
            "symbol_mismatch_count": symbol_mismatches,
            "rotation_discrepancy": rotation_discrepancy,
        }

    def detailed_score(self, other: "Grid") -> float:
        """Return weighted similarity score accounting for symbolic context."""
        diff = self.diff_summary(other)
        score = diff["cell_match_ratio"] * 0.6 + diff["zone_coverage_match"] * 0.3
        score -= diff["symbol_mismatch_count"] * 0.02
        if diff["rotation_discrepancy"]:
            score -= 0.1
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        return score

    def structural_diff(self, other: "Grid") -> List[List[bool]]:
        """Return a mask marking cell mismatches with ``other``."""
        h1, w1 = self.shape()
        h2, w2 = other.shape()
        mask: List[List[bool]] = [[True for _ in range(w1)] for _ in range(h1)]
        for r in range(h1):
            for c in range(w1):
                if r >= h2 or c >= w2:
                    mask[r][c] = True
                else:
                    mask[r][c] = self.get(r, c) != other.get(r, c)
        return mask

    def __repr__(self) -> str:
        return f"Grid(shape={self.shape()})"

