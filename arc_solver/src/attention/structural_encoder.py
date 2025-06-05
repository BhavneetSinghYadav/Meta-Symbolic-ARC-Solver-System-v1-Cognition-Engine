from __future__ import annotations

"""Simple structural context encoder for ARC grids."""

from typing import Any, List, Optional

from arc_solver.src.core.grid import Grid


def validate_overlay(grid: Grid, overlay: List[List[Any]]) -> bool:
    """Return True if ``overlay`` matches ``grid`` and has valid values."""
    h, w = grid.shape()
    if len(overlay) != h or any(len(row) != w for row in overlay):
        return False
    for row in overlay:
        for val in row:
            if val is not None and not isinstance(val, (str, int)):
                return False
    return True

import numpy as np


class StructuralEncoder:
    """Encode symbolic overlays and masks into a dense vector."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def _embed_token(self, token: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(token)) % (2**32))
        return rng.standard_normal(self.dim)

    def encode(
        self,
        grid: Grid,
        zone_overlay: List[List[Optional[str]]],
        entropy_mask: Optional[List[List[float]]] | None = None,
        symbolic_overlay: Optional[List[List[Optional[str]]]] | None = None,
    ) -> np.ndarray:
        """Return deterministic embedding of task structure."""

        if not validate_overlay(grid, zone_overlay):
            raise ValueError("Invalid overlay structure")

        vec = np.zeros(self.dim, dtype=float)
        height = len(zone_overlay)
        width = len(zone_overlay[0]) if height > 0 else 0
        for r in range(height):
            for c in range(width):
                label = zone_overlay[r][c]
                if label is not None:
                    vec += self._embed_token(str(label))
                if symbolic_overlay is not None:
                    sym = symbolic_overlay[r][c]
                    if sym is not None:
                        vec += self._embed_token(str(sym))
                if entropy_mask is not None:
                    vec += float(entropy_mask[r][c])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


__all__ = ["StructuralEncoder", "validate_overlay"]
