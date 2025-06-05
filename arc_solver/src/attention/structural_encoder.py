from __future__ import annotations

"""Simple structural context encoder for ARC grids."""

from typing import List, Optional

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
        zone_overlay: List[List[Optional[str]]],
        entropy_mask: Optional[List[List[float]]] | None = None,
        symbolic_overlay: Optional[List[List[Optional[str]]]] | None = None,
    ) -> np.ndarray:
        """Return deterministic embedding of task structure."""

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


__all__ = ["StructuralEncoder"]
