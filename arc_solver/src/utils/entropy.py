"""Entropy computation helpers for zone-based heuristics."""

from __future__ import annotations

from typing import Dict, List
import math

from arc_solver.src.core.grid import Grid
from arc_solver.src.segment.segmenter import zone_overlay


def _shannon_entropy(values: List[int]) -> float:
    """Return Shannon entropy of the provided color values."""
    counts: Dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    total = len(values)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def compute_zone_entropy_map(grid: Grid) -> Dict[str, float]:
    """Return normalized entropy per predefined grid zone."""
    overlay = zone_overlay(grid)
    h = len(overlay)
    w = len(overlay[0]) if h else 0
    zone_cells: Dict[str, List[int]] = {}
    for r in range(h):
        for c in range(w):
            sym = overlay[r][c]
            if sym is None:
                continue
            zone_cells.setdefault(sym.value, []).append(grid.get(r, c))

    max_ent = math.log2(10)
    entropies: Dict[str, float] = {}
    for zone, cells in zone_cells.items():
        if not cells:
            entropies[zone] = 0.0
            continue
        ent = _shannon_entropy(cells)
        entropies[zone] = ent / max_ent if max_ent else ent
    return entropies


__all__ = ["compute_zone_entropy_map"]
