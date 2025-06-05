from __future__ import annotations

"""Utilities to inject structural attention into rule ranking."""

from typing import List, Tuple

from arc_solver.src.core.grid import Grid
from arc_solver.src.segment.segmenter import zone_overlay
from arc_solver.src.attention.structural_encoder import StructuralEncoder
from arc_solver.src.attention.symbolic_attention import SymbolicAttention


_DEF_DIM = 32


def apply_structural_attention(
    grid: Grid, ranked_rules: List[Tuple[List, float]], weight: float = 0.2
) -> List[Tuple[List, float]]:
    """Return attention-modulated ``ranked_rules``."""

    overlay = zone_overlay(grid)
    encoder = StructuralEncoder(dim=_DEF_DIM)
    context = encoder.encode([[z.value if z else None for z in row] for row in overlay])
    attention = SymbolicAttention(weight=weight, dim=_DEF_DIM)
    return attention.apply(ranked_rules, context)


__all__ = ["apply_structural_attention"]
