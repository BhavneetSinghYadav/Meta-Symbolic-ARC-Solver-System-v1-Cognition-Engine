"""Library of simple symbolic transformations."""

from __future__ import annotations

from typing import Optional

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)


class ReplaceColor:
    """Replace one color with another across the grid."""

    @staticmethod
    def match(input_grid: Grid, output_grid: Grid) -> Optional[SymbolicRule]:
        from .abstractor import extract_color_change_rules

        rules = extract_color_change_rules(input_grid, output_grid)
        return rules[0] if len(rules) == 1 else None

    @staticmethod
    def apply(input_grid: Grid, rule: SymbolicRule) -> Grid:
        if not rule.source or not rule.target:
            return input_grid
        src_color = int(rule.source[-1].value)
        tgt_color = int(rule.target[-1].value)
        result = [row[:] for row in input_grid.data]
        out = Grid(result)
        h, w = out.shape()
        for r in range(h):
            for c in range(w):
                if out.get(r, c) == src_color:
                    out.set(r, c, tgt_color)
        return out


TRANSFORMATIONS = [ReplaceColor]

__all__ = ["ReplaceColor", "TRANSFORMATIONS"]
