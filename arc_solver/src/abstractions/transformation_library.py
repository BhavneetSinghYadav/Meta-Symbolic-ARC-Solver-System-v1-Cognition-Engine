"""Library of simple symbolic transformations."""

from __future__ import annotations

from typing import Optional

import numpy as np
from arc_solver.src.utils.logger import get_logger

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)

logger = get_logger(__name__)


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
        return replace_color(input_grid, src_color, tgt_color)


def replace_color(grid: Grid, src_color: int, tgt_color: int) -> Grid:
    """Return ``grid`` with all occurrences of ``src_color`` replaced by ``tgt_color``.

    If ``src_color`` does not appear, the original grid is returned and a warning
    is logged.
    """

    arr = np.array(grid.data)
    if src_color not in arr:
        logger.warning(f"replace_color: source color {src_color} not found")
        return grid
    replaced = np.where(arr == src_color, tgt_color, arr)
    return Grid(replaced.tolist())


TRANSFORMATIONS = [ReplaceColor]

__all__ = ["ReplaceColor", "TRANSFORMATIONS"]
