from __future__ import annotations

"""Simple symbolic rule simulator for ARC grids."""

from typing import List

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    SymbolType,
    SymbolicRule,
    TransformationType,
)


def _apply_replace(grid: Grid, rule: SymbolicRule) -> Grid:
    src_color = None
    tgt_color = None
    for sym in rule.source:
        if sym.type is SymbolType.COLOR:
            src_color = int(sym.value)
            break
    for sym in rule.target:
        if sym.type is SymbolType.COLOR:
            tgt_color = int(sym.value)
            break
    if src_color is None or tgt_color is None:
        return grid

    h, w = grid.shape()
    new_data = [row[:] for row in grid.data]
    for r in range(h):
        for c in range(w):
            if new_data[r][c] == src_color:
                new_data[r][c] = tgt_color
    return Grid(new_data)


def _apply_translate(grid: Grid, rule: SymbolicRule) -> Grid:
    try:
        dx = int(rule.transformation.params.get("dx", "0"))
        dy = int(rule.transformation.params.get("dy", "0"))
    except ValueError:
        return grid
    h, w = grid.shape()
    new_data = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            nr = r + dy
            nc = c + dx
            if 0 <= nr < h and 0 <= nc < w:
                new_data[nr][nc] = grid.data[r][c]
    return Grid(new_data)


def simulate_rules(input_grid: Grid, rules: List[SymbolicRule]) -> Grid:
    """Apply a list of symbolic rules to ``input_grid``."""
    grid = Grid([row[:] for row in input_grid.data])
    for rule in rules:
        if rule.transformation.ttype is TransformationType.REPLACE:
            grid = _apply_replace(grid, rule)
        elif rule.transformation.ttype is TransformationType.TRANSLATE:
            grid = _apply_translate(grid, rule)
    return grid


def score_prediction(predicted: Grid, target: Grid) -> float:
    """Return match ratio between ``predicted`` and ``target``."""
    return predicted.compare_to(target)
