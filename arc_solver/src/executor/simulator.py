from __future__ import annotations

"""Simple symbolic rule simulator for ARC grids."""

from typing import List

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.segment.segmenter import zone_overlay


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
    zone = rule.condition.get("zone") if rule.condition else None
    overlay = zone_overlay(grid) if zone else None
    for r in range(h):
        for c in range(w):
            if zone and (overlay[r][c] is None or overlay[r][c].value != zone):
                continue
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
    zone = rule.condition.get("zone") if rule.condition else None
    overlay = zone_overlay(grid) if zone else None
    for r in range(h):
        for c in range(w):
            if zone and (overlay[r][c] is None or overlay[r][c].value != zone):
                new_data[r][c] = grid.data[r][c]
                continue
            nr = r + dy
            nc = c + dx
            if 0 <= nr < h and 0 <= nc < w:
                new_data[nr][nc] = grid.data[r][c]
            else:
                # cells translated outside remain 0
                pass
    return Grid(new_data)


def _apply_conditional(grid: Grid, rule: SymbolicRule) -> Grid:
    """Apply a simple conditional replace rule."""
    src_color = None
    tgt_color = None
    neighbor_color = rule.transformation.params.get("neighbor")
    for sym in rule.source:
        if sym.type is SymbolType.COLOR:
            src_color = int(sym.value)
        elif sym.type is SymbolType.ZONE:
            # zone scoping is handled in _apply_region
            pass
    for sym in rule.target:
        if sym.type is SymbolType.COLOR:
            tgt_color = int(sym.value)
    if src_color is None or tgt_color is None:
        return grid

    h, w = grid.shape()
    zone = rule.condition.get("zone") if rule.condition else None
    overlay = zone_overlay(grid) if zone else None
    new_data = [row[:] for row in grid.data]
    for r in range(h):
        for c in range(w):
            if zone and (overlay[r][c] is None or overlay[r][c].value != zone):
                continue
            if new_data[r][c] != src_color:
                continue
            if neighbor_color is not None:
                neigh_match = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid.get(nr, nc) == int(neighbor_color):
                        neigh_match = True
                        break
                if not neigh_match:
                    continue
            new_data[r][c] = tgt_color
    return Grid(new_data)


def _apply_region(grid: Grid, rule: SymbolicRule) -> Grid:
    """Apply a rule only to cells within a labelled region overlay."""
    if grid.overlay is None:
        return grid
    region = None
    for sym in rule.source:
        if sym.type in (SymbolType.REGION, SymbolType.ZONE):
            region = sym.value
            break
    if region is None:
        return grid

    inner_rule = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[s for s in rule.source if s.type is SymbolType.COLOR],
        target=rule.target,
        nature=rule.nature,
    )

    h, w = grid.shape()
    new_data = [row[:] for row in grid.data]
    for r in range(h):
        for c in range(w):
            sym = grid.overlay[r][c]
            if sym is None or sym.value != region:
                continue
            cell_grid = Grid([row[:] for row in grid.data])
            cell_grid.set(r, c, grid.get(r, c))
            cell_grid = _apply_replace(cell_grid, inner_rule)
            new_data[r][c] = cell_grid.get(r, c)
    return Grid(new_data)


def _apply_functional(grid: Grid, rule: SymbolicRule) -> Grid:
    op = rule.transformation.params.get("op")
    if op == "invert_diagonal":
        h, w = grid.shape()
        new_data = [row[:] for row in grid.data]
        for r in range(h):
            for c in range(w):
                if r == c or r == w - c - 1:
                    new_data[r][c] = grid.get(r, c)
                else:
                    new_data[r][c] = grid.get(r, c)
        return Grid(new_data)
    elif op == "flip_horizontal":
        return grid.flip_horizontal()
    return grid


def simulate_rules(input_grid: Grid, rules: List[SymbolicRule]) -> Grid:
    """Apply a list of symbolic rules to ``input_grid``."""
    grid = Grid([row[:] for row in input_grid.data])
    for rule in rules:
        if rule.transformation.ttype is TransformationType.REPLACE:
            grid = _apply_replace(grid, rule)
        elif rule.transformation.ttype is TransformationType.TRANSLATE:
            grid = _apply_translate(grid, rule)
        elif rule.transformation.ttype is TransformationType.CONDITIONAL:
            grid = _apply_conditional(grid, rule)
        elif rule.transformation.ttype is TransformationType.REGION:
            grid = _apply_region(grid, rule)
        elif rule.transformation.ttype is TransformationType.FUNCTIONAL:
            grid = _apply_functional(grid, rule)
    return grid


def score_prediction(predicted: Grid, target: Grid) -> float:
    """Return match ratio between ``predicted`` and ``target``."""
    return predicted.compare_to(target)


def simulate_symbolic_program(grid: Grid, rules: List[SymbolicRule]) -> Grid:
    """Alias of :func:`simulate_rules` for program semantics."""
    return simulate_rules(grid, rules)


__all__ = ["simulate_rules", "simulate_symbolic_program", "score_prediction"]
