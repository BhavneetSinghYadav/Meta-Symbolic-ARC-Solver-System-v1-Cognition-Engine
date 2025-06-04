from __future__ import annotations

"""Simple symbolic rule simulator for ARC grids."""

from typing import List, Optional

from arc_solver.src.utils.logger import get_logger

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.segment.segmenter import zone_overlay


logger = get_logger(__name__)


class ReflexOverrideException(Exception):
    """Raised when a rule violates a reflex constraint."""


def _is_vertically_symmetric(grid: Grid) -> bool:
    h, w = grid.shape()
    for r in range(h):
        for c in range(w // 2):
            if grid.get(r, c) != grid.get(r, w - c - 1):
                return False
    return True


def _is_horizontally_symmetric(grid: Grid) -> bool:
    h, w = grid.shape()
    for c in range(w):
        for r in range(h // 2):
            if grid.get(r, c) != grid.get(h - r - 1, c):
                return False
    return True


def violates_symmetry(after: Grid, before: Grid) -> bool:
    return (
        (_is_vertically_symmetric(before) and not _is_vertically_symmetric(after))
        or (
            _is_horizontally_symmetric(before)
            and not _is_horizontally_symmetric(after)
        )
    )


def breaks_training_constraint(after: Grid) -> bool:
    # Placeholder constraint: ensure color values remain within 0-9
    h, w = after.shape()
    for r in range(h):
        for c in range(w):
            val = after.get(r, c)
            if val < 0 or val > 9:
                return True
    return False


def _apply_replace(
    grid: Grid, rule: SymbolicRule, attention_mask: Optional[List[List[bool]]] = None
) -> Grid:
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
            if attention_mask and not attention_mask[r][c]:
                continue
            if zone and (overlay[r][c] is None or overlay[r][c].value != zone):
                continue
            if new_data[r][c] == src_color:
                new_data[r][c] = tgt_color
    return Grid(new_data)


def _apply_translate(
    grid: Grid, rule: SymbolicRule, attention_mask: Optional[List[List[bool]]] = None
) -> Grid:
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
            if attention_mask and not attention_mask[r][c]:
                new_data[r][c] = grid.data[r][c]
                continue
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


def _apply_conditional(
    grid: Grid, rule: SymbolicRule, attention_mask: Optional[List[List[bool]]] = None
) -> Grid:
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
            if attention_mask and not attention_mask[r][c]:
                continue
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


def _apply_region(
    grid: Grid, rule: SymbolicRule, attention_mask: Optional[List[List[bool]]] = None
) -> Grid:
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
            if attention_mask and not attention_mask[r][c]:
                continue
            sym = grid.overlay[r][c]
            if sym is None or sym.value != region:
                continue
            cell_grid = Grid([row[:] for row in grid.data])
            cell_grid.set(r, c, grid.get(r, c))
            cell_grid = _apply_replace(cell_grid, inner_rule)
            new_data[r][c] = cell_grid.get(r, c)
    return Grid(new_data)


def _apply_functional(
    grid: Grid, rule: SymbolicRule, attention_mask: Optional[List[List[bool]]] = None
) -> Grid:
    op = rule.transformation.params.get("op")
    if op == "invert_diagonal":
        h, w = grid.shape()
        new_data = [row[:] for row in grid.data]
        for r in range(h):
            for c in range(w):
                if attention_mask and not attention_mask[r][c]:
                    new_data[r][c] = grid.get(r, c)
                    continue
                if r == c or r == w - c - 1:
                    new_data[r][c] = grid.get(r, c)
                else:
                    new_data[r][c] = grid.get(r, c)
        return Grid(new_data)
    elif op == "flip_horizontal":
        return grid.flip_horizontal()
    return grid


def _safe_apply_rule(
    grid: Grid, rule: SymbolicRule, attention_mask: Optional[List[List[bool]]]
) -> Grid:
    before = Grid([row[:] for row in grid.data])

    if rule.transformation.ttype is TransformationType.REPLACE:
        after = _apply_replace(grid, rule, attention_mask)
    elif rule.transformation.ttype is TransformationType.TRANSLATE:
        after = _apply_translate(grid, rule, attention_mask)
    elif rule.transformation.ttype is TransformationType.CONDITIONAL:
        after = _apply_conditional(grid, rule, attention_mask)
    elif rule.transformation.ttype is TransformationType.REGION:
        after = _apply_region(grid, rule, attention_mask)
    elif rule.transformation.ttype is TransformationType.FUNCTIONAL:
        after = _apply_functional(grid, rule, attention_mask)
    else:
        after = grid

    if violates_symmetry(after, before):
        raise ReflexOverrideException("Symmetry violation")
    if breaks_training_constraint(after):
        raise ReflexOverrideException("Training constraint mismatch")

    return after


def simulate_rules(
    input_grid: Grid,
    rules: List[SymbolicRule],
    *,
    attention_mask: Optional[List[List[bool]]] = None,
) -> Grid:
    """Apply a list of symbolic rules to ``input_grid`` with reflex checks."""
    grid = Grid([row[:] for row in input_grid.data])
    for rule in rules:
        try:
            grid = _safe_apply_rule(grid, rule, attention_mask)
        except ReflexOverrideException as e:
            logger.warning(f"Reflex override triggered: {e}")
            continue
    return grid


def score_prediction(predicted: Grid, target: Grid) -> float:
    """Return match ratio between ``predicted`` and ``target``."""
    return predicted.compare_to(target)


def simulate_symbolic_program(grid: Grid, rules: List[SymbolicRule]) -> Grid:
    """Alias of :func:`simulate_rules` for program semantics."""
    return simulate_rules(grid, rules)


__all__ = [
    "simulate_rules",
    "simulate_symbolic_program",
    "score_prediction",
    "ReflexOverrideException",
]
