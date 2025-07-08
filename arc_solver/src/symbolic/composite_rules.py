from __future__ import annotations

"""Composite symbolic rules combining repeat tiling and recoloring."""

from typing import List

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.repeat_rule import repeat_tile, generate_repeat_rules
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationNature,
    TransformationType,
)


def generate_repeat_composite_rules(input_grid: Grid, output_grid: Grid) -> List[SymbolicRule]:
    """Return rules that repeat ``input_grid`` then recolor to match ``output_grid``."""
    repeat_rules = generate_repeat_rules(input_grid, output_grid)
    if not repeat_rules:
        return []
    rule = repeat_rules[0]
    try:
        kx = int(rule.transformation.params.get("kx", "1"))
        ky = int(rule.transformation.params.get("ky", "1"))
    except Exception:
        return []
    tiled = repeat_tile(input_grid, kx, ky)
    if tiled.shape() != output_grid.shape():
        return []
    mappings = {}
    h, w = tiled.shape()
    for r in range(h):
        for c in range(w):
            src = tiled.get(r, c)
            tgt = output_grid.get(r, c)
            if src != tgt:
                if src in mappings and mappings[src] != tgt:
                    return []
                mappings[src] = tgt
    if not mappings:
        return []
    # For simplicity only generate composite rule when a single color mapping exists
    if len(mappings) != 1:
        return []
    src_color, tgt_color = next(iter(mappings.items()))
    composite_rule = SymbolicRule(
        transformation=Transformation(
            TransformationType.COMPOSITE,
            params={"steps": ["REPEAT", "REPLACE"], "kx": str(kx), "ky": str(ky)},
        ),
        source=[Symbol(SymbolType.COLOR, str(src_color))],
        target=[Symbol(SymbolType.COLOR, str(tgt_color))],
        nature=TransformationNature.SPATIAL,
    )
    return [composite_rule]


__all__ = ["generate_repeat_composite_rules"]

