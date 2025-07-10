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
from arc_solver.src.symbolic.rule_language import CompositeRule


def generate_repeat_composite_rules(input_grid: Grid, output_grid: Grid) -> List[CompositeRule]:
    """Return composite rules repeating ``input_grid`` then recolouring to match ``output_grid``.

    When multiple colour substitutions are required the function generates a chain
    of ``REPLACE`` steps after the initial ``REPEAT``.  Each mapping is applied
    sequentially so colour dependencies are tracked correctly during validation.
    """
    repeat_rules = generate_repeat_rules(input_grid, output_grid)
    if not repeat_rules:
        return []
    base_rule = repeat_rules[0]
    try:
        kx = int(base_rule.transformation.params.get("kx", "1"))
        ky = int(base_rule.transformation.params.get("ky", "1"))
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
    steps: List[SymbolicRule] = [
        SymbolicRule(
            transformation=Transformation(
                TransformationType.REPEAT,
                params={"kx": str(kx), "ky": str(ky)},
            ),
            source=[Symbol(SymbolType.REGION, "All")],
            target=[Symbol(SymbolType.REGION, "All")],
            nature=TransformationNature.SPATIAL,
        )
    ]

    for src_color, tgt_color in mappings.items():
        steps.append(
            SymbolicRule(
                transformation=Transformation(TransformationType.REPLACE),
                source=[Symbol(SymbolType.COLOR, str(src_color))],
                target=[Symbol(SymbolType.COLOR, str(tgt_color))],
            )
        )

    composite = CompositeRule(steps)
    return [composite]


__all__ = ["generate_repeat_composite_rules"]

