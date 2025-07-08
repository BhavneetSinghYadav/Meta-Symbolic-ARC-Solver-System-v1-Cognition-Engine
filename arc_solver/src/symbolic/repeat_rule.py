from __future__ import annotations

"""Repeat tiling transformation utilities."""

from typing import List

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationNature,
    TransformationType,
)


def repeat_tile(grid: Grid, kx: int, ky: int) -> Grid:
    """Return ``grid`` tiled ``kx`` times horizontally and ``ky`` times vertically."""
    if kx <= 0 or ky <= 0:
        return grid
    h, w = grid.shape()
    new_h = h * ky
    new_w = w * kx
    if new_h > 30 or new_w > 30:
        return grid

    new_data = [[0 for _ in range(new_w)] for _ in range(new_h)]
    for ty in range(ky):
        for tx in range(kx):
            for r in range(h):
                for c in range(w):
                    new_data[ty * h + r][tx * w + c] = grid.get(r, c)
    return Grid(new_data)


def generate_repeat_rules(input_grid: Grid, output_grid: Grid) -> List[SymbolicRule]:
    """Return repeat rules transforming ``input_grid`` into ``output_grid``."""
    h1, w1 = input_grid.shape()
    h2, w2 = output_grid.shape()
    if h2 % h1 != 0 or w2 % w1 != 0:
        return []

    ky = h2 // h1
    kx = w2 // w1
    if (kx, ky) == (1, 1) or kx == 1 or ky == 1:
        return []

    tiled = repeat_tile(input_grid, kx, ky)
    if tiled.shape() != output_grid.shape():
        return []

    diff_pixels = sum(
        1 for r in range(h2) for c in range(w2)
        if tiled.get(r, c) != output_grid.get(r, c)
    )
    if diff_pixels and diff_pixels > h2 * w2 * 0.34:
        return []

    rule = SymbolicRule(
        transformation=Transformation(
            TransformationType.REPEAT,
            params={"kx": str(kx), "ky": str(ky)},
        ),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
        nature=TransformationNature.SPATIAL,
    )
    return [rule]


__all__ = ["repeat_tile", "generate_repeat_rules"]

