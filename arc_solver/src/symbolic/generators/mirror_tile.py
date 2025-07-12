from __future__ import annotations

from typing import List

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.operators import mirror_tile
from arc_solver.src.symbolic.vocabulary import (
    SymbolicRule,
    Transformation,
    TransformationNature,
    TransformationType,
    Symbol,
    SymbolType,
)
from arc_solver.src.symbolic.rule_language import rule_to_dsl
from arc_solver.src.executor.scoring import score_rule


def generate_mirror_tile_rules(grid_in: Grid, grid_out: Grid) -> List[SymbolicRule]:
    """Detect horizontal or vertical mirror tiling."""
    rules: List[SymbolicRule] = []
    ih, iw = grid_in.shape()
    oh, ow = grid_out.shape()
    for axis in ("horizontal", "vertical"):
        if axis == "horizontal":
            if oh != ih or ow % iw != 0:
                continue
            count = ow // iw
        else:
            if ow != iw or oh % ih != 0:
                continue
            count = oh // ih
        if count <= 1 or count > 25:
            continue
        try:
            pred = mirror_tile(grid_in, axis, count)
        except Exception:
            continue
        if pred == grid_out:
            rule = SymbolicRule(
                transformation=Transformation(
                    TransformationType.FUNCTIONAL,
                    params={"op": "mirror_tile", "axis": axis, "repeats": str(count)},
                ),
                source=[Symbol(SymbolType.REGION, "All")],
                target=[Symbol(SymbolType.REGION, "All")],
                nature=TransformationNature.SPATIAL,
            )
            rule.meta["derivation"] = {"heuristic_used": "mirror_tile"}
            rule.dsl_str = rule_to_dsl(rule)
            rule.meta["score_trace"] = score_rule(grid_in, grid_out, rule, return_trace=True)
            rules.append(rule)
    return rules


__all__ = ["generate_mirror_tile_rules"]
