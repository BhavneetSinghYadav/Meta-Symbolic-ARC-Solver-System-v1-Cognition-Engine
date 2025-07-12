from __future__ import annotations

from typing import List

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.draw_line import draw_line
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


def generate_draw_line_rules(grid_in: Grid, grid_out: Grid) -> List[SymbolicRule]:
    """Detect drawing a straight line connecting two existing cells."""
    if grid_in.shape() != grid_out.shape():
        return []
    h, w = grid_in.shape()
    diff = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid_in.get(r, c) != grid_out.get(r, c)
    ]
    if not diff:
        return []
    color = grid_out.get(diff[0][0], diff[0][1])
    if any(grid_out.get(r, c) != color for r, c in diff):
        return []
    points = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid_in.get(r, c) != 0
    ]
    rules: List[SymbolicRule] = []
    for i, p1 in enumerate(points):
        for p2 in points[i + 1 :]:
            try:
                pred = draw_line(grid_in.to_list(), p1, p2, color)
            except Exception:
                continue
            if Grid(pred if isinstance(pred, list) else pred.tolist()) == grid_out:
                rule = SymbolicRule(
                    transformation=Transformation(
                        TransformationType.FUNCTIONAL,
                        params={"op": "draw_line", "p1": str(p1), "p2": str(p2), "color": str(color)},
                    ),
                    source=[Symbol(SymbolType.REGION, "All")],
                    target=[Symbol(SymbolType.REGION, "All")],
                    nature=TransformationNature.SPATIAL,
                )
                rule.meta["derivation"] = {"heuristic_used": "draw_line"}
                rule.dsl_str = rule_to_dsl(rule)
                rule.meta["score_trace"] = score_rule(grid_in, grid_out, rule, return_trace=True)
                rules.append(rule)
                if len(rules) >= 25:
                    return rules
    return rules


__all__ = ["generate_draw_line_rules"]
