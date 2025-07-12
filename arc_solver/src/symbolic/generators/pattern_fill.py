from __future__ import annotations

from typing import List

from arc_solver.src.core.grid import Grid
from arc_solver.src.segment.segmenter import label_connected_regions
from arc_solver.src.symbolic.pattern_fill import pattern_fill
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


def generate_pattern_fill_rules(grid_in: Grid, grid_out: Grid) -> List[SymbolicRule]:
    """Detect pattern fill between segmented regions."""
    if grid_in.shape() != grid_out.shape():
        return []

    overlay = label_connected_regions(grid_in)
    h, w = grid_in.shape()
    zone_ids = {z for row in overlay for z in row if z is not None}

    rules: List[SymbolicRule] = []
    for src in zone_ids:
        for tgt in zone_ids:
            if src == tgt:
                continue
            try:
                pred = pattern_fill(grid_in.to_list(), src, tgt, overlay)
            except Exception:
                continue
            pred_grid = Grid(pred if isinstance(pred, list) else pred.tolist())
            if pred_grid != grid_out:
                continue

            mask = [[1 if overlay[r][c] == tgt else 0 for c in range(w)] for r in range(h)]
            cells = [
                (r, c)
                for r in range(h)
                for c in range(w)
                if overlay[r][c] == src and grid_in.get(r, c) != 0
            ]
            if not cells:
                continue
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            top, left, bottom, right = min(rows), min(cols), max(rows) + 1, max(cols) + 1
            pattern = [
                [grid_in.get(r, c) for c in range(left, right)]
                for r in range(top, bottom)
            ]

            rule = SymbolicRule(
                transformation=Transformation(
                    TransformationType.FUNCTIONAL,
                    params={"op": "pattern_fill"},
                ),
                source=[Symbol(SymbolType.REGION, "All")],
                target=[Symbol(SymbolType.REGION, "All")],
                nature=TransformationNature.SPATIAL,
                meta={"mask": Grid(mask), "pattern": Grid(pattern)},
            )
            rule.meta["derivation"] = {"heuristic_used": "pattern_fill"}
            rule.dsl_str = rule_to_dsl(rule)
            rule.meta["score_trace"] = score_rule(grid_in, grid_out, rule, return_trace=True)
            rules.append(rule)
            if len(rules) >= 25:
                return rules
    return rules


__all__ = ["generate_pattern_fill_rules"]
