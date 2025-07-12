from __future__ import annotations

from typing import Dict, List

from arc_solver.src.core.grid import Grid
from arc_solver.src.segment.segmenter import label_connected_regions, zone_overlay
from arc_solver.src.symbolic.morphology_ops import dilate_zone, erode_zone
from arc_solver.src.symbolic.rotate_about_point import rotate_about_point
from arc_solver.src.symbolic.zone_remap import zone_remap
from arc_solver.src.symbolic.vocabulary import (
    SymbolicRule,
    Transformation,
    TransformationNature,
    TransformationType,
    Symbol,
    SymbolType,
)
from arc_solver.src.symbolic.rule_language import rule_to_dsl, CompositeRule
from arc_solver.src.executor.scoring import score_rule
from arc_solver.src.executor.simulator import simulate_rules


def generate_dilate_zone_rules(grid_in: Grid, grid_out: Grid) -> List[SymbolicRule]:
    if grid_in.shape() != grid_out.shape():
        return []
    overlay = label_connected_regions(grid_in)
    zone_ids = {z for row in overlay for z in row if z is not None}
    rules: List[SymbolicRule] = []
    for zid in zone_ids:
        try:
            pred = dilate_zone(grid_in.to_list(), zid, overlay)
        except Exception:
            continue
        if Grid(pred if isinstance(pred, list) else pred.tolist()) == grid_out:
            rule = SymbolicRule(
                transformation=Transformation(
                    TransformationType.FUNCTIONAL,
                    params={"op": "dilate_zone", "zone": str(zid)},
                ),
                source=[Symbol(SymbolType.REGION, "All")],
                target=[Symbol(SymbolType.REGION, "All")],
                nature=TransformationNature.SPATIAL,
                meta={"zone_overlay": overlay, "input_zones": [str(zid)], "output_zones": [str(zid)]},
            )
            rule.meta["derivation"] = {"heuristic_used": "dilate_zone"}
            rule.dsl_str = rule_to_dsl(rule)
            rule.meta["score_trace"] = score_rule(grid_in, grid_out, rule, return_trace=True)
            rules.append(rule)
            if len(rules) >= 25:
                break
    return rules


def generate_erode_zone_rules(grid_in: Grid, grid_out: Grid) -> List[SymbolicRule]:
    if grid_in.shape() != grid_out.shape():
        return []
    overlay = label_connected_regions(grid_in)
    zone_ids = {z for row in overlay for z in row if z is not None}
    rules: List[SymbolicRule] = []
    for zid in zone_ids:
        try:
            pred = erode_zone(grid_in.to_list(), zid, overlay)
        except Exception:
            continue
        if Grid(pred if isinstance(pred, list) else pred.tolist()) == grid_out:
            rule = SymbolicRule(
                transformation=Transformation(
                    TransformationType.FUNCTIONAL,
                    params={"op": "erode_zone", "zone": str(zid)},
                ),
                source=[Symbol(SymbolType.REGION, "All")],
                target=[Symbol(SymbolType.REGION, "All")],
                nature=TransformationNature.SPATIAL,
                meta={"zone_overlay": overlay, "input_zones": [str(zid)], "output_zones": [str(zid)]},
            )
            rule.meta["derivation"] = {"heuristic_used": "erode_zone"}
            rule.dsl_str = rule_to_dsl(rule)
            rule.meta["score_trace"] = score_rule(grid_in, grid_out, rule, return_trace=True)
            rules.append(rule)
            if len(rules) >= 25:
                break
    return rules


def generate_rotate_about_point_rules(grid_in: Grid, grid_out: Grid) -> List[SymbolicRule]:
    if grid_in.shape() != grid_out.shape():
        return []
    h, w = grid_in.shape()
    diff = [(r, c) for r in range(h) for c in range(w) if grid_in.get(r, c) != grid_out.get(r, c)]
    if diff:
        min_r = max(min(r for r, _ in diff) - 1, 0)
        max_r = min(max(r for r, _ in diff) + 1, h - 1)
        min_c = max(min(c for _, c in diff) - 1, 0)
        max_c = min(max(c for _, c in diff) + 1, w - 1)
        pivots = [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)]
    else:
        pivots = [(h // 2, w // 2)]
    rules: List[SymbolicRule] = []
    for cx, cy in pivots[:25]:
        for angle in (90, 180, 270):
            try:
                pred = rotate_about_point(grid_in, (cx, cy), angle)
            except Exception:
                continue
            if pred == grid_out:
                rule = SymbolicRule(
                    transformation=Transformation(
                        TransformationType.ROTATE,
                        params={"cx": str(cx), "cy": str(cy), "angle": str(angle)},
                    ),
                    source=[Symbol(SymbolType.REGION, "All")],
                    target=[Symbol(SymbolType.REGION, "All")],
                    nature=TransformationNature.SPATIAL,
                    meta={"pivot": f"{cx},{cy}"},
                )
                rule.meta["derivation"] = {"heuristic_used": "rotate_about_point"}
                rule.dsl_str = rule_to_dsl(rule)
                rule.meta["score_trace"] = score_rule(grid_in, grid_out, rule, return_trace=True)
                rules.append(rule)
                if len(rules) >= 25:
                    return rules
    return rules


def generate_zone_remap_rules(grid_in: Grid, grid_out: Grid) -> List[SymbolicRule]:
    if grid_in.shape() != grid_out.shape():
        return []
    overlay_syms = zone_overlay(grid_in)
    h, w = grid_in.shape()
    label_to_id: Dict[str, int] = {}
    overlay_ids: List[List[int]] = [[-1 for _ in range(w)] for _ in range(h)]
    zone_out: Dict[int, set[int]] = {}
    zone_in: Dict[int, set[int]] = {}
    for r in range(h):
        for c in range(w):
            sym = overlay_syms[r][c]
            if sym is None:
                continue
            label = str(sym.value)
            zid = label_to_id.setdefault(label, len(label_to_id) + 1)
            overlay_ids[r][c] = zid
            zone_in.setdefault(zid, set()).add(grid_in.get(r, c))
            zone_out.setdefault(zid, set()).add(grid_out.get(r, c))
    mapping: Dict[int, int] = {}
    for zid in label_to_id.values():
        in_colors = zone_in.get(zid, set())
        out_colors = zone_out.get(zid, set())
        if len(out_colors) == 1 and len(in_colors) == 1:
            out_color = next(iter(out_colors))
            if out_color != next(iter(in_colors)):
                mapping[zid] = out_color
    if not mapping:
        return []
    predicted = zone_remap(grid_in.to_list(), overlay_ids, mapping)
    pred_grid = Grid(predicted if isinstance(predicted, list) else predicted.tolist())
    if pred_grid != grid_out:
        return []
    str_keys = {str(k): v for k, v in mapping.items()}
    rule = SymbolicRule(
        transformation=Transformation(
            TransformationType.FUNCTIONAL, params={"op": "zone_remap"}
        ),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
        nature=TransformationNature.SPATIAL,
        meta={
            "mapping": str_keys,
            "zone_overlay": overlay_ids,
            "input_zones": list(str_keys.keys()),
            "output_zones": list(str_keys.keys()),
        },
    )
    rule.dsl_str = rule_to_dsl(rule)
    rule.meta["score_trace"] = score_rule(grid_in, grid_out, rule, return_trace=True)
    return [rule]


def generate_morph_remap_composites(grid_in: Grid, grid_out: Grid) -> List[CompositeRule]:
    """Return composites applying a zone morph then recolouring."""

    if grid_in.shape() != grid_out.shape():
        return []

    overlay = label_connected_regions(grid_in)
    zone_ids = {z for row in overlay for z in row if z is not None}

    composites: List[CompositeRule] = []
    for zid in zone_ids:
        for op, func in ("dilate_zone", dilate_zone), ("erode_zone", erode_zone):
            try:
                mid = func(grid_in.to_list(), zid, overlay)
            except Exception:
                continue
            mid_grid = Grid(mid if isinstance(mid, list) else mid.tolist())
            remaps = generate_zone_remap_rules(mid_grid, grid_out)
            for zr in remaps:
                base_rule = SymbolicRule(
                    transformation=Transformation(
                        TransformationType.FUNCTIONAL,
                        params={"op": op, "zone": str(zid)},
                    ),
                    source=[Symbol(SymbolType.REGION, "All")],
                    target=[Symbol(SymbolType.REGION, "All")],
                    nature=TransformationNature.SPATIAL,
                    meta={"zone_overlay": overlay, "input_zones": [str(zid)], "output_zones": [str(zid)]},
                )
                base_rule.dsl_str = rule_to_dsl(base_rule)
                chain = CompositeRule([base_rule, zr])
                chain.meta["derivation"] = {"heuristic_used": f"{op}+zone_remap"}
                chain.meta["zone_overlay"] = overlay
                chain.dsl_str = " ; ".join(rule_to_dsl(s) for s in chain.steps)
                chain.meta["score_trace"] = score_rule(grid_in, grid_out, chain, return_trace=True)
                composites.append(chain)
    return composites


__all__ = [
    "generate_dilate_zone_rules",
    "generate_erode_zone_rules",
    "generate_zone_remap_rules",
    "generate_rotate_about_point_rules",
    "generate_morph_remap_composites",
]

