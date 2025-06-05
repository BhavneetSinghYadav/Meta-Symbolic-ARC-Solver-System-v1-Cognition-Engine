"""Symbolic rule extraction utilities for ARC problems."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationNature,
    TransformationType,
)


def _heuristic_fallback_rules(inp: Grid, out: Grid) -> List[SymbolicRule]:
    """Return a simple color replacement rule as fallback."""
    src_counts = inp.count_colors()
    tgt_counts = out.count_colors()
    if not src_counts or not tgt_counts:
        return []
    src = max(src_counts, key=src_counts.get)
    tgt = max(tgt_counts, key=tgt_counts.get)
    return [
        SymbolicRule(
            transformation=Transformation(TransformationType.REPLACE),
            source=[Symbol(SymbolType.COLOR, str(src))],
            target=[Symbol(SymbolType.COLOR, str(tgt))],
            nature=TransformationNature.LOGICAL,
        )
    ]
from arc_solver.src.segment.segmenter import zone_overlay
from arc_solver.src.executor.simulator import simulate_rules


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def extract_color_change_rules(
    input_grid: Grid,
    output_grid: Grid,
    zone_overlay: Optional[List[List[Symbol]]] = None,
) -> List[SymbolicRule]:
    """Return rules describing consistent color replacements.

    If ``zone_overlay`` is provided, replacements are recorded per zone and
    returned as conditional rules.
    """
    if input_grid.shape() != output_grid.shape():
        return []

    h, w = input_grid.shape()

    if zone_overlay is None:
        mappings: Dict[int, set[int]] = {}
        for r in range(h):
            for c in range(w):
                src = input_grid.get(r, c)
                tgt = output_grid.get(r, c)
                if src != tgt:
                    mappings.setdefault(src, set()).add(tgt)

        rules: List[SymbolicRule] = []
        for src_color, tgts in mappings.items():
            if len(tgts) == 1:
                tgt_color = next(iter(tgts))
                rule = SymbolicRule(
                    transformation=Transformation(TransformationType.REPLACE),
                    source=[Symbol(SymbolType.COLOR, str(src_color))],
                    target=[Symbol(SymbolType.COLOR, str(tgt_color))],
                    nature=TransformationNature.LOGICAL,
                )
                rules.append(rule)
        return rules

    # Zone-aware extraction
    zone_maps: Dict[str, Dict[int, set[int]]] = {}
    for r in range(h):
        for c in range(w):
            zone_sym = zone_overlay[r][c]
            if zone_sym is None:
                continue
            zone = zone_sym.value
            src = input_grid.get(r, c)
            tgt = output_grid.get(r, c)
            if src != tgt:
                zmap = zone_maps.setdefault(zone, {})
                zmap.setdefault(src, set()).add(tgt)

    # Check if all zones share a consistent mapping
    global_map: Dict[int, set[int]] = {}
    for zone, mapping in zone_maps.items():
        for src_color, tgts in mapping.items():
            gm = global_map.setdefault(src_color, set())
            gm.update(tgts)

    rules: List[SymbolicRule] = []
    if all(len(tgts) == 1 for tgts in global_map.values()):
        # produce unconditional rules when mappings are globally consistent
        for src_color, tgts in global_map.items():
            tgt_color = next(iter(tgts))
            rules.append(
                SymbolicRule(
                    transformation=Transformation(TransformationType.REPLACE),
                    source=[Symbol(SymbolType.COLOR, str(src_color))],
                    target=[Symbol(SymbolType.COLOR, str(tgt_color))],
                    nature=TransformationNature.LOGICAL,
                )
            )
        return rules

    for zone, mapping in zone_maps.items():
        for src_color, tgts in mapping.items():
            if len(tgts) == 1:
                tgt_color = next(iter(tgts))
                rules.append(
                    SymbolicRule(
                        transformation=Transformation(TransformationType.REPLACE),
                        source=[Symbol(SymbolType.COLOR, str(src_color))],
                        target=[Symbol(SymbolType.COLOR, str(tgt_color))],
                        nature=TransformationNature.LOGICAL,
                        condition={"zone": zone},
                    )
                )
    return rules


def extract_zonewise_rules(
    input_grid: Grid,
    output_grid: Grid,
    zone_overlay: Optional[List[List[Symbol]]] = None,
) -> List[SymbolicRule]:
    """Return color replacement rules conditioned on zone overlays."""
    if zone_overlay is None or input_grid.shape() != output_grid.shape():
        return []

    h, w = input_grid.shape()
    zone_mappings: Dict[str, Dict[int, set[int]]] = {}
    for r in range(h):
        for c in range(w):
            zone = zone_overlay[r][c].value
            src = input_grid.get(r, c)
            tgt = output_grid.get(r, c)
            if src != tgt:
                zone_map = zone_mappings.setdefault(zone, {})
                zone_map.setdefault(src, set()).add(tgt)

    rules: List[SymbolicRule] = []
    for zone, mapping in zone_mappings.items():
        for src_color, tgts in mapping.items():
            if len(tgts) == 1:
                tgt_color = next(iter(tgts))
                rules.append(
                    SymbolicRule(
                        transformation=Transformation(TransformationType.REPLACE),
                        source=[
                            Symbol(SymbolType.ZONE, zone),
                            Symbol(SymbolType.COLOR, str(src_color)),
                        ],
                        target=[Symbol(SymbolType.COLOR, str(tgt_color))],
                        nature=TransformationNature.LOGICAL,
                    )
                )
    return rules


def _find_translation(input_grid: Grid, output_grid: Grid) -> Optional[Tuple[int, int]]:
    """Return translation offset if output is a translated version of input."""
    if input_grid.shape() != output_grid.shape():
        return None

    h, w = input_grid.shape()
    points_in: List[Tuple[int, int, int]] = []
    points_out: List[Tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            val_in = input_grid.get(r, c)
            val_out = output_grid.get(r, c)
            if val_in != 0:
                points_in.append((r, c, val_in))
            if val_out != 0:
                points_out.append((r, c, val_out))

    if len(points_in) != len(points_out):
        return None
    if not points_in:
        return None

    dy = points_out[0][0] - points_in[0][0]
    dx = points_out[0][1] - points_in[0][1]
    for (ri, ci, vi), (ro, co, vo) in zip(points_in, points_out):
        if vi != vo:
            return None
        if ro - ri != dy or co - ci != dx:
            return None
    return dx, dy


def extract_shape_based_rules(input_grid: Grid, output_grid: Grid) -> List[SymbolicRule]:
    """Return translation rules when the entire grid is shifted."""
    offset = _find_translation(input_grid, output_grid)
    if offset is None:
        return []

    dx, dy = offset
    rule = SymbolicRule(
        transformation=Transformation(
            TransformationType.TRANSLATE,
            params={"dx": str(dx), "dy": str(dy)},
        ),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
        nature=TransformationNature.SPATIAL,
    )
    return [rule]


def split_rule_by_overlay(
    rule: SymbolicRule, grid: Grid, overlay: List[List[Symbol]]
) -> List[SymbolicRule]:
    """Return zone-scoped variants of ``rule`` when applicable."""
    h, w = grid.shape()
    predicted = simulate_rules(grid, [rule])
    zone_changes: Dict[str | None, List[Tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if predicted.get(r, c) == grid.get(r, c):
                continue
            zone_sym = overlay[r][c]
            zone = zone_sym.value if zone_sym is not None else None
            zone_changes.setdefault(zone, []).append((r, c))

    zones = [z for z in zone_changes.keys() if z is not None]
    if not zones:
        return [rule]

    new_rules: List[SymbolicRule] = []
    for zone in zones:
        new_rules.append(
            SymbolicRule(
                transformation=rule.transformation,
                source=rule.source,
                target=rule.target,
                nature=rule.nature,
                condition={"zone": zone},
            )
        )
    if None in zone_changes:
        new_rules.append(rule)
    return new_rules


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def abstract(objects, *, logger=None) -> List[SymbolicRule]:
    """Return symbolic abstractions of a grid pair.

    When ``logger`` is provided, messages describing which extraction heuristics
    were used are emitted for debugging purposes.
    """
    if not isinstance(objects, (list, tuple)) or len(objects) < 2:
        return []

    input_grid, output_grid = objects[0], objects[1]
    overlay = zone_overlay(input_grid)
    try:
        rules: List[SymbolicRule] = []
        cc_rules = extract_color_change_rules(
            input_grid, output_grid, zone_overlay=overlay
        )
        if logger:
            logger.info(f"color_change_rules: {len(cc_rules)}")
        rules.extend(cc_rules)
        shape_rules = extract_shape_based_rules(input_grid, output_grid)
        if logger:
            logger.info(f"shape_based_rules: {len(shape_rules)}")
        rules.extend(shape_rules)
        split: List[SymbolicRule] = []
        for r in rules:
            if r.transformation.ttype in (
                TransformationType.REPLACE,
                TransformationType.TRANSLATE,
            ):
                split.extend(split_rule_by_overlay(r, input_grid, overlay))
            else:
                split.append(r)
        rules = split
    except Exception:
        if logger:
            logger.warning("abstraction failure, falling back")
        rules = []

    if not rules or any(not isinstance(r, SymbolicRule) for r in rules):
        if logger:
            logger.info("using heuristic fallback rules")
        rules = _heuristic_fallback_rules(input_grid, output_grid)
    return rules


__all__ = [
    "extract_color_change_rules",
    "extract_shape_based_rules",
    "extract_zonewise_rules",
    "split_rule_by_overlay",
    "abstract",
]
