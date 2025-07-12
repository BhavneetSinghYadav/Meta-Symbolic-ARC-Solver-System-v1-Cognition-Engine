"""Symbolic rule extraction utilities for ARC problems."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
try:  # pragma: no cover - optional
    from scipy.stats import entropy as scipy_entropy
except Exception:  # pragma: no cover - fallback if scipy unavailable
    def scipy_entropy(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[arr > 0]
        if arr.size == 0:
            return 0.0
        p = arr / arr.sum()
        return float(-(p * np.log(p)).sum())

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationNature,
    TransformationType,
)
from arc_solver.src.symbolic.repeat_rule import generate_repeat_rules
from arc_solver.src.configs.defaults import ENABLE_COMPOSITE_REPEAT
from arc_solver.src.symbolic.zone_remap import zone_remap
from arc_solver.src.symbolic.morphology_ops import dilate_zone, erode_zone
from arc_solver.src.symbolic.draw_line import draw_line
from arc_solver.src.symbolic.rotate_about_point import rotate_about_point
from arc_solver.src.symbolic.operators import mirror_tile
from arc_solver.src.symbolic.pattern_fill import pattern_fill
from arc_solver.src.segment.segmenter import label_connected_regions
from arc_solver.src.symbolic.generators import (
    generate_mirror_tile_rules,
    generate_draw_line_rules,
    generate_dilate_zone_rules,
    generate_erode_zone_rules,
    generate_zone_remap_rules,
    generate_rotate_about_point_rules,
    generate_morph_remap_composites,
    generate_pattern_fill_rules,
)
from arc_solver.src.abstractions.zone_shape_auto_discover import auto_discover_rules
from arc_solver.src.abstractions.rule_generator import generate_all_rules
from arc_solver.src.utils.logger import get_logger

logger = get_logger(__name__)

# Optional toggles controlling abstraction behaviour
ENABLE_FALLBACK_COMPOSITES = True
RELIABILITY_THRESHOLD = 0.4




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


def zone_entropy(zone: np.ndarray) -> float:
    """Return entropy of color distribution in ``zone``."""
    return float(scipy_entropy(np.bincount(zone.flatten())))


def merge_with_neighbors(overlay: List[List[Optional[Symbol]]], label: str) -> None:
    """Remove ``label`` from overlay to merge with surrounding cells."""
    h = len(overlay)
    w = len(overlay[0]) if h else 0
    for r in range(h):
        for c in range(w):
            sym = overlay[r][c]
            if sym is not None and sym.value == label:
                overlay[r][c] = None


def align_segments(
    input_overlay: List[List[Optional[Symbol]]],
    output_overlay: List[List[Optional[Symbol]]],
) -> List[List[Optional[Symbol]]]:
    """Return overlay where zones align across input and output."""
    h = len(input_overlay)
    w = len(input_overlay[0]) if h else 0
    matched: List[List[Optional[Symbol]]] = [
        [None for _ in range(w)] for _ in range(h)
    ]
    for r in range(h):
        for c in range(w):
            iz = input_overlay[r][c]
            oz = output_overlay[r][c] if r < len(output_overlay) and c < len(output_overlay[0]) else None
            if iz is not None and oz is not None and iz.value == oz.value:
                matched[r][c] = iz
    return matched


def segment_and_overlay(
    input_grid: Grid | GridProxy, output_grid: Grid | GridProxy
) -> Tuple[List[List[Optional[Symbol]]], Dict[str, float]]:
    """Return IO-aligned zone overlay and entropy map."""
    inp_overlay = (
        input_grid.get_zone_overlay() if isinstance(input_grid, GridProxy) else zone_overlay(input_grid)
    )
    out_overlay = (
        output_grid.get_zone_overlay() if isinstance(output_grid, GridProxy) else zone_overlay(output_grid)
    )
    overlay = align_segments(inp_overlay, out_overlay)
    entropies: Dict[str, float] = {}
    zones: Dict[str, List[int]] = {}
    h = len(overlay)
    w = len(overlay[0]) if h else 0
    for r in range(h):
        for c in range(w):
            sym = overlay[r][c]
            if sym is None:
                continue
            zones.setdefault(sym.value, []).append(input_grid.get(r, c))
    for label, cells in zones.items():
        arr = np.array(cells)
        ent = zone_entropy(arr)
        entropies[label] = ent
        if ent < 0.15:
            merge_with_neighbors(overlay, label)
    if not any(sym is not None for row in overlay for sym in row):
        overlay = None
    return overlay, entropies


def generate_fallback_rules(pair: Tuple[Grid, Grid]) -> List[SymbolicRule]:
    """Return simple fallback rules when extraction fails."""
    inp, out = pair
    heuristics = _heuristic_fallback_rules(inp, out)
    if heuristics:
        for r in heuristics:
            r.meta["fallback_reason"] = "heuristic"
        return heuristics

    fallback_templates = [
        SymbolicRule(
            transformation=Transformation(TransformationType.REPLACE),
            source=[Symbol(SymbolType.COLOR, "1")],
            target=[Symbol(SymbolType.COLOR, "2")],
            nature=TransformationNature.LOGICAL,
        ),
        SymbolicRule(
            transformation=Transformation(
                TransformationType.TRANSLATE, params={"dx": "1", "dy": "0"}
            ),
            source=[Symbol(SymbolType.REGION, "All")],
            target=[Symbol(SymbolType.REGION, "All")],
            nature=TransformationNature.SPATIAL,
        ),
    ]
    for r in fallback_templates:
        r.meta["fallback_reason"] = "no_rule_found"
    return fallback_templates

from arc_solver.src.segment.segmenter import zone_overlay, expand_zone_overlay
from arc_solver.src.core.grid_proxy import GridProxy
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.executor.scoring import score_rule, preferred_rule_types
from arc_solver.src.symbolic.rule_language import rule_to_dsl
from arc_solver.src.utils import config_loader


def zone_change_coverage_map(
    input_grid: Grid,
    output_grid: Grid,
    overlay: List[List[Optional[Symbol]]],
) -> Dict[str, float]:
    """Return mapping of zone label to change coverage ratio."""
    h = len(overlay)
    w = len(overlay[0]) if h else 0
    totals: Dict[str, int] = {}
    changed: Dict[str, int] = {}
    for r in range(h):
        for c in range(w):
            sym = overlay[r][c]
            if sym is None:
                continue
            zone = sym.value
            totals[zone] = totals.get(zone, 0) + 1
            if input_grid.get(r, c) != output_grid.get(r, c):
                changed[zone] = changed.get(zone, 0) + 1
    return {z: changed.get(z, 0) / totals[z] for z in totals}


def fuse_low_entropy_zones(
    overlay: List[List[Optional[Symbol]]],
    grid: Grid,
    entropies: Dict[str, float],
    threshold: float = 0.2,
) -> List[List[Optional[Symbol]]]:
    """Return ``overlay`` with adjacent low-entropy zones fused."""
    if overlay is None:
        return overlay

    h = len(overlay)
    w = len(overlay[0]) if h else 0
    zone_cells: Dict[str, List[int]] = {}
    for r in range(h):
        for c in range(w):
            sym = overlay[r][c]
            if sym is None:
                continue
            zone_cells.setdefault(sym.value, []).append(grid.get(r, c))

    dominant = {z: max(vals, key=vals.count) for z, vals in zone_cells.items() if vals}

    def adjacent(z1: str, z2: str) -> bool:
        for r in range(h):
            for c in range(w):
                sym = overlay[r][c]
                if sym is None or sym.value != z1:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        sym2 = overlay[nr][nc]
                        if sym2 is not None and sym2.value == z2:
                            return True
        return False

    zones = list(zone_cells.keys())
    for i, z1 in enumerate(zones):
        for z2 in zones[i + 1 :]:
            if (
                entropies.get(z1, 1.0) < threshold
                and entropies.get(z2, 1.0) < threshold
                and dominant.get(z1) == dominant.get(z2)
                and adjacent(z1, z2)
            ):
                for r in range(h):
                    for c in range(w):
                        if overlay[r][c] is not None and overlay[r][c].value == z2:
                            overlay[r][c] = Symbol(SymbolType.ZONE, z1)
                entropies.pop(z2, None)
    return overlay


# ---------------------------------------------------------------------------
# Layout detection helpers
# ---------------------------------------------------------------------------

def extract_layout_rules(input_grid: Grid, target_grid: Grid) -> List[SymbolicRule]:
    """Return simple layout level rules between input and target."""
    from arc_solver.src.utils.patterns import (
        detect_mirrored_regions,
        detect_repeating_blocks,
    )

    layout_rules: List[SymbolicRule] = []

    mirror_zones = detect_mirrored_regions(input_grid, target_grid)
    repeat_zones = detect_repeating_blocks(input_grid, target_grid)

    for axis in mirror_zones:
        rule = SymbolicRule(
            transformation=Transformation(
                TransformationType.FUNCTIONAL, params={"op": f"mirror_{axis}"}
            ),
            source=[Symbol(SymbolType.REGION, "All")],
            target=[Symbol(SymbolType.REGION, "All")],
            nature=TransformationNature.SPATIAL,
        )
        layout_rules.append(rule)
    for rep in repeat_zones:
        rule = SymbolicRule(
            transformation=Transformation(
                TransformationType.FUNCTIONAL, params={"op": f"repeat_{rep}"}
            ),
            source=[Symbol(SymbolType.REGION, "All")],
            target=[Symbol(SymbolType.REGION, "All")],
            nature=TransformationNature.SPATIAL,
        )
        layout_rules.append(rule)

    return layout_rules


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def extract_color_change_rules(
    input_grid: Grid,
    output_grid: Grid,
    zone_overlay: Optional[List[List[Symbol]]] = None,
    zone_entropy_map: Optional[Dict[str, float]] = None,
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
                rule.meta["derivation"] = {
                    "heuristic_used": "color_change",
                    "zone_entropy": None,
                }
                rules.append(rule)
        return rules

    # Zone-aware extraction
    zone_maps: Dict[str, Dict[int, set[int]]] = {}
    zone_totals: Dict[str, int] = {}
    zone_changes: Dict[str, int] = {}
    for r in range(h):
        for c in range(w):
            zone_sym = zone_overlay[r][c]
            if zone_sym is None:
                continue
            zone = zone_sym.value
            zone_totals[zone] = zone_totals.get(zone, 0) + 1
            src = input_grid.get(r, c)
            tgt = output_grid.get(r, c)
            if src != tgt:
                zone_changes[zone] = zone_changes.get(zone, 0) + 1
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
            rule = SymbolicRule(
                transformation=Transformation(TransformationType.REPLACE),
                source=[Symbol(SymbolType.COLOR, str(src_color))],
                target=[Symbol(SymbolType.COLOR, str(tgt_color))],
                nature=TransformationNature.LOGICAL,
            )
            rule.meta["derivation"] = {
                "heuristic_used": "color_change",
                "zone_entropy": None,
            }
            rules.append(rule)
        return rules

    for zone, mapping in zone_maps.items():
        for src_color, tgts in mapping.items():
            if len(tgts) == 1:
                tgt_color = next(iter(tgts))
                rule = SymbolicRule(
                    transformation=Transformation(TransformationType.REPLACE),
                    source=[Symbol(SymbolType.COLOR, str(src_color))],
                    target=[Symbol(SymbolType.COLOR, str(tgt_color))],
                    nature=TransformationNature.LOGICAL,
                    condition={"zone": zone},
                )
                ent = None
                if zone_entropy_map is not None:
                    ent = zone_entropy_map.get(zone)
                rule.meta["derivation"] = {
                    "heuristic_used": "color_change",
                    "zone_entropy": ent,
                }
                coverage = zone_changes.get(zone, 0) / zone_totals.get(zone, 1)
                rule.meta["zone_coverage"] = coverage
                if coverage < config_loader.ZONE_COVERAGE_THRESHOLD:
                    rule.condition.pop("zone", None)
                rules.append(rule)
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
    zone_totals: Dict[str, int] = {}
    zone_changes: Dict[str, int] = {}
    for r in range(h):
        for c in range(w):
            zone = zone_overlay[r][c].value
            zone_totals[zone] = zone_totals.get(zone, 0) + 1
            src = input_grid.get(r, c)
            tgt = output_grid.get(r, c)
            if src != tgt:
                zone_changes[zone] = zone_changes.get(zone, 0) + 1
                zone_map = zone_mappings.setdefault(zone, {})
                zone_map.setdefault(src, set()).add(tgt)

    rules: List[SymbolicRule] = []
    for zone, mapping in zone_mappings.items():
        for src_color, tgts in mapping.items():
            if len(tgts) == 1:
                tgt_color = next(iter(tgts))
                rule = SymbolicRule(
                    transformation=Transformation(TransformationType.REPLACE),
                    source=[
                        Symbol(SymbolType.ZONE, zone),
                        Symbol(SymbolType.COLOR, str(src_color)),
                    ],
                    target=[Symbol(SymbolType.COLOR, str(tgt_color))],
                    nature=TransformationNature.LOGICAL,
                )
                coverage = zone_changes.get(zone, 0) / zone_totals.get(zone, 1)
                rule.meta["zone_coverage"] = coverage
                if coverage < config_loader.ZONE_COVERAGE_THRESHOLD:
                    rule.condition.pop("zone", None)
                rules.append(rule)
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


def _detect_zone_morphology(
    input_grid: Grid, output_grid: Grid
) -> List[SymbolicRule]:
    """Detect simple dilation or erosion of a connected region."""
    if input_grid.shape() != output_grid.shape():
        return []
    overlay = label_connected_regions(input_grid)
    zone_ids = {z for row in overlay for z in row if z is not None}
    for zid in zone_ids:
        try:
            pred = dilate_zone(input_grid.to_list(), zid, overlay)
            if Grid(pred if isinstance(pred, list) else pred.tolist()) == output_grid:
                rule = SymbolicRule(
                    transformation=Transformation(
                        TransformationType.FUNCTIONAL,
                        params={"op": "dilate_zone", "zone": str(zid)},
                    ),
                    source=[Symbol(SymbolType.REGION, "All")],
                    target=[Symbol(SymbolType.REGION, "All")],
                    nature=TransformationNature.SPATIAL,
                )
                rule.meta["derivation"] = {"heuristic_used": "dilate_zone"}
                return [rule]
        except Exception:
            pass
        try:
            pred = erode_zone(input_grid.to_list(), zid, overlay)
            if Grid(pred if isinstance(pred, list) else pred.tolist()) == output_grid:
                rule = SymbolicRule(
                    transformation=Transformation(
                        TransformationType.FUNCTIONAL,
                        params={"op": "erode_zone", "zone": str(zid)},
                    ),
                    source=[Symbol(SymbolType.REGION, "All")],
                    target=[Symbol(SymbolType.REGION, "All")],
                    nature=TransformationNature.SPATIAL,
                )
                rule.meta["derivation"] = {"heuristic_used": "erode_zone"}
                return [rule]
        except Exception:
            pass
    return []


def generate_dilate_zone_rule(input_grid: Grid, output_grid: Grid) -> List[SymbolicRule]:
    """Return rule if ``output_grid`` is ``input_grid`` with a zone dilated."""
    if input_grid.shape() != output_grid.shape():
        return []
    overlay = label_connected_regions(input_grid)
    zone_ids = {z for row in overlay for z in row if z is not None}
    rules: List[SymbolicRule] = []
    for zid in zone_ids:
        try:
            pred = dilate_zone(input_grid.to_list(), zid, overlay)
        except Exception:
            continue
        if Grid(pred if isinstance(pred, list) else pred.tolist()) == output_grid:
            rule = SymbolicRule(
                transformation=Transformation(
                    TransformationType.FUNCTIONAL,
                    params={"op": "dilate_zone", "zone": str(zid)},
                ),
                source=[Symbol(SymbolType.REGION, "All")],
                target=[Symbol(SymbolType.REGION, "All")],
                nature=TransformationNature.SPATIAL,
            )
            rule.meta["derivation"] = {"heuristic_used": "dilate_zone"}
            logger.debug("dilate_zone matched zone=%s", zid)
            rules.append(rule)
            if len(rules) >= 25:
                break
    return rules


def generate_erode_zone_rule(input_grid: Grid, output_grid: Grid) -> List[SymbolicRule]:
    """Return rule if ``output_grid`` is ``input_grid`` with a zone eroded."""
    if input_grid.shape() != output_grid.shape():
        return []
    overlay = label_connected_regions(input_grid)
    zone_ids = {z for row in overlay for z in row if z is not None}
    rules: List[SymbolicRule] = []
    for zid in zone_ids:
        try:
            pred = erode_zone(input_grid.to_list(), zid, overlay)
        except Exception:
            continue
        if Grid(pred if isinstance(pred, list) else pred.tolist()) == output_grid:
            rule = SymbolicRule(
                transformation=Transformation(
                    TransformationType.FUNCTIONAL,
                    params={"op": "erode_zone", "zone": str(zid)},
                ),
                source=[Symbol(SymbolType.REGION, "All")],
                target=[Symbol(SymbolType.REGION, "All")],
                nature=TransformationNature.SPATIAL,
            )
            rule.meta["derivation"] = {"heuristic_used": "erode_zone"}
            logger.debug("erode_zone matched zone=%s", zid)
            rules.append(rule)
            if len(rules) >= 25:
                break
    return rules


def generate_mirror_tile_rule(input_grid: Grid, output_grid: Grid) -> List[SymbolicRule]:
    """Detect horizontal or vertical mirror tiling."""
    rules: List[SymbolicRule] = []
    ih, iw = input_grid.shape()
    oh, ow = output_grid.shape()
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
            pred = mirror_tile(input_grid, axis, count)
        except Exception:
            continue
        if pred == output_grid:
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
            logger.debug("mirror_tile matched axis=%s repeats=%s", axis, count)
            rules.append(rule)
    return rules[:25]


def generate_rotate_about_point_rule(input_grid: Grid, output_grid: Grid) -> List[SymbolicRule]:
    """Detect rotation around an arbitrary pivot point."""
    if input_grid.shape() != output_grid.shape():
        return []
    h, w = input_grid.shape()
    diff = [(r, c) for r in range(h) for c in range(w) if input_grid.get(r, c) != output_grid.get(r, c)]
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
                pred = rotate_about_point(input_grid, (cx, cy), angle)
            except Exception:
                continue
            if pred == output_grid:
                rule = SymbolicRule(
                    transformation=Transformation(
                        TransformationType.ROTATE,
                        params={"cx": str(cx), "cy": str(cy), "angle": str(angle)},
                    ),
                    source=[Symbol(SymbolType.REGION, "All")],
                    target=[Symbol(SymbolType.REGION, "All")],
                    nature=TransformationNature.SPATIAL,
                )
                rule.meta["derivation"] = {"heuristic_used": "rotate_about_point"}
                logger.debug("rotate_about_point matched pivot=(%s,%s) angle=%s", cx, cy, angle)
                rules.append(rule)
                if len(rules) >= 25:
                    return rules
    return rules


def generate_pattern_fill_rule(input_grid: Grid, output_grid: Grid) -> List[SymbolicRule]:
    """Wrapper for :func:`generate_pattern_fill_rules` for backward compatibility."""

    rules = generate_pattern_fill_rules(input_grid, output_grid)
    if rules:
        logger.debug("pattern_fill matched %d rule(s)", len(rules))
        return rules
    return []


def _detect_draw_line(
    input_grid: Grid, output_grid: Grid
) -> List[SymbolicRule]:
    """Detect drawing a straight line connecting two existing cells."""
    if input_grid.shape() != output_grid.shape():
        return []
    h, w = input_grid.shape()
    diff = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if input_grid.get(r, c) != output_grid.get(r, c)
    ]
    if not diff:
        return []
    color = output_grid.get(diff[0][0], diff[0][1])
    if any(output_grid.get(r, c) != color for r, c in diff):
        return []
    points = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if input_grid.get(r, c) != 0
    ]
    for i, p1 in enumerate(points):
        for p2 in points[i + 1 :]:
            try:
                pred = draw_line(input_grid.to_list(), p1, p2, color)
            except Exception:
                continue
            if Grid(pred if isinstance(pred, list) else pred.tolist()) == output_grid:
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
                return [rule]
    return []


def _detect_rotate_patch(
    input_grid: Grid, output_grid: Grid
) -> List[SymbolicRule]:
    """Detect small rotation around a pivot."""
    if input_grid.shape() != output_grid.shape():
        return []
    h, w = input_grid.shape()
    diff = [(r, c) for r in range(h) for c in range(w) if input_grid.get(r, c) != output_grid.get(r, c)]
    if not diff or len(diff) > 12:
        return []
    min_r = min(r for r, _ in diff)
    max_r = max(r for r, _ in diff)
    min_c = min(c for _, c in diff)
    max_c = max(c for _, c in diff)
    if max_r - min_r > 3 or max_c - min_c > 3:
        return []
    for cx in range(min_r, max_r + 1):
        for cy in range(min_c, max_c + 1):
            for angle in (90, 180, 270):
                try:
                    pred = rotate_about_point(input_grid, (cx, cy), angle)
                except Exception:
                    continue
                if pred == output_grid:
                    rule = SymbolicRule(
                        transformation=Transformation(
                            TransformationType.ROTATE,
                            params={"cx": str(cx), "cy": str(cy), "angle": str(angle)},
                        ),
                        source=[Symbol(SymbolType.REGION, "All")],
                        target=[Symbol(SymbolType.REGION, "All")],
                        nature=TransformationNature.SPATIAL,
                    )
                    rule.meta["derivation"] = {"heuristic_used": "rotate_about_point"}
                    return [rule]
    return []


def extract_shape_based_rules(input_grid: Grid, output_grid: Grid) -> List[SymbolicRule]:
    """Return shape abstraction, translation or rotation rules."""

    rules: List[SymbolicRule] = []

    # translation detection
    offset = _find_translation(input_grid, output_grid)
    if offset is not None:
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
        rule.meta["derivation"] = {"heuristic_used": "translation"}
        rules.append(rule)
        return rules

    morph = _detect_zone_morphology(input_grid, output_grid)
    if morph:
        for r in morph:
            r.dsl_str = rule_to_dsl(r)
            r.meta["score_trace"] = score_rule(input_grid, output_grid, r, return_trace=True)
        return morph

    candidates = generate_all_rules(
        input_grid,
        output_grid,
        allowlist=[
            "dilate_zone",
            "erode_zone",
            "draw_line",
            "rotate_about_point",
            "mirror_tile",
        ],
    )
    if candidates:
        return candidates

    rot_rule = _detect_rotate_patch(input_grid, output_grid)
    if rot_rule:
        for r in rot_rule:
            r.dsl_str = rule_to_dsl(r)
            r.meta["score_trace"] = score_rule(input_grid, output_grid, r, return_trace=True)
        return rot_rule

    # shape abstraction / rotation heuristics
    if input_grid.shape() == output_grid.shape():
        h, w = input_grid.shape()
        in_shape = [[1 if input_grid.get(r, c) != 0 else 0 for c in range(w)] for r in range(h)]
        out_shape = [[1 if output_grid.get(r, c) != 0 else 0 for c in range(w)] for r in range(h)]

        if in_shape == out_shape:
            rule = SymbolicRule(
                transformation=Transformation(TransformationType.SHAPE_ABSTRACT),
                source=[Symbol(SymbolType.SHAPE, "Any")],
                target=[Symbol(SymbolType.SHAPE, "Any")],
                nature=TransformationNature.LOGICAL,
            )
            rule.meta["derivation"] = {"heuristic_used": "shape"}
            rules.append(rule)
            return rules

        for t in range(1, 4):
            if Grid(in_shape).rotate90(t).data == out_shape:
                rule = SymbolicRule(
                    transformation=Transformation(
                        TransformationType.ROTATE90,
                        params={"times": str(t)},
                    ),
                    source=[Symbol(SymbolType.SHAPE, "Any")],
                    target=[Symbol(SymbolType.SHAPE, "Any")],
                    nature=TransformationNature.SPATIAL,
                )
                rule.meta["derivation"] = {"heuristic_used": "rotation"}
                rules.append(rule)
                return rules

    return rules


def extract_zone_remap_rules(
    input_grid: Grid | GridProxy, output_grid: Grid | GridProxy
) -> List[SymbolicRule]:
    """Return functional zone remap rules where entire zones recolour."""

    if input_grid.shape() != output_grid.shape():
        return []

    overlay_syms = (
        input_grid.get_zone_overlay() if isinstance(input_grid, GridProxy) else zone_overlay(input_grid)
    )
    h, w = input_grid.shape()

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
            zone_in.setdefault(zid, set()).add(input_grid.get(r, c))
            zone_out.setdefault(zid, set()).add(output_grid.get(r, c))

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

    predicted = zone_remap(input_grid.to_list(), overlay_ids, mapping)
    pred_grid = Grid(predicted if isinstance(predicted, list) else predicted.tolist())
    if pred_grid != output_grid:
        return []

    rule = SymbolicRule(
        transformation=Transformation(
            TransformationType.FUNCTIONAL, params={"op": "zone_remap"}
        ),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
        nature=TransformationNature.SPATIAL,
        meta={"mapping": mapping},
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


def retest_rules_on_pairs(
    rules: List[SymbolicRule], other_pairs: List[Tuple[Grid, Grid]]
) -> None:
    """Annotate ``rules`` with reliability across ``other_pairs``."""
    if not other_pairs:
        return
    for rule in rules:
        success = 0
        total = 0
        for inp, out in other_pairs:
            try:
                pred = simulate_rules(inp, [rule])
            except Exception:
                continue
            total += 1
            if pred == out:
                success += 1
        rule.meta["rule_reliability"] = success / total if total else 0.0


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def abstract(
    objects,
    *,
    logger=None,
    other_pairs: Optional[List[Tuple[Grid, Grid]]] = None,
    trace: bool = False,
) -> List[SymbolicRule]:
    """Return symbolic abstractions of a grid pair.

    When ``logger`` is provided, messages describing which extraction heuristics
    were used are emitted for debugging purposes.
    """
    if not isinstance(objects, (list, tuple)) or len(objects) < 2:
        return []

    input_grid, output_grid = objects[0], objects[1]
    trace_entry: Dict[str, int] | None = {} if trace else None
    layout = extract_layout_rules(input_grid, output_grid)
    overlay, zone_info = segment_and_overlay(input_grid, output_grid)
    if overlay is not None:
        coverage_map = zone_change_coverage_map(input_grid, output_grid, overlay)
        for z, cov in coverage_map.items():
            if cov < config_loader.ZONE_COVERAGE_THRESHOLD:
                overlay = expand_zone_overlay(overlay, z)
        overlay = fuse_low_entropy_zones(overlay, input_grid, zone_info)
    try:
        rules: List[SymbolicRule] = []
        cc_rules = extract_color_change_rules(
            input_grid, output_grid, zone_overlay=overlay, zone_entropy_map=zone_info
        )
        if logger:
            logger.info(f"color_change_rules: {len(cc_rules)}")
        rules.extend(cc_rules)
        mid_grid = simulate_rules(input_grid, cc_rules) if cc_rules else input_grid
        shape_rules = extract_shape_based_rules(mid_grid, output_grid)
        if logger:
            logger.info(f"shape_based_rules: {len(shape_rules)}")
        rules.extend(shape_rules)
        pf_rules = generate_pattern_fill_rule(mid_grid, output_grid)
        if logger:
            logger.info(f"pattern_fill_rules: {len(pf_rules)}")
        rules.extend(pf_rules)
        repeat_rules = generate_repeat_rules(mid_grid, output_grid, post_process=True)
        if logger:
            logger.info(f"repeat_rules: {len(repeat_rules)}")
        rules.extend([r for r in repeat_rules if not hasattr(r, "steps")])

        composite_candidates = [r for r in repeat_rules if hasattr(r, "steps")]
        for comp in composite_candidates:
            if not comp.is_well_formed():
                continue
            pred = comp.simulate(mid_grid)
            if pred.compare_to(output_grid) > mid_grid.compare_to(output_grid):
                rules.append(comp)
        if ENABLE_COMPOSITE_REPEAT:
            for rr in repeat_rules:
                rmap = rr.meta.get("replace_map") if hasattr(rr, "meta") else None
                if not rmap:
                    continue
                for src, tgt in rmap.items():
                    repl = SymbolicRule(
                        transformation=Transformation(TransformationType.REPLACE),
                        source=[Symbol(SymbolType.COLOR, str(src))],
                        target=[Symbol(SymbolType.COLOR, str(tgt))],
                        nature=TransformationNature.LOGICAL,
                    )
                    rr.meta.setdefault("secondary_rules", []).append(repl)
                    rules.append(repl)
        zone_rules = generate_zone_remap_rules(input_grid, output_grid)
        if logger:
            logger.info(f"zone_remap_rules: {len(zone_rules)}")
        rules.extend(zone_rules)

        composites = generate_morph_remap_composites(input_grid, output_grid)
        if logger:
            logger.info(f"morph_remap_composites: {len(composites)}")
        rules.extend(composites)
        split: List[SymbolicRule] = []
        for r in rules:
            if (
                r.transformation.ttype in (TransformationType.REPLACE, TransformationType.TRANSLATE)
                and overlay is not None
            ):
                split.extend(split_rule_by_overlay(r, input_grid, overlay))
            else:
                split.append(r)
        rules = split

        # Consolidate near-identical rules before scoring
        try:
            from arc_solver.src.utils.rule_utils import generalize_rules as _gen_rules

            rules = _gen_rules(rules)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Prioritize rules using scoring and strategy registry
        # ------------------------------------------------------------------
        scored: List[Tuple[SymbolicRule, float]] = []
        pref = preferred_rule_types(input_grid, output_grid)
        for r in rules:
            s = score_rule(input_grid, output_grid, r)
            bonus = 0.2 if r.transformation.ttype.value in pref else 0.0
            chain_pen = len(getattr(r, "steps", []))
            scored.append((r, s + bonus - 0.05 * chain_pen))
        scored.sort(key=lambda x: x[1], reverse=True)
        if logger:
            ranking = [
                {"type": r.transformation.ttype.name, "score": float(s)}
                for r, s in scored
            ]
            logger.debug("ranked_rules=%s", ranking)
        TOP_N = 25
        rules = [r for r, _ in scored[:TOP_N]]
    except Exception:
        if logger:
            logger.warning("abstraction failure, falling back")
        rules = []

    if not rules or any(not getattr(r, "is_well_formed", lambda: False)() for r in rules):
        if logger:
            logger.warning("No rules extracted. Triggering fallback generator.")
        rules = _heuristic_fallback_rules(input_grid, output_grid)
        if not rules:
            rules = generate_fallback_rules((input_grid, output_grid))
    filtered: List[SymbolicRule] = []
    for rule in rules:
        if hasattr(rule, "is_well_formed") and not rule.is_well_formed():
            continue
        filtered.append(rule)

    valid_rules = filtered
    fallback_rules: List[SymbolicRule] = []
    if ENABLE_FALLBACK_COMPOSITES and len(valid_rules) == 0:
        from arc_solver.src.symbolic.composite_rules import (
            generate_repeat_composite_rules,
        )
        fallback_rules = generate_repeat_composite_rules(input_grid, output_grid)
        fallback_rules = [
            r
            for r in fallback_rules
            if score_rule(input_grid, output_grid, r) > 0.5
        ]
        valid_rules.extend(fallback_rules)

    if (
        ENABLE_FALLBACK_COMPOSITES
        and valid_rules
        and max(score_rule(input_grid, output_grid, r) for r in valid_rules)
        < RELIABILITY_THRESHOLD
    ):
        from arc_solver.src.abstractions.rule_generator import fallback_composite_rules

        comps = fallback_composite_rules(
            valid_rules, input_grid, output_grid, score_threshold=RELIABILITY_THRESHOLD
        )
        fallback_rules.extend(comps)
        valid_rules.extend(comps)

    if trace_entry is not None:
        trace_entry["fallback_rules_used"] = len(fallback_rules)

    if RELIABILITY_THRESHOLD:
        before = len(valid_rules)
        valid_rules = [
            r
            for r in valid_rules
            if getattr(r, "reliability", r.meta.get("rule_reliability", 1.0))
            >= RELIABILITY_THRESHOLD
        ]
        if trace_entry is not None:
            trace_entry["filtered_by_reliability"] = before - len(valid_rules)

    if other_pairs:
        retest_rules_on_pairs(valid_rules, other_pairs)

    if not valid_rules:
        try:
            discovered = auto_discover_rules("unknown", input_grid, output_grid, {})
            valid_rules.extend(discovered)
        except Exception:
            pass

    for r in valid_rules:
        if not hasattr(r, "dsl_str"):
            if hasattr(r, "as_symbolic_proxy"):
                proxy = r.as_symbolic_proxy()
                r.dsl_str = rule_to_dsl(proxy)
            else:
                r.dsl_str = rule_to_dsl(r)
        if "score_trace" not in r.meta:
            r.meta["score_trace"] = score_rule(input_grid, output_grid, r, return_trace=True)

    return layout + valid_rules


__all__ = [
    "extract_layout_rules",
    "extract_color_change_rules",
    "extract_shape_based_rules",
    "extract_zone_remap_rules",
    "generate_dilate_zone_rule",
    "generate_erode_zone_rule",
    "generate_mirror_tile_rule",
    "generate_rotate_about_point_rule",
    "generate_mirror_tile_rules",
    "generate_draw_line_rules",
    "generate_dilate_zone_rules",
    "generate_erode_zone_rules",
    "generate_zone_remap_rules",
    "generate_rotate_about_point_rules",
    "generate_morph_remap_composites",
    "generate_pattern_fill_rule",
    "extract_zonewise_rules",
    "split_rule_by_overlay",
    "retest_rules_on_pairs",
    "abstract",
]
