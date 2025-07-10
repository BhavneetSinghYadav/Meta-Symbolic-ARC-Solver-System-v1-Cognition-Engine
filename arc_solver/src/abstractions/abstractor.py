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
    input_grid: Grid, output_grid: Grid
) -> Tuple[List[List[Optional[Symbol]]], Dict[str, float]]:
    """Return IO-aligned zone overlay and entropy map."""
    inp_overlay = zone_overlay(input_grid)
    out_overlay = zone_overlay(output_grid)
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
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.executor.scoring import score_rule, preferred_rule_types
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

    return layout + valid_rules


__all__ = [
    "extract_layout_rules",
    "extract_color_change_rules",
    "extract_shape_based_rules",
    "extract_zonewise_rules",
    "split_rule_by_overlay",
    "retest_rules_on_pairs",
    "abstract",
]
