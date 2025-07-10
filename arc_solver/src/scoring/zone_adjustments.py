from __future__ import annotations

"""Spatial zone-based scoring adjustments."""

from typing import Optional

from arc_solver.src.core.grid import Grid
from arc_solver.src.segment.segmenter import zone_overlay
from arc_solver.src.utils.entropy import compute_zone_entropy_map


def _rule_zone(rule) -> Optional[str]:
    cond = getattr(rule, "condition", None)
    if not cond:
        return None
    return cond.get("zone")


def zone_entropy_penalty(input_grid: Grid, rule) -> float:
    """Return penalty based on entropy of the rule's target zone."""
    zone = _rule_zone(rule)
    if not zone:
        return 0.0
    ent_map = compute_zone_entropy_map(input_grid)
    entropy = ent_map.get(zone, 1.0)
    return (1.0 - entropy) * 0.1


def zone_alignment_bonus(input_grid: Grid, output_grid: Grid, rule) -> float:
    """Return bonus for alignment of ``rule`` zone across ``input`` and ``output``."""
    zone = _rule_zone(rule)
    if not zone:
        return 0.0
    if input_grid.shape() != output_grid.shape():
        return 0.0
    ov_in = zone_overlay(input_grid)
    ov_out = zone_overlay(output_grid)
    h = len(ov_in)
    w = len(ov_in[0]) if h else 0
    total = 0
    matches = 0
    for r in range(h):
        for c in range(w):
            if ov_in[r][c] is not None and ov_in[r][c].value == zone:
                total += 1
                if ov_out[r][c] is not None and ov_out[r][c].value == zone:
                    matches += 1
    if total == 0:
        return 0.0
    ratio = matches / total
    return ratio * 0.05


def zone_coverage_weight(input_grid: Grid, output_grid: Grid) -> float:
    """Return multiplicative weight based on zone alignment coverage."""
    if input_grid.shape() != output_grid.shape():
        return 1.0
    ov_in = zone_overlay(input_grid)
    ov_out = zone_overlay(output_grid)
    h = len(ov_out)
    w = len(ov_out[0]) if h else 0
    total = 0
    matches = 0
    for r in range(h):
        for c in range(w):
            if ov_out[r][c] is not None:
                total += 1
                if ov_in[r][c] is not None and ov_in[r][c].value == ov_out[r][c].value:
                    matches += 1
    if total == 0:
        return 1.0
    ratio = matches / total
    return 1.0 + 0.1 * ratio


__all__ = [
    "zone_entropy_penalty",
    "zone_alignment_bonus",
    "zone_coverage_weight",
]

