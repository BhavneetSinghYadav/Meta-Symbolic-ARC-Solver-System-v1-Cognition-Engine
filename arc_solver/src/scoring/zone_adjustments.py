from __future__ import annotations

"""Zone-aware scoring helpers."""

from collections import Counter
from math import log2
from typing import Any, List

from arc_solver.src.core.grid import Grid
from arc_solver.src.core.grid_proxy import GridProxy
from arc_solver.src.segment.segmenter import zone_overlay


def _label_of(zone: Any) -> str:
    """Return string label for ``zone``."""
    try:
        return zone.label  # type: ignore[attr-defined]
    except Exception:
        return str(zone)


def zone_entropy_penalty(grid: Grid | GridProxy, zones: List[Any]) -> float:
    """Return negative entropy penalty for ``zones`` within ``grid``.

    Each zone's entropy is computed from its pixel color distribution. The
    penalty is the average normalized entropy scaled by ``-0.1`` so that chaotic
    zones slightly reduce the score.
    """
    if not zones:
        return 0.0

    overlay = grid.get_zone_overlay() if isinstance(grid, GridProxy) else zone_overlay(grid)
    h = len(overlay)
    w = len(overlay[0]) if h else 0

    zone_cells: dict[str, List[int]] = { _label_of(z): [] for z in zones }
    for r in range(h):
        for c in range(w):
            sym = overlay[r][c]
            if sym is None:
                continue
            label = sym.value
            if label in zone_cells:
                zone_cells[label].append(grid.get(r, c))

    entropies: List[float] = []
    for label, cells in zone_cells.items():
        if not cells:
            entropies.append(0.0)
            continue
        counts = Counter(cells)
        total = len(cells)
        ent = 0.0
        for cnt in counts.values():
            p = cnt / total
            ent -= p * log2(p)
        max_ent = log2(len(counts)) if len(counts) > 1 else 0.0
        entropies.append(ent / max_ent if max_ent else 0.0)

    avg_ent = sum(entropies) / len(entropies) if entropies else 0.0
    return -0.1 * avg_ent


def zone_alignment_bonus(
    predicted: Grid | GridProxy, target: Grid | GridProxy, zones: List[Any]
) -> float:
    """Return bonus based on how well ``predicted`` aligns to ``target`` per zone."""
    if not zones:
        return 0.0

    pred_overlay = (
        predicted.get_zone_overlay() if isinstance(predicted, GridProxy) else zone_overlay(predicted)
    )
    tgt_overlay = (
        target.get_zone_overlay() if isinstance(target, GridProxy) else zone_overlay(target)
    )
    h = len(pred_overlay)
    w = len(pred_overlay[0]) if h else 0

    overlap_scores: List[float] = []
    for zone in zones:
        label = _label_of(zone)
        pred_cells = {
            (r, c)
            for r in range(h)
            for c in range(w)
            if pred_overlay[r][c] is not None and pred_overlay[r][c].value == label
        }
        tgt_cells = {
            (r, c)
            for r in range(h)
            for c in range(w)
            if tgt_overlay[r][c] is not None and tgt_overlay[r][c].value == label
        }
        union = pred_cells | tgt_cells
        if not union:
            continue
        inter = pred_cells & tgt_cells
        overlap_scores.append(len(inter) / len(union))

    if not overlap_scores:
        return 0.0
    avg_overlap = sum(overlap_scores) / len(overlap_scores)
    return 0.1 * avg_overlap


def zone_coverage_weight(predicted: Grid | GridProxy, zones: List[Any]) -> float:
    """Return coverage weight of ``predicted`` across ``zones``.

    The weight is the average ratio of non-zero pixels inside each zone,
    yielding a value in ``[0.0, 1.0]``. Higher coverage indicates that the rule
    impacted most cells within the zone.
    """
    if not zones:
        return 0.0

    overlay = (
        predicted.get_zone_overlay() if isinstance(predicted, GridProxy) else zone_overlay(predicted)
    )
    h = len(overlay)
    w = len(overlay[0]) if h else 0

    coverages: List[float] = []
    for zone in zones:
        label = _label_of(zone)
        cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if overlay[r][c] is not None and overlay[r][c].value == label
        ]
        if not cells:
            continue
        covered = sum(1 for r, c in cells if predicted.get(r, c) != 0)
        coverages.append(covered / len(cells))

    if not coverages:
        return 0.0
    return sum(coverages) / len(coverages)


__all__ = [
    "zone_entropy_penalty",
    "zone_alignment_bonus",
    "zone_coverage_weight",
]
