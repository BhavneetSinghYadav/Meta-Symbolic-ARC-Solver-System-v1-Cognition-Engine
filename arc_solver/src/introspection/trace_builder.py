"""Utilities for constructing symbolic execution traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import Symbol, SymbolType, SymbolicRule
from arc_solver.src.segment.segmenter import zone_overlay
import logging


@dataclass
class RuleTrace:
    """Structured record of a single symbolic rule application."""

    rule: SymbolicRule
    affected_cells: List[Tuple[int, int]]
    predicted_grid: Grid
    ground_truth: Optional[Grid]
    delta_mask: List[List[bool]]
    match_score: float
    symbolic_context: Dict[str, Any] = field(default_factory=dict)


def build_trace(
    rule: SymbolicRule,
    grid_in: Grid,
    grid_out: Grid,
    grid_true: Optional[Grid],
    symbolic_overlay: Optional[List[List[Symbol]]] = None,
) -> RuleTrace:
    """Return a :class:`RuleTrace` describing ``rule``'s effect."""

    logger = logging.getLogger("trace_builder")
    height, width = grid_in.shape()

    affected: List[Tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            if grid_in.get(r, c) != grid_out.get(r, c):
                affected.append((r, c))
                logger.debug("cell %d,%d changed from %s to %s", r, c, grid_in.get(r, c), grid_out.get(r, c))

    if grid_true is not None and grid_true.shape() == grid_out.shape():
        delta_mask = [
            [grid_out.get(r, c) != grid_true.get(r, c) for c in range(width)]
            for r in range(height)
        ]
        match_score = grid_out.compare_to(grid_true)
    else:
        delta_mask = [[False for _ in range(width)] for _ in range(height)]
        match_score = 1.0 if grid_true is None else 0.0

    context: Dict[str, Any] = {}
    if rule.condition:
        context["condition"] = dict(rule.condition)
    if symbolic_overlay is None and rule.condition.get("zone"):
        symbolic_overlay = zone_overlay(grid_in)

    if symbolic_overlay is not None and len(symbolic_overlay) == height:
        zones: set[str] = set()
        regions: set[str] = set()
        cell_labels: Dict[Tuple[int, int], List[str]] = {}
        for r, c in affected:
            syms = symbolic_overlay[r][c]
            if syms is None:
                continue
            if not isinstance(syms, list):
                syms = [syms]
            cell_labels[(r, c)] = [str(s) for s in syms]
            for s in syms:
                if s.type is SymbolType.ZONE:
                    zones.add(s.value)
                elif s.type is SymbolType.REGION:
                    regions.add(s.value)
            logger.debug("trace cell %d,%d labels=%s", r, c, [str(s) for s in syms])
        if zones:
            context["zones"] = sorted(zones)
        if regions:
            context["regions"] = sorted(regions)
        if cell_labels:
            context["labels"] = cell_labels
        if zones:
            hierarchy = {"rule_set": {z: {"rule": str(rule)}} for z in zones}
            context["hierarchy"] = hierarchy

    if getattr(rule, "meta", None):
        context.setdefault("rule_derivations", []).append(rule.meta)

    return RuleTrace(
        rule=rule,
        affected_cells=affected,
        predicted_grid=grid_out,
        ground_truth=grid_true,
        delta_mask=delta_mask,
        match_score=match_score,
        symbolic_context=context,
    )


__all__ = ["RuleTrace", "build_trace"]
