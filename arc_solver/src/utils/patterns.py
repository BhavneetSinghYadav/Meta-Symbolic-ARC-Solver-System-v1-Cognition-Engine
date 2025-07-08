from __future__ import annotations

from typing import Dict, List
import logging

from arc_solver.src.core.grid import Grid


def detect_mirrored_regions(inp: Grid, out: Grid) -> List[str]:
    """Return list describing mirror relationship between ``inp`` and ``out``."""
    if inp.shape() != out.shape():
        return []
    inp_rows = inp.to_list()
    out_rows = out.to_list()
    zones: List[str] = []
    if out_rows == [row[::-1] for row in inp_rows]:
        zones.append("horizontal")
    if out_rows == inp_rows[::-1]:
        zones.append("vertical")
    return zones


def detect_repeating_blocks(inp: Grid, out: Grid) -> List[str]:
    """Return list of repeating row or column patterns in ``out``."""
    rows = out.to_list()
    row_repeat = any(rows[i] == rows[i + 1] for i in range(len(rows) - 1)) if len(rows) > 1 else False
    cols = list(zip(*rows)) if rows else []
    col_repeat = any(list(cols[i]) == list(cols[i + 1]) for i in range(len(cols) - 1)) if len(cols) > 1 else False
    zones: List[str] = []
    if row_repeat:
        zones.append("row")
    if col_repeat:
        zones.append("column")
    return zones


def detect_replace_map(
    grid1: Grid, grid2: Grid, *, max_substitutions: int = 3
) -> Dict[int, int]:
    """Return color mapping from ``grid1`` to ``grid2`` if consistent."""
    logger = logging.getLogger(__name__)
    if grid1.shape() != grid2.shape():
        return {}
    h, w = grid1.shape()
    mapping: Dict[int, int] = {}
    for r in range(h):
        for c in range(w):
            a = grid1.get(r, c)
            b = grid2.get(r, c)
            if a == b:
                continue
            if a in mapping and mapping[a] != b:
                return {}
            mapping[a] = b
    if len(mapping) > max_substitutions:
        logger.warning("replace_map exceeds max_substitutions")
    return mapping


__all__ = [
    "detect_mirrored_regions",
    "detect_repeating_blocks",
    "detect_replace_map",
]
