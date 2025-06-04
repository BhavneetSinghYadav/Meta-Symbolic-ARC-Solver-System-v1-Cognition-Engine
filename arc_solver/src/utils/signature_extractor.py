from __future__ import annotations

"""Utilities for deriving symbolic signatures from tasks."""

from typing import Any, Dict, List, Tuple

from arc_solver.src.core.grid import Grid


def _collect_grids(task: Dict[str, Any]) -> List[Grid]:
    grids: List[Grid] = []
    for pair in task.get("train", []):
        grids.append(Grid(pair["input"]))
        grids.append(Grid(pair["output"]))
    for item in task.get("test", []):
        data = item["input"] if isinstance(item, dict) and "input" in item else item
        grids.append(Grid(data))
    return grids


def extract_task_signature(task: Dict[str, Any]) -> str:
    """Return a symbolic fingerprint string for ``task``."""
    grids = _collect_grids(task)
    if not grids:
        return ""
    colors = set()
    shapes = set()
    symmetric = False
    for g in grids:
        colors.update(v for row in g.data for v in row)
        shapes.add(g.shape())
        if g.data == g.flip_horizontal().data:
            symmetric = True
    num_colors = len(colors)
    shape_str = "-".join(f"{h}x{w}" for h, w in sorted(shapes))
    sym_str = "vsym" if symmetric else "nosym"
    return f"{num_colors}-colors_{shape_str}_{sym_str}"


def similarity_score(sig1: str, sig2: str) -> float:
    """Simple Jaccard similarity between underscore-separated tokens."""
    set1 = set(sig1.split("_"))
    set2 = set(sig2.split("_"))
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


__all__ = ["extract_task_signature", "similarity_score"]
