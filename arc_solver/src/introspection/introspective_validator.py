"""Heuristics for validating symbolic execution traces.

`validate_trace` analyses a :class:`RuleTrace` and returns a structured
dictionary describing how well the rule's output matches the ground truth.
The metrics attempt to capture whether the rule was applied in the correct
symbolic region and if any semantic conflicts occurred during execution.
"""

from __future__ import annotations

from typing import Any, Dict, List
import math

from arc_solver.src.core.grid import Grid

from .trace_builder import RuleTrace


def validate_trace(trace: RuleTrace) -> Dict[str, Any]:
    """Return validation metrics for ``trace``.

    The returned dictionary has the following keys::

        {
            "coverage_score": float,
            "correct_cells": int,
            "total_cells": int,
            "symbolic_consistency": bool,
            "conflict_flags": List[str],
            "interpretation_valid": bool,
        }
    """

    height, width = trace.predicted_grid.shape()
    total_cells = height * width

    # ------------------------------------------------------------------
    # Evaluate cell level agreement with the ground truth
    # ------------------------------------------------------------------
    mismatched: List[tuple[int, int]] = []
    if trace.ground_truth is not None:
        for r in range(height):
            for c in range(width):
                if trace.delta_mask[r][c]:
                    mismatched.append((r, c))

    correct_cells = total_cells - len(mismatched)
    coverage_score = correct_cells / total_cells if total_cells else 1.0

    # ------------------------------------------------------------------
    # Symbolic consistency heuristics
    # ------------------------------------------------------------------
    # ``symbolically_matched`` is True when all mismatched cells were within the
    # set of cells transformed by the rule.  In this case the rule logically
    # targeted the right area but produced the wrong pixel values.
    symbolically_matched = all(loc in trace.affected_cells for loc in mismatched)
    symbolic_consistency = symbolically_matched

    # Determine simple conflict flags
    conflict_flags: List[str] = []

    if mismatched and symbolically_matched:
        conflict_flags.append("visual_failure")
    elif mismatched and not symbolically_matched:
        conflict_flags.append("structural_mismatch")

    # Detect repeated region labels in the symbolic context
    labels = trace.symbolic_context.get("labels", {})
    if labels:
        region_count: Dict[str, int] = {}
        for labs in labels.values():
            for lab in labs:
                region_count[lab] = region_count.get(lab, 0) + 1
        for label, count in region_count.items():
            if count > len(labels):
                conflict_flags.append("region_repeated")
                break

    # ``interpretation_valid`` approximates whether the rule behaved as
    # intended. High coverage and no conflicts imply success.
    interpretation_valid = (
        coverage_score >= 0.99 and symbolic_consistency and not conflict_flags
    )

    def _grid_entropy(grid: Grid) -> float:
        counts = grid.count_colors()
        total = sum(counts.values())
        ent = 0.0
        for v in counts.values():
            if v == 0:
                continue
            p = v / total
            ent -= p * math.log2(p)
        return ent

    entropy_change = 0.0
    if trace.ground_truth is not None:
        try:
            entropy_pred = _grid_entropy(trace.predicted_grid)
            entropy_true = _grid_entropy(trace.ground_truth)
            entropy_change = entropy_true - entropy_pred
        except Exception:
            entropy_change = 0.0

    return {
        "coverage_score": coverage_score,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "symbolic_consistency": symbolic_consistency,
        "conflict_flags": conflict_flags,
        "interpretation_valid": interpretation_valid,
        "entropy_change": entropy_change,
    }


__all__ = ["validate_trace"]
