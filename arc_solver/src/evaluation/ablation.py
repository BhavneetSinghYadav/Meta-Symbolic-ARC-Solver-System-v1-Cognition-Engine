"""Utilities for ablation testing of structural attention."""

from __future__ import annotations

from typing import Tuple

from arc_solver.src.executor.full_pipeline import solve_task
from arc_solver.src.utils import config_loader
from arc_solver.src.evaluation.metrics import task_score


def compare_attention(task: dict) -> Tuple[float, float]:
    """Return task scores with and without structural attention."""

    config_loader.set_use_structural_attention(False)
    preds_no, targets, _, _ = solve_task(task)
    score_no = task_score(preds_no, targets)

    config_loader.set_use_structural_attention(True)
    preds_yes, targets, _, _ = solve_task(task)
    score_yes = task_score(preds_yes, targets)

    return score_no, score_yes


__all__ = ["compare_attention"]
