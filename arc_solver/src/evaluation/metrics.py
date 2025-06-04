from __future__ import annotations

"""Evaluation utilities for ARC task predictions."""

from typing import List

from arc_solver.src.core.grid import Grid


def accuracy_score(predicted: Grid, target: Grid) -> float:
    """Return cell-wise accuracy between two grids."""
    return predicted.compare_to(target)


def task_score(preds: List[Grid], targets: List[Grid]) -> float:
    """Return the proportion of grids predicted exactly."""
    if not preds or not targets:
        return 0.0
    correct = 0
    for p, t in zip(preds, targets):
        if p.compare_to(t) == 1.0:
            correct += 1
    return correct / len(targets)


def aggregate_accuracy(task_scores: List[float]) -> float:
    """Return mean accuracy over multiple tasks."""
    return sum(task_scores) / len(task_scores) if task_scores else 0.0

__all__ = ["accuracy_score", "task_score", "aggregate_accuracy"]
