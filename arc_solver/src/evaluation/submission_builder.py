from __future__ import annotations

"""Utilities for building ARC AGI competition submission files."""

from typing import Dict, List

from arc_solver.src.core.grid import Grid
from arc_solver.src.data.agi_loader import ARCAGITask


def build_submission_json(
    tasks: List[ARCAGITask], predictions: Dict[tuple[str, int], Grid | dict | list]
) -> dict:
    """Return a submission dictionary compatible with the ARC AGI leaderboard."""

    submission: Dict[str, Dict[str, List[List[int]] | List[List[List[int]]]]] = {}
    assert predictions, "Predictions dictionary is empty"
    for task in tasks:
        outputs: List[List[List[int]]] = []
        for i in range(len(task.test)):
            key = (task.task_id, i)
            if key not in predictions:
                raise KeyError(f"Missing prediction for {task.task_id} index {i}")
            pred = predictions[key]
            # Unwrap structures like {"output": grid} returned by some solvers
            if isinstance(pred, dict) and "output" in pred:
                pred = pred["output"]
            if isinstance(pred, Grid):
                grid = pred.to_list()
            elif isinstance(pred, list):
                grid = pred
            else:
                raise TypeError(
                    f"Prediction for {task.task_id} index {i} has invalid type"
                )
            if not grid or not isinstance(grid, list) or not isinstance(grid[0], list):
                raise ValueError(
                    f"Prediction for {task.task_id} index {i} is malformed: {pred}"
                )
            outputs.append(grid)
        submission[task.task_id] = {
            "output": outputs if len(outputs) > 1 else outputs[0]
        }
    return submission

__all__ = ["build_submission_json"]
