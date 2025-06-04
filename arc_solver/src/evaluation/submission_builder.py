from __future__ import annotations

"""Utilities for building ARC AGI competition submission files."""

from typing import Dict, List

from arc_solver.src.core.grid import Grid
from arc_solver.src.data.agi_loader import ARCAGITask


def build_submission_json(tasks: List[ARCAGITask], predictions: Dict[tuple[str, int], Grid]) -> dict:
    """Return a submission dictionary compatible with the ARC AGI leaderboard."""

    submission: Dict[str, Dict[str, List[List[int]] | List[List[List[int]]]]] = {}
    for task in tasks:
        outputs = [predictions[(task.task_id, i)].to_list() for i in range(len(task.test))]
        submission[task.task_id] = {
            "output": outputs if len(outputs) > 1 else outputs[0]
        }
    return submission

__all__ = ["build_submission_json"]
