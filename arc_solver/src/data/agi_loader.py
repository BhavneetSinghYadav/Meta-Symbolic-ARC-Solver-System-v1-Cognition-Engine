from __future__ import annotations

"""Loader utilities for the ARC AGI dataset format."""

from pathlib import Path
import json
from typing import Any, List, Optional

from arc_solver.src.core.grid import Grid


class ARCAGITask:
    """Container for a single ARC AGI challenge."""

    def __init__(
        self,
        task_id: str,
        train_pairs: list,
        test_inputs: list,
        ground_truth: Optional[list] = None,
    ) -> None:
        self.task_id = task_id

        def _to_grid(obj: Any) -> Grid:
            data = obj
            if isinstance(obj, dict):
                data = obj.get("input", obj.get("output"))
            if not isinstance(data, list):
                raise ValueError(f"Malformed grid in task {task_id}")
            return Grid(data)

        self.train = [(_to_grid(p["input"] if isinstance(p, dict) else p[0]),
                       _to_grid(p["output"] if isinstance(p, dict) else p[1]))
                      for p in train_pairs]
        self.test = [
            _to_grid(t["input"] if isinstance(t, dict) and "input" in t else t)
            for t in test_inputs
        ]
        self.ground_truth = [_to_grid(g) for g in ground_truth] if ground_truth else None


def load_agi_tasks(challenges_path: Path | str, solutions_path: Optional[Path | str] = None) -> List[ARCAGITask]:
    """Load ARC AGI tasks from challenge and optional solution files."""

    with open(Path(challenges_path), "r", encoding="utf-8") as f:
        challenge_data = json.load(f)

    solution_data = {}
    if solutions_path is not None:
        with open(Path(solutions_path), "r", encoding="utf-8") as f:
            solution_data = json.load(f)

    tasks: List[ARCAGITask] = []
    for task_id, task in challenge_data.items():
        train = task["train"]
        test = task.get("test", [])
        gt = solution_data.get(task_id, {}).get("output", None)
        tasks.append(ARCAGITask(task_id, train, test, gt))
    return tasks


__all__ = ["ARCAGITask", "load_agi_tasks"]

