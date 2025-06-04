from __future__ import annotations

"""Run the symbolic solver on the official ARC AGI dataset."""

import argparse
import json
from pathlib import Path
from typing import List

from arc_solver.src.core.grid import Grid

from arc_solver.src.data.agi_loader import load_agi_tasks, ARCAGITask
from arc_solver.src.executor.full_pipeline import solve_task as pipeline_solve_task
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.evaluation.metrics import accuracy_score
from arc_solver.src.evaluation.submission_builder import build_submission_json


def _predict(task: ARCAGITask, *, introspect: bool = False, threshold: float = 0.9):
    """Return predictions for ``task`` optionally refining with introspection."""

    train_dicts = [
        {"input": inp.data, "output": out.data} for inp, out in task.train
    ]
    test_dicts = [{"input": g.data} for g in task.test]
    json_task = {"train": train_dicts, "test": test_dicts}

    preds, _, _, rules = pipeline_solve_task(json_task, introspect=False)

    if introspect and task.train:
        score = sum(
            accuracy_score(simulate_rules(inp, rules), out)
            for inp, out in task.train
        ) / len(task.train)
        if score < threshold:
            preds, _, _, _ = pipeline_solve_task(json_task, introspect=True)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the solver on AGI dataset")
    parser.add_argument(
        "--split",
        choices=["train", "evaluation", "test"],
        default="test",
        help="Which dataset split to run",
    )
    parser.add_argument("--introspect", action="store_true", help="Enable introspection refinement")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for introspection",
    )
    args = parser.parse_args()

    split_map = {
        "train": "arc-agi_training-challenges.json",
        "evaluation": "arc-agi_evaluation-challenges.json",
        "test": "arc-agi_test-challenges.json",
    }
    challenges_path = Path(split_map[args.split])

    tasks = load_agi_tasks(challenges_path)

    predictions: dict[tuple[str, int], Grid] = {}
    for task in tasks:
        outputs = _predict(task, introspect=args.introspect, threshold=args.threshold)
        for i, grid in enumerate(outputs):
            predictions[(task.task_id, i)] = grid

    submission = build_submission_json(tasks, predictions)

    with open("sample_submission.json", "w", encoding="utf-8") as f:
        json.dump(submission, f)

    print("Submission written to sample_submission.json")


if __name__ == "__main__":
    main()
