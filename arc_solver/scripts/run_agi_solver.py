from __future__ import annotations

"""Run the symbolic solver on the official ARC AGI dataset."""

import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

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
    # Normalize predictions to ``Grid`` objects in case the pipeline returns
    # dictionaries like {"output": grid} or raw list data
    norm_preds: List[Grid] = []
    for p in preds:
        if isinstance(p, dict) and "output" in p:
            p = p["output"]
        if isinstance(p, Grid):
            norm_preds.append(p)
        else:
            norm_preds.append(Grid(p))

    if introspect and task.train:
        score = sum(
            accuracy_score(simulate_rules(inp, rules), out)
            for inp, out in task.train
        ) / len(task.train)
        if score < threshold:
            preds, _, _, _ = pipeline_solve_task(json_task, introspect=True)
            norm_preds = []
            for p in preds:
                if isinstance(p, dict) and "output" in p:
                    p = p["output"]
                if isinstance(p, Grid):
                    norm_preds.append(p)
                else:
                    norm_preds.append(Grid(p))
    return norm_preds


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
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing dataset JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_submission.json",
        help="Path for the submission JSON output",
    )
    args = parser.parse_args()

    split_prefix = {
        "train": "arc-agi_training",
        "evaluation": "arc-agi_evaluation",
        "test": "arc-agi_test",
    }[args.split]

    hyphen = Path(args.data_dir) / f"{split_prefix}-challenges.json"
    underscore = Path(args.data_dir) / f"{split_prefix}_challenges.json"
    if hyphen.exists():
        challenges_path = hyphen
    elif underscore.exists():
        challenges_path = underscore
    else:
        raise FileNotFoundError(f"Dataset file not found: {hyphen} or {underscore}")

    tasks = load_agi_tasks(challenges_path)

    predictions: dict[tuple[str, int], Grid] = {}
    for task in tasks:
        outputs = _predict(task, introspect=args.introspect, threshold=args.threshold)
        for i, grid in enumerate(outputs):
            predictions[(task.task_id, i)] = grid

    submission = build_submission_json(tasks, predictions)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(submission, f)

    print(f"Submission written to {args.output}")


if __name__ == "__main__":
    main()
