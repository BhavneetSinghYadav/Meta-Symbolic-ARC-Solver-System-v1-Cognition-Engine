"""Entrypoint for executing the ARC solver over a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from arc_solver.src.data.arc_dataset import ARCDataset
from arc_solver.src.executor.full_pipeline import solve_task
from arc_solver.src.evaluation.metrics import accuracy_score, aggregate_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ARC solver")
    parser.add_argument("data_dir", type=Path, help="Directory of ARC JSON tasks")
    parser.add_argument("--introspect", action="store_true", help="Enable introspection")
    args = parser.parse_args()

    dataset = ARCDataset(args.data_dir)
    task_scores = []
    for task in dataset:
        preds, targets, traces, rules = solve_task(task, introspect=args.introspect)
        scores = [accuracy_score(p, t) for p, t in zip(preds, targets)]
        task_score = sum(scores) / len(scores) if scores else 0.0
        task_scores.append(task_score)
        tid = task.get("id", "unknown")
        print(f"Task {tid} accuracy: {task_score:.3f}")

    overall = aggregate_accuracy(task_scores)
    print(f"Overall accuracy: {overall:.3f}")


if __name__ == "__main__":
    main()
