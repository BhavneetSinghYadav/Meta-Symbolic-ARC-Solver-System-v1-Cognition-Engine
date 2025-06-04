"""Entrypoint for executing the ARC solver over a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from arc_solver.src.executor.full_pipeline import solve_task
from arc_solver.src.evaluation.metrics import accuracy_score, aggregate_accuracy
from arc_solver.scripts.utils import iter_arc_task_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ARC solver")
    parser.add_argument("data_dir", type=Path, help="Directory of ARC JSON tasks")
    parser.add_argument("--introspect", action="store_true", help="Enable introspection")
    args = parser.parse_args()

    task_scores = []
    for tid, task, skipped in iter_arc_task_files(args.data_dir):
        if skipped:
            continue
        try:
            preds, targets, traces, rules = solve_task(task, introspect=args.introspect)
        except Exception as exc:
            print(f"[ERROR] Task {tid} — exception during solve(): {exc}", file=sys.stderr)
            continue
        scores = [accuracy_score(p, t) for p, t in zip(preds, targets)]
        task_score = sum(scores) / len(scores) if scores else 0.0
        task_scores.append(task_score)
        print(f"Task {tid} accuracy: {task_score:.3f}")

    overall = aggregate_accuracy(task_scores)
    print(f"Overall accuracy: {overall:.3f}")


if __name__ == "__main__":
    main()
