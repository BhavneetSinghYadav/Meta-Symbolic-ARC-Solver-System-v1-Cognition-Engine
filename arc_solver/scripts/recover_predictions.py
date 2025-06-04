from __future__ import annotations

"""Regenerate predictions for tasks missing from a prediction JSON file."""

import argparse
import json
from pathlib import Path
import sys

from arc_solver.src.executor.full_pipeline import solve_task
from arc_solver.scripts.utils import iter_arc_task_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover missing ARC predictions")
    parser.add_argument("existing", type=Path, help="Existing predictions JSON")
    parser.add_argument("tasks", type=Path, help="Directory of ARC task JSON files")
    parser.add_argument("--output", type=Path, default="merged_predictions.json", help="Output path for merged predictions")
    args = parser.parse_args()

    preds: dict[str, dict] = {}
    if args.existing.exists():
        preds = json.loads(args.existing.read_text())

    for tid, task, skipped in iter_arc_task_files(args.tasks):
        if skipped:
            continue
        if tid in preds and preds[tid].get("output"):
            continue
        try:
            outputs, _, _, _ = solve_task(task)
            preds[tid] = {"output": [g.data for g in outputs]}
        except Exception as exc:
            print(f"[ERROR] Task {tid} — exception during solve(): {exc}", file=sys.stderr)

    args.output.write_text(json.dumps(preds))
    print(f"Merged predictions written to {args.output}")


if __name__ == "__main__":
    main()
