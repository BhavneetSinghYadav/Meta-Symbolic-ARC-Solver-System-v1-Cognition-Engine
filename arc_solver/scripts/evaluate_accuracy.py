from __future__ import annotations

"""Compute accuracy of predictions against ARC AGI solutions."""

import argparse
import json
from pathlib import Path
from typing import List

import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from arc_solver.src.core.grid import Grid


def load_predictions(path: Path) -> dict[str, List[List[List[int]]]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    preds: dict[str, List[List[List[int]]]] = {}
    for tid, obj in raw.items():
        data = obj.get("output")
        if data is None:
            print(f"[WARN] Task {tid} — prediction missing from output JSON", file=sys.stderr)
            continue
        try:
            if data and isinstance(data[0][0], int):
                preds[tid] = [data]
            else:
                preds[tid] = data
        except Exception:
            print(f"[WARN] Task {tid} — malformed prediction entry", file=sys.stderr)
    return preds


def load_solutions(path: Path) -> dict[str, List[List[List[int]]]]:
    """Load ground truth solutions from ``path``.

    The JSON file may either be a dictionary mapping task ids to objects
    containing an ``"output"`` field or a list of such objects with explicit
    ``"task_id"`` fields. In both cases a dictionary keyed by task id and whose
    values are the raw ``output`` entries is returned.
    """

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sols: dict[str, List[List[List[int]]]] = {}

    if isinstance(raw, dict):
        # Legacy dictionary format {tid: {"output": ...}}
        for tid, obj in raw.items():
            if not isinstance(obj, dict) or "output" not in obj:
                raise ValueError(f"Missing 'output' for task {tid}")
            data = obj["output"]
            # normalize a single grid into a list of grids
            if data and isinstance(data[0][0], int):
                sols[tid] = [data]
            else:
                sols[tid] = data
    elif isinstance(raw, list):
        # New list format [{"task_id": tid, "output": ...}, ...]
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("Solution list items must be dictionaries")
            tid = item.get("task_id")
            data = item.get("output")
            if tid is None or data is None:
                raise ValueError("Each solution entry requires 'task_id' and 'output'")
            if tid in sols:
                raise ValueError(f"Duplicate task id {tid} in solutions")
            if data and isinstance(data[0][0], int):
                sols[tid] = [data]
            else:
                sols[tid] = data
    else:
        raise ValueError("Unsupported JSON format for solutions")

    return sols


def evaluate(pred_path: Path, sol_path: Path) -> None:
    preds = load_predictions(pred_path)
    sols = load_solutions(sol_path)

    all_ids = set(preds) | set(sols)
    total = 0
    correct = 0
    very_bad: list[str] = []
    borderline: list[str] = []
    positive: list[str] = []

    for tid in sorted(all_ids):
        pred_grids = preds.get(tid)
        gt_grids = sols.get(tid)

        if pred_grids is None and gt_grids is None:
            print(f"[WARN] Task {tid} missing from predictions and solutions", file=sys.stderr)
            very_bad.append(tid)
            continue
        if pred_grids is None:
            print(f"[WARN] Task {tid} — prediction missing", file=sys.stderr)
            borderline.append(tid)
            continue
        if gt_grids is None:
            print(f"[WARN] Task {tid} — ground truth missing from solutions", file=sys.stderr)
            borderline.append(tid)
            continue

        if len(pred_grids) != len(gt_grids):
            print(
                f"Count mismatch for {tid}: predicted {len(pred_grids)} vs gt {len(gt_grids)}",
                file=sys.stderr,
            )

        for i, gt in enumerate(gt_grids):
            if i >= len(pred_grids):
                print(f"[WARN] Task {tid} index {i} — prediction missing", file=sys.stderr)
                continue
            pg = pred_grids[i]
            try:
                if len(pg) != len(gt) or len(pg[0]) != len(gt[0]):
                    print(f"Shape mismatch {tid} index {i}: pred={pg} gt={gt}", file=sys.stderr)
                if Grid(pg).compare_to(Grid(gt)) == 1.0:
                    correct += 1
                total += 1
            except Exception as exc:
                print(f"[ERROR] Task {tid} index {i}: {exc}", file=sys.stderr)

        if pred_grids and gt_grids and correct > 0:
            positive.append(tid)

    acc = correct / total * 100 if total else 0.0
    print(f"Accuracy: {correct}/{total} = {acc:.2f}%")

    if very_bad or borderline:
        print("Summary:")
        if very_bad:
            print("VERY_BAD:", ", ".join(very_bad))
        if borderline:
            print("BORDERLINE:", ", ".join(borderline))
        if positive:
            print("POSITIVE:", ", ".join(positive))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions accuracy")
    parser.add_argument("predictions", type=Path, help="Path to predictions JSON")
    parser.add_argument("solutions", type=Path, help="Path to ground truth JSON")
    args = parser.parse_args()
    evaluate(args.predictions, args.solutions)


if __name__ == "__main__":
    main()
