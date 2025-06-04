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
    preds = {}
    for tid, obj in raw.items():
        data = obj.get("output")
        if data is None:
            raise ValueError(f"Missing 'output' for task {tid}")
        # normalize single grid vs list of grids
        if data and isinstance(data[0][0], int):
            preds[tid] = [data]
        else:
            preds[tid] = data
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
    total = 0
    correct = 0
    for tid, gt_grids in sols.items():
        pred_grids = preds.get(tid)
        if pred_grids is None:
            print(f"Missing prediction for {tid}")
            continue
        if len(pred_grids) != len(gt_grids):
            print(
                f"Count mismatch for {tid}: predicted {len(pred_grids)} vs gt {len(gt_grids)}"
            )
        for i, gt in enumerate(gt_grids):
            if i >= len(pred_grids):
                break
            pg = pred_grids[i]
            if len(pg) != len(gt) or len(pg[0]) != len(gt[0]):
                print(f"Shape mismatch {tid} index {i}: pred={pg} gt={gt}")
            if Grid(pg).compare_to(Grid(gt)) == 1.0:
                correct += 1
            total += 1
    acc = correct / total * 100 if total else 0.0
    print(f"Accuracy: {correct}/{total} = {acc:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions accuracy")
    parser.add_argument("predictions", type=Path, help="Path to predictions JSON")
    parser.add_argument("solutions", type=Path, help="Path to ground truth JSON")
    args = parser.parse_args()
    evaluate(args.predictions, args.solutions)


if __name__ == "__main__":
    main()
