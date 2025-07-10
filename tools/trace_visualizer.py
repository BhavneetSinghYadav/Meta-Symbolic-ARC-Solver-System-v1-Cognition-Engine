"""Visualize scoring traces for individual ARC tasks.

This utility loads a JSON lines trace file produced when score tracing is
enabled and renders a diagnostic figure with prediction overlays and rule
breakdowns.  It is primarily meant to help debugging failed rules.

Usage once installed with ``pip``:

```
trace_visualizer --task_id 00000001 \
                 --trace_file logs/trace.jsonl \
                 --task_file arc-agi_training_challenges.json \
                 --solution_file arc-agi_training_solutions.json
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from arc_solver.src.core.grid import Grid
from arc_solver.src.segment.segmenter import zone_overlay
from arc_solver.src.utils.grid_utils import validate_grid
from arc_solver.src.debug.visualizer import visual_diff_report


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_tasks(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _load_score_trace(path: Path) -> pd.DataFrame:
    """Return dataframe of trace entries from ``path``."""
    records: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(records)


def _grid_from_list(obj: List[List[int]]) -> Grid:
    return Grid([list(row) for row in obj])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_grids(inp: Grid, pred: Grid, sol: Grid, title: str, out_path: Path) -> None:
    """Render input, prediction and solution side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, grid, name in zip(axes, [inp, pred, sol], ["input", "prediction", "solution"]):
        ax.imshow(grid.data, cmap="tab20", interpolation="none")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_zone_mismatch(pred: Grid, sol: Grid, out_path: Path) -> None:
    """Render zone overlay mismatch heatmap."""
    pred_o = zone_overlay(pred)
    sol_o = zone_overlay(sol)
    h, w = pred.shape()
    mask = [[0 for _ in range(w)] for _ in range(h)]

    def _val(cell: Any) -> str:
        return str(cell.value) if cell is not None else ""

    for r in range(h):
        for c in range(w):
            if _val(pred_o[r][c]) != _val(sol_o[r][c]):
                mask[r][c] = 1

    plt.figure(figsize=(3, 3))
    plt.imshow(mask, cmap="Reds", interpolation="none")
    plt.title("zone mismatch")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize score trace for a task")
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--trace_file", required=True)
    parser.add_argument("--task_file", required=True)
    parser.add_argument("--solution_file", required=True)
    parser.add_argument("--out_file", default=None)
    args = parser.parse_args()

    tasks = _load_tasks(Path(args.task_file))
    solutions = _load_tasks(Path(args.solution_file))
    task = tasks.get(args.task_id)
    if not task:
        raise SystemExit(f"Task {args.task_id} not found in {args.task_file}")

    inp = _grid_from_list(task["train"][0]["input"])
    sol = _grid_from_list(solutions[args.task_id][0])

    trace_df = _load_score_trace(Path(args.trace_file))
    entries = trace_df[trace_df.get("task_id") == args.task_id]
    if entries.empty:
        raise SystemExit(f"No trace entry for task {args.task_id}")

    best = entries.iloc[entries["final_score"].idxmax()]
    pred_grid = _grid_from_list(best["prediction"])

    if not validate_grid(pred_grid):
        print("Warning: predicted grid failed validation")

    diff_report = visual_diff_report(pred_grid, sol)
    print(diff_report)

    out_dir = Path(args.out_file or f"out/trace_{args.task_id}.pdf").resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    title = (
        f"{args.task_id} score {best['final_score']:.2f}"
        f" | cost {best.get('op_cost', 0):.2f}"
    )
    _plot_grids(inp, pred_grid, sol, title, out_dir)

    zone_path = out_dir.with_name(out_dir.stem + "_zones" + out_dir.suffix)
    try:
        _plot_zone_mismatch(pred_grid, sol, zone_path)
    except Exception:
        pass

    print(f"Saved visualization to {out_dir}")


if __name__ == "__main__":
    main()
