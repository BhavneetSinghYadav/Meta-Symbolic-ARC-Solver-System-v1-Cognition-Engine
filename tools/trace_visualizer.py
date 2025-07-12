from __future__ import annotations

"""Interactive trace visualisation utilities.

This module exposes helpers to inspect how a symbolic program executes on a
single ARC task.  It provides step by step grid snapshots, entropy deltas and
scoring information.  The CLI entry point can be invoked as::

    python trace_visualizer.py <task_id>

The script expects the standard training challenge/solution files in the current
working directory as well as ``submission.json`` with predictions.  If a
``failure_log.jsonl`` is present it will use the most recent entry for the given
``task_id`` to reconstruct the program DSL for simulation.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt

from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.scoring import score_rule
from arc_solver.src.executor.simulator import safe_apply_rule
from arc_solver.src.scoring.entropy_utils import grid_color_entropy
from arc_solver.src.segment.segmenter import zone_overlay, segment_connected_regions
from arc_solver.src.symbolic.rule_language import (
    CompositeRule,
    SymbolicRule,
    parse_rule,
    rule_to_dsl,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

@dataclass
class TaskBundle:
    task_id: str
    input_grid: Grid
    target_grid: Grid
    prediction_grid: Optional[Grid]
    failure_trace: Optional[Dict[str, Any]]


def _grid_from_list(obj: Iterable[Iterable[int]]) -> Grid:
    return Grid([list(row) for row in obj])


def load_task_data(task_id: str) -> TaskBundle:
    """Return :class:`TaskBundle` for ``task_id`` if found."""
    tasks = json.loads(Path("arc-agi_training_challenges.json").read_text())
    solutions = json.loads(Path("arc-agi_training_solutions.json").read_text())
    preds = json.loads(Path("submission.json").read_text()) if Path("submission.json").is_file() else {}

    tdata = tasks.get(task_id)
    if not tdata:
        raise KeyError(f"task {task_id} not found")

    inp = _grid_from_list(tdata["train"][0]["input"])
    tgt = _grid_from_list(solutions[task_id][0])

    pred_grid = None
    if task_id in preds:
        pred_entry = preds[task_id][0]
        if isinstance(pred_entry, dict):
            pred_grid = _grid_from_list(next(iter(pred_entry.values())))
        else:
            pred_grid = _grid_from_list(pred_entry)

    failure = None
    flog = Path("failure_log.jsonl")
    if flog.is_file():
        for line in flog.read_text().splitlines()[::-1]:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("task_id") == task_id:
                failure = obj
                break

    return TaskBundle(task_id, inp, tgt, pred_grid, failure)


# ---------------------------------------------------------------------------
# Trace simulation
# ---------------------------------------------------------------------------

def _shape_entropy(grid: Grid) -> float:
    regions = segment_connected_regions(grid)
    sizes = [len(c) for c in regions.values()]
    if not sizes:
        return 0.0
    from math import log2

    total = sum(sizes)
    ent = 0.0
    for s in sizes:
        p = s / total
        ent -= p * log2(p)
    max_ent = log2(len(sizes)) if len(sizes) > 1 else 0.0
    return ent / max_ent if max_ent else 0.0


def simulate_trace(
    rule: SymbolicRule | CompositeRule,
    input_grid: Grid,
    target_grid: Optional[Grid] = None,
) -> Dict[str, Any]:
    """Simulate ``rule`` step by step and return trace records."""
    steps = rule.steps if isinstance(rule, CompositeRule) else [rule]
    grid = input_grid
    trace: List[Dict[str, Any]] = []

    for idx, step in enumerate(steps):
        before = grid
        after = safe_apply_rule(step, grid, perform_checks=False)
        ce_before = grid_color_entropy(before)
        ce_after = grid_color_entropy(after)
        se_before = _shape_entropy(before)
        se_after = _shape_entropy(after)
        sim = after.compare_to(target_grid) if target_grid is not None else None
        trace.append(
            {
                "index": idx,
                "dsl": rule_to_dsl(step),
                "zone": (step.condition or {}).get("zone"),
                "before": before.to_list(),
                "after": after.to_list(),
                "color_entropy_delta": ce_after - ce_before,
                "shape_entropy_delta": se_after - se_before,
                "similarity": sim,
            }
        )
        grid = after

    final_score = (
        score_rule(input_grid, target_grid, rule, return_trace=True)
        if target_grid is not None
        else None
    )
    return {"steps": trace, "final": final_score, "output": grid.to_list()}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _show_grid(ax, grid: Grid, title: str) -> None:
    ax.imshow(grid.data, cmap="tab20", interpolation="none")
    ax.set_title(title)
    ax.axis("off")


def render_trace(task_id: str, jupyter: bool = False, out_file: str | None = None) -> None:
    bundle = load_task_data(task_id)
    rule_dsl = None
    if bundle.failure_trace:
        rule_dsl = bundle.failure_trace.get("rule_dsl") or bundle.failure_trace.get("rule_id")
    if not rule_dsl:
        print("No rule information found; displaying prediction only")
    rule = parse_rule(rule_dsl) if rule_dsl else None

    trace_data = simulate_trace(rule, bundle.input_grid, bundle.target_grid) if rule else None

    fig_rows = 1
    if trace_data:
        fig_rows = len(trace_data["steps"]) + 1

    fig, axes = plt.subplots(fig_rows, 3, figsize=(9, 3 * fig_rows))
    if fig_rows == 1:
        axes = [axes]

    _show_grid(axes[0][0], bundle.input_grid, "input")
    _show_grid(axes[0][1], bundle.prediction_grid or bundle.input_grid, "prediction")
    _show_grid(axes[0][2], bundle.target_grid, "target")

    if trace_data:
        grid = bundle.input_grid
        for idx, step in enumerate(trace_data["steps"], 1):
            before = _grid_from_list(step["before"])
            after = _grid_from_list(step["after"])
            axb, axa, _ = axes[idx]
            _show_grid(axb, before, f"step {idx-1} before")
            _show_grid(axa, after, f"{step['dsl']}")
            axes[idx][2].axis("off")

    fig.tight_layout()
    if out_file:
        out = Path(out_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        print(f"saved trace to {out}")
    if jupyter:
        return fig
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Render program trace for a task")
    parser.add_argument("task_id")
    parser.add_argument("--output", help="save PDF/PNG instead of showing interactively")
    parser.add_argument("--jupyter", action="store_true", help="return matplotlib figure")
    args = parser.parse_args()

    render_trace(args.task_id, jupyter=args.jupyter, out_file=args.output)


if __name__ == "__main__":
    main()
