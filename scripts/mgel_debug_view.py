"""MGEL Debug Viewer — works with both bundled and per-file ARC datasets.

Usage
-----
Legacy layout:
    python mgel_debug_view.py --task_id 5bd6f4ac --data_dir arc_data/

Bundled JSON:
    python mgel_debug_view.py --task_id 093a4d8 \
        --data_file /kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json \
        --solutions_file /kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from arc_solver.src.data.arc_dataset import ARCDataset, load_arc_task
from arc_solver.src.core.grid import Grid
from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic.rule_language import rule_to_dsl

logger = logging.getLogger("mgel_debug_view")


# ────────────────────────────────────────────────────────────
def _load_single_task(
    task_id: str,
    data_dir: Optional[Path] = None,
    data_file: Optional[Path] = None,
) -> Tuple[Grid, Grid]:
    """Return (input_grid, target_grid) of the first *train* pair."""
    if data_file is not None:
        bundle = json.load(open(data_file, "r"))
        if task_id not in bundle:
            raise KeyError(f"{task_id} not found in {data_file}")
        task_dict = bundle[task_id]
    elif data_dir is not None:
        task_path = Path(data_dir) / f"{task_id}.json"
        task_dict = load_arc_task(task_path)
    else:
        raise ValueError("Provide --data_dir or --data_file")

    grids = ARCDataset.to_grids(task_dict)
    if not grids["train"]:
        raise ValueError("Task has no training pairs")
    return grids["train"][0]


def _ansi(val: int) -> str:
    return f"\033[3{val % 8}m{val}\033[0m"


def _print_grid(title: str, g: Grid, color: bool = False) -> None:
    print(f"\n🔹{title}:")
    for row in g.data:
        if color:
            print(" ", *(_ansi(v) for v in row))
        else:
            print(" ", row)


def _print_diff(pred: Grid, tgt: Grid) -> None:
    diff = [
        ["✓" if a == b else "✗" for a, b in zip(rp, rt)]
        for rp, rt in zip(pred.data, tgt.data)
    ]
    for pr, tg, df in zip(pred.data, tgt.data, diff):
        print(" P:", pr, "\n T:", tg, "\n D:", df, "\n")


def _maybe_show_solutions(task_id: str, solutions_file: Optional[Path]) -> None:
    if solutions_file is None or not solutions_file.exists():
        return
    sol_bundle = json.load(open(solutions_file, "r"))
    if task_id not in sol_bundle:
        print("⚠️  Task not found in solutions bundle.")
        return
    print("\n🎯 Official ground-truth test grid(s):")
    for idx, grid in enumerate(sol_bundle[task_id], 1):
        print(f"  • Test {idx}:")
        for row in grid:
            print("   ", row)


# ────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser("MGEL symbolic debug viewer")
    p.add_argument("--task_id")
    p.add_argument("--data_dir", type=Path)
    p.add_argument("--data_file", type=Path)
    p.add_argument("--solutions_file", type=Path)
    p.add_argument("--manual_input", type=Path)
    p.add_argument("--manual_target", type=Path)
    p.add_argument("--color", action="store_true")
    p.add_argument("--step_by_step", action="store_true")
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()

    if args.trace:
        logging.basicConfig(level=logging.DEBUG)

    if args.manual_input and args.manual_target:
        with open(args.manual_input, "r", encoding="utf-8") as f:
            inp_grid = Grid(json.load(f))
        with open(args.manual_target, "r", encoding="utf-8") as f:
            tgt_grid = Grid(json.load(f))
        task_label = "manual"
    else:
        if not args.task_id:
            raise ValueError("--task_id required unless using --manual_input and --manual_target")
        inp_grid, tgt_grid = _load_single_task(
            args.task_id, args.data_dir, args.data_file
        )
        task_label = args.task_id

    print("=" * 40)
    print(f"TASK {task_label}  —  MGEL Debug Trace")
    print("=" * 40)
    _print_grid("Input", inp_grid, args.color)
    _print_grid("Target", tgt_grid, args.color)

    print("\n🧠  Extracting symbolic rules …")
    rule_programs = abstract([inp_grid, tgt_grid])
    print(f"Found {len(rule_programs)} rules")

    if not rule_programs:
        print("❌ No symbolic rules extracted.")
        return

    best_score = -1.0
    best_pred: Optional[Grid] = None

    for idx, rule in enumerate(rule_programs, 1):
        print(f"\nRule {idx}: {rule_to_dsl(rule)}")
        try:
            pred = simulate_rules(inp_grid, [rule])
        except Exception as err:  # pylint: disable=broad-except
            print("   ⚠️  Simulation error:", err)
            continue

        if args.step_by_step:
            _print_grid("Predicted (step)", pred, args.color)

        score = pred.compare_to(tgt_grid)
        print(f"   Score = {score:.3f}")

        if score > best_score:
            best_score = score
            best_pred = pred

    if best_pred is not None:
        print("\n✅  Best prediction diff:")
        _print_diff(best_pred, tgt_grid)
        print(f"Prediction Score: {best_score:.3f}")
    else:
        print("No symbolic rules")

    if args.solutions_file is not None and args.task_id:
        _maybe_show_solutions(args.task_id, args.solutions_file)


if __name__ == "__main__":
    main()
