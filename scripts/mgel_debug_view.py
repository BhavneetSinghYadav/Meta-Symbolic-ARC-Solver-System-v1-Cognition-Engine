"""Minimal Grounded Execution Loop (MGEL) debug viewer.

This script loads a single ARC task, extracts symbolic rules using the
``abstract`` function, and then applies each rule while printing a detailed
trace.  It can load a task by ``task_id`` from a directory of ARC JSON files
or use manually injected matrices.  Optional ``--color`` and ``--step`` flags
enable coloured grid output and step-by-step rule execution.

Usage::

    python scripts/mgel_debug_view.py --task_id 5bd6f4ac --data_dir <dir>

If ``--task_id`` is omitted the manual matrices defined in the script are
used instead.
"""

import argparse
from pathlib import Path

from arc_solver.src.data.arc_dataset import ARCDataset, load_arc_task
from arc_solver.src.core.grid import Grid
from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic.rule_language import rule_to_dsl


def load_single_task(task_id: str, data_dir: Path) -> tuple[Grid, Grid]:
    """Return the first train pair of ``task_id`` from ``data_dir``."""
    task_path = data_dir / f"{task_id}.json"
    task = load_arc_task(task_path)
    grids = ARCDataset.to_grids(task)
    if not grids["train"]:
        raise ValueError("Task has no training pairs")
    return grids["train"][0]


def print_colored_grid(grid: Grid) -> None:
    for row in grid.data:
        print(" ", "".join(f"\033[3{val % 8}m{val}\033[0m" for val in row))


def print_grid(label: str, grid: Grid, *, use_color: bool = False) -> None:
    print(f"\n🔹{label}:")
    if use_color:
        print_colored_grid(grid)
    else:
        for row in grid.data:
            print(" ", row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MGEL Debug View")
    parser.add_argument("--task_id", type=str, help="Task ID to load (e.g., '5bd6f4ac')")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("arc_solver/tests"),
        help="Directory containing ARC task JSON files",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Print grids using ANSI color codes",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Apply rules one by one with intermediate grids",
    )
    args = parser.parse_args()

    if args.task_id:
        input_grid, target_grid = load_single_task(args.task_id, args.data_dir)
        task_id = args.task_id
    else:
        input_matrix = [
            [0, 0, 0],
            [1, 2, 3],
            [0, 0, 0],
        ]
        target_matrix = [
            [3, 3, 3],
            [1, 2, 3],
            [3, 3, 3],
        ]
        input_grid = Grid(input_matrix)
        target_grid = Grid(target_matrix)
        task_id = "MANUAL_INJECTION"

    print("=" * 40)
    print(f"TASK {task_id} :: MGEL Symbolic Debug Trace")
    print("=" * 40)
    print_grid("Input Grid", input_grid, use_color=args.color)
    print_grid("Target Grid", target_grid, use_color=args.color)

    print("\n\U0001F9E0 Extracting Symbolic Rules...")
    rule_programs = abstract(input_grid, target_grid)

    if not rule_programs:
        print("\u274C No symbolic rules extracted.")
        return

    for idx, rule_set in enumerate(rule_programs):
        print(f"\nRule Program {idx + 1}:")
        for r in rule_set.rules:
            print(" \u25AA", rule_to_dsl(r))

        if args.step:
            working_grid = Grid([row[:] for row in input_grid.data])
            for rule in rule_set.rules:
                dsl = rule_to_dsl(rule)
                zone = rule.condition.get("zone")
                print(
                    f" \u25AA Rule: {dsl} | Zone: {zone} | Type: {rule.__class__.__name__}"
                )
                try:
                    working_grid = simulate_rules(working_grid, [rule])
                    print("    \u2705 Rule applied successfully")
                    print_grid(
                        "Intermediate Grid", working_grid, use_color=args.color
                    )
                except Exception as e:
                    print(f"    \u274C Rule failed: {e}")
        try:
            pred_grid = simulate_rules(input_grid, rule_set.rules)
            print_grid("Predicted Grid", pred_grid, use_color=args.color)
            score = pred_grid.compare_to(target_grid)
            print(f"\u2705 Prediction Score: {score:.3f}")
            for i, (row_pred, row_target) in enumerate(zip(pred_grid.data, target_grid.data)):
                diff = ["\u2713" if a == b else "\u2717" for a, b in zip(row_pred, row_target)]
                print(f"    Row {i}: Pred = {row_pred}, Target = {row_target}, Diff = {diff}")
        except Exception as exc:
            print(f"\u1F6D1 Rule simulation failed with error: {exc}")


if __name__ == "__main__":
    main()
