"""Minimal Grounded Execution Loop (MGEL) debug viewer.

This script loads a single ARC task, extracts symbolic rules using the
``abstract`` function, and then applies each rule sequentially while
printing a detailed trace.  It can load a task by ``task_id`` from a
directory of ARC JSON files or use manually injected matrices.

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


def load_single_task(task_id: str, data_dir: Path) -> tuple[Grid, Grid]:
    """Return the first train pair of ``task_id`` from ``data_dir``."""
    task_path = data_dir / f"{task_id}.json"
    task = load_arc_task(task_path)
    grids = ARCDataset.to_grids(task)
    if not grids["train"]:
        raise ValueError("Task has no training pairs")
    return grids["train"][0]


def print_grid(label: str, grid: Grid) -> None:
    print(f"\n🔹{label}:")
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
    print_grid("Input Grid", input_grid)
    print_grid("Target Grid", target_grid)

    print("\n\U0001F9E0 Extracting Symbolic Rules...")
    rule_programs = abstract(input_grid, target_grid)

    if not rule_programs:
        print("\u274C No symbolic rules extracted.")
        return

    for idx, rule_set in enumerate(rule_programs):
        print(f"\nRule Program {idx + 1}:")
        for r in rule_set.rules:
            print(" \u25AA", str(r))
        try:
            pred_grid = simulate_rules(input_grid, rule_set.rules)
            print_grid("Predicted Grid", pred_grid)
            score = pred_grid.compare_to(target_grid)
            print(f"\u2705 Prediction Score: {score:.3f}")
        except Exception as exc:
            print(f"\u1F6D1 Rule simulation failed with error: {exc}")


if __name__ == "__main__":
    main()
