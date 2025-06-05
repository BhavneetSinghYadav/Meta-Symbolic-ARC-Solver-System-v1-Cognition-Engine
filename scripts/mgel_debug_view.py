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
from arc_solver.src.symbolic.rule_language import rule_to_dsl


def load_single_task(task_id: str, data_dir: Path) -> tuple[Grid, Grid]:
    """Return the first train pair of ``task_id`` from ``data_dir``."""
    task_path = data_dir / f"{task_id}.json"
    task = load_arc_task(task_path)
    grids = ARCDataset.to_grids(task)
    if not grids["train"]:
        raise ValueError("Task has no training pairs")
    return grids["train"][0]


def print_grid(label: str, grid: Grid, *, color: bool = False) -> None:
    """Pretty-print ``grid`` with optional ANSI colors."""
    print(f"\n🔹{label}:")
    for row in grid.data:
        if color:
            text = "".join(f"\033[3{val % 8}m{val}\033[0m" for val in row)
            print(" ", text)
        else:
            print(" ", row)


def print_row_diff(pred: Grid, target: Grid) -> None:
    """Display row-wise differences between ``pred`` and ``target``."""
    if pred.shape() != target.shape():
        print("\nShapes differ; cannot compute row diff.")
        return
    print("\n🔻 Row Differences:")
    for i, (prow, trow) in enumerate(zip(pred.data, target.data)):
        diff = ["✓" if a == b else "✗" for a, b in zip(prow, trow)]
        print(f" Row {i}: Pred = {prow}, Target = {trow}, Diff = {diff}")


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
        help="Display grids with ANSI color codes",
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
    print_grid("Input Grid", input_grid, color=args.color)
    print_grid("Target Grid", target_grid, color=args.color)

    print("\n\U0001F9E0 Extracting Symbolic Rules...")
    rules = abstract([input_grid, target_grid])

    if not rules:
        print("\u274C No symbolic rules extracted.")
        return

    print("\nExtracted Rules:")
    for r in rules:
        zone = r.condition.get("zone") if r.condition else None
        print(f" \u25AA {rule_to_dsl(r)} | Zone: {zone} | Type: {r.__class__.__name__}")

    grid_state = Grid([row[:] for row in input_grid.data])
    for rule in rules:
        try:
            grid_state = simulate_rules(grid_state, [rule])
            print(f" \u2705 Rule applied successfully: {rule_to_dsl(rule)}")
            print_grid("Intermediate Grid", grid_state, color=args.color)
        except Exception as e:
            print(f" \u274C Rule failed: {rule_to_dsl(rule)} \u2014 {e}")
            break

    pred_grid = grid_state
    print_grid("Predicted Grid", pred_grid, color=args.color)
    score = pred_grid.compare_to(target_grid)
    print(f"\u2705 Prediction Score: {score:.3f}")
    print_row_diff(pred_grid, target_grid)


if __name__ == "__main__":
    main()
