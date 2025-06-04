from __future__ import annotations

"""Evaluate solver accuracy on the ARC AGI dataset using provided solutions."""

from arc_solver.src.core.grid import Grid
from arc_solver.src.data.agi_loader import load_agi_tasks
from arc_solver.scripts.run_agi_solver import _predict


def compute_accuracy(pred: Grid, gt: Grid) -> int:
    """Return 1 if ``pred`` exactly matches ``gt``."""
    return int(pred.compare_to(gt) == 1.0)


def main() -> None:
    tasks = load_agi_tasks("arc-agi_training-challenges.json", "arc-agi_training-solutions.json")

    total = 0
    correct = 0
    for task in tasks:
        outputs = _predict(task)
        for i, pred in enumerate(outputs):
            gt = task.ground_truth[i]
            correct += compute_accuracy(pred, gt)
            total += 1

    acc = correct / total * 100 if total else 0.0
    print(f"Accuracy: {correct}/{total} = {acc:.2f}%")


if __name__ == "__main__":
    main()

