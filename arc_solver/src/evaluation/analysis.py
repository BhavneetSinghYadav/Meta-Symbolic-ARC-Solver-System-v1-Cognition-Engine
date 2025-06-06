from __future__ import annotations

"""Visualization and analysis helpers for ARC solver evaluation."""

import json
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # ensure headless operation for tests
import matplotlib.pyplot as plt

from arc_solver.src.core.grid import Grid
from arc_solver.src.utils.grid_utils import compute_grid_entropy


def grid_diff_heatmap(
    predicted: Grid, target: Grid, *, return_data: bool = False
) -> Tuple[plt.Figure, list[list[int]] | None]:
    """Return a heatmap showing mismatched cells between ``predicted`` and ``target``."""

    if predicted.shape() != target.shape():
        raise ValueError("grid shapes must match")

    h, w = predicted.shape()
    heat = [
        [1 if predicted.get(r, c) != target.get(r, c) else 0 for c in range(w)]
        for r in range(h)
    ]

    fig = plt.figure()
    plt.imshow(heat, cmap="Reds", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()

    if return_data:
        return fig, heat
    return fig, None


def entropy_change(input_grid: Grid, pred_grid: Grid) -> Tuple[float, float]:
    """Return entropy of input and predicted grids for comparison."""

    ent_in = compute_grid_entropy(input_grid)
    ent_out = compute_grid_entropy(pred_grid)
    return ent_in, ent_out


def save_failure_case(
    task_id: str,
    index: int,
    input_grid: Grid,
    target_grid: Grid,
    pred_grid: Grid,
    score: float | None = None,
    out_dir: str | Path = "failures",
) -> None:
    """Persist grids and metadata for inspection of failed predictions."""

    case_dir = Path(out_dir) / f"{task_id}.csv"
    case_dir.mkdir(parents=True, exist_ok=True)

    def _save(grid: Grid, name: str) -> None:
        fig = plt.figure(figsize=(2, 2))
        plt.imshow(grid.data, interpolation="nearest")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(case_dir / f"{name}.png")
        plt.close(fig)

    _save(input_grid, "input")
    _save(target_grid, "target")
    _save(pred_grid, "pred")

    meta = {
        "task_id": task_id,
        "index": index,
        "score": score,
        "entropy_in": compute_grid_entropy(input_grid),
        "entropy_pred": compute_grid_entropy(pred_grid),
    }

    with open(case_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


__all__ = ["grid_diff_heatmap", "entropy_change", "save_failure_case"]

