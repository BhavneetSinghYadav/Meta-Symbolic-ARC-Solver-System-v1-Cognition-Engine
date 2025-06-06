from __future__ import annotations

"""Utilities for visually scoring predicted grids."""

from typing import List

import numpy as np
from skimage.metrics import structural_similarity as ssim

from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules


def compute_visual_score(pred_grid: Grid, target_grid: Grid) -> float:
    """Return a structural similarity score between two grids."""
    pred_np = np.array(pred_grid.data)
    tgt_np = np.array(target_grid.data)
    min_dim = min(pred_np.shape[0], pred_np.shape[1])
    if min_dim < 3:
        return pred_grid.compare_to(target_grid)
    win = min(7, min_dim)
    if win % 2 == 0:
        win -= 1
    data_range = float(
        max(pred_np.max(), tgt_np.max()) - min(pred_np.min(), tgt_np.min())
    ) or 1.0
    score, _ = ssim(pred_np, tgt_np, win_size=win, data_range=data_range, full=True)
    return float(score)


def rerank_by_visual_score(
    rule_sets: List[List],
    input_grid: Grid,
    target_grid: Grid,
) -> List[List]:
    """Order ``rule_sets`` by visual similarity of their predictions."""
    ranked = []
    for rules in rule_sets:
        try:
            pred = simulate_rules(input_grid, rules)
            visual_score = compute_visual_score(pred, target_grid)
            ranked.append((visual_score, rules))
        except Exception:
            continue
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked]


__all__ = ["compute_visual_score", "rerank_by_visual_score"]
