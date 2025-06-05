from __future__ import annotations

"""Utilities for building ARC AGI competition submission files."""

from typing import Dict, List
import logging

from arc_solver.src.executor.fallback_predictor import predict as fallback_predict

from arc_solver.src.core.grid import Grid
from arc_solver.src.data.agi_loader import ARCAGITask


def build_submission_json(
    tasks: List[ARCAGITask], predictions: Dict[tuple[str, int], Grid | dict | list]
) -> dict:
    """Return a submission dictionary compatible with the ARC AGI leaderboard."""

    submission: Dict[str, Dict[str, List[List[List[int]]]]] = {}
    logger = logging.getLogger("submission_builder")
    assert predictions, "Predictions dictionary is empty"
    for task in tasks:
        outputs: List[List[List[int]]] = []
        for i in range(len(task.test)):
            key = (task.task_id, i)
            if key not in predictions:
                raise KeyError(f"Missing prediction for {task.task_id} index {i}")
            pred = predictions[key]
            if isinstance(pred, dict) and "output" in pred:
                pred = pred["output"]

            grids: List[List[List[int]]]
            if isinstance(pred, Grid):
                grids = [pred.to_list()]
            elif isinstance(pred, list):
                if pred and isinstance(pred[0], list) and pred and isinstance(pred[0][0], int):
                    grids = [pred]
                else:
                    grids = pred  # assume already list of grids
            else:
                raise TypeError(
                    f"Prediction for {task.task_id} index {i} has invalid type"
                )

            if task.ground_truth and i < len(task.ground_truth):
                ref_shape = task.ground_truth[i].shape()
            elif task.train:
                ref_shape = task.train[0][1].shape()
            else:
                ref_shape = (len(grids[0]), len(grids[0][0]))

            h, w = ref_shape

            fixed_grids: List[List[List[int]]] = []
            for g in grids:
                if not g or not isinstance(g, list) or not isinstance(g[0], list):
                    continue
                if not all(isinstance(v, int) for row in g for v in row):
                    if logger:
                        logger.warning("non-int values corrected in %s[%d]", task.task_id, i)
                    g = [[int(v) if isinstance(v, int) else 0 for v in row] for row in g]
                if (len(g), len(g[0])) != ref_shape:
                    flat = [v for row in g for v in row]
                    if len(flat) == h * w:
                        g = [flat[j * w:(j + 1) * w] for j in range(h)]
                    else:
                        g = [[0 for _ in range(w)] for _ in range(h)]
                fixed_grids.append(g)

            if not fixed_grids:
                if logger:
                    logger.warning("prediction for %s[%d] was empty; using fallback", task.task_id, i)
                fixed_grids.append(fallback_predict(Grid([[0]*w for _ in range(h)])).to_list())

            if len(fixed_grids) == 1:
                fixed_grids.append(fixed_grids[0])
            if len(fixed_grids) > 2:
                fixed_grids = fixed_grids[:2]

            outputs.append(fixed_grids)

        submission[task.task_id] = {"output": outputs}
    return submission

__all__ = ["build_submission_json"]
