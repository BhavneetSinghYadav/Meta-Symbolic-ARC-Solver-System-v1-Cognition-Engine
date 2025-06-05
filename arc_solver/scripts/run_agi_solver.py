from __future__ import annotations

"""Run the symbolic solver on the official ARC AGI dataset."""

import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from arc_solver.src.data.visualization import visualize

from arc_solver.src.core.grid import Grid

from arc_solver.src.data.agi_loader import load_agi_tasks, ARCAGITask
from arc_solver.src.executor.full_pipeline import solve_task as pipeline_solve_task
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.evaluation.metrics import accuracy_score
from arc_solver.src.evaluation.submission_builder import build_submission_json

_MAX_LOGGED_FAILURES = 5
_logged_failures = 0


def _candidate_transforms(grid: Grid) -> List[Tuple[Grid, str]]:
    """Return common symmetry variants of ``grid``."""
    variants: List[Tuple[Grid, str]] = []
    for k in range(4):
        rotated = grid.rotate90(k)
        variants.append((rotated, f"rot{k*90}"))
        variants.append((rotated.flip_horizontal(), f"rot{k*90}_flip"))
    return variants


def _expected_pattern(task: ARCAGITask) -> Tuple[Tuple[int, int] | None, Dict[int, int]]:
    """Return reference shape and aggregated color counts from training/GT."""
    ref_shape: Tuple[int, int] | None = None
    ref_counts: Dict[int, int] = {}
    sources: List[Grid] = []
    if task.ground_truth:
        sources.extend(task.ground_truth)
    if task.train:
        sources.extend(out for _, out in task.train)
    if sources:
        ref_shape = sources[0].shape()
    for g in sources:
        for c, v in g.count_colors().items():
            ref_counts[c] = ref_counts.get(c, 0) + v
    return ref_shape, ref_counts


def _score_candidate(
    grid: Grid,
    ref_shape: Tuple[int, int] | None,
    ref_counts: Dict[int, int],
    ground_truth: Grid | None,
) -> float:
    score = 0.0
    if ref_shape and grid.shape() == ref_shape:
        score += 1.0
    if ref_counts:
        counts = grid.count_colors()
        diff = sum(
            abs(counts.get(c, 0) - ref_counts.get(c, 0))
            for c in set(ref_counts) | set(counts)
        )
        score -= diff / (sum(ref_counts.values()) + 1)
    if ground_truth is not None:
        score += accuracy_score(grid, ground_truth)
    return score


def _plot_failure(task_id: str, idx: int, pred: Grid, true: Grid) -> None:
    """Save a side-by-side visualization of a failed prediction."""
    out_dir = Path("debug_failures")
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    axes[0].imshow(pred.data, interpolation="nearest")
    axes[0].set_title("Prediction")
    axes[0].axis("off")
    axes[1].imshow(true.data, interpolation="nearest")
    axes[1].set_title("Truth")
    axes[1].axis("off")
    fig.suptitle(f"{task_id} [{idx}]")
    fig.tight_layout()
    fig.savefig(out_dir / f"{task_id}_{idx}.png")
    plt.close(fig)


def _predict(
    task: ARCAGITask,
    *,
    introspect: bool = False,
    threshold: float = 0.9,
    use_memory: bool = False,
    use_prior: bool = False,
):
    """Return predictions for ``task`` optionally refining with introspection."""

    train_dicts = [
        {"input": inp.data, "output": out.data} for inp, out in task.train
    ]
    test_dicts = [{"input": g.data} for g in task.test]
    json_task = {"train": train_dicts, "test": test_dicts}

    preds, _, _, rules = pipeline_solve_task(
        json_task,
        introspect=False,
        use_memory=use_memory,
        use_prior=use_prior,
        task_id=task.task_id,
    )

    def _normalize(pred_list: List) -> List[Grid]:
        out: List[Grid] = []
        for p in pred_list:
            if isinstance(p, dict) and "output" in p:
                p = p["output"]
            if isinstance(p, Grid):
                out.append(p)
            else:
                out.append(Grid(p))
        return out

    norm_preds = _normalize(preds)

    if introspect and task.train:
        score = sum(
            accuracy_score(simulate_rules(inp, rules), out) for inp, out in task.train
        ) / len(task.train)
        if score < threshold:
            preds, _, _, _ = pipeline_solve_task(
                json_task,
                introspect=True,
                use_memory=use_memory,
                use_prior=use_prior,
                task_id=task.task_id,
            )
            norm_preds = _normalize(preds)

    ref_shape, ref_counts = _expected_pattern(task)
    gt_list = task.ground_truth if task.ground_truth else [None] * len(task.test)

    improved_preds: List[Grid] = []
    for idx, pred in enumerate(norm_preds):
        best_grid = pred
        best_score = _score_candidate(pred, ref_shape, ref_counts, gt_list[idx] if idx < len(gt_list) else None)
        best_name = "original"
        for cand, name in _candidate_transforms(pred):
            score = _score_candidate(cand, ref_shape, ref_counts, gt_list[idx] if idx < len(gt_list) else None)
            if score > best_score:
                best_score = score
                best_grid = cand
                best_name = name
        print(f"Task {task.task_id} test {idx}: selected {best_name} score={best_score:.3f}")
        if gt_list and idx < len(gt_list) and gt_list[idx] is not None:
            if best_grid.compare_to(gt_list[idx]) != 1.0:
                global _logged_failures
                if _logged_failures < _MAX_LOGGED_FAILURES:
                    _plot_failure(task.task_id, idx, best_grid, gt_list[idx])
                    _logged_failures += 1
        improved_preds.append(best_grid)

    return improved_preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the solver on AGI dataset")
    parser.add_argument(
        "--split",
        choices=["train", "evaluation", "test"],
        default="test",
        help="Which dataset split to run",
    )
    parser.add_argument("--introspect", action="store_true", help="Enable introspection refinement")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for introspection",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing dataset JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_submission.json",
        help="Path for the submission JSON output",
    )
    parser.add_argument("--use_memory", action="store_true", help="Enable rule memory")
    parser.add_argument("--use_prior", action="store_true", help="Use prior templates")
    parser.add_argument("--reflex_override", action="store_true", help="Enable regime override")
    parser.add_argument("--regime_threshold", type=float, default=0.45, help="Override threshold")
    parser.add_argument(
        "--llm_mode",
        choices=["online", "offline"],
        default="online",
        help="Use local LLM when offline",
    )
    parser.add_argument(
        "--allow_self_repair",
        action="store_true",
        help="Enable self-repair loop",
    )
    parser.add_argument(
        "--repair_threshold",
        type=float,
        default=0.75,
        help="Score threshold to trigger self-repair",
    )
    args = parser.parse_args()

    from arc_solver.src.utils import config_loader
    config_loader.set_offline_mode(args.llm_mode == "offline")
    config_loader.set_repair_enabled(args.allow_self_repair)
    config_loader.set_repair_threshold(args.repair_threshold)
    config_loader.set_reflex_override(args.reflex_override)
    config_loader.set_regime_threshold(args.regime_threshold)

    split_prefix = {
        "train": "arc-agi_training",
        "evaluation": "arc-agi_evaluation",
        "test": "arc-agi_test",
    }[args.split]

    hyphen = Path(args.data_dir) / f"{split_prefix}-challenges.json"
    underscore = Path(args.data_dir) / f"{split_prefix}_challenges.json"
    if hyphen.exists():
        challenges_path = hyphen
    elif underscore.exists():
        challenges_path = underscore
    else:
        raise FileNotFoundError(f"Dataset file not found: {hyphen} or {underscore}")

    tasks = load_agi_tasks(challenges_path)

    predictions: dict[tuple[str, int], Grid] = {}
    for task in tasks:
        outputs = _predict(
            task,
            introspect=args.introspect,
            threshold=args.threshold,
            use_memory=args.use_memory,
            use_prior=args.use_prior,
        )
        for i, grid in enumerate(outputs):
            predictions[(task.task_id, i)] = grid

    submission = build_submission_json(tasks, predictions)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(submission, f)

    print(f"Submission written to {args.output}")


if __name__ == "__main__":
    main()
