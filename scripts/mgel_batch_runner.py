import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from arc_solver.src.data.arc_dataset import ARCDataset, load_arc_task
from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic.rule_language import rule_to_dsl
from arc_solver.src.core.grid import Grid
from arc_solver.src.evaluation.perceptual_score import perceptual_similarity_score


def load_first_pair(task_path: Path) -> tuple[str, Grid, Grid]:
    """Return (task_id, input_grid, target_grid) for the first train pair."""
    task = load_arc_task(task_path)
    grids = ARCDataset.to_grids(task)
    if not grids["train"]:
        raise ValueError(f"Task {task_path.stem} has no training pairs")
    inp, tgt = grids["train"][0]
    return task_path.stem, inp, tgt


def diff_grid(pred: Grid, target: Grid) -> List[List[str]]:
    """Return a grid of ✓/✗ comparisons between ``pred`` and ``target``."""
    return [
        ["✓" if a == b else "✗" for a, b in zip(pr, tg)]
        for pr, tg in zip(pred.data, target.data)
    ]


def process_task(task_path: Path, trace: bool = False, perceptual: bool = False) -> Dict[str, Any]:
    """Run MGEL on ``task_path`` and return the best rule result."""
    task_id, inp, tgt = load_first_pair(task_path)
    programs = abstract([inp, tgt])
    best: Dict[str, Any] | None = None
    best_score = -1.0

    for rule in programs:
        try:
            pred = simulate_rules(inp, [rule])
        except Exception:
            continue
        score = (
            perceptual_similarity_score(pred, tgt)
            if perceptual
            else pred.compare_to(tgt)
        )
        diff = pred.diff_summary(tgt)
        trace_details: Dict[str, Any] = {}
        fix_suggestion = ""
        if trace:
            try:
                from arc_solver.src.introspection import (
                    build_trace,
                    validate_trace,
                    suggest_fix_from_trace,
                )

                trace_obj = build_trace(rule, inp, pred, tgt)
                metrics = validate_trace(trace_obj)
                trace_details = {
                    "coverage": metrics.get("coverage_score"),
                    "conflicts": metrics.get("conflict_flags"),
                    "entropy_delta": metrics.get("entropy_change"),
                }
                try:
                    fix_suggestion = suggest_fix_from_trace(trace_obj)
                except Exception:
                    fix_suggestion = ""
            except Exception:
                trace_details = {}
                metrics = {}
        result = {
            "task_id": task_id,
            "rules": [rule_to_dsl(rule)],
            "prediction_score": score,
            "grid_input": inp.data,
            "grid_target": tgt.data,
            "grid_pred": pred.data,
            "trace": trace_details,
            "grid_diff": diff_grid(pred, tgt),
            "diff_summary": diff,
        }
        if fix_suggestion:
            result["llm_fix"] = fix_suggestion
        if trace_details:
            if score < 0.5 and "zone_miss" in trace_details.get("conflicts", []):
                result["failure_type"] = "zone overfit"
            elif "shape_mismatch" in trace_details.get("conflicts", []):
                result["failure_type"] = "shape logic error"
        if score > best_score:
            best = result
            best_score = score
    return best if best is not None else {"task_id": task_id, "error": "no_rules"}


def dump_results(results: List[Dict[str, Any]], path: Path) -> None:
    """Write ``results`` to ``path`` as JSON or CSV."""
    if path.suffix == ".json":
        with open(path, "w") as fh:
            json.dump(results, fh, indent=2)
    elif path.suffix == ".csv":
        import csv

        fieldnames = list(results[0].keys()) if results else []
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
    else:
        raise ValueError("Unsupported file format; use .json or .csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MGEL batch evaluation")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("arc_solver/tests"),
        help="Directory with ARC task JSON files",
    )
    parser.add_argument("--dump_to", type=Path, default=Path("results.json"))
    parser.add_argument("--trace", action="store_true", help="Enable trace introspection")
    parser.add_argument(
        "--perceptual",
        action="store_true",
        help="Use visual perceptual scoring instead of raw .compare_to()",
    )
    parser.add_argument("--limit", type=int, help="Max number of tasks to process")
    args = parser.parse_args()

    task_files = sorted(Path(args.data_dir).glob("*.json"))
    if args.limit:
        task_files = task_files[: args.limit]

    results: List[Dict[str, Any]] = []
    for path in task_files:
        try:
            res = process_task(path, trace=args.trace, perceptual=args.perceptual)
            results.append(res)
        except Exception as e:
            results.append({"task_id": path.stem, "error": str(e)})

    dump_results(results, args.dump_to)
    print(f"Results saved to {args.dump_to}")


if __name__ == "__main__":
    main()
