"""Generate ARC AGI leaderboard submissions using the MGEL solver."""

import argparse
import json
from pathlib import Path
from copy import deepcopy

from arc_solver.src.data.agi_loader import load_agi_tasks, ARCAGITask
from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.abstractions.rule_generator import generalize_rules
from arc_solver.src.rank_rule_sets import probabilistic_rank_rule_sets
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.executor.fallback_predictor import predict as fallback_predict
from arc_solver.src.evaluation.submission_builder import build_submission_json


def _derive_program(train_pair):
    """Return ranked rule sets from the first train pair."""
    inp, out = train_pair
    rules = abstract([inp, out])
    if not rules:
        return [], []
    rules = generalize_rules(rules)
    # fallback variant without zone condition
    no_zone = []
    for r in rules:
        if r.condition and "zone" in r.condition:
            nr = deepcopy(r)
            nr.condition.pop("zone", None)
            no_zone.append(nr)
    rule_sets = [rules]
    if no_zone:
        rule_sets.append(no_zone)
    ranked = probabilistic_rank_rule_sets(rule_sets, [(inp, out)])
    if not ranked:
        return [], []
    best = ranked[0][0]
    fallback = ranked[1][0] if len(ranked) > 1 else (no_zone or best)
    return best, fallback


def _predict_task(task: ARCAGITask):
    """Return predictions for each test input of ``task``."""
    if task.train:
        best, fallback_rules = _derive_program(task.train[0])
    else:
        best, fallback_rules = [], []
    outputs = []
    for g in task.test:
        try:
            p1 = simulate_rules(g, best) if best else fallback_predict(g)
        except Exception:
            p1 = fallback_predict(g)
        try:
            p2 = simulate_rules(g, fallback_rules) if fallback_rules else p1
        except Exception:
            p2 = p1
        outputs.append([p1.to_list(), p2.to_list()])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="MGEL leaderboard submission generator")
    parser.add_argument(
        "--challenges",
        type=Path,
        default=Path("arc-agi_test-challenges.json"),
        help="Path to challenge JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.json"),
        help="Destination submission file",
    )
    args = parser.parse_args()

    tasks = load_agi_tasks(args.challenges)
    predictions = {}
    for task in tasks:
        try:
            preds = _predict_task(task)
        except Exception:
            preds = [[fallback_predict(g).to_list(), fallback_predict(g).to_list()] for g in task.test]
        for i, grids in enumerate(preds):
            predictions[(task.task_id, i)] = grids

    submission = build_submission_json(tasks, predictions)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(submission, f)
    print(f"Submission written to {args.output}")


if __name__ == "__main__":
    main()
