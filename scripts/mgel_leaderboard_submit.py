"""Generate ARC AGI leaderboard submissions using the MGEL solver."""

import argparse
import json
from pathlib import Path

from arc_solver.src.data.agi_loader import load_agi_tasks, ARCAGITask
from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.abstractions.rule_generator import generalize_rules
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.executor.fallback_predictor import predict as fallback_predict
from arc_solver.src.evaluation.submission_builder import build_submission_json


def _extract_candidate_rule_sets(train_pairs):
    """Return generalized rule sets extracted from each training pair."""
    rule_sets = []
    for inp, out in train_pairs:
        extracted = abstract([inp, out])
        if extracted:
            generalized = generalize_rules(extracted)
            rule_sets.append(generalized)
    return rule_sets


def _validate_program_on_all_pairs(rule_program, train_pairs, threshold=1.0):
    """Return True if program predicts each pair with score >= ``threshold``."""
    for inp, out in train_pairs:
        try:
            pred = simulate_rules(inp, rule_program)
            if pred.compare_to(out) < threshold:
                return False
        except Exception:
            return False
    return True


def _derive_generalized_program(train_pairs):
    """Return a rule program that works across all training pairs."""
    candidate_sets = _extract_candidate_rule_sets(train_pairs)
    valid_sets = []

    for rule_set in candidate_sets:
        if not rule_set:
            continue
        if _validate_program_on_all_pairs(rule_set, train_pairs):
            valid_sets.append(rule_set)

    if valid_sets:
        return valid_sets[0], valid_sets[0]
    return (candidate_sets[-1], candidate_sets[-1]) if candidate_sets else ([], [])


def _predict_task(task: ARCAGITask):
    """Return predictions for each test input of ``task``."""
    if task.train:
        best, fallback_rules = _derive_generalized_program(task.train)
    else:
        best, fallback_rules = [], []

    outputs = []
    for test_input in task.test[:2]:
        try:
            p1 = simulate_rules(test_input, best) if best else fallback_predict(test_input)
        except Exception:
            p1 = fallback_predict(test_input)

        try:
            p2 = simulate_rules(test_input, fallback_rules) if fallback_rules else p1
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
