from __future__ import annotations

"""Probabilistic ranking of rule sets."""

from typing import List, Tuple

import numpy as np

from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.search.feature_mapper import rule_feature_vector

TEMPERATURE = 1.0


def _score_rules(rules: List, pairs: List[Tuple[Grid, Grid]]) -> float:
    total = 0.0
    for inp, out in pairs:
        pred = simulate_rules(inp, rules)
        total += pred.compare_to(out)
    return total / len(pairs) if pairs else 0.0


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / exp.sum()


def probabilistic_rank_rule_sets(
    rule_sets: List[List],
    pairs: List[Tuple[Grid, Grid]],
) -> List[Tuple[List, float]]:
    """Return rule sets with softmax-weighted probabilities."""

    scored: List[Tuple[List, float]] = []
    for rules in rule_sets:
        score = _score_rules(rules, pairs)
        feat_bonus = sum(sum(rule_feature_vector(r)) for r in rules)
        scored.append((rules, score + 0.1 * feat_bonus))
    scored.sort(key=lambda x: x[1], reverse=True)

    scores = np.array([s for _, s in scored])
    probs = _softmax(scores / TEMPERATURE)

    return [(rules, float(prob)) for (rules, _), prob in zip(scored, probs)]


__all__ = ["probabilistic_rank_rule_sets"]
