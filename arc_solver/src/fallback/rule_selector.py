from __future__ import annotations

"""Helpers for ranking and selecting rule programs."""

from collections import Counter
from typing import List

from arc_solver.src.core.grid import Grid


def prioritize(rule_sets: List[List], training_scores: List[float]) -> List[List]:
    """Order rule sets by descending ``training_scores``."""
    scored = list(zip(rule_sets, training_scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [rs for rs, _ in scored]


def soft_vote(prediction_candidates: List[Grid]) -> Grid:
    """Return the most common grid among candidates."""
    if not prediction_candidates:
        raise ValueError("No candidates provided")
    hashes = [tuple(map(tuple, g.data)) for g in prediction_candidates]
    most = Counter(hashes).most_common(1)[0][0]
    for g, h in zip(prediction_candidates, hashes):
        if h == most:
            return g
    return prediction_candidates[0]


__all__ = ["prioritize", "soft_vote"]
