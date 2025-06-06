"""Scoring utilities for symbolic rule programs."""

from .compositional import (
    RuleInfo,
    extract_all_rules,
    simulate_and_trace,
    score_rule,
    evaluate_on_all_pairs,
    compose_programs,
    justify_selection,
    run_pipeline,
)
from .diff_penalty import SymbolicDiffPenaltyEngine

__all__ = [
    "RuleInfo",
    "extract_all_rules",
    "simulate_and_trace",
    "score_rule",
    "evaluate_on_all_pairs",
    "compose_programs",
    "justify_selection",
    "run_pipeline",
    "SymbolicDiffPenaltyEngine",
]
