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
from .zone_adjustments import (
    zone_entropy_penalty,
    zone_alignment_bonus,
    zone_coverage_weight,
)

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
    "zone_entropy_penalty",
    "zone_alignment_bonus",
    "zone_coverage_weight",
]
