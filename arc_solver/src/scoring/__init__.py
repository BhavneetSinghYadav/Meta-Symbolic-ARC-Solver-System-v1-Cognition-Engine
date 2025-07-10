"""Scoring utilities for symbolic rule programs."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
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


def __getattr__(name: str):  # pragma: no cover - thin lazy loader
    if name == "SymbolicDiffPenaltyEngine":
        from .diff_penalty import SymbolicDiffPenaltyEngine

        return SymbolicDiffPenaltyEngine
    if name in {
        "RuleInfo",
        "extract_all_rules",
        "simulate_and_trace",
        "score_rule",
        "evaluate_on_all_pairs",
        "compose_programs",
        "justify_selection",
        "run_pipeline",
    }:
        from . import compositional as _comp

        return getattr(_comp, name)
    raise AttributeError(name)
