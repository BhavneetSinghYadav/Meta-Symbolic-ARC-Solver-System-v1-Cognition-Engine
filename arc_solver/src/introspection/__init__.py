
"""Introspection utilities for tracing and explaining solver behaviour."""

from .trace_builder import RuleTrace, build_trace
from .symbolic_trace import SymbolicTrace
from .validate_trace import validate_trace
from .narrator_llm import narrate_trace
from .refinement import (
    FeedbackSignal,
    inject_feedback,
    llm_refine_program,
    evaluate_refinements,
)
from .self_repair import (
    compute_discrepancy,
    trace_prediction,
    localize_faulty_rule,
    refine_rule,
    llm_suggest_rule_fix,
    evaluate_repair_candidates,
    run_meta_repair,
    RuleTraceEntry,
    FaultHypothesis,
)

__all__ = [
    "RuleTrace",
    "SymbolicTrace",
    "build_trace",
    "validate_trace",
    "narrate_trace",
    "FeedbackSignal",
    "inject_feedback",
    "llm_refine_program",
    "evaluate_refinements",
    "compute_discrepancy",
    "trace_prediction",
    "localize_faulty_rule",
    "refine_rule",
    "llm_suggest_rule_fix",
    "evaluate_repair_candidates",
    "run_meta_repair",
    "RuleTraceEntry",
    "FaultHypothesis",
]
