
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
from arc_solver.src.symbolic.rule_language import rule_to_dsl


def suggest_fix_from_trace(trace: RuleTrace) -> str:
    """Return DSL suggestion to repair ``trace.rule`` based on mismatches."""

    if trace.ground_truth is None:
        return ""
    discrepancy = compute_discrepancy(trace.predicted_grid, trace.ground_truth)
    fix = refine_rule(trace.rule, discrepancy)
    return rule_to_dsl(fix) if fix else ""

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
    "suggest_fix_from_trace",
]
