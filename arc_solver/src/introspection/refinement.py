from __future__ import annotations

"""Rule refinement utilities leveraging feedback and optional LLMs."""

from dataclasses import dataclass
from typing import List

from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationNature,
    TransformationType,
    parse_rule,
)
from .trace_builder import RuleTrace
from .introspective_validator import validate_trace
from .narrator_llm import narrate_trace

try:  # pragma: no cover - optional dependency
    import openai
except Exception:  # pragma: no cover - if openai is unavailable
    openai = None


@dataclass
class FeedbackSignal:
    """Structured signal describing deficiencies in a rule."""

    rule_id: str
    coverage_score: float
    conflict_zones: List[Symbol]
    missing_transforms: List[TransformationType]
    suggested_nature_adjustments: List[TransformationNature]
    verbal_description: str


def inject_feedback(trace: RuleTrace) -> FeedbackSignal:
    """Return a :class:`FeedbackSignal` derived from ``trace`` metrics."""

    metrics = validate_trace(trace)
    zones = [Symbol(SymbolType.ZONE, z) for z in trace.symbolic_context.get("zones", [])]
    missing_transforms: List[TransformationType] = []
    if "structural_mismatch" in metrics.get("conflict_flags", []):
        missing_transforms.append(TransformationType.TRANSLATE)
    suggested_nature: List[TransformationNature] = []
    if metrics["coverage_score"] < 0.8:
        suggested_nature.append(TransformationNature.LOGICAL)
    summary = narrate_trace(trace)
    return FeedbackSignal(
        rule_id=str(trace.rule),
        coverage_score=metrics["coverage_score"],
        conflict_zones=zones,
        missing_transforms=missing_transforms,
        suggested_nature_adjustments=suggested_nature,
        verbal_description=summary,
    )


def _heuristic_refinements(trace: RuleTrace, feedback: FeedbackSignal) -> List[SymbolicRule]:
    """Deterministically generate candidate rules using simple heuristics."""

    rule = trace.rule
    candidates: List[SymbolicRule] = []

    # Adjust nature if suggested
    for nature in feedback.suggested_nature_adjustments:
        candidates.append(
            SymbolicRule(
                transformation=rule.transformation,
                source=rule.source,
                target=rule.target,
                nature=nature,
            )
        )

    # If predicted color differs from ground truth, try correcting
    if trace.ground_truth is not None:
        # determine most common color in ground truth
        counts = {}
        h, w = trace.ground_truth.shape()
        for r in range(h):
            for c in range(w):
                val = trace.ground_truth.get(r, c)
                counts[val] = counts.get(val, 0) + 1
        if counts:
            tgt_color = max(counts, key=counts.get)
            src_color = next(
                (int(s.value) for s in rule.source if s.type is SymbolType.COLOR),
                None,
            )
            if src_color is not None:
                candidates.append(
                    SymbolicRule(
                        transformation=Transformation(TransformationType.REPLACE),
                        source=[Symbol(SymbolType.COLOR, str(src_color))],
                        target=[Symbol(SymbolType.COLOR, str(tgt_color))],
                    )
                )

    if not candidates:
        candidates.append(rule)
    return candidates


def llm_refine_program(trace: RuleTrace, feedback: FeedbackSignal) -> List[SymbolicRule]:
    """Return refined rule candidates using GPT or heuristic fallback."""

    prompt = None
    if openai is not None:
        dsl = str(trace.rule)
        prompt = (
            "You are Codex. Revise the following symbolic rule based on feedback.\n"
            f"Rule: {dsl}\n"
            f"Feedback: {feedback.verbal_description}\n"
            "Provide up to 3 rules in DSL format separated by newlines."
        )
        try:  # pragma: no cover - external call
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response["choices"][0]["message"]["content"].strip()
            rules: List[SymbolicRule] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rules.append(parse_rule(line))
                except Exception:
                    continue
            if rules:
                return rules
        except Exception:
            pass

    # Fallback heuristics if LLM unavailable or fails
    return _heuristic_refinements(trace, feedback)


def evaluate_refinements(rules: List[SymbolicRule], grid_in: Grid, grid_out: Grid) -> SymbolicRule:
    """Return the candidate rule with the highest score."""

    best_rule = rules[0]
    best_score = -1.0
    for rule in rules:
        pred = simulate_rules(grid_in, [rule])
        score = pred.compare_to(grid_out)
        if score > best_score:
            best_score = score
            best_rule = rule
    return best_rule


__all__ = [
    "FeedbackSignal",
    "inject_feedback",
    "llm_refine_program",
    "evaluate_refinements",
]
