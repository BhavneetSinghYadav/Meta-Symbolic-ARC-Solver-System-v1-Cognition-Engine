from __future__ import annotations

"""Utilities for translating score traces into readable explanations."""

from typing import Dict, Any


def _rating(value: float) -> str:
    if value >= 0.85:
        return "high"
    if value >= 0.6:
        return "moderate"
    return "low"


def score_trace_explainer(trace: Dict[str, Any], task_id: str) -> str:
    """Return a concise human readable summary for ``trace``.

    Parameters
    ----------
    trace:
        Trace dictionary returned by :func:`score_rule`.
    task_id:
        Identifier of the analysed task (unused but kept for context).
    """

    steps = trace.get("steps")
    op_types = trace.get("op_types") or trace.get("rule_steps") or []
    if steps is None:
        steps = len(op_types) if op_types else 1

    composite = trace.get("composite")
    if composite is None:
        composite = steps > 1

    title = "Composite rule" if composite else "Atomic rule"
    if op_types:
        title += f" with {steps} step{'s' if steps != 1 else ''} ({' \u2192 '.join(op_types)})"
    else:
        title += ""

    final = trace.get("final_score")
    explanation_parts = [f"{title} scored {final:.2f}" if final is not None else title]

    similarity = trace.get("similarity")
    zone_match = trace.get("zone_match")
    shape_bonus = trace.get("shape_bonus")

    components = []
    if similarity is not None:
        components.append(f"{_rating(similarity)} similarity ({similarity:.2f})")
    if zone_match is not None:
        components.append(f"zone alignment {zone_match:.2f}")
    if shape_bonus:
        if shape_bonus >= 0.1:
            components.append("shape bonus")
        else:
            components.append("small shape bonus")

    if components:
        explanation_parts.append("due to " + ", ".join(components))

    op_cost = trace.get("op_cost") or trace.get("cost")
    penalties = trace.get("penalties") or {}
    penalty_total = trace.get("penalty")
    if op_cost is not None and not penalties.get("cost"):
        # if cost penalty not listed separately mention op_cost
        explanation_parts.append(f"operation cost ({op_cost:.2f})")

    penalty_descriptions = []
    for name, value in penalties.items():
        penalty_descriptions.append(f"{name} penalty of {value:.2f}")
    if penalty_total and not penalty_descriptions:
        penalty_descriptions.append(f"penalty of {penalty_total:.2f}")

    if penalty_descriptions:
        explanation_parts.append("; ".join(penalty_descriptions))

    bonus = trace.get("bonus")
    if bonus:
        explanation_parts.append(f"bonus {bonus:.2f}")

    if penalty_descriptions:
        explanation_parts.append("These suppressed the final score.")

    return " ".join(explanation_parts)


__all__ = ["score_trace_explainer"]
