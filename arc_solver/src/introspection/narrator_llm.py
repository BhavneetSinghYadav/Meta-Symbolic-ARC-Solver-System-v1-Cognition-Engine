"""Generate natural language explanations for rule traces.

The :func:`narrate_trace` helper converts a :class:`RuleTrace` and the
associated validation metrics into a concise text description.  If the
``openai`` package is available a GPT completion can be used for richer
language, otherwise a deterministic summary string is returned.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from .trace_builder import RuleTrace
from .introspective_validator import validate_trace

try:  # pragma: no cover - optional dependency
    import openai
except Exception:  # pragma: no cover - if openai is unavailable
    openai = None


_PROMPT_TEMPLATE = (
    "You are a symbolic reasoning assistant analysing a transformation rule applied to a grid.\n"
    "Rule: {rule}\n"
    "Affected cells: {cells}\n"
    "Metrics: {metrics}\n"
    "Symbolic context: {context}\n"
    "Provide a concise explanation of the rule's effect, coverage and any conflicts."
)


def narrate_trace(
    trace: RuleTrace, *, model: str = "gpt-4", use_llm: bool = False
) -> str:
    """Return a natural language summary for ``trace``.

    Parameters
    ----------
    trace:
        The rule execution trace.
    model:
        GPT model to use when ``use_llm`` is ``True`` and ``openai`` is
        available.
    use_llm:
        When ``True`` and ``openai`` is installed, use GPT to craft the
        narrative.  Otherwise a deterministic summary string is returned.
    """

    metrics: Dict[str, Any] = validate_trace(trace)

    prompt = _PROMPT_TEMPLATE.format(
        rule=str(trace.rule),
        cells=json.dumps(trace.affected_cells),
        metrics=json.dumps(metrics),
        context=json.dumps(trace.symbolic_context),
    )

    if openai is not None and use_llm:
        try:  # pragma: no cover - optional external call
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception:  # pragma: no cover - fall back if API fails
            pass

    # ------------------------------------------------------------------
    # Deterministic fallback summary
    # ------------------------------------------------------------------
    zones = ", ".join(trace.symbolic_context.get("zones", []))
    regions = ", ".join(trace.symbolic_context.get("regions", []))

    parts = [f"Rule {trace.rule} affected {len(trace.affected_cells)} cells"]
    if zones:
        parts.append(f"in zones {zones}")
    if regions:
        parts.append(f"across regions {regions}")

    coverage_pct = metrics["coverage_score"] * 100
    if metrics["interpretation_valid"]:
        parts.append("with full coverage")
    else:
        parts.append(f"covering {coverage_pct:.1f}% of the grid")
        if metrics["conflict_flags"]:
            parts.append(f"conflicts: {', '.join(metrics['conflict_flags'])}")

    return " ".join(parts) + "."


__all__ = ["narrate_trace"]
