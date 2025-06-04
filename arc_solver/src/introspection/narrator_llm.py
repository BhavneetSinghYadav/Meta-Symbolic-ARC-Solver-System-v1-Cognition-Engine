"""LLM-based narrative descriptions."""

from __future__ import annotations

import os
from typing import Any, Sequence

from arc_solver.src.introspection.trace_builder import build_trace


def _narrate_with_openai(prompt: str) -> str:
    """Return narration using OpenAI if available."""

    try:
        import openai  # type: ignore

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return ""
        openai.api_key = api_key
        result = openai.Completion.create(
            model="text-davinci-003", prompt=prompt, max_tokens=50
        )
        return result.choices[0].text.strip()
    except Exception:
        return ""


def narrate(plan: Sequence[Any]) -> str:
    """Return a narration for ``plan``.

    The plan is first converted to a trace.  If the ``openai`` package is
    available and an API key is provided via ``OPENAI_API_KEY`` the narration is
    generated using the OpenAI API.  Otherwise a simple textual summary is
    produced locally.
    """

    trace = build_trace(list(plan))
    prompt_lines = ["Describe the following solver steps:"]
    for step in trace:
        prompt_lines.append(f"Step {step['step']}: {step['action']}")
    prompt = "\n".join(prompt_lines)

    narration = _narrate_with_openai(prompt)
    if narration:
        return narration

    # Fallback summarisation -------------------------------------------------
    return " -> ".join(str(step["action"]) for step in trace)


__all__ = ["narrate"]
