"""Validates solver steps using introspection."""

from __future__ import annotations

from typing import Any, Sequence

from arc_solver.src.introspection.trace_builder import build_trace


def validate(plan: Sequence[Any]) -> bool:
    """Return ``True`` if ``plan`` passes a basic sanity check.

    The plan is first converted to a trace using :func:`build_trace`.  Each
    action must be a non-empty string and the ``step`` fields of the trace must
    form a consecutive sequence starting from zero.
    """

    trace = build_trace(list(plan))
    for idx, step in enumerate(trace):
        if step.get("step") != idx:
            return False
        action = step.get("action")
        if not isinstance(action, str) or not action:
            return False
    return True


__all__ = ["validate"]
