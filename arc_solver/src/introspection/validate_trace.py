"""Extended trace validation wrapper."""

from __future__ import annotations

from typing import Any, Dict

from .trace_builder import RuleTrace
from .introspective_validator import validate_trace as _validate_trace


def validate_trace(trace: RuleTrace) -> Dict[str, Any]:
    """Return validation metrics for ``trace``.

    This thin wrapper exists for backward compatibility and forwards
    to :func:`introspective_validator.validate_trace`.
    """
    return _validate_trace(trace)


__all__ = ["validate_trace"]
