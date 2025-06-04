"""Utilities for building execution traces."""

from __future__ import annotations

from typing import Any, Dict, List


def build_trace(actions: List[Any]) -> List[Dict[str, Any]]:
    """Return a simple ordered trace of ``actions``.

    Each element in ``actions`` is wrapped into a dictionary with the
    corresponding ``step`` index.  The format is intentionally lightweight
    so that other components can easily consume it for validation or
    narration.
    """

    trace: List[Dict[str, Any]] = []
    for idx, action in enumerate(actions):
        trace.append({"step": idx, "action": action})
    return trace


__all__ = ["build_trace"]
