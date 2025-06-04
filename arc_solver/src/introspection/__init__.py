
"""Introspection utilities for tracing and explaining solver behaviour."""

from .trace_builder import RuleTrace, build_trace
from .introspective_validator import validate_trace
from .narrator_llm import narrate_trace

__all__ = ["RuleTrace", "build_trace", "validate_trace", "narrate_trace"]
