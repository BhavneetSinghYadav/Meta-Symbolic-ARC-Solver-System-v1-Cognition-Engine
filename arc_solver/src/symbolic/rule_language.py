"""Parsing utilities for the symbolic rule DSL.

This module now performs basic validation and sanitisation of DSL
expressions to avoid malformed rules crashing the solver.  It exposes
helpers used by :mod:`abstraction_dsl` for round trip checking and
program parsing.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

from .vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationNature,
    TransformationType,
    validate_color_range as vocab_color_range,
)

logger = logging.getLogger(__name__)

DSL_GRAMMAR = {
    "type": [t.name.lower() for t in TransformationType],
    "symbol_range": range(0, 10),
    "operators": ["->", "|", "&"],
    "fields": ["color", "zone", "transform", "position"],
}

MAX_SYMBOL_VALUE = 10


def validate_color_range(color: int | str) -> bool:
    """Return ``True`` if ``color`` is a valid ARC color index (0-9)."""

    return vocab_color_range(color)


def clean_dsl_string(s: str) -> str:
    """Return ``s`` trimmed of surrounding and newline whitespace."""

    return s.strip().replace("\n", "").replace("\t", "")


def validate_dsl_program(program_str: str) -> bool:
    """Return ``True`` if ``program_str`` looks like a well formed DSL rule."""

    if not isinstance(program_str, str):
        return False
    if "->" not in program_str:
        return False
    if program_str.count("[") != program_str.count("]"):
        return False
    if any(tok in program_str for tok in ["NaN", "None"]):
        return False
    op = clean_dsl_string(program_str).split("[", 1)[0].strip()
    if op.upper() not in {t.name for t in TransformationType}:
        return False
    return True


def extract_symbol_value(s: str) -> int:
    """Parse ``s`` as an integer within ``MAX_SYMBOL_VALUE``."""

    try:
        val = int(s)
        assert 0 <= val < MAX_SYMBOL_VALUE
        return val
    except Exception as exc:  # noqa: PERF203 - simple guard
        raise ValueError(f"Invalid symbol value in: {s}") from exc


def _parse_symbol(token: str) -> Symbol:
    if "=" not in token:
        raise ValueError(f"Invalid symbol token: {token}")
    key, value = token.split("=", 1)
    key = key.strip().upper()
    value = value.strip()
    try:
        stype = SymbolType[key]
    except KeyError as exc:
        raise ValueError(f"Unknown symbol type: {key}") from exc
    if stype is SymbolType.COLOR:
        if value.isdigit():
            value_int = extract_symbol_value(value)
            if not validate_color_range(value_int):
                raise ValueError(f"Invalid color value: {value}")
            value = str(value_int)
        else:
            if not validate_color_range(value):
                raise ValueError(f"Invalid color value: {value}")
    return Symbol(stype, value)


def _parse_symbol_list(text: str) -> List[Symbol]:
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError(f"Invalid symbol list: {text}")
    content = text[1:-1].strip()
    if not content:
        return []
    parts = [p.strip() for p in content.split(",")]
    return [_parse_symbol(part) for part in parts]


def parse_rule(text: str) -> SymbolicRule:
    text = clean_dsl_string(text)
    if not validate_dsl_program(text):
        raise ValueError("Malformed rule expression")
    if " " not in text:
        raise ValueError("Invalid rule format")
    op, remainder = text.split(" ", 1)
    try:
        ttype = TransformationType[op.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown transformation type: {op}") from exc
    if "->" not in remainder:
        raise ValueError("Rule must contain '->'")
    left_str, right_str = [part.strip() for part in remainder.split("->", 1)]
    left_syms = _parse_symbol_list(left_str)
    right_syms = _parse_symbol_list(right_str)
    transformation = Transformation(ttype)
    return SymbolicRule(transformation=transformation, source=left_syms, target=right_syms)


def rule_to_dsl(rule: SymbolicRule) -> str:
    return str(rule)


def generate_fallback_rules(inp: "Grid", out: "Grid") -> list[SymbolicRule]:
    """Return heuristic fallback rules for use outside the abstractor."""
    from arc_solver.src.abstractions.abstractor import _heuristic_fallback_rules

    return _heuristic_fallback_rules(inp, out)


@dataclass
class CompositeRule:
    """A rule composed of multiple symbolic transformations."""

    steps: List[SymbolicRule]
    nature: TransformationNature | None = None
    meta: Dict[str, Any] = field(default_factory=dict)

    transformation: Transformation = field(
        default_factory=lambda: Transformation(TransformationType.COMPOSITE)
    )

    def simulate(self, grid: "Grid") -> "Grid":
        """Return grid after sequentially applying steps."""
        from arc_solver.src.executor.simulator import safe_apply_rule

        out = grid
        for step in self.steps:
            out = safe_apply_rule(step, out, perform_checks=False)
        return out

    def get_sources(self) -> List[Symbol]:
        """Return merged source symbols across all steps."""
        return list({s for step in self.steps for s in getattr(step, "source", [])})

    def get_targets(self) -> List[Symbol]:
        """Return merged target symbols across all steps."""
        return list({s for step in self.steps for s in getattr(step, "target", [])})

    def get_condition(self) -> Optional[Any]:
        """Return the first non-null condition among steps."""
        for step in self.steps:
            if getattr(step, "condition", None):
                return step.condition
        return None

    def is_well_formed(self) -> bool:
        return all(getattr(step, "is_well_formed", lambda: False)() for step in self.steps)

    def as_symbolic_proxy(self) -> SymbolicRule:
        """Create a SymbolicRule-like proxy for use in dependency utilities."""
        return SymbolicRule(
            transformation=self.transformation,
            source=self.get_sources(),
            target=self.get_targets(),
            condition=self.get_condition() or {},
            nature=TransformationNature.SPATIAL,
        )

    def to_string(self) -> str:
        return " -> ".join(str(s) for s in self.steps)

    def __str__(self) -> str:  # pragma: no cover - simple
        return self.to_string()


__all__ = [
    "parse_rule",
    "rule_to_dsl",
    "generate_fallback_rules",
    "validate_dsl_program",
    "validate_color_range",
    "clean_dsl_string",
    "CompositeRule",
]
