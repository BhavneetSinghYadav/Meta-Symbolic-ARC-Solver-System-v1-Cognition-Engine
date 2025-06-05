"""Parsing utilities for the symbolic rule DSL.

This module now performs basic validation and sanitisation of DSL
expressions to avoid malformed rules crashing the solver.  It exposes
helpers used by :mod:`abstraction_dsl` for round trip checking and
program parsing.
"""

from __future__ import annotations

from typing import List
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


__all__ = [
    "parse_rule",
    "rule_to_dsl",
    "validate_dsl_program",
    "validate_color_range",
    "clean_dsl_string",
]
