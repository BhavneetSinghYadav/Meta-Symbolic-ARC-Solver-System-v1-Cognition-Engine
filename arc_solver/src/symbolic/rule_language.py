"""Parsing utilities for the symbolic rule DSL."""

from __future__ import annotations

from typing import List

from .vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationNature,
    TransformationType,
)


def _parse_symbol(token: str) -> Symbol:
    key, value = token.split("=", 1)
    key = key.strip().upper()
    value = value.strip()
    try:
        stype = SymbolType[key]
    except KeyError as exc:
        raise ValueError(f"Unknown symbol type: {key}") from exc
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
    text = text.strip()
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


__all__ = ["parse_rule", "rule_to_dsl"]
