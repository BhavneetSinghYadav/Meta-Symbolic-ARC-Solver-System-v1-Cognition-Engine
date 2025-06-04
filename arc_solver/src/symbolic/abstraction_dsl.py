"""Helpers for composing symbolic rule programs."""

from __future__ import annotations

from typing import List

from .rule_language import parse_rule, rule_to_dsl
from .vocabulary import SymbolicRule


def rules_to_program(rules: List[SymbolicRule]) -> str:
    """Serialize rules into a simple program string."""
    return " | ".join(rule_to_dsl(r) for r in rules)


def program_to_rules(text: str) -> List[SymbolicRule]:
    """Parse a program string back into rules."""
    if not text.strip():
        return []
    return [parse_rule(part.strip()) for part in text.split("|") if part.strip()]


__all__ = ["rules_to_program", "program_to_rules"]
