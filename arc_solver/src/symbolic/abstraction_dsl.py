"""Helpers for composing symbolic rule programs."""

from __future__ import annotations

from typing import List
import logging

from .rule_language import (
    parse_rule,
    rule_to_dsl,
    validate_dsl_program,
    clean_dsl_string,
)
from .vocabulary import SymbolicRule

logger = logging.getLogger(__name__)


def rules_to_program(rules: List[SymbolicRule]) -> str:
    """Serialize rules into a simple program string."""
    return " | ".join(rule_to_dsl(r) for r in rules)


def program_to_rules(text: str) -> List[SymbolicRule]:
    """Parse a program string back into rules."""
    if not text.strip():
        return []
    rules: List[SymbolicRule] = []
    for part in text.split("|"):
        raw = part.strip()
        if not raw:
            continue
        cleaned = clean_dsl_string(raw)
        if not validate_dsl_program(cleaned):
            logger.warning(f"Skipping malformed rule: {raw}")
            continue
        try:
            rules.append(parse_rule(cleaned))
        except Exception as exc:  # noqa: PERF203 - defensive parse
            logger.warning(f"Skipping invalid rule: {raw} ({exc})")
    return rules


__all__ = ["rules_to_program", "program_to_rules"]
