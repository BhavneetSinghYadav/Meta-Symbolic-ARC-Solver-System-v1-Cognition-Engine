from __future__ import annotations

"""Parsers for simple symbolic program expressions."""

from typing import List

from .rule_language import parse_rule
from .vocabulary import SymbolicRule


def parse_program_expression(text: str) -> List[SymbolicRule]:
    """Parse a very small DSL describing a rule program.

    Parameters
    ----------
    text:
        Expression such as ``"if color == 3 and in region(Center): replace with 2"``.

    Returns
    -------
    List[SymbolicRule]
        Parsed rules in execution order. Unsupported constructs are ignored.
    """

    text = text.strip()
    if not text:
        return []

    # Currently the parser only understands two patterns:
    # 1. "if color == X and in region(NAME): replace with Y"
    # 2. ``REPLACE [...] -> [...]`` style rules delegated to :func:`parse_rule`.

    if text.lower().startswith("if color"):
        try:
            prefix, action = text.split(":", 1)
        except ValueError:
            return []
        parts = prefix.split("and in region(")
        if len(parts) != 2:
            return []
        color_part = parts[0].strip()
        region_part = parts[1].rstrip(")").strip()
        try:
            src_color = int(color_part.split("==")[-1])
        except Exception:
            return []
        tgt_color = None
        if "replace with" in action:
            try:
                tgt_color = int(action.split("replace with")[-1].strip())
            except Exception:
                pass
        if tgt_color is None:
            return []
        dsl = f"CONDITIONAL [COLOR={src_color}, REGION={region_part}] -> [COLOR={tgt_color}]"
        return [parse_rule(dsl)]

    # Fallback to simple rule parser which also handles newly added types
    try:
        return [parse_rule(text)]
    except Exception:
        return []


__all__ = ["parse_program_expression"]
