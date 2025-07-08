"""Symbolic reasoning primitives and DSL parsing."""

from .vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationNature,
    TransformationType,
)
from .rule_language import parse_rule, rule_to_dsl, CompositeRule
from .repeat_rule import repeat_tile, generate_repeat_rules
from .composite_rules import generate_repeat_composite_rules
from .abstraction_dsl import rules_to_program, program_to_rules
from .program_dsl import parse_program_expression

__all__ = [
    "Symbol",
    "SymbolType",
    "SymbolicRule",
    "Transformation",
    "TransformationNature",
    "TransformationType",
    "parse_rule",
    "rule_to_dsl",
    "rules_to_program",
    "program_to_rules",
    "parse_program_expression",
    "repeat_tile",
    "generate_repeat_rules",
    "generate_repeat_composite_rules",
    "CompositeRule",
]
