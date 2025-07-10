import pytest

from arc_solver.src.symbolic.rule_language import (
    parse_rule,
    rule_to_dsl,
    validate_dsl_program,
    clean_dsl_string,
)
from arc_solver.src.symbolic.vocabulary import SymbolicRule, TransformationType


def test_parse_valid_rule():
    rule = parse_rule("REPLACE [COLOR=1] -> [COLOR=2]")
    assert isinstance(rule, SymbolicRule)


def test_parse_invalid_symbol():
    with pytest.raises(ValueError):
        parse_rule("REPLACE [COLOR=11] -> [COLOR=1]")


def test_roundtrip_consistency():
    rule = parse_rule("REPLACE [COLOR=3] -> [COLOR=4]")
    stringified = rule_to_dsl(rule)
    parsed = parse_rule(stringified)
    assert parsed == rule


def test_edge_tokens():
    assert not validate_dsl_program("NaN")
    assert not validate_dsl_program("None")
    with pytest.raises(ValueError):
        parse_rule(clean_dsl_string("FOO [COLOR=1]->[COLOR=2]"))


def test_parse_new_transforms():
    r1 = parse_rule("ROTATE90 [SHAPE=A] -> [SHAPE=A]")
    r2 = parse_rule("SHAPE_ABSTRACT [SHAPE=A] -> [SHAPE=A]")
    assert r1.transformation.ttype is TransformationType.ROTATE90
    assert r2.transformation.ttype is TransformationType.SHAPE_ABSTRACT
