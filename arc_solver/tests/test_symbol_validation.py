import pytest
from arc_solver.src.symbolic.rule_language import parse_rule
from arc_solver.src.symbolic.vocabulary import Symbol


def test_parse_rule_valid_colors():
    rule = parse_rule("REPLACE [COLOR=1] -> [COLOR=2]")
    assert all(sym.is_valid() for sym in rule.source + rule.target)


def test_parse_rule_invalid_color():
    with pytest.raises(ValueError):
        parse_rule("REPLACE [COLOR=11] -> [COLOR=1]")
