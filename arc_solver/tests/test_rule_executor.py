from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic.vocabulary import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType, TransformationNature
from arc_solver.src.symbolic.rule_language import parse_rule
import pytest


def test_apply_replace_invalid_symbol():
    grid = Grid([[4, 4], [4, 4]])
    rule = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "4|LOGICAL")],
        target=[Symbol(SymbolType.COLOR, "2")],
        nature=TransformationNature.LOGICAL,
    )
    out = simulate_rules(grid, [rule])
    assert out.data == grid.data


def test_parse_rule_invalid_token():
    rule = parse_rule("REPLACE [COLOR=4|LOGICAL] -> [COLOR=1]")
    out = simulate_rules(Grid([[4]]), [rule])
    assert out.data == [[4]]
