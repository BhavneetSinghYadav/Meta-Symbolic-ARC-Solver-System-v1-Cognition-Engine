import pytest
from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.validator import validate_color_dependencies
from arc_solver.src.symbolic.vocabulary import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType
from arc_solver.src.symbolic.rule_language import CompositeRule


def _rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_validate_color_dependencies_pass():
    grid = Grid([[1, 2]])
    chain = [_rule(1, 1), _rule(2, 2)]
    assert validate_color_dependencies(chain, grid)


def test_validate_color_dependencies_fail():
    grid = Grid([[1, 2]])
    chain = [_rule(1, 3), _rule(3, 3)]
    assert not validate_color_dependencies(chain, grid)
