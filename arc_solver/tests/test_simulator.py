import logging
from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules, validate_color_dependencies
import pytest
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.symbolic.rule_language import CompositeRule


def test_replace_missing_source(caplog):
    grid = Grid([[1, 1], [1, 1]])
    rule = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "3")],
        target=[Symbol(SymbolType.COLOR, "5")],
    )
    logger = logging.getLogger("sim_test")
    with caplog.at_level(logging.WARNING):
        out = simulate_rules(grid, [rule], logger=logger)
    assert out.data == grid.data
    assert any("skipped" in rec.message for rec in caplog.records)


def test_color_dependency_skip(caplog):
    grid = Grid([[1]])
    r1 = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )
    r2 = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "3")],
    )
    logger = logging.getLogger("dep_skip")
    with caplog.at_level(logging.WARNING):
        out = simulate_rules(grid, [r1, r2], logger=logger)
    assert out.data == [[2]]
    assert any("skipped" in rec.message for rec in caplog.records)


def test_color_dependency_strict():
    grid = Grid([[1]])
    r1 = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )
    r2 = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "3")],
    )
    with pytest.raises(ValueError):
        simulate_rules(grid, [r1, r2], strict=True)


def test_composite_chain_validation_accept():
    grid = Grid([[1]])

    step1 = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )
    step2 = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "2")],
        target=[Symbol(SymbolType.COLOR, "3")],
    )
    comp = CompositeRule([step1, step2])

    validated = validate_color_dependencies([comp], grid)
    assert validated == [comp]
