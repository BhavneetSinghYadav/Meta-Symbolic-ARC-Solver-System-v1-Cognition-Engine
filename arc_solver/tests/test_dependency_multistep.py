import logging
from arc_solver.src.executor.dependency import sort_rules_by_dependency
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.symbolic.rule_language import CompositeRule


def _color_rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_order_and_simulation_no_warning(caplog):
    step1 = _color_rule(1, 2)
    step2 = _color_rule(2, 3)
    comp = CompositeRule([step1, step2])
    rule = _color_rule(1, 4)
    ordered = sort_rules_by_dependency([rule, comp])
    assert ordered == [rule, comp]

    grid = Grid([[1, 1], [1, 1]])
    logger = logging.getLogger("dep_multi")
    with caplog.at_level(logging.WARNING):
        simulate_rules(grid, ordered, logger=logger)
    assert not any("Source color" in rec.message for rec in caplog.records)
