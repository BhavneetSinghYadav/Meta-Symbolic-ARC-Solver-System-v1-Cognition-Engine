import logging
from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic.vocabulary import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType


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
    assert any("Skipping rule" in rec.message for rec in caplog.records)
