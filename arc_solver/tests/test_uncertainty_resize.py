from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import SymbolicRule, Transformation, TransformationType, Symbol, SymbolType
from arc_solver.src.executor.simulator import simulate_rules


def test_uncertainty_grid_resizes():
    inp = Grid([[1]])
    repeat = SymbolicRule(
        transformation=Transformation(TransformationType.REPEAT, params={"kx": "2", "ky": "2"}),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "1")],
    )
    replace = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )
    out = simulate_rules(inp, [repeat, replace])
    assert out.shape() == (2, 2)
