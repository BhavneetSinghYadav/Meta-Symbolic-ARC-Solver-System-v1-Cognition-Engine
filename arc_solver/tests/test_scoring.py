from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
    TransformationNature,
)
from arc_solver.src.executor.scoring import score_rule, preferred_rule_types


def test_rule_scoring_on_tiling_vs_replace():
    inp = Grid([[1]])
    out = Grid([[1, 1], [1, 1]])

    repeat = SymbolicRule(
        transformation=Transformation(TransformationType.REPEAT, params={"kx": "2", "ky": "2"}),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
        nature=TransformationNature.SPATIAL,
    )

    replace = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )

    s_repeat = score_rule(inp, out, repeat)
    s_replace = score_rule(inp, out, replace)
    assert s_repeat > s_replace


def test_grow_rules_prioritization():
    inp = Grid([[1]])
    out = Grid([[1, 1], [1, 1]])
    prefs = preferred_rule_types(inp, out)
    assert "REPEAT" in prefs or "REPEAT→REPLACE" in prefs
