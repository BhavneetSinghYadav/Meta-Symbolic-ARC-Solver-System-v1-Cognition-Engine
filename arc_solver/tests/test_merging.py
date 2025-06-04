from arc_solver.src.core.grid import Grid
from arc_solver.src.abstractions.abstractor import split_rule_by_overlay
from arc_solver.src.executor.merger import merge_rule_sets
from arc_solver.src.symbolic import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.segment.segmenter import zone_overlay


def _color_rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_split_rule_by_overlay():
    grid = Grid([[1, 0], [0, 0]])
    overlay = zone_overlay(grid)
    rule = _color_rule(1, 2)
    split = split_rule_by_overlay(rule, grid, overlay)
    assert len(split) == 1
    assert split[0].condition.get("zone") == "Center"


def test_merge_rule_sets():
    r1 = SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
        condition={"zone": "TopLeft"},
    )
    r2 = SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
        condition={"zone": "BottomRight"},
    )
    merged = merge_rule_sets([[r1], [r2]])
    assert len(merged) == 1
    cond = merged[0].condition.get("zone")
    assert "TopLeft" in cond and "BottomRight" in cond
