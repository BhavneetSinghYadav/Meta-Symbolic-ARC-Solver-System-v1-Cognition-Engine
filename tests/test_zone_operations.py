from arc_solver.src.symbolic.zone_remap import zone_remap
from arc_solver.src.symbolic.morphology_ops import dilate_zone, erode_zone
from arc_solver.src.symbolic.pattern_fill import pattern_fill
from arc_solver.src.memory.rule_memory import RuleMemory
from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    SymbolicRule,
    Transformation,
    TransformationType,
    Symbol,
    SymbolType,
)


def test_zone_remap_basic():
    base = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    overlay = [[1, 1, 2], [1, 2, 2], [1, 2, 2]]
    mapping = {1: 3, 2: 5}
    assert zone_remap(base, overlay, mapping) == [
        [3, 3, 5],
        [3, 5, 5],
        [3, 5, 5],
    ]


def test_dilate_erode_zone():
    grid = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    overlay = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    assert dilate_zone(grid, 1, overlay) == [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    assert erode_zone(grid, 1, overlay) == [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]


def test_pattern_fill_then_remap():
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    overlay = [[1, 1, 2], [1, 2, 2], [1, 2, 2]]
    grid[0][0] = 3
    grid[1][0] = 4
    filled = pattern_fill(grid, 1, 2, overlay)
    remapped = zone_remap(filled, overlay, {1: 7, 2: 8})
    assert remapped == [
        [7, 7, 8],
        [7, 8, 8],
        [7, 8, 8],
    ]


def test_rule_memory_cycle(tmp_path):
    mem_path = tmp_path / 'mem.json'
    mem = RuleMemory(mem_path)
    rule = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, '1')],
        target=[Symbol(SymbolType.COLOR, '2')],
    )
    inp = Grid([[1, 1], [1, 1]])
    out = Grid([[2, 2], [2, 2]])
    mem.record('t1', rule, inp, out)
    suggested = mem.suggest(inp, min_score=0.7)
    assert suggested and str(suggested[0]) == str(rule)
