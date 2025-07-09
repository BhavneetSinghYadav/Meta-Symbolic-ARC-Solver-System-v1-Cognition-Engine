from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.simulator import ColorLineageTracker
from arc_solver.src.symbolic.vocabulary import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType
from arc_solver.src.symbolic.rule_language import CompositeRule


def _rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_sequential_lineage():
    grid = Grid([[1]])
    tracker = ColorLineageTracker(grid)
    simulate_rules(grid, [_rule(1, 2), _rule(2, 3)], lineage_tracker=tracker)
    assert tracker.get_lineage(3) == [
        "origin",
        f"1→2 by {_rule(1,2)}",
        f"2→3 by {_rule(2,3)}",
    ]


def test_composite_lineage():
    grid = Grid([[1, 2]])
    step1 = _rule(1, 3)
    step2 = _rule(2, 4)
    comp = CompositeRule([step1, step2])
    tracker = ColorLineageTracker(grid)
    simulate_rules(grid, [comp], lineage_tracker=tracker)
    line3 = tracker.get_lineage(3)
    line4 = tracker.get_lineage(4)
    assert line3 and line3[-1].endswith(str(comp))
    assert line4 and line4[-1].endswith(str(comp))


def test_parallel_changes():
    grid = Grid([[1, 2]])
    r1 = _rule(1, 3)
    r2 = _rule(2, 4)
    tracker = ColorLineageTracker(grid)
    simulate_rules(grid, [r1, r2], lineage_tracker=tracker)
    summary = tracker.render_lineage_summary()
    assert "3" in summary and "4" in summary
