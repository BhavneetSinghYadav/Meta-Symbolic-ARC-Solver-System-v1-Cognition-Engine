from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic.repeat_rule import repeat_tile
from arc_solver.src.symbolic.composite_rules import generate_repeat_composite_rules


def test_composite_repeat_recolor():
    inp = Grid([[1, 0], [0, 1]])
    tiled = repeat_tile(inp, 3, 2)
    tgt = Grid([[2 if v == 1 else v for v in row] for row in tiled.data])
    rules = generate_repeat_composite_rules(inp, tgt)
    assert rules, "No composite rule generated"
    pred = simulate_rules(inp, rules)
    assert pred.compare_to(tgt) == 1.0
