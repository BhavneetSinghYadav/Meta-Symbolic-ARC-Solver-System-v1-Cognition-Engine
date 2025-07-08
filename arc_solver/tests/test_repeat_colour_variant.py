from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.repeat_rule import generate_repeat_rules
from arc_solver.src.executor.simulator import simulate_rules


def test_repeat_colour_variant():
    inp = Grid([[1, 2], [3, 4]])
    tgt = Grid([
        [5, 2, 5, 2, 5, 2],
        [3, 4, 3, 4, 3, 4],
        [5, 2, 5, 2, 5, 2],
        [3, 4, 3, 4, 3, 4],
        [5, 2, 5, 2, 5, 2],
        [3, 4, 3, 4, 3, 4],
    ])
    rules = generate_repeat_rules(inp, tgt)
    assert rules, "repeat rule not generated"
    tiled = simulate_rules(inp, rules)
    assert tiled.shape() == tgt.shape()
