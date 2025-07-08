from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic.repeat_rule import repeat_tile, generate_repeat_rules


def test_repeat_rule_basic():
    inp = Grid([[1, 2], [3, 4]])
    expected = repeat_tile(inp, 3, 3)
    rules = generate_repeat_rules(inp, expected)
    assert rules
    out = simulate_rules(inp, rules)
    assert out.compare_to(expected) == 1.0

