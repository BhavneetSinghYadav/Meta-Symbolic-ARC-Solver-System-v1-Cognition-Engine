from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.core.grid import Grid


def test_trace_meta():
    inp = Grid([[1, 1], [1, 1]])
    out = Grid([[0, 0], [0, 0]])
    rules = abstract([inp, out])
    assert any("derivation" in r.meta for r in rules)
