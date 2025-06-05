from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.core.grid import Grid


def test_abstract_fallback_on_failure():
    inp = Grid([[1]])
    out = Grid([[2, 2]])
    rules = abstract([inp, out])
    assert rules, "Fallback should provide at least one rule"
