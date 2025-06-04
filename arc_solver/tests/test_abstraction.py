from arc_solver.src.abstractions.abstractor import abstract

def test_abstract_returns_list():
    assert isinstance(abstract([]), list)
