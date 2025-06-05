from arc_solver.src.core.grid import Grid
from arc_solver.src.executor import fallback_predictor


def test_fallback_padding_shape():
    g = Grid([[1, 2, 3], [4, 5, 6]])
    out = fallback_predictor.predict(g)
    assert out.shape() == (3, 3)
    assert out.get(0, 0) == 1
    assert out.get(2, 2) == 1
