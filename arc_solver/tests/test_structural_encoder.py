import numpy as np
from arc_solver.src.attention.structural_encoder import StructuralEncoder
from arc_solver.src.core.grid import Grid


def test_encoding_deterministic():
    encoder = StructuralEncoder(dim=8)
    overlay = [["A", None], [None, "B"]]
    grid = Grid([[0, 0], [0, 0]])
    vec1 = encoder.encode(grid, overlay)
    vec2 = encoder.encode(grid, overlay)
    assert vec1.shape == (8,)
    assert np.allclose(vec1, vec2)
