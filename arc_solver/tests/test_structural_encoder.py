import numpy as np
from arc_solver.src.attention.structural_encoder import StructuralEncoder


def test_encoding_deterministic():
    encoder = StructuralEncoder(dim=8)
    overlay = [["A", None], [None, "B"]]
    vec1 = encoder.encode(overlay)
    vec2 = encoder.encode(overlay)
    assert vec1.shape == (8,)
    assert np.allclose(vec1, vec2)
