from arc_solver.src.core.grid import Grid
from arc_solver.src.regime.regime_classifier import (
    compute_task_signature,
    predict_regime_category,
    RegimeType,
)


def test_low_entropy_symbolic():
    inp = Grid([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    out = Grid([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    sig = compute_task_signature([(inp, out)])
    assert predict_regime_category(sig) is RegimeType.SymbolicallyTractable


def test_high_entropy_fragmented():
    data_in = [[(r * c) % 10 for c in range(30)] for r in range(30)]
    data_out = [[(r + c) % 10 for c in range(30)] for r in range(30)]
    sig = compute_task_signature([(Grid(data_in), Grid(data_out))])
    assert predict_regime_category(sig) is RegimeType.Fragmented


def test_delta_requires_heuristic():
    inp = Grid([[0 for _ in range(5)] for _ in range(5)])
    out = Grid([[i + j + 2 for j in range(5)] for i in range(5)])
    sig = compute_task_signature([(inp, out)])
    assert predict_regime_category(sig) is RegimeType.RequiresHeuristic


def test_misaligned_pair_no_crash():
    inp = Grid([[1, 1], [1, 1]])
    out = Grid([[1]])
    sig = compute_task_signature([(inp, out)])
    assert isinstance(sig, dict)


def test_corrupted_task_zero_width():
    inp = Grid([[]])
    out = Grid([[]])
    sig = compute_task_signature([(inp, out)])
    assert isinstance(sig, dict)
