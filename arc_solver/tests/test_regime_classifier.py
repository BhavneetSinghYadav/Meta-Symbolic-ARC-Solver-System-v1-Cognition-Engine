from arc_solver.src.core.grid import Grid
import json
import arc_solver.src.regime.regime_classifier as rc
from arc_solver.src.regime.regime_classifier import (
    compute_task_signature,
    predict_regime_category,
    regime_confidence_score,
    add_signature_to_index,
    find_closest_signature,
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
    assert predict_regime_category(sig) is RegimeType.RequiresHeuristic


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


def test_low_entropy_classification():
    inp = Grid([[0]])
    out = Grid([[0]])
    sig = compute_task_signature([(inp, out)])
    assert predict_regime_category(sig) is RegimeType.SymbolicallyTractable


def test_symmetry_violation_case():
    inp = Grid([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    out = Grid([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    sig = compute_task_signature([(inp, out)])
    assert predict_regime_category(sig) is RegimeType.LikelyConflicted


def test_confidence_filtering():
    sig = {k: 0.0 for k in compute_task_signature([(Grid([[0]]), Grid([[0]]))]).keys()}
    conf = regime_confidence_score(sig, RegimeType.SymbolicallyTractable)
    assert conf < 0.55
    assert predict_regime_category(sig) is RegimeType.Unknown


def test_signature_indexing(tmp_path):
    rc._SIGNATURE_INDEX = tmp_path / "idx.json"
    inp = Grid([[1, 1], [1, 1]])
    out = Grid([[1, 1], [1, 1]])
    sig = compute_task_signature([(inp, out)])
    add_signature_to_index("t1", sig, RegimeType.Fragmented)
    label = find_closest_signature(sig, threshold=0.8)
    assert label == "Fragmented"


def test_corrupted_task_zero_width():
    inp = Grid([[]])
    out = Grid([[]])
    sig = compute_task_signature([(inp, out)])
    assert isinstance(sig, dict)
