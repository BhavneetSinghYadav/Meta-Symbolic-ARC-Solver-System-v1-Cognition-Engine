from arc_solver.src.core.grid import Grid
from arc_solver.src.executor import fallback_predictor


def test_dataset_transform_ranking_applied():
    g = Grid([[1, 2], [3, 4]])
    out = fallback_predictor.predict(g)
    ranking = getattr(fallback_predictor, "_RANKED_TRANSFORMS", [])
    transforms = getattr(fallback_predictor, "_TRANSFORMS", {})
    if ranking:
        tname = ranking[0]
        transformed = transforms[tname](g)
    else:
        transformed = g
    counts = transformed.count_colors()
    mode = max(counts, key=counts.get) if counts else 0
    expected = fallback_predictor.pad_to_expected(transformed, fill=mode)
    assert out.data == expected.data
