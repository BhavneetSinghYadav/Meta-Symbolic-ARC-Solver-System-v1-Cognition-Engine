from debug.score_trace_utils import score_trace_explainer


def test_score_trace_explainer_composite():
    trace = {
        "similarity": 0.63,
        "cost": 0.12,
        "op_types": ["REPEAT", "RECOLOR"],
        "zone_match": 0.7,
        "shape_bonus": 0.05,
        "penalties": {"length": 0.06, "cost": 0.04},
        "composite": True,
        "steps": 2,
        "final_score": 0.57,
    }
    text = score_trace_explainer(trace, "T")
    assert "Composite rule" in text
    assert "REPEAT" in text and "RECOLOR" in text
    assert "penalty" in text
    assert "0.57" in text


def test_score_trace_explainer_atomic():
    trace = {
        "similarity": 0.9,
        "cost": 0.08,
        "op_types": ["REPLACE"],
        "zone_match": 0.85,
        "penalties": {"cost": 0.02},
        "final_score": 0.88,
    }
    text = score_trace_explainer(trace, "T")
    assert "Atomic rule" in text
    assert "REPLACE" in text
    assert "0.88" in text
    assert "penalty" in text
