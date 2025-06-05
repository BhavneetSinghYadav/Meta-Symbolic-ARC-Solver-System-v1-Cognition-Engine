import json

from arc_solver.src.regime.regime_classifier import RegimeType
import arc_solver.src.regime.decision_controller as dc


def test_decision_controller_logging(tmp_path):
    dc._LOG_PATH = tmp_path / "log.json"
    controller = dc.DecisionReflexController("t1", RegimeType.SymbolicallyTractable, 0.9)
    policy = controller.decide()
    assert policy == "symbolic"
    data = json.loads(dc._LOG_PATH.read_text())
    assert data and data[0]["policy"] == "symbolic"
