from arc_solver.src.regime.policy_router import decide_policy
from arc_solver.src.regime.regime_classifier import RegimeType


def test_policy_router_mapping():
    assert decide_policy(RegimeType.RequiresHeuristic, 0.2) == "fallback"
    assert decide_policy(RegimeType.LikelyConflicted, 0.3) == "repair_then_simulate"
    assert decide_policy(RegimeType.Fragmented, 0.2) == "fallback_then_prior"
    assert decide_policy(RegimeType.SymbolicallyTractable, 0.8) == "symbolic"
