from __future__ import annotations

"""Policy routing utilities mapping regimes to execution policies."""

from .regime_classifier import RegimeType


def decide_policy(regime: RegimeType | str, score: float) -> str:
    """Return policy label for given ``regime`` and detection ``score``."""
    name = regime.name if isinstance(regime, RegimeType) else str(regime)
    if name in {"RequiresHeuristic", "LowSymbolSupport"}:
        return "fallback"
    if name == "LikelyConflicted":
        return "repair_then_simulate"
    if name in {"Fragmented", "EntropyHighComplexity"}:
        return "fallback_then_prior"
    return "symbolic"


__all__ = ["decide_policy"]
