from __future__ import annotations

"""Unified decision controller for regime-based policy selection."""

import json
from pathlib import Path
from typing import Any

from .regime_classifier import RegimeType
from .policy_router import decide_policy
from arc_solver.src.utils import config_loader

_LOG_PATH = Path("logs/regime_decision_log.json")


class DecisionReflexController:
    """Decide execution policy based on regime and scores."""

    def __init__(self, task_id: str, regime: RegimeType, score: float) -> None:
        self.task_id = task_id
        self.regime = regime
        self.score = score

    def decide(self) -> str:
        policy = decide_policy(self.regime, self.score)
        if (
            config_loader.REFLEX_OVERRIDE_ENABLED
            and self.score < config_loader.REGIME_THRESHOLD
            and policy == "symbolic"
        ):
            policy = "fallback"
        self._log({"task_id": self.task_id, "regime": self.regime.name, "score": self.score, "policy": policy})
        return policy

    def _log(self, data: dict[str, Any]) -> None:
        _LOG_PATH.parent.mkdir(exist_ok=True)
        if _LOG_PATH.exists():
            try:
                existing = json.loads(_LOG_PATH.read_text())
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        else:
            existing = []
        existing.append(data)
        with _LOG_PATH.open("w", encoding="utf-8") as f:
            json.dump(existing, f)


__all__ = ["DecisionReflexController"]
