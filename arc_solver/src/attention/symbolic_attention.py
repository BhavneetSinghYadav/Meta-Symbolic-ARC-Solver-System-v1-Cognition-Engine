from __future__ import annotations

"""Rule ranking attention mechanism."""

from typing import List, Tuple

import numpy as np

from arc_solver.src.symbolic.rule_language import rule_to_dsl


class SymbolicAttention:
    """Apply structural context to rule ranking."""

    def __init__(self, weight: float = 0.2, dim: int = 32) -> None:
        self.weight = weight
        self.dim = dim

    def _embed_token(self, token: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(token)) % (2**32))
        return rng.standard_normal(self.dim)

    def _embed_rule(self, rules: List) -> np.ndarray:
        if not rules:
            return np.zeros(self.dim, dtype=float)
        vec = np.zeros(self.dim, dtype=float)
        for r in rules:
            vec += self._embed_token(rule_to_dsl(r))
        vec /= len(rules)
        return vec

    def apply(
        self, ranked_rules: List[Tuple[List, float]], context_vec: np.ndarray
    ) -> List[Tuple[List, float]]:
        """Return reranked rule sets with attention-adjusted scores."""

        adjusted: List[Tuple[List, float]] = []
        for rules, base in ranked_rules:
            rule_vec = self._embed_rule(rules)
            score = float(np.dot(rule_vec, context_vec))
            adjusted.append((rules, base + self.weight * score))
        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted


__all__ = ["SymbolicAttention"]
