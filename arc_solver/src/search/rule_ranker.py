from __future__ import annotations

"""Simple heuristics for ranking sets of symbolic rules."""

import math
from pathlib import Path
from typing import List

import yaml

from arc_solver.src.memory.policy_cache import PolicyCache
from arc_solver.src.symbolic.vocabulary import SymbolicRule
from arc_solver.src.symbolic.rule_language import rule_to_dsl
from arc_solver.src.search.feature_mapper import rule_feature_vector


_MOTIF_PATH = Path(__file__).resolve().parents[2] / "configs" / "motif_db.yaml"


def _load_motifs(path: Path = _MOTIF_PATH) -> List[dict]:
    """Load motif rules from ``path`` if available."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or []
    except Exception:  # pragma: no cover - best effort
        return []


def _tokenize(text: str) -> List[str]:
    for ch in "[],()":
        text = text.replace(ch, " ")
    return [t for t in text.lower().split() if t]


def _cosine(a: List[str], b: List[str]) -> float:
    """Cosine similarity between token lists."""
    if not a or not b:
        return 0.0
    vocab = set(a) | set(b)
    vec_a = [a.count(tok) for tok in vocab]
    vec_b = [b.count(tok) for tok in vocab]
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


_MOTIFS = _load_motifs()


def _motif_bonus(rule: SymbolicRule, threshold: float = 0.6) -> float:
    """Return a small bonus if ``rule`` matches a known motif."""
    tokens = _tokenize(rule_to_dsl(rule))
    for m in _MOTIFS:
        mtokens = _tokenize(str(m.get("rule_dsl", "")))
        if _cosine(tokens, mtokens) >= threshold:
            return 0.2
    return 0.0


def rank_rule_sets(
    rule_sets: List[List[SymbolicRule]],
    policy_cache: PolicyCache | None = None,
    task_hash: str | None = None,
) -> List[List[SymbolicRule]]:
    """Return ``rule_sets`` sorted by heuristic score."""

    def _rule_score(rule: SymbolicRule) -> float:
        feats = rule_feature_vector(rule)
        dsl_len = len(rule_to_dsl(rule))
        cond_complexity = len(rule.condition) if rule.condition else 0
        minimality = -0.01 * dsl_len - 0.1 * cond_complexity
        prior_bonus = _motif_bonus(rule)
        return sum(feats) + minimality + prior_bonus

    def score(rs: List[SymbolicRule]) -> tuple:
        base = len(rs) + len({r.transformation.ttype for r in rs})
        scores = [_rule_score(r) for r in rs]
        if scores:
            mean = sum(scores) / len(scores)
            var = sum((s - mean) ** 2 for s in scores) / len(scores)
            stab = mean - var
        else:
            stab = 0.0
        reuse = 0
        if policy_cache and task_hash and policy_cache.is_failed(task_hash, rs):
            reuse = -1
        return (reuse, stab + base)

    return sorted(rule_sets, key=score, reverse=True)
