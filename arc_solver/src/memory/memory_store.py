from __future__ import annotations

"""Persistent storage of successful rule programs."""

import json
from pathlib import Path
from typing import List, Dict, Any

from arc_solver.src.symbolic.rule_language import parse_rule, rule_to_dsl
from arc_solver.src.symbolic.vocabulary import SymbolicRule
from arc_solver.src.utils.signature_extractor import similarity_score


_DEFAULT_PATH = Path("rule_memory.json")


def load_memory(path: str | Path = _DEFAULT_PATH) -> List[Dict[str, Any]]:
    """Return list of stored rule programs."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_rule_program(
    task_id: str,
    signature: str,
    rules: List[SymbolicRule],
    score: float,
    path: str | Path = _DEFAULT_PATH,
) -> None:
    """Append a rule program entry to the memory store."""
    memory = load_memory(path)
    entry = {
        "task_id": task_id,
        "signature": signature,
        "rules": [rule_to_dsl(r) for r in rules],
        "score": score,
    }
    memory.append(entry)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(memory, f)


def retrieve_similar_signatures(
    current_signature: str,
    path: str | Path = _DEFAULT_PATH,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Return stored programs with signatures similar to ``current_signature``."""
    memory = load_memory(path)
    scored = []
    for entry in memory:
        sim = similarity_score(current_signature, entry.get("signature", ""))
        scored.append((sim, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    results: List[Dict[str, Any]] = []
    for sim, entry in scored[:top_k]:
        rules = [parse_rule(r) for r in entry.get("rules", [])]
        results.append({"rules": rules, "score": entry.get("score", 0.0), "similarity": sim})
    return results


__all__ = ["save_rule_program", "load_memory", "retrieve_similar_signatures"]
