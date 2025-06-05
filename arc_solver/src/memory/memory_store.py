from __future__ import annotations

"""Persistent storage of successful rule programs."""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any

from arc_solver.src.symbolic.rule_language import parse_rule, rule_to_dsl
from arc_solver.src.symbolic.vocabulary import SymbolicRule, is_valid_symbol
from arc_solver.src.utils import config_loader
from arc_solver.src.utils.signature_extractor import similarity_score


_DEFAULT_PATH = Path("rule_memory.json")


_LAST_LOAD_STATS: tuple[int, int] = (0, 0)


def load_memory(path: str | Path = _DEFAULT_PATH, *, verbose: bool = False) -> List[Dict[str, Any]]:
    """Return list of stored rule programs, filtering malformed entries."""
    global _LAST_LOAD_STATS
    p = Path(path)
    if not p.exists():
        _LAST_LOAD_STATS = (0, 0)
        return []
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if config_loader.LAZY_MEMORY_LOADING:
        loaded = sum(len(e.get("rules", [])) for e in data)
        _LAST_LOAD_STATS = (loaded, 0)
        if verbose:
            print(f"Loaded {loaded} rules (lazy)")
        return data

    cleaned: List[Dict[str, Any]] = []
    discarded = 0
    loaded = 0
    for entry in data:
        valid_rules: List[str] = []
        for r in entry.get("rules", []):
            try:
                rule = parse_rule(r)
            except Exception:
                discarded += 1
                continue
            if not rule.is_well_formed() or not all(
                is_valid_symbol(s) for s in rule.source + rule.target
            ):
                discarded += 1
                continue
            valid_rules.append(rule_to_dsl(rule))
        loaded += len(valid_rules)
        entry["rules"] = valid_rules
        cleaned.append(entry)

    _LAST_LOAD_STATS = (loaded, discarded)
    if verbose:
        print(f"Loaded {loaded} rules, {discarded} discarded as malformed")
    return cleaned


def get_last_load_stats() -> tuple[int, int]:
    """Return counts of rules loaded and discarded in the last load operation."""
    return _LAST_LOAD_STATS


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
        parsed = []
        for r in entry.get("rules", []):
            try:
                rule = parse_rule(r)
            except Exception:
                continue
            if not rule.is_well_formed():
                continue
            parsed.append(rule)
        if not parsed:
            continue
        results.append({"rules": parsed, "score": entry.get("score", 0.0), "similarity": sim})
    return results


__all__ = [
    "save_rule_program",
    "load_memory",
    "retrieve_similar_signatures",
    "get_last_load_stats",
]


def preload_memory_from_kaggle_input() -> None:
    """Copy Kaggle dataset memory file into the working directory if present."""
    src = "/kaggle/input/arc-memory/rule_memory.json"
    if os.path.exists(src):
        shutil.copy(src, _DEFAULT_PATH)


__all__.append("preload_memory_from_kaggle_input")

