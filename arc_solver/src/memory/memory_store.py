from __future__ import annotations

"""Persistent storage of successful rule programs."""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any

from sklearn.metrics.pairwise import cosine_similarity

from arc_solver.src.symbolic.rule_language import parse_rule, rule_to_dsl
from arc_solver.src.symbolic.vocabulary import SymbolicRule, is_valid_symbol
from arc_solver.src.utils import config_loader
from arc_solver.src.utils.signature_extractor import similarity_score
from arc_solver.src.utils.logger import get_logger


_DEFAULT_PATH = Path("rule_memory.json")


_LAST_LOAD_STATS: tuple[int, int] = (0, 0)

logger = get_logger(__name__)


def validate_program_entry(entry: Any) -> bool:
    """Return ``True`` if ``entry`` looks like a valid memory item."""
    required_keys = {"task_id", "rules", "score", "signature"}
    return isinstance(entry, dict) and required_keys.issubset(entry.keys())


def auto_clean_memory_file(path: Path) -> None:
    """Attempt to back up a corrupted memory file and reset it."""
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
    except Exception as e:
        logger.error(f"MEMORY corrupted memory file: {e}")
        backup = path.with_suffix(".bak")
        try:
            shutil.move(path, backup)
            logger.warning(f"MEMORY file backed up to {backup}")
        except Exception as exc:
            logger.error(f"MEMORY failed to backup memory file: {exc}")
        path.write_text("[]", encoding="utf-8")


def extract_task_constraints(grid: Any) -> Dict[str, Any]:
    """Return simple constraints dict for ``grid`` (shape only)."""
    try:
        shape = grid.shape()
    except Exception:
        shape = None
    return {"shape": shape}


def load_memory(path: str | Path = _DEFAULT_PATH, *, verbose: bool = False) -> List[Dict[str, Any]]:
    """Return list of stored rule programs, filtering malformed entries."""
    global _LAST_LOAD_STATS
    p = Path(path)
    if not p.exists():
        _LAST_LOAD_STATS = (0, 0)
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"MEMORY failed to load memory: {e}")
        auto_clean_memory_file(p)
        _LAST_LOAD_STATS = (0, 0)
        return []

    if not isinstance(raw_data, list):
        logger.error("MEMORY file format invalid")
        _LAST_LOAD_STATS = (0, 0)
        return []

    valid_memory = [e for e in raw_data if validate_program_entry(e)]
    skipped = len(raw_data) - len(valid_memory)
    if config_loader.MEMORY_DIAGNOSTICS:
        logger.info(f"MEMORY loaded: {len(valid_memory)} valid, {skipped} skipped")

    if config_loader.LAZY_MEMORY_LOADING:
        loaded = sum(len(e.get("rules", [])) for e in valid_memory)
        _LAST_LOAD_STATS = (loaded, skipped)
        if verbose:
            print(f"Loaded {loaded} rules (lazy)")
        if len(valid_memory) == 0:
            logger.warning("MEMORY no valid memory entries found in memory_store.json")
        return valid_memory

    cleaned: List[Dict[str, Any]] = []
    discarded = 0
    loaded = 0
    for entry in valid_memory:
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

    _LAST_LOAD_STATS = (loaded, discarded + skipped)
    if verbose:
        print(
            f"Loaded {loaded} rules, {discarded} discarded as malformed, {skipped} entries skipped"
        )
    if len(cleaned) == 0:
        logger.warning("MEMORY no valid memory entries found in memory_store.json")
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
    constraints: Dict[str, Any] | None = None,
) -> None:
    """Append a rule program entry to the memory store."""
    memory = load_memory(path)
    reliability = 1.0
    if rules:
        try:
            reliability = min(
                r.meta.get("rule_reliability", 1.0) for r in rules
            )
        except Exception:
            reliability = 1.0
    entry = {
        "task_id": task_id,
        "signature": signature,
        "rules": [rule_to_dsl(r) for r in rules],
        "score": score,
        "reliability": reliability,
        "uses": 0,
        "failures": 0,
        "reflex_errors": 0,
    }
    if constraints is not None:
        entry["constraints"] = constraints
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


def normalize_signature(sig: Any) -> List[float]:
    """Return numeric vector for signature."""
    if isinstance(sig, str):
        parts = [p for p in sig.replace(",", " ").split() if p]
    else:
        parts = list(sig)
    out: List[float] = []
    for p in parts:
        try:
            out.append(round(float(p), 3))
        except Exception:
            continue
    return out


def get_best_memory_match(
    current_sig: Any,
    memory_entries: List[Dict[str, Any]],
    threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """Return entries whose signature cosine similarity exceeds ``threshold``."""

    if threshold is None:
        threshold = config_loader.MEMORY_SIMILARITY_THRESHOLD

    norm_cur = [normalize_signature(current_sig)]
    matches = []
    sims: List[float] = []
    for entry in memory_entries:
        sig = normalize_signature(entry.get("signature", []))
        if not sig or not norm_cur[0]:
            # Fallback to token similarity
            sim = similarity_score(str(current_sig), str(entry.get("signature", "")))
        else:
            try:
                sim = cosine_similarity(norm_cur, [sig])[0][0]
            except Exception:
                sim = similarity_score(str(current_sig), str(entry.get("signature", "")))

        sims.append(sim)
        if config_loader.MEMORY_DIAGNOSTICS:
            logger.info(f"MEMORY match score: {sim:.4f} for task {entry.get('task_id')}")

        if sim >= threshold:
            matches.append((sim, entry))

    if config_loader.MEMORY_DIAGNOSTICS:
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        max_sim = max(sims) if sims else 0.0
        logger.info(
            f"MEMORY diagnostics - stored: {len(memory_entries)}, avg_similarity: {avg_sim:.4f}, max_similarity: {max_sim:.4f}"
        )

    matches.sort(key=lambda x: x[0], reverse=True)

    if not matches:
        logger.warning("MEMORY No memory match above threshold")

    return [e for _, e in matches]


def match_signature(
    current_signature: Any,
    path: str | Path = _DEFAULT_PATH,
    threshold: float | None = None,
    constraints: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Retrieve memory programs matching signature and constraints."""
    if threshold is None:
        threshold = config_loader.MEMORY_SIMILARITY_THRESHOLD
    memory = load_memory(path)
    candidates = get_best_memory_match(current_signature, memory, threshold)
    ignore_shape = config_loader.IGNORE_MEMORY_SHAPE_CONSTRAINT
    shape_skipped = 0
    if constraints:
        filtered = []
        for entry in candidates:
            c = entry.get("constraints")
            if not c:
                filtered.append(entry)
                continue
            if c.get("shape") and constraints.get("shape"):
                if tuple(c["shape"]) != tuple(constraints["shape"]) and not ignore_shape:
                    shape_skipped += 1
                    continue
            filtered.append(entry)
        candidates = filtered
    reliability_threshold = config_loader.MEMORY_RELIABILITY_THRESHOLD
    candidates = [
        c
        for c in candidates
        if c.get("reliability", 1.0) >= reliability_threshold
        and c.get("reflex_errors", 0) == 0
        and not (
            c.get("uses", 0) >= 3
            and c.get("failures", 0) / max(c.get("uses", 1), 1) > 0.6
        )
    ]
    parsed_candidates: List[Dict[str, Any]] = []
    for entry in candidates:
        parsed_rules = []
        for r in entry.get("rules", []):
            try:
                parsed_rules.append(parse_rule(r))
            except Exception:
                continue
        if parsed_rules:
            parsed_candidates.append({**entry, "rules": parsed_rules})
    if config_loader.MEMORY_DIAGNOSTICS:
        logger.info(
            f"MEMORY loaded: {len(memory)} entries; {len(parsed_candidates)} injected"
        )
        if shape_skipped:
            logger.info(f"MEMORY shape skipped: {shape_skipped}")
    return parsed_candidates


__all__ = [
    "save_rule_program",
    "load_memory",
    "retrieve_similar_signatures",
    "normalize_signature",
    "get_best_memory_match",
    "match_signature",
    "extract_task_constraints",
    "get_last_load_stats",
]


def update_memory_stats(
    task_id: str,
    coverage_score: float,
    *,
    reflex_error: bool = False,
    path: str | Path = _DEFAULT_PATH,
) -> None:
    """Update usage statistics for a stored program and prune if needed."""
    memory = load_memory(path)
    updated: List[Dict[str, Any]] = []
    for entry in memory:
        if entry.get("task_id") == task_id:
            entry["uses"] = entry.get("uses", 0) + 1
            if coverage_score < 0.3:
                entry["failures"] = entry.get("failures", 0) + 1
            if reflex_error:
                entry["reflex_errors"] = entry.get("reflex_errors", 0) + 1
        updated.append(entry)
    pruned: List[Dict[str, Any]] = []
    for entry in updated:
        if entry.get("reflex_errors", 0) > 0:
            continue
        uses = entry.get("uses", 0)
        fails = entry.get("failures", 0)
        if uses >= 3 and fails / max(uses, 1) > 0.6:
            continue
        pruned.append(entry)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(pruned, f)


__all__.append("update_memory_stats")


def preload_memory_from_kaggle_input() -> None:
    """Copy Kaggle dataset memory file into the working directory if present."""
    src = "/kaggle/input/arc-memory/rule_memory.json"
    if os.path.exists(src):
        shutil.copy(src, _DEFAULT_PATH)


__all__.append("preload_memory_from_kaggle_input")

