"""Memory utilities for storing symbolic policies."""

from .policy_cache import PolicyCache
from .lineage import LineageTracker, RuleLineageTracker
from .rule_memory import RuleMemory

try:  # Optional import; memory_store depends on heavy packages like sklearn
    from .memory_store import (
        save_rule_program,
        load_memory,
        retrieve_similar_signatures,
        match_signature,
        get_best_memory_match,
        extract_task_constraints,
        get_last_load_stats,
        update_memory_stats,
        preload_memory_from_kaggle_input,
    )
except Exception:  # pragma: no cover - optional dependency missing
    save_rule_program = None
    load_memory = None
    retrieve_similar_signatures = None
    match_signature = None
    get_best_memory_match = None
    extract_task_constraints = None
    get_last_load_stats = None
    update_memory_stats = None
    preload_memory_from_kaggle_input = None

__all__ = [
    "PolicyCache",
    "LineageTracker",
    "RuleLineageTracker",
    "RuleMemory",
    "save_rule_program",
    "load_memory",
    "retrieve_similar_signatures",
    "match_signature",
    "get_best_memory_match",
    "extract_task_constraints",
    "get_last_load_stats",
    "update_memory_stats",
    "preload_memory_from_kaggle_input",
]

