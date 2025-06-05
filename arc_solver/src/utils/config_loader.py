"""Loads YAML/JSON configuration files and global meta settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON configuration file."""
    path_p = Path(path)
    with open(path_p, "r", encoding="utf-8") as f:
        if path_p.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        if path_p.suffix == ".json":
            return json.load(f)
        raise ValueError("Unsupported config format")


def load_meta_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Return the solver's meta configuration."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "meta_config.yaml"
    if path.exists():
        return load_config(str(path))
    return {}


META_CONFIG: Dict[str, Any] = load_meta_config()
OFFLINE_MODE: bool = META_CONFIG.get("llm_mode", "online") == "offline"
REPAIR_ENABLED: bool = META_CONFIG.get("repair_enabled", False)
REPAIR_THRESHOLD: float = float(META_CONFIG.get("repair_threshold", 0.7))
_REFLEX_CONF = META_CONFIG.get("reflex_override", {})
REFLEX_OVERRIDE_ENABLED: bool = bool(_REFLEX_CONF.get("enabled", False))
REGIME_THRESHOLD: float = float(_REFLEX_CONF.get("threshold", 0.45))
DEFAULT_OVERRIDE_PATH: str = str(_REFLEX_CONF.get("default_path", "fallback"))

_PRIOR_CONF = META_CONFIG.get("prior_injection", {})
PRIOR_INJECTION_ENABLED: bool = bool(_PRIOR_CONF.get("enabled", False))
PRIOR_USE_MOTIFS: bool = bool(_PRIOR_CONF.get("use_motifs", False))
PRIOR_FROM_MEMORY: bool = bool(_PRIOR_CONF.get("inject_from_memory", False))
PRIOR_FALLBACK_ONLY: bool = bool(_PRIOR_CONF.get("fallback_only", False))
PRIOR_MAX_INJECT: int = int(_PRIOR_CONF.get("max_inject", 3))

USE_STRUCTURAL_ATTENTION: bool = bool(META_CONFIG.get("use_structural_attention", False))
STRUCTURAL_ATTENTION_WEIGHT: float = float(META_CONFIG.get("structural_attention_weight", 0.2))

INTROSPECTION_ENABLED: bool = bool(META_CONFIG.get("introspect", False))
MEMORY_ENABLED: bool = bool(META_CONFIG.get("use_memory", False))
LAZY_MEMORY_LOADING: bool = bool(META_CONFIG.get("lazy_memory", False))
MEMORY_SIMILARITY_THRESHOLD: float = float(
    META_CONFIG.get("memory_similarity_threshold", 0.95)
)
MEMORY_DIAGNOSTICS: bool = bool(META_CONFIG.get("memory_diagnostics", False))
SPARSE_MODE: bool = bool(META_CONFIG.get("sparse_mode", False))
FALLBACK_ON_ABSTRACTION_FAIL: bool = bool(
    META_CONFIG.get("fallback_on_abstraction_fail", False)
)
IGNORE_MEMORY_SHAPE_CONSTRAINT: bool = bool(
    META_CONFIG.get("ignore_memory_shape_constraint", False)
)



def set_offline_mode(value: bool) -> None:
    """Override offline mode at runtime."""
    global OFFLINE_MODE
    OFFLINE_MODE = value
    META_CONFIG["llm_mode"] = "offline" if value else "online"


def set_repair_enabled(value: bool) -> None:
    """Override repair loop enabled flag at runtime."""
    global REPAIR_ENABLED
    REPAIR_ENABLED = value
    META_CONFIG["repair_enabled"] = value


def set_repair_threshold(value: float) -> None:
    """Override repair loop threshold at runtime."""
    global REPAIR_THRESHOLD
    REPAIR_THRESHOLD = value
    META_CONFIG["repair_threshold"] = value


def set_reflex_override(value: bool) -> None:
    """Enable or disable reflex override logic."""
    global REFLEX_OVERRIDE_ENABLED
    REFLEX_OVERRIDE_ENABLED = value
    META_CONFIG.setdefault("reflex_override", {})["enabled"] = value


def set_regime_threshold(value: float) -> None:
    """Override regime routing threshold."""
    global REGIME_THRESHOLD
    REGIME_THRESHOLD = value
    META_CONFIG.setdefault("reflex_override", {})["threshold"] = value


def set_prior_injection(value: bool) -> None:
    """Enable or disable deep prior injection."""
    global PRIOR_INJECTION_ENABLED
    PRIOR_INJECTION_ENABLED = value
    META_CONFIG.setdefault("prior_injection", {})["enabled"] = value


def set_prior_threshold(value: int) -> None:
    """Override number of max injected priors."""
    global PRIOR_MAX_INJECT
    PRIOR_MAX_INJECT = value
    META_CONFIG.setdefault("prior_injection", {})["max_inject"] = value


def set_use_structural_attention(value: bool) -> None:
    """Enable or disable structural attention."""
    global USE_STRUCTURAL_ATTENTION
    USE_STRUCTURAL_ATTENTION = value
    META_CONFIG["use_structural_attention"] = value


def set_attention_weight(value: float) -> None:
    """Override structural attention weight."""
    global STRUCTURAL_ATTENTION_WEIGHT
    STRUCTURAL_ATTENTION_WEIGHT = value
    META_CONFIG["structural_attention_weight"] = value

def set_introspection_enabled(value: bool) -> None:
    """Enable or disable pipeline introspection."""
    global INTROSPECTION_ENABLED
    INTROSPECTION_ENABLED = value
    META_CONFIG["introspect"] = value


def set_memory_enabled(value: bool) -> None:
    """Enable or disable rule memory usage."""
    global MEMORY_ENABLED
    MEMORY_ENABLED = value
    META_CONFIG["use_memory"] = value


def set_sparse_mode(value: bool) -> None:
    """Enable or disable sparse ranking mode."""
    global SPARSE_MODE
    SPARSE_MODE = value
    META_CONFIG["sparse_mode"] = value


def set_fallback_on_abstraction_fail(value: bool) -> None:
    """Enable or disable fallback rule injection when abstraction fails."""
    global FALLBACK_ON_ABSTRACTION_FAIL
    FALLBACK_ON_ABSTRACTION_FAIL = value
    META_CONFIG["fallback_on_abstraction_fail"] = value


def set_ignore_memory_shape_constraint(value: bool) -> None:
    """Enable or disable ignoring shape constraints during memory recall."""
    global IGNORE_MEMORY_SHAPE_CONSTRAINT
    IGNORE_MEMORY_SHAPE_CONSTRAINT = value
    META_CONFIG["ignore_memory_shape_constraint"] = value


def print_runtime_config() -> None:
    """Print a summary of the current runtime configuration."""
    info = {
        "introspect": INTROSPECTION_ENABLED,
        "llm_mode": "offline" if OFFLINE_MODE else "online",
        "use_memory": MEMORY_ENABLED,
        "self_repair": REPAIR_ENABLED,
        "regime_override": REFLEX_OVERRIDE_ENABLED,
        "use_structural_attention": USE_STRUCTURAL_ATTENTION,
    }
    print("Runtime configuration:")
    for k, v in info.items():
        print(f"  {k}: {v}")


def print_system_health(memory_loaded: int, memory_skipped: int) -> None:
    """Print runtime system health overview."""
    print(
        f"\u2714 Memory rules: {memory_loaded} loaded, {memory_skipped} skipped"
    )
    print(
        f"\u2714 Structural attention: {'ENABLED' if USE_STRUCTURAL_ATTENTION else 'DISABLED'}"
    )
    print("\u2714 Fallback predictor: ACTIVATED")
    print(f"\u2714 Self-repair: {'ON' if REPAIR_ENABLED else 'OFF'}")

