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


def set_offline_mode(value: bool) -> None:
    """Override offline mode at runtime."""
    global OFFLINE_MODE
    OFFLINE_MODE = value


def set_repair_enabled(value: bool) -> None:
    """Override repair loop enabled flag at runtime."""
    global REPAIR_ENABLED
    REPAIR_ENABLED = value


def set_repair_threshold(value: float) -> None:
    """Override repair loop threshold at runtime."""
    global REPAIR_THRESHOLD
    REPAIR_THRESHOLD = value


def set_reflex_override(value: bool) -> None:
    """Enable or disable reflex override logic."""
    global REFLEX_OVERRIDE_ENABLED
    REFLEX_OVERRIDE_ENABLED = value


def set_regime_threshold(value: float) -> None:
    """Override regime routing threshold."""
    global REGIME_THRESHOLD
    REGIME_THRESHOLD = value
