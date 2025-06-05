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


def set_offline_mode(value: bool) -> None:
    """Override offline mode at runtime."""
    global OFFLINE_MODE
    OFFLINE_MODE = value
