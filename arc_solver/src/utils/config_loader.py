"""Loads YAML/JSON configuration files."""

import json
import yaml
from pathlib import Path


def load_config(path: str):
    path = Path(path)
    with open(path, "r") as f:
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        elif path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError("Unsupported config format")
