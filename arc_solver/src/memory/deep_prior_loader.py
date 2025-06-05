from __future__ import annotations

"""Utilities for loading and matching deep prior rule templates and motifs."""

from pathlib import Path
from typing import Any, Dict, List

import yaml
import json

from arc_solver.src.symbolic.rule_language import parse_rule
from arc_solver.src.symbolic.vocabulary import SymbolicRule
from arc_solver.src.utils.signature_extractor import similarity_score

_DEFAULT_PRIOR_PATH = Path(__file__).resolve().parents[2] / "configs" / "prior_templates.yaml"
_DEFAULT_MOTIF_PATH = Path(__file__).resolve().parents[2] / "configs" / "motif_db.yaml"


def _load_data(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(f) or []
        else:
            data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def load_prior_templates(path: str | Path | None = None) -> List[Dict[str, Any]]:
    """Return prior rule templates from file."""
    path = Path(path) if path else _DEFAULT_PRIOR_PATH
    entries = _load_data(path)
    valid: List[Dict[str, Any]] = []
    for ent in entries:
        rules: List[SymbolicRule] = []
        for text in ent.get("rules", []):
            try:
                rules.append(parse_rule(text))
            except Exception:
                rules = []
                break
        if rules:
            ent = ent.copy()
            ent["rules"] = rules
            valid.append(ent)
    # sort by frequency if available
    valid.sort(key=lambda x: x.get("frequency", 0), reverse=True)
    return valid


def load_motifs(path: str | Path | None = None) -> List[Dict[str, Any]]:
    """Return motif database entries."""
    path = Path(path) if path else _DEFAULT_MOTIF_PATH
    return _load_data(path)


def match_task_signature_to_prior(
    signature: str,
    threshold: float = 0.4,
    templates: List[Dict[str, Any]] | None = None,
) -> List[List[SymbolicRule]]:
    """Return prior programs best matching ``signature`` sorted by score."""
    templates = templates or load_prior_templates()
    scored: List[tuple[float, List[SymbolicRule]]] = []
    for ent in templates:
        sig = ent.get("signature", "")
        sim = similarity_score(signature, sig) if sig else 0.0
        if sim >= threshold:
            scored.append((sim, ent["rules"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [rules for _, rules in scored]


def select_motifs(signature: str, motifs: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    """Return motifs relevant to ``signature``."""
    motifs = motifs or load_motifs()
    selected: List[Dict[str, Any]] = []
    sym = "vsym" in signature
    for m in motifs:
        tag = m.get("abstract_tag", "")
        if sym and "mirror" in tag:
            selected.append(m)
        elif not sym and "color_shift" in tag:
            selected.append(m)
    selected.sort(key=lambda x: x.get("frequency", 0), reverse=True)
    return selected


__all__ = [
    "load_prior_templates",
    "load_motifs",
    "match_task_signature_to_prior",
    "select_motifs",
]
