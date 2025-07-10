from __future__ import annotations

"""Rule reuse and reliability tracking cache.

This module records successful rules from past tasks and suggests them for
reuse when encountering structurally similar inputs. Similarity is estimated
from basic grid features such as shape, color histogram and zone layout hash.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import hashlib

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.rule_language import rule_to_dsl, parse_rule
from arc_solver.src.symbolic.vocabulary import SymbolicRule
from arc_solver.src.segment.segmenter import zone_overlay


class RuleMemory:
    """Persistent cache of successful rules keyed by task id."""

    def __init__(self, memory_file: str | Path = "memory/rule_cache.json") -> None:
        self.path = Path(memory_file)
        self.memory: Dict[str, List[Dict[str, Any]]] = self._load(self.path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def suggest(self, input_grid: Grid, *, min_score: float = 0.9) -> List[SymbolicRule]:
        """Return stored rules ranked by similarity to ``input_grid``."""
        features = self._features(input_grid)
        scored: List[Tuple[float, SymbolicRule]] = []
        for entries in self.memory.values():
            for item in entries:
                if item.get("score", 0.0) < min_score:
                    continue
                sim = self._score_similarity(features, item.get("features", {}))
                if sim > 0.5:
                    try:
                        rule = parse_rule(item["rule_dsl"])
                    except Exception:
                        continue
                    scored.append((sim, rule))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rule for _, rule in scored]

    def record(
        self,
        task_id: str,
        rule: SymbolicRule,
        input_grid: Grid,
        output_grid: Grid,
    ) -> None:
        """Store ``rule`` with context information and reliability score."""
        pred = rule.apply(input_grid)
        score = pred.detailed_score(output_grid)
        entry = {
            "rule_dsl": rule_to_dsl(rule),
            "score": score,
            "features": self._features(input_grid),
        }
        self.memory.setdefault(task_id, []).append(entry)
        self._save(self.path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self, path: Path) -> Dict[str, List[Dict[str, Any]]]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.memory, f)

    # feature extraction -------------------------------------------------
    def _features(self, grid: Grid) -> Dict[str, Any]:
        shape = grid.shape()
        hist = [0] * 10
        counts = grid.count_colors()
        for c, v in counts.items():
            if 0 <= c < 10:
                hist[c] = v
        overlay = zone_overlay(grid)
        flat = [str(cell.value) if cell else "-" for row in overlay for cell in row]
        zone_hash = hashlib.sha1("".join(flat).encode()).hexdigest()[:8]
        return {"shape": shape, "colors": hist, "zone_hash": zone_hash}

    def _score_similarity(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        shape_sim = 1.0 if a.get("shape") == b.get("shape") else 0.0
        hist_a = a.get("colors") or []
        hist_b = b.get("colors") or []
        inter = sum(min(x, y) for x, y in zip(hist_a, hist_b))
        union = sum(hist_a) + sum(hist_b)
        color_sim = (2 * inter / union) if union else 0.0
        zone_sim = 1.0 if a.get("zone_hash") == b.get("zone_hash") else 0.0
        return 0.5 * shape_sim + 0.3 * color_sim + 0.2 * zone_sim


__all__ = ["RuleMemory"]
