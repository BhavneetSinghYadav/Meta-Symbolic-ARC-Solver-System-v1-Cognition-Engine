"""Simple overlay scoring utilities."""

from __future__ import annotations

from typing import Dict


def zone_overlap_score(zone_label: str, entropy_map: Dict[str, float]) -> float:
    """Return normalized entropy score for ``zone_label``."""
    return float(entropy_map.get(zone_label, 0.0))


__all__ = ["zone_overlap_score"]
