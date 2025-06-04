"""Segmentation utilities for ARC grids."""

from typing import Any, List

from .segmenter import assign_zone_labels, segment_connected_regions, segment_fixed_zones


def segment(grid: Any) -> List:
    """Placeholder segmentation entry point."""
    return []

__all__ = [
    "segment_fixed_zones",
    "segment_connected_regions",
    "assign_zone_labels",
    "segment",
]
