"""Core grid utilities and data structures."""

from .grid import Grid
from .grid_utils import compute_conflict_map
from .grid_proxy import GridProxy

__all__ = ["Grid", "GridProxy", "compute_conflict_map"]

