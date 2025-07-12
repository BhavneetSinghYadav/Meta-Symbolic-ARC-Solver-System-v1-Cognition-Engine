from __future__ import annotations

from typing import Any, Dict

from arc_solver.src.core.grid import Grid

class FunctionalOp:
    """Base class for functional ARC operators."""

    def simulate(self, grid: Grid, params: Dict[str, Any]) -> Grid:
        """Return result of applying operator on ``grid``."""
        raise NotImplementedError

    def validate_params(self, grid: Grid, params: Dict[str, Any]) -> None:
        """Validate ``params`` with respect to ``grid``.

        Should raise ``ValueError`` if parameters are malformed or
        ``RuntimeError`` if the transformation would be unsafe.
        """
        return None

    def proxy_meta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return metadata for proxy generation."""
        return {}
