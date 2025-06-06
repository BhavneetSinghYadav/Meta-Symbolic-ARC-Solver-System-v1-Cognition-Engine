from __future__ import annotations

"""Symbolic vocabulary definitions for the ARC cognitive engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

MAX_COLOR = 10


def validate_color_range(value: int) -> bool:
    """Return True if ``value`` represents a legal color index."""
    try:
        iv = int(value)
        return 0 <= iv < MAX_COLOR
    except Exception:
        # Allow non-integer color tokens (e.g. "Red")
        return True


class SymbolType(Enum):
    """Types of symbolic attributes."""

    COLOR = "COLOR"
    SHAPE = "SHAPE"
    REGION = "REGION"
    ZONE = "ZONE"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class TransformationType(Enum):
    """Supported symbolic transformation primitives.

    New types extend the basic set with conditional and functional logic as well
    as region-scoped transformations.
    """

    REPLACE = "REPLACE"
    TRANSLATE = "TRANSLATE"
    MERGE = "MERGE"
    FILTER = "FILTER"
    ROTATE = "ROTATE"
    REFLECT = "REFLECT"
    CONDITIONAL = "CONDITIONAL"
    REGION = "REGION"
    FUNCTIONAL = "FUNCTIONAL"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class TransformationNature(Enum):
    """Nature or category of a transformation."""

    SPATIAL = "SPATIAL"
    LOGICAL = "LOGICAL"
    SYMMETRIC = "SYMMETRIC"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class Symbol:
    """Represents an abstract symbolic attribute."""

    type: SymbolType
    value: str

    def __post_init__(self) -> None:
        if self.type is SymbolType.COLOR:
            if not validate_color_range(self.value):
                raise ValueError(f"Invalid symbol value: {self.value}")

    def is_valid(self) -> bool:
        """Return ``True`` if this symbol represents a valid token."""
        if self.type is SymbolType.COLOR:
            return validate_color_range(self.value)
        return True

    def __str__(self) -> str:
        return f"{self.type.value}={self.value}"

    def __repr__(self) -> str:  # pragma: no cover - simple
        return f"Symbol(type={self.type}, value={self.value!r})"


def is_valid_symbol(symbol: Symbol) -> bool:
    """Return ``True`` if ``symbol`` has a legal value."""
    return symbol.is_valid()


@dataclass(frozen=True)
class Transformation:
    """A symbolic transformation with optional parameters."""

    ttype: TransformationType
    params: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.ttype.value}({param_str})"
        return self.ttype.value

    def __repr__(self) -> str:  # pragma: no cover - simple
        return f"Transformation(ttype={self.ttype}, params={self.params})"


@dataclass
class SymbolicRule:
    """Structured rule describing a symbolic transformation."""

    transformation: Transformation
    source: List[Symbol]
    target: List[Symbol]
    nature: TransformationNature | None = None
    condition: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        left = ", ".join(str(s) for s in self.source)
        right = ", ".join(str(s) for s in self.target)
        nature_str = f" [{self.nature.value}]" if self.nature else ""
        return f"{self.transformation.ttype.value} [{left}] -> [{right}]{nature_str}"

    def __repr__(self) -> str:  # pragma: no cover - simple
        return (
            "SymbolicRule("
            f"transformation={self.transformation!r}, "
            f"source={self.source!r}, "
            f"target={self.target!r}, "
            f"nature={self.nature!r}, condition={self.condition!r})"
        )

    def is_well_formed(self) -> bool:
        """Return True if color tokens contain valid integer values."""
        try:
            for sym in self.source + self.target:
                if not sym.is_valid():
                    return False
        except Exception:
            return False
        return True

    def apply(self, grid: "Grid") -> "Grid":
        """Return grid after applying this rule without strict checks."""
        from arc_solver.src.executor.simulator import safe_apply_rule

        return safe_apply_rule(self, grid, perform_checks=False)

    def triggers_large_conflict(
        self, conflict_map: list[list[int]], radius: int = 2
    ) -> bool:
        """Return True if conflict region exceeds ``radius`` heuristically."""
        points = [
            (r, c)
            for r, row in enumerate(conflict_map)
            for c, v in enumerate(row)
            if v
        ]
        if not points:
            return False
        rmin = min(p[0] for p in points)
        rmax = max(p[0] for p in points)
        cmin = min(p[1] for p in points)
        cmax = max(p[1] for p in points)
        if (rmax - rmin) > radius or (cmax - cmin) > radius:
            return True
        if len(points) > (radius + 1) ** 2:
            return True
        return False


__all__ = [
    "SymbolType",
    "TransformationType",
    "TransformationNature",
    "Symbol",
    "Transformation",
    "SymbolicRule",
    "validate_color_range",
    "MAX_COLOR",
    "is_valid_symbol",
]
