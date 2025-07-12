from __future__ import annotations

"""Symbolic vocabulary definitions for the ARC cognitive engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Callable

# Import extended symbolic operators
try:  # mirror_tile used to live in operators.py in older versions
    from .mirror_tile import mirror_tile
except Exception:  # pragma: no cover - fallback for legacy layout
    from .operators import mirror_tile

try:
    from .pattern_fill_operator import pattern_fill
except Exception:  # pragma: no cover - handle absent optional module
    pattern_fill = None  # type: ignore

try:
    from .draw_line import draw_line  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    draw_line = None  # type: ignore

try:
    from .morphology_ops import dilate_zone, erode_zone  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    dilate_zone = erode_zone = None  # type: ignore

try:
    from .rotate_about_point import rotate_about_point
except Exception:  # pragma: no cover - optional dependency
    rotate_about_point = None  # type: ignore

try:
    from .zone_remap import zone_remap
except Exception:  # pragma: no cover - optional dependency
    zone_remap = None  # type: ignore

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
    ROTATE90 = "ROTATE90"
    REFLECT = "REFLECT"
    REPEAT = "REPEAT"
    SHAPE_ABSTRACT = "SHAPE_ABSTRACT"
    CONDITIONAL = "CONDITIONAL"
    REGION = "REGION"
    FUNCTIONAL = "FUNCTIONAL"
    COMPOSITE = "COMPOSITE"

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

    @property
    def dsl(self) -> str:
        """Return DSL representation of this rule."""
        from arc_solver.src.symbolic.rule_language import rule_to_dsl

        if hasattr(self, "dsl_str") and isinstance(self.dsl_str, str):
            return self.dsl_str
        dsl = rule_to_dsl(self)
        self.dsl_str = dsl
        return dsl

    def generalize_with(self, other: "SymbolicRule") -> "SymbolicRule | None":
        """Return merged rule if ``other`` shares this rule's DSL."""

        if self.dsl != other.dsl:
            return None

        merged_meta: Dict[str, Any] = {}
        keys = set(self.meta) | set(other.meta)
        for k in keys:
            v1 = self.meta.get(k)
            v2 = other.meta.get(k)
            if v1 is None:
                merged_meta[k] = v2
            elif v2 is None or v1 == v2:
                merged_meta[k] = v1
            else:
                merged_meta[k] = [v1, v2]

        merged = SymbolicRule(
            transformation=self.transformation,
            source=self.source,
            target=self.target,
            nature=self.nature,
            condition=dict(self.condition),
            meta=merged_meta,
        )
        merged.dsl_str = self.dsl
        return merged

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

    def as_symbolic_proxy(self) -> "SymbolicRule":
        """Return proxy describing this rule with zone metadata."""
        from arc_solver.src.executor.proxy_ext import as_symbolic_proxy as _proxy

        return _proxy(self)


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
    "get_extended_operators",
]


# === EXTENDED_OPERATORS ===
EXTENDED_OPERATORS: Dict[str, Dict[str, Any]] = {
    # Functional operator: mirror the grid while tiling
    "mirror_tile": {
        "factory": mirror_tile,
        "rule": lambda axis, repeats: SymbolicRule(
            transformation=Transformation(
                TransformationType.FUNCTIONAL,
                params={"op": "mirror_tile", "axis": str(axis), "repeats": str(int(repeats))},
            ),
            source=[Symbol(SymbolType.REGION, "All")],
            target=[Symbol(SymbolType.REGION, "All")],
            nature=TransformationNature.SPATIAL,
        ),
        "desc": "Tile the grid along an axis mirroring every alternate copy.",
    },
    # Fill regions of a grid using a repeating pattern mask
    "pattern_fill": {
        "factory": pattern_fill,
        "rule": lambda mask, pattern: SymbolicRule(
            transformation=Transformation(
                TransformationType.FUNCTIONAL,
                params={"op": "pattern_fill"},
            ),
            source=[Symbol(SymbolType.REGION, "All")],
            target=[Symbol(SymbolType.REGION, "All")],
            nature=TransformationNature.SPATIAL,
            meta={"mask": mask, "pattern": pattern},
        ),
        "desc": "Fill masked cells with a tiled pattern.",
    },
    # Draw a straight line between two points of a specific colour
    "draw_line": {
        "factory": draw_line,
        "rule": lambda p1, p2, color: SymbolicRule(
            transformation=Transformation(
                TransformationType.FUNCTIONAL,
                params={
                    "op": "draw_line",
                    "p1": str(p1),
                    "p2": str(p2),
                    "color": str(color),
                },
            ),
            source=[Symbol(SymbolType.REGION, "All")],
            target=[Symbol(SymbolType.REGION, "All")],
            nature=TransformationNature.SPATIAL,
        ),
        "desc": "Draw a coloured line segment between two coordinates.",
    },
    # Morphological dilate operation on labelled zone
    "dilate_zone": {
        "factory": dilate_zone,
        "rule": lambda zone_id: SymbolicRule(
            transformation=Transformation(
                TransformationType.FUNCTIONAL,
                params={"op": "dilate_zone", "zone": str(zone_id)},
            ),
            source=[Symbol(SymbolType.ZONE, str(zone_id))],
            target=[Symbol(SymbolType.ZONE, str(zone_id))],
            nature=TransformationNature.SPATIAL,
        ),
        "desc": "Dilate a segmented zone by one cell.",
    },
    # Morphological erode operation on labelled zone
    "erode_zone": {
        "factory": erode_zone,
        "rule": lambda zone_id: SymbolicRule(
            transformation=Transformation(
                TransformationType.FUNCTIONAL,
                params={"op": "erode_zone", "zone": str(zone_id)},
            ),
            source=[Symbol(SymbolType.ZONE, str(zone_id))],
            target=[Symbol(SymbolType.ZONE, str(zone_id))],
            nature=TransformationNature.SPATIAL,
        ),
        "desc": "Erode a segmented zone by one cell.",
    },
    # Rotate grid by multiples of 90 degrees around a point
    "rotate_about_point": {
        "factory": rotate_about_point,
        "rule": lambda center, angle: SymbolicRule(
            transformation=Transformation(
                TransformationType.ROTATE,
                params={"cx": str(center[0]), "cy": str(center[1]), "angle": str(int(angle))},
            ),
            source=[Symbol(SymbolType.REGION, "All")],
            target=[Symbol(SymbolType.REGION, "All")],
            nature=TransformationNature.SPATIAL,
        ),
        "desc": "Rotate the grid around a pivot point.",
    },
    # Recolour segments based on zone IDs
    "zone_remap": {
        "factory": zone_remap,
        "rule": lambda overlay, mapping: SymbolicRule(
            transformation=Transformation(
                TransformationType.FUNCTIONAL,
                params={"op": "zone_remap"},
            ),
            source=[Symbol(SymbolType.REGION, "All")],
            target=[Symbol(SymbolType.REGION, "All")],
            nature=TransformationNature.SPATIAL,
            meta={"mapping": mapping},
        ),
        "desc": "Remap colours of zones via an overlay mapping.",
    },
    # Placeholder for future operators
}


def get_extended_operators() -> Dict[str, Dict[str, Any]]:
    """Return registry of extended symbolic operators."""

    return EXTENDED_OPERATORS

