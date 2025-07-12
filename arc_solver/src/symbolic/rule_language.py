"""Parsing utilities for the symbolic rule DSL.

This module now performs basic validation and sanitisation of DSL
expressions to avoid malformed rules crashing the solver.  It exposes
helpers used by :mod:`abstraction_dsl` for round trip checking and
program parsing.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import ast

from .vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationNature,
    TransformationType,
    validate_color_range as vocab_color_range,
)

logger = logging.getLogger(__name__)

DSL_GRAMMAR = {
    "type": [t.name.lower() for t in TransformationType],
    "symbol_range": range(0, 10),
    "operators": ["->", "|", "&"],
    "fields": ["color", "zone", "transform", "position"],
}

MAX_SYMBOL_VALUE = 10


def validate_color_range(color: int | str) -> bool:
    """Return ``True`` if ``color`` is a valid ARC color index (0-9)."""

    return vocab_color_range(color)


def clean_dsl_string(s: str) -> str:
    """Return ``s`` trimmed of surrounding and newline whitespace."""

    return s.strip().replace("\n", "").replace("\t", "")


def validate_dsl_program(program_str: str) -> bool:
    """Return ``True`` if ``program_str`` looks like a well formed DSL rule."""

    if not isinstance(program_str, str):
        return False
    if "->" not in program_str:
        return False
    if program_str.count("[") != program_str.count("]"):
        return False
    if any(tok in program_str for tok in ["NaN", "None"]):
        return False
    op = clean_dsl_string(program_str).split("[", 1)[0].strip()
    base_op = op.split("(", 1)[0]
    if base_op.upper() not in {t.name for t in TransformationType} and base_op.lower() not in {
        "mirror_tile",
        "pattern_fill",
        "rotate_about_point",
        "zone_remap",
        "dilate_zone",
        "erode_zone",
    }:
        return False
    return True


def extract_symbol_value(s: str) -> int:
    """Parse ``s`` as an integer within ``MAX_SYMBOL_VALUE``."""

    try:
        val = int(s)
        assert 0 <= val < MAX_SYMBOL_VALUE
        return val
    except Exception as exc:  # noqa: PERF203 - simple guard
        raise ValueError(f"Invalid symbol value in: {s}") from exc


def _parse_symbol(token: str) -> Symbol:
    if "=" not in token:
        raise ValueError(f"Invalid symbol token: {token}")
    key, value = token.split("=", 1)
    key = key.strip().upper()
    value = value.strip()
    try:
        stype = SymbolType[key]
    except KeyError as exc:
        raise ValueError(f"Unknown symbol type: {key}") from exc
    if stype is SymbolType.COLOR:
        if value.isdigit():
            value_int = extract_symbol_value(value)
            if not validate_color_range(value_int):
                raise ValueError(f"Invalid color value: {value}")
            value = str(value_int)
        else:
            if not validate_color_range(value):
                raise ValueError(f"Invalid color value: {value}")
    return Symbol(stype, value)


def _parse_symbol_list(text: str) -> List[Symbol]:
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError(f"Invalid symbol list: {text}")
    content = text[1:-1].strip()
    if not content:
        return []
    parts = [p.strip() for p in content.split(",")]
    return [_parse_symbol(part) for part in parts]


def parse_rule(text: str) -> SymbolicRule:
    """Parse ``text`` into a :class:`SymbolicRule` instance.

    The parser understands optional transformation parameters using a
    ``NAME(key=value, ...)`` syntax.  Unknown operators fall back to the
    ``FUNCTIONAL`` transformation type with ``op`` capturing the raw
    operator name.  This keeps older DSL strings compatible while
    allowing new functional primitives to round trip correctly.
    """

    text = clean_dsl_string(text)
    if not validate_dsl_program(text):
        raise ValueError("Malformed rule expression")
    if "[" not in text or "]" not in text:
        raise ValueError("Invalid rule format")

    idx = text.index("[")
    op_token = text[:idx].strip()
    remainder = text[idx:].lstrip()

    params: Dict[str, Any] = {}
    if op_token.endswith(")") and "(" in op_token:
        op_name, param_str = op_token.split("(", 1)
        op_name = op_name.strip()
        param_str = param_str[:-1]
        for part in param_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                logger.warning("Malformed parameter '%s' in '%s'", part, text)
                continue
            k, v = [p.strip() for p in part.split("=", 1)]
            params[k] = v
    else:
        op_name = op_token.strip()

    ttype: TransformationType
    meta: Dict[str, Any] = {}
    lower_op = op_name.lower()
    if op_name.upper() in TransformationType.__members__:
        ttype = TransformationType[op_name.upper()]
    elif lower_op == "rotate_about_point":
        ttype = TransformationType.ROTATE
        pivot = params.pop("pivot", None)
        if pivot is not None:
            try:
                cx, cy = [p.strip() for p in str(pivot).strip("() ").split(",")]
                params["cx"] = cx
                params["cy"] = cy
            except Exception:
                logger.warning("Malformed pivot parameter: %s", pivot)
        angle = params.pop("angle", None)
        if angle is not None:
            params["angle"] = angle
    elif lower_op in {"mirror_tile", "pattern_fill", "zone_remap", "dilate_zone", "erode_zone"}:
        ttype = TransformationType.FUNCTIONAL
        params = {**{k: v for k, v in params.items() if v is not None}, "op": lower_op, **{}}
        if lower_op in {"dilate_zone", "erode_zone"}:
            z = params.pop("zone_id", None)
            if z is not None:
                params["zone"] = z
        if lower_op == "zone_remap":
            mapping = params.pop("mapping", None)
            if mapping is not None:
                try:
                    meta["mapping"] = ast.literal_eval(mapping)
                except Exception:
                    logger.warning("Failed to parse mapping: %s", mapping)
                    meta["mapping"] = mapping
        if lower_op == "pattern_fill":
            mapping = params.pop("mapping", None)
            if mapping is not None:
                try:
                    meta["mapping"] = ast.literal_eval(mapping)
                except Exception:
                    meta["mapping"] = mapping
    else:
        raise ValueError(f"Unknown transformation type: {op_name}")

    if "->" not in remainder:
        raise ValueError("Rule must contain '->'")

    left_str, right_str = [part.strip() for part in remainder.split("->", 1)]
    left_syms = _parse_symbol_list(left_str)
    right_syms = _parse_symbol_list(right_str)

    transformation = Transformation(ttype, params={k: str(v) for k, v in params.items()})
    return SymbolicRule(transformation=transformation, source=left_syms, target=right_syms, meta=meta)


def rule_to_dsl(rule: SymbolicRule) -> str:
    """Serialise ``rule`` back into DSL form including parameters."""

    op_name = rule.transformation.ttype.value
    params = dict(rule.transformation.params)

    # Detect functional shorthand operators
    functional_op = None
    if rule.transformation.ttype is TransformationType.FUNCTIONAL:
        functional_op = params.get("op")
    if rule.transformation.ttype is TransformationType.ROTATE and {
        "cx",
        "cy",
        "angle",
    }.issubset(params):
        functional_op = "rotate_about_point"

    param_parts: list[str] = []

    if functional_op == "mirror_tile":
        op_name = "mirror_tile"
        axis = params.get("axis")
        repeats = params.get("repeats")
        if axis is not None:
            param_parts.append(f"axis={axis}")
        if repeats is not None:
            param_parts.append(f"repeats={repeats}")
    elif functional_op == "pattern_fill":
        op_name = "pattern_fill"
        mapping = rule.meta.get("mapping") or params.get("mapping")
        if mapping is not None:
            if isinstance(mapping, dict):
                items = ", ".join(f"{k}: {v}" for k, v in sorted(mapping.items()))
                mapping_str = "{" + items + "}"
            else:
                mapping_str = str(mapping)
            param_parts.append(f"mapping={mapping_str}")
    elif functional_op == "zone_remap":
        op_name = "zone_remap"
        mapping = rule.meta.get("mapping") or params.get("mapping")
        if mapping is not None:
            if isinstance(mapping, dict):
                items = ", ".join(f"{k}: {v}" for k, v in sorted(mapping.items()))
                mapping_str = "{" + items + "}"
            else:
                mapping_str = str(mapping)
            param_parts.append(f"mapping={mapping_str}")
    elif functional_op in {"dilate_zone", "erode_zone"}:
        op_name = functional_op
        zone = params.get("zone") or params.get("zone_id")
        if zone is not None:
            param_parts.append(f"zone_id={zone}")
    elif functional_op:
        param_parts = [f"{k}={v}" for k, v in params.items() if k != "op"]
        op_name = functional_op
    elif functional_op is None and rule.transformation.ttype is TransformationType.ROTATE:
        cx = params.get("cx")
        cy = params.get("cy")
        angle = params.get("angle")
        op_name = "rotate_about_point"
        if cx is not None and cy is not None:
            param_parts.append(f"pivot=({cx},{cy})")
        if angle is not None:
            param_parts.append(f"angle={angle}")
    else:
        param_parts = [f"{k}={v}" for k, v in params.items()]

    op_str = op_name
    if param_parts:
        op_str += "(" + ", ".join(param_parts) + ")"

    left = ", ".join(str(s) for s in rule.source)
    right = ", ".join(str(s) for s in rule.target)
    base = f"{op_str} [{left}] -> [{right}]"
    if rule.nature:
        base += f" [{rule.nature.value}]"
    return base


def generate_fallback_rules(inp: "Grid", out: "Grid") -> list[SymbolicRule]:
    """Return heuristic fallback rules for use outside the abstractor."""
    from arc_solver.src.abstractions.abstractor import _heuristic_fallback_rules

    return _heuristic_fallback_rules(inp, out)


@dataclass
class CompositeRule:
    """A rule composed of multiple symbolic transformations."""

    steps: List[SymbolicRule]
    nature: TransformationNature | None = None
    meta: Dict[str, Any] = field(default_factory=dict)

    transformation: Transformation = field(
        default_factory=lambda: Transformation(TransformationType.COMPOSITE)
    )

    def simulate(self, grid: "Grid") -> "Grid":
        """Return grid after sequentially applying steps."""
        from arc_solver.src.executor.simulator import safe_apply_rule

        out = grid
        for step in self.steps:
            out = safe_apply_rule(step, out, perform_checks=False)
        return out

    def get_sources(self) -> List[Symbol]:
        """Return merged source symbols across all steps."""
        return list({s for step in self.steps for s in getattr(step, "source", [])})

    def get_targets(self) -> List[Symbol]:
        """Return merged target symbols across all steps."""
        return list({s for step in self.steps for s in getattr(step, "target", [])})


    def get_condition(self) -> Optional[Any]:
        """Return the first non-null condition among steps."""
        for step in self.steps:
            if getattr(step, "condition", None):
                return step.condition
        return None

    def is_well_formed(self) -> bool:
        return all(getattr(step, "is_well_formed", lambda: False)() for step in self.steps)

    def final_targets(self) -> List[Symbol]:
        """Return target symbols from the final step."""
        return self.steps[-1].target if self.steps else []

    def as_symbolic_proxy(self) -> SymbolicRule:
        """Create a SymbolicRule-like proxy for use in dependency utilities."""
        from arc_solver.src.executor.proxy_ext import as_symbolic_proxy as _proxy

        return _proxy(self)

    def to_string(self) -> str:
        return " -> ".join(str(s) for s in self.steps)

    def __str__(self) -> str:  # pragma: no cover - simple
        return self.to_string()


def final_targets(rule: CompositeRule) -> List[Symbol]:
    """Return target symbols from the final step of ``rule``."""
    return rule.final_targets()


__all__ = [
    "parse_rule",
    "rule_to_dsl",
    "generate_fallback_rules",
    "validate_dsl_program",
    "validate_color_range",
    "clean_dsl_string",
    "CompositeRule",
    "final_targets",
]
