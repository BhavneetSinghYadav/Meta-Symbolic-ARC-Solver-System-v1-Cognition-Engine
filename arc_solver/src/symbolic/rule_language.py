"""Parsing utilities for the symbolic rule DSL.

This module now performs basic validation and sanitisation of DSL
expressions to avoid malformed rules crashing the solver.  It exposes
helpers used by :mod:`abstraction_dsl` for round trip checking and
program parsing.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Callable
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


@dataclass
class OperatorSpec:
    """Metadata describing a functional DSL operator."""

    ttype: TransformationType
    params: List[str]
    parser: Callable[[Dict[str, str]], tuple[Dict[str, str], Dict[str, Any]]]
    serializer: Callable[[Dict[str, str], Dict[str, Any]], List[str]]
    description: str = ""


def get_extended_operators() -> Dict[str, OperatorSpec]:
    """Return mapping of functional operator names to their specs."""

    def _mirror_tile_parse(p: Dict[str, str]) -> tuple[Dict[str, str], Dict[str, Any]]:
        return {
            "op": "mirror_tile",
            "axis": p.get("axis", "horizontal"),
            "repeats": p.get("repeats", "2"),
        }, {}

    def _mirror_tile_ser(params: Dict[str, str], meta: Dict[str, Any]) -> List[str]:
        out = []
        if axis := params.get("axis"):
            out.append(f"axis={axis}")
        if reps := params.get("repeats"):
            out.append(f"repeats={reps}")
        return out

    def _rotate_parse(p: Dict[str, str]) -> tuple[Dict[str, str], Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        params: Dict[str, str] = {}
        pivot = p.get("pivot")
        if pivot:
            try:
                cx, cy = [x.strip() for x in pivot.strip("() ").split(",")]
                params["cx"] = cx
                params["cy"] = cy
            except Exception:
                meta["pivot"] = pivot
        angle = p.get("angle")
        if angle:
            params["angle"] = angle
        return params, meta

    def _rotate_ser(params: Dict[str, str], meta: Dict[str, Any]) -> List[str]:
        out = []
        if "cx" in params and "cy" in params:
            out.append(f"pivot=({params['cx']},{params['cy']})")
        if angle := params.get("angle"):
            out.append(f"angle={angle}")
        return out

    def _zone_remap_parse(p: Dict[str, str]) -> tuple[Dict[str, str], Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        mapping = p.get("mapping")
        if mapping is not None:
            try:
                meta["mapping"] = ast.literal_eval(mapping)
            except Exception:
                meta["mapping"] = mapping
        return {"op": "zone_remap"}, meta

    def _zone_remap_ser(params: Dict[str, str], meta: Dict[str, Any]) -> List[str]:
        mapping = meta.get("mapping") or params.get("mapping")
        if mapping is None:
            return []
        if isinstance(mapping, dict):
            items = ", ".join(f"{k}: {v}" for k, v in sorted(mapping.items()))
            return ["mapping={" + items + "}"]
        return [f"mapping={mapping}"]

    def _pattern_fill_parse(p: Dict[str, str]) -> tuple[Dict[str, str], Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        mapping = p.get("mapping")
        if mapping is not None:
            try:
                meta["mapping"] = ast.literal_eval(mapping)
            except Exception:
                meta["mapping"] = mapping
        return {"op": "pattern_fill"}, meta

    def _pattern_fill_ser(params: Dict[str, str], meta: Dict[str, Any]) -> List[str]:
        mapping = meta.get("mapping") or params.get("mapping")
        if mapping is None:
            return []
        if isinstance(mapping, dict):
            items = ", ".join(f"{k}: {v}" for k, v in sorted(mapping.items()))
            return ["mapping={" + items + "}"]
        return [f"mapping={mapping}"]

    def _morph_parse(name: str) -> Callable[[Dict[str, str]], tuple[Dict[str, str], Dict[str, Any]]]:
        def inner(p: Dict[str, str]) -> tuple[Dict[str, str], Dict[str, Any]]:
            zone = p.get("zone") or p.get("zone_id")
            d: Dict[str, str] = {"op": name}
            if zone is not None:
                d["zone"] = zone
            return d, {}
        return inner

    def _morph_ser(params: Dict[str, str], meta: Dict[str, Any]) -> List[str]:
        zone = params.get("zone") or params.get("zone_id")
        return [f"zone_id={zone}"] if zone is not None else []

    return {
        "mirror_tile": OperatorSpec(
            ttype=TransformationType.FUNCTIONAL,
            params=["axis", "repeats"],
            parser=_mirror_tile_parse,
            serializer=_mirror_tile_ser,
            description="Tile the grid while mirroring every second tile",
        ),
        "rotate_about_point": OperatorSpec(
            ttype=TransformationType.ROTATE,
            params=["pivot", "angle"],
            parser=_rotate_parse,
            serializer=_rotate_ser,
            description="Rotate the grid around a pivot point",
        ),
        "draw_line": OperatorSpec(
            ttype=TransformationType.FUNCTIONAL,
            params=["p1", "p2", "color"],
            parser=lambda p: (
                {
                    "op": "draw_line",
                    "p1": p.get("p1", "(0,0)"),
                    "p2": p.get("p2", "(1,1)"),
                    "color": p.get("color", "1"),
                },
                {},
            ),
            serializer=lambda params, meta: [
                f"p1={params.get('p1')}",
                f"p2={params.get('p2')}",
                f"color={params.get('color')}",
            ],
            description="Draw a straight line between two points",
        ),
        "zone_remap": OperatorSpec(
            ttype=TransformationType.FUNCTIONAL,
            params=["mapping"],
            parser=_zone_remap_parse,
            serializer=_zone_remap_ser,
            description="Recolour zones according to a mapping",
        ),
        "pattern_fill": OperatorSpec(
            ttype=TransformationType.FUNCTIONAL,
            params=["mapping"],
            parser=_pattern_fill_parse,
            serializer=_pattern_fill_ser,
            description="Fill target zone with pattern from source zone",
        ),
        "dilate_zone": OperatorSpec(
            ttype=TransformationType.FUNCTIONAL,
            params=["zone_id"],
            parser=_morph_parse("dilate_zone"),
            serializer=_morph_ser,
            description="Dilate pixels of the given zone",
        ),
        "erode_zone": OperatorSpec(
            ttype=TransformationType.FUNCTIONAL,
            params=["zone_id"],
            parser=_morph_parse("erode_zone"),
            serializer=_morph_ser,
            description="Erode pixels of the given zone",
        ),
    }


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
    if base_op.upper() not in {t.name for t in TransformationType}:
        registry = get_extended_operators()
        if base_op.lower() not in registry:
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

    idx = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "[" and depth == 0:
            idx = i
            break
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    if idx is None:
        raise ValueError("Invalid rule format")
    op_token = text[:idx].strip()
    remainder = text[idx:].lstrip()

    params: Dict[str, Any] = {}
    if op_token.endswith(")") and "(" in op_token:
        op_name, param_str = op_token.split("(", 1)
        op_name = op_name.strip()
        param_str = param_str[:-1]
        parts = []
        buf = ""
        depth = 0
        for ch in param_str:
            if ch == "," and depth == 0:
                parts.append(buf)
                buf = ""
                continue
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            buf += ch
        if buf:
            parts.append(buf)
        for part in parts:
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
    else:
        registry = get_extended_operators()
        if lower_op in registry:
            spec = registry[lower_op]
            ttype = spec.ttype
            parsed_params, meta = spec.parser(params)
            params = {k: v for k, v in parsed_params.items() if v is not None}
        else:
            raise ValueError(f"Unknown transformation type: {op_name}")

    if "->" not in remainder:
        raise ValueError("Rule must contain '->'")

    left_str, right_str = [part.strip() for part in remainder.split("->", 1)]
    left_syms = _parse_symbol_list(left_str)

    nature = None
    if right_str.count("[") > 1:
        idx = right_str.rfind("[")
        nature_token = right_str[idx + 1 : -1].strip()
        right_str = right_str[:idx].strip()
        try:
            nature = TransformationNature[nature_token]
        except Exception:
            nature = None

    right_syms = _parse_symbol_list(right_str)

    transformation = Transformation(ttype, params={k: str(v) for k, v in params.items()})
    return SymbolicRule(
        transformation=transformation,
        source=left_syms,
        target=right_syms,
        nature=nature,
        meta=meta,
    )


def rule_to_dsl(rule: SymbolicRule) -> str:
    """Serialise ``rule`` back into DSL form including parameters."""

    op_name = rule.transformation.ttype.value
    params = dict(rule.transformation.params)

    registry = get_extended_operators()
    functional_op = None
    if rule.transformation.ttype is TransformationType.FUNCTIONAL:
        functional_op = params.get("op")
    elif rule.transformation.ttype is TransformationType.ROTATE and {
        "cx",
        "cy",
        "angle",
    }.issubset(params):
        functional_op = "rotate_about_point"

    param_parts: list[str] = []

    if functional_op in registry:
        spec = registry[functional_op]
        op_name = functional_op
        param_parts.extend(spec.serializer(params, rule.meta))
    elif functional_op:
        op_name = functional_op
        param_parts = [f"{k}={v}" for k, v in params.items() if k != "op"]
    elif rule.transformation.ttype is TransformationType.ROTATE:
        spec = registry.get("rotate_about_point")
        if spec:
            op_name = "rotate_about_point"
            param_parts.extend(spec.serializer(params, rule.meta))
        else:
            param_parts = [f"{k}={v}" for k, v in params.items()]
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
    "get_extended_operators",
    "OperatorSpec",
    "CompositeRule",
    "final_targets",
]
