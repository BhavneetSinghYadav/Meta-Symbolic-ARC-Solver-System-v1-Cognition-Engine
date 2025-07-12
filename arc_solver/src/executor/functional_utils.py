from __future__ import annotations

"""Validation helpers for functional symbolic operators."""

from typing import Any, Dict

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import SymbolicRule, TransformationType
from arc_solver.src.segment.segmenter import label_connected_regions

__all__ = [
    "InvalidParameterError",
    "UnsafeTransformationError",
    "validate_functional_params",
]


class InvalidParameterError(ValueError):
    """Raised when a functional operator has malformed parameters."""


class UnsafeTransformationError(RuntimeError):
    """Raised when a functional operator would create an unsafe state."""


def _parse_int(val: Any) -> int:
    if isinstance(val, int):
        return val
    return int(str(val))


def validate_functional_params(rule: SymbolicRule, grid: Grid) -> None:
    """Validate parameters for functional ``rule`` relative to ``grid``."""

    if rule.transformation.ttype is not TransformationType.FUNCTIONAL:
        # rotate_about_point is encoded as ROTATE but treated like a functional op
        if rule.transformation.ttype is TransformationType.ROTATE and {
            "cx",
            "cy",
            "angle",
        }.issubset(rule.transformation.params):
            op = "rotate_about_point"
        else:
            return
    else:
        op = rule.transformation.params.get("op")

    from arc_solver.src.executor.functional_ops import FUNCTIONAL_OPS

    params = {**rule.transformation.params, **(getattr(rule, "meta", {}) or {})}
    wrapper = FUNCTIONAL_OPS.get(op)
    if wrapper is not None:
        wrapper.validate_params(grid, params)
        return

    meta: Dict[str, Any] = getattr(rule, "meta", {})

    if op == "rotate_about_point":
        pivot_raw = params.get("pivot")
        if pivot_raw is not None:
            try:
                cx, cy = [_parse_int(x) for x in str(pivot_raw).strip("() ").split(",")]
            except Exception as exc:
                raise InvalidParameterError(f"invalid pivot: {exc}") from exc
        else:
            try:
                cx = _parse_int(params.get("cx"))
                cy = _parse_int(params.get("cy"))
            except Exception as exc:
                raise InvalidParameterError(f"invalid pivot: {exc}") from exc
        try:
            angle = _parse_int(params.get("angle"))
        except Exception as exc:
            raise InvalidParameterError(f"invalid angle: {exc}") from exc
        if angle not in {90, 180, 270}:
            raise InvalidParameterError("angle must be 90, 180 or 270")
        h, w = grid.shape()
        if not (0 <= cx < h and 0 <= cy < w):
            raise InvalidParameterError("pivot out of bounds")

    elif op == "zone_remap":
        mapping = meta.get("mapping") or params.get("mapping") or params.get("map")
        if not isinstance(mapping, dict) or not mapping:
            raise InvalidParameterError(
                "zone_remap requires a 'mapping' dict with zone_id->color keys"
            )
        overlay = label_connected_regions(grid)
        zones_present = {z for row in overlay for z in row if z is not None}
        for zid, col in mapping.items():
            try:
                zid_int = _parse_int(zid)
            except Exception as exc:
                raise InvalidParameterError(f"invalid zone id {zid}: {exc}") from exc
            if zid_int not in zones_present:
                raise InvalidParameterError(f"zone id {zid_int} not present in overlay")
            try:
                col_int = _parse_int(col)
            except Exception as exc:
                raise InvalidParameterError(f"invalid colour {col}: {exc}") from exc
            if not (0 <= col_int <= 9):
                raise InvalidParameterError(f"colour values must be 0-9: {col_int}")

    elif op in {"dilate_zone", "erode_zone"}:
        zone_val = params.get("zone") or params.get("zone_id")
        if zone_val is None:
            raise InvalidParameterError("zone id missing")
        try:
            zid = _parse_int(zone_val)
        except Exception as exc:
            raise InvalidParameterError(f"invalid zone id: {exc}") from exc
        overlay = label_connected_regions(grid)
        zones_present = {z for row in overlay for z in row if z is not None}
        if zid not in zones_present:
            raise InvalidParameterError(f"zone id {zid} not present in overlay")

    elif op == "pattern_fill":
        mask = meta.get("mask")
        pattern = meta.get("pattern")
        if mask is None or pattern is None:
            raise InvalidParameterError(
                "pattern_fill requires 'mask' and 'pattern' Grid objects"
            )
        if not isinstance(mask, Grid) or not isinstance(pattern, Grid):
            raise InvalidParameterError("mask and pattern must be Grid objects")
        if mask.shape() != grid.shape():
            raise InvalidParameterError("mask shape must match grid")
        ph, pw = pattern.shape()
        for r in range(ph):
            for c in range(pw):
                val = pattern.get(r, c)
                if not (0 <= int(val) <= 9):
                    raise InvalidParameterError("pattern contains invalid colours")

    # Unknown functional operators fall back to no validation
