from __future__ import annotations

from typing import Any, Dict

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.operators import mirror_tile
from arc_solver.src.symbolic.draw_line import draw_line
from arc_solver.src.symbolic.rotate_about_point import rotate_about_point
from arc_solver.src.symbolic.zone_remap import zone_remap
from arc_solver.src.symbolic.pattern_fill_operator import pattern_fill
from arc_solver.src.symbolic.morphology_ops import dilate_zone, erode_zone
from arc_solver.src.segment.segmenter import label_connected_regions

from .functional_base import FunctionalOp
from .functional_utils import InvalidParameterError


class MirrorTileOp(FunctionalOp):
    def simulate(self, grid: Grid, params: Dict[str, Any]) -> Grid:
        axis = params.get("axis")
        repeats = int(params.get("repeats", 1))
        if axis is None:
            raise ValueError("axis missing")
        return mirror_tile(grid, axis, repeats)

    def validate_params(self, grid: Grid, params: Dict[str, Any]) -> None:
        axis = params.get("axis")
        if axis not in {"horizontal", "vertical"}:
            raise InvalidParameterError("axis must be 'horizontal' or 'vertical'")
        repeats = int(params.get("repeats", 1))
        if repeats <= 0:
            raise InvalidParameterError("repeats must be >0")

    def proxy_meta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class PatternFillOp(FunctionalOp):
    def simulate(self, grid: Grid, params: Dict[str, Any]) -> Grid:
        mask = params.get("mask")
        pattern = params.get("pattern")
        if mask is None or pattern is None:
            raise ValueError("mask and pattern required")
        return pattern_fill(grid, mask, pattern)

    def validate_params(self, grid: Grid, params: Dict[str, Any]) -> None:
        mask = params.get("mask")
        pattern = params.get("pattern")
        if not isinstance(mask, Grid) or not isinstance(pattern, Grid):
            raise InvalidParameterError("mask and pattern must be Grid")
        if mask.shape() != grid.shape():
            raise InvalidParameterError("mask shape mismatch")

    def proxy_meta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class DrawLineOp(FunctionalOp):
    def simulate(self, grid: Grid, params: Dict[str, Any]) -> Grid:
        p1_raw = params.get("p1")
        p2_raw = params.get("p2")
        color_raw = params.get("color")
        if p1_raw is None or p2_raw is None or color_raw is None:
            raise ValueError("p1, p2 and color required")
        p1 = tuple(int(x) for x in str(p1_raw).strip("() ").split(","))
        p2 = tuple(int(x) for x in str(p2_raw).strip("() ").split(","))
        color = int(color_raw)
        raw = draw_line(grid.to_list() if hasattr(grid, "to_list") else grid, p1, p2, color)
        return Grid(raw if isinstance(raw, list) else raw.tolist())

    def validate_params(self, grid: Grid, params: Dict[str, Any]) -> None:
        p1_raw = params.get("p1")
        p2_raw = params.get("p2")
        color_raw = params.get("color")
        try:
            p1 = tuple(int(x) for x in str(p1_raw).strip("() ").split(","))
            p2 = tuple(int(x) for x in str(p2_raw).strip("() ").split(","))
            color = int(color_raw)
        except Exception as exc:
            raise InvalidParameterError(f"invalid parameters: {exc}") from exc
        h, w = grid.shape()
        if not (0 <= p1[0] < h and 0 <= p1[1] < w and 0 <= p2[0] < h and 0 <= p2[1] < w):
            raise InvalidParameterError("points out of bounds")
        if not (0 <= color <= 9):
            raise InvalidParameterError("color out of range")

    def proxy_meta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class RotateAboutPointOp(FunctionalOp):
    def simulate(self, grid: Grid, params: Dict[str, Any]) -> Grid:
        pivot_raw = params.get("pivot")
        if pivot_raw is not None:
            cx, cy = [int(x) for x in str(pivot_raw).strip("() ").split(",")]
        else:
            cx = int(params.get("cx"))
            cy = int(params.get("cy"))
        angle = int(params.get("angle"))
        return rotate_about_point(grid, (cx, cy), angle)

    def validate_params(self, grid: Grid, params: Dict[str, Any]) -> None:
        angle = int(params.get("angle"))
        if angle not in {90, 180, 270}:
            raise InvalidParameterError("angle must be 90, 180 or 270")
        pivot_raw = params.get("pivot")
        if pivot_raw is not None:
            cx, cy = [int(x) for x in str(pivot_raw).strip("() ").split(",")]
        else:
            cx = int(params.get("cx"))
            cy = int(params.get("cy"))
        h, w = grid.shape()
        if not (0 <= cx < h and 0 <= cy < w):
            raise InvalidParameterError("pivot out of bounds")

    def proxy_meta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pivot_raw = params.get("pivot")
        if pivot_raw is not None:
            cx, cy = [int(x) for x in str(pivot_raw).strip("() ").split(",")]
        else:
            cx = int(params.get("cx"))
            cy = int(params.get("cy"))
        return {"pivot": f"{cx},{cy}"}


class ZoneRemapOp(FunctionalOp):
    def simulate(self, grid: Grid, params: Dict[str, Any]) -> Grid:
        overlay = params.get("overlay")
        mapping = params.get("mapping") or params.get("map")
        if overlay is None:
            overlay = label_connected_regions(grid)
        if mapping is None:
            raise InvalidParameterError("overlay and mapping required")
        new_grid = zone_remap(grid.to_list(), overlay, mapping)
        return Grid(new_grid if isinstance(new_grid, list) else new_grid.tolist())

    def validate_params(self, grid: Grid, params: Dict[str, Any]) -> None:
        overlay = params.get("overlay")
        mapping = params.get("mapping") or params.get("map")
        if overlay is None:
            overlay = label_connected_regions(grid)
        if not isinstance(mapping, dict):
            raise InvalidParameterError("overlay and mapping required")
        if hasattr(overlay, "shape"):
            oh, ow = overlay.shape()
        else:
            oh = len(overlay)
            ow = len(overlay[0])
        if (oh, ow) != grid.shape():
            raise InvalidParameterError("overlay shape mismatch")
        zones_present = {z for row in overlay for z in row if z is not None}
        missing = [z for z in mapping.keys() if int(z) not in zones_present]
        if missing:
            raise InvalidParameterError(f"zone id {missing[0]} not present in overlay")

    def proxy_meta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mapping = params.get("mapping") or params.get("map")
        if isinstance(mapping, dict):
            zones = [str(z) for z in mapping.keys()]
        else:
            zones = []
        return {"input_zones": zones, "output_zones": zones}


class DilateZoneOp(FunctionalOp):
    def simulate(self, grid: Grid, params: Dict[str, Any]) -> Grid:
        zone_id = int(params.get("zone"))
        overlay = label_connected_regions(grid)
        new = dilate_zone(grid.to_list(), zone_id, overlay)
        return Grid(new if isinstance(new, list) else new.tolist())

    def validate_params(self, grid: Grid, params: Dict[str, Any]) -> None:
        zone_id = int(params.get("zone"))
        overlay = label_connected_regions(grid)
        zones_present = {z for row in overlay for z in row if z is not None}
        if zone_id not in zones_present:
            raise InvalidParameterError(f"zone id {zone_id} not present")

    def proxy_meta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        zid = params.get("zone")
        return {"input_zones": [str(zid)], "output_zones": [str(zid)]}


class ErodeZoneOp(DilateZoneOp):
    def simulate(self, grid: Grid, params: Dict[str, Any]) -> Grid:
        zone_id = int(params.get("zone"))
        overlay = label_connected_regions(grid)
        new = erode_zone(grid.to_list(), zone_id, overlay)
        return Grid(new if isinstance(new, list) else new.tolist())


FUNCTIONAL_OPS: Dict[str, FunctionalOp] = {
    "mirror_tile": MirrorTileOp(),
    "pattern_fill": PatternFillOp(),
    "draw_line": DrawLineOp(),
    "rotate_about_point": RotateAboutPointOp(),
    "zone_remap": ZoneRemapOp(),
    "dilate_zone": DilateZoneOp(),
    "erode_zone": ErodeZoneOp(),
}

__all__ = ["FUNCTIONAL_OPS", "FunctionalOp"]
