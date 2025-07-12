"""Fallback rule discovery using zone overlays and shape metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from arc_solver.src.core.grid import Grid
from arc_solver.src.segment.segmenter import (
    zone_overlay,
    segment_morphological_regions,
    label_connected_regions,
)
from arc_solver.src.utils.patterns import detect_mirrored_regions
from arc_solver.src.symbolic.vocabulary import (
    SymbolicRule,
    Transformation,
    TransformationNature,
    TransformationType,
    Symbol,
    SymbolType,
)
from arc_solver.src.symbolic.morphology_ops import dilate_zone, erode_zone
from arc_solver.src.symbolic.rule_language import rule_to_dsl


def _skeleton_segments(grid: Grid) -> Dict[str, List[tuple[int, int]]]:
    seg = segment_morphological_regions(grid)
    zones: Dict[str, List[tuple[int, int]]] = {}
    for (r, c), sym in seg.items():
        zones.setdefault(str(sym.value), []).append((r, c))
    return zones


def auto_discover_rules(
    task_id: str,
    input_grid: Grid,
    output_grid: Grid,
    overlays: Dict[str, List[List[int]]] | None = None,
) -> List[SymbolicRule]:
    """Return unscored rule suggestions based on zone and shape cues."""

    overlays = overlays or {}
    in_overlay = overlays.get("input") or zone_overlay(input_grid, use_morphology=True)
    out_overlay = overlays.get("output") or zone_overlay(output_grid, use_morphology=True)

    rules: List[SymbolicRule] = []

    # Skeleton to line -----------------------------------------------------
    sk_in = _skeleton_segments(input_grid)
    sk_out = _skeleton_segments(output_grid)
    for label, cells in sk_out.items():
        if label in sk_in or len(cells) < 2:
            continue
        p1, p2 = cells[0], cells[-1]
        color = output_grid.get(p1[0], p1[1])
        rule = SymbolicRule(
            transformation=Transformation(
                TransformationType.FUNCTIONAL,
                params={"op": "draw_line", "p1": str(p1), "p2": str(p2), "color": str(color)},
            ),
            source=[Symbol(SymbolType.REGION, "All")],
            target=[Symbol(SymbolType.REGION, "All")],
            nature=TransformationNature.SPATIAL,
            meta={"trigger": "skeleton"},
        )
        rule.dsl_str = rule_to_dsl(rule)
        rules.append(rule)

    # Symmetry and tiling --------------------------------------------------
    for axis in detect_mirrored_regions(input_grid, output_grid):
        rule = SymbolicRule(
            transformation=Transformation(
                TransformationType.FUNCTIONAL,
                params={"op": "mirror_tile", "axis": axis, "repeats": "2"},
            ),
            source=[Symbol(SymbolType.REGION, "All")],
            target=[Symbol(SymbolType.REGION, "All")],
            nature=TransformationNature.SPATIAL,
            meta={"trigger": "symmetry"},
        )
        rule.dsl_str = rule_to_dsl(rule)
        rules.append(rule)

    # Recolouring hints ----------------------------------------------------
    cnt_in = input_grid.count_colors()
    cnt_out = output_grid.count_colors()
    if len(cnt_out) == 1 and cnt_in:
        tgt = next(iter(cnt_out))
        rule = SymbolicRule(
            transformation=Transformation(
                TransformationType.REPLACE,
            ),
            source=[Symbol(SymbolType.COLOR, "*")],
            target=[Symbol(SymbolType.COLOR, str(tgt))],
            nature=TransformationNature.LOGICAL,
            meta={"trigger": "uniform_fill"},
        )
        rule.dsl_str = rule_to_dsl(rule)
        rules.append(rule)

    # Zone morph proposals -------------------------------------------------
    regions = label_connected_regions(input_grid)
    zone_ids = {z for row in regions for z in row if z is not None}
    for zid in zone_ids:
        try:
            pred = dilate_zone(input_grid.to_list(), zid, regions)
        except Exception:
            pred = None
        if pred is not None and Grid(pred if isinstance(pred, list) else pred.tolist()) == output_grid:
            rule = SymbolicRule(
                transformation=Transformation(
                    TransformationType.FUNCTIONAL,
                    params={"op": "dilate_zone", "zone": str(zid)},
                ),
                source=[Symbol(SymbolType.REGION, "All")],
                target=[Symbol(SymbolType.REGION, "All")],
                nature=TransformationNature.SPATIAL,
                meta={"trigger": "dilate"},
            )
            rule.dsl_str = rule_to_dsl(rule)
            rules.append(rule)
        try:
            pred = erode_zone(input_grid.to_list(), zid, regions)
        except Exception:
            pred = None
        if pred is not None and Grid(pred if isinstance(pred, list) else pred.tolist()) == output_grid:
            rule = SymbolicRule(
                transformation=Transformation(
                    TransformationType.FUNCTIONAL,
                    params={"op": "erode_zone", "zone": str(zid)},
                ),
                source=[Symbol(SymbolType.REGION, "All")],
                target=[Symbol(SymbolType.REGION, "All")],
                nature=TransformationNature.SPATIAL,
                meta={"trigger": "erode"},
            )
            rule.dsl_str = rule_to_dsl(rule)
            rules.append(rule)

    # Logging --------------------------------------------------------------
    trace = {
        "task_id": task_id,
        "candidate_count": len(rules),
        "rules": [rule.dsl for rule in rules],
    }
    try:
        path = Path("logs/auto_discover_trace.jsonl")
        path.parent.mkdir(exist_ok=True)
        path.open("a", encoding="utf-8").write(json.dumps(trace) + "\n")
    except Exception:
        pass

    return rules


def _cli() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Auto discover rules for a task")
    parser.add_argument("task_id")
    parser.add_argument("--visualize", action="store_true", help="show overlays")
    args = parser.parse_args()

    tasks = json.loads(Path("arc-agi_training_challenges.json").read_text())
    solutions = json.loads(Path("arc-agi_training_solutions.json").read_text())

    t = tasks.get(args.task_id)
    if not t:
        raise KeyError(f"task {args.task_id} not found")
    inp = Grid(t["train"][0]["input"])
    tgt = Grid(solutions[args.task_id][0])

    rules = auto_discover_rules(args.task_id, inp, tgt, {})
    for r in rules:
        print(rule_to_dsl(r))

    if args.visualize:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(inp.data, cmap="tab20", interpolation="none")
        axes[0].axis("off")
        axes[0].set_title("input")
        axes[1].imshow(tgt.data, cmap="tab20", interpolation="none")
        axes[1].axis("off")
        axes[1].set_title("output")
        plt.show()


if __name__ == "__main__":
    _cli()
