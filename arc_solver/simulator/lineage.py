from __future__ import annotations

from typing import Dict, List, Tuple

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import SymbolicRule
from arc_solver.src.symbolic.rule_language import CompositeRule


class ColorLineageTracker:
    """Track the lineage of colors across rule applications."""

    def __init__(self, initial_grid: Grid | None = None) -> None:
        self.lineage: Dict[int, List[str]] = {}
        self.removed_by: Dict[int, str] = {}
        if initial_grid is not None:
            self.init_from_grid(initial_grid)

    def init_from_grid(self, grid: Grid) -> None:
        colors = {v for row in grid.data for v in row}
        for c in colors:
            self.lineage.setdefault(c, ["origin"])

    def observe_rule(
        self,
        rule: SymbolicRule | CompositeRule,
        grid_before: Grid,
        grid_after: Grid,
    ) -> None:
        """Update lineage based on ``rule`` applied to ``grid_before`` producing ``grid_after``."""
        bh, bw = grid_before.shape()
        ah, aw = grid_after.shape()
        h, w = max(bh, ah), max(bw, aw)
        changed_pairs: set[Tuple[int, int]] = set()
        for r in range(h):
            for c in range(w):
                b = grid_before.get(r, c)
                a = grid_after.get(r, c)
                if b != a:
                    changed_pairs.add((b, a))
        before_colors = {v for row in grid_before.data for v in row}
        after_colors = {v for row in grid_after.data for v in row}
        removed = before_colors - after_colors
        for col in removed:
            self.removed_by[col] = str(rule)
        added = after_colors - before_colors
        for b, a in changed_pairs:
            if a in added or a not in before_colors or b != a:
                base = self.lineage.get(b, ["origin"])
                self.lineage[a] = base + [f"{b}→{a} by {rule}"]
        for col in after_colors:
            self.lineage.setdefault(col, ["origin"])

    def get_lineage(self, color: int) -> List[str]:
        path = list(self.lineage.get(color, []))
        if color in self.removed_by:
            path.append(f"removed by {self.removed_by[color]}")
        return path

    def render_lineage_summary(self) -> str:
        lines = []
        for col in sorted(self.lineage.keys()):
            trace = "; ".join(self.get_lineage(col))
            lines.append(f"{col}: {trace}")
        return "\n".join(lines)

    def to_dot_graph(self) -> str:
        """Return a DOT format string visualizing color transitions."""
        edges = []
        for col, trace in self.lineage.items():
            for step in trace:
                if "→" in step:
                    pair, info = step.split(" by ", 1)
                    src, tgt = pair.split("\u2192")
                    edges.append(f"    {src} -> {tgt} [label=\"{info}\"]")
        body = "\n".join(edges)
        return "digraph{\n" + body + "\n}"

