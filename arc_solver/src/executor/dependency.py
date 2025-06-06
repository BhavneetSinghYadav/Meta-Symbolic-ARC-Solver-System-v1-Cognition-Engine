from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set

from arc_solver.src.symbolic.vocabulary import (
    SymbolType,
    SymbolicRule,
    TransformationType,
)


class RuleDependencyGraph:
    """Simple undirected conflict graph between rules."""

    def __init__(self, rules: List[SymbolicRule]) -> None:
        self.edges: Dict[int, Set[int]] = defaultdict(set)
        self.build(rules)

    def build(self, rules: List[SymbolicRule]) -> None:
        for i, r1 in enumerate(rules):
            for j, r2 in enumerate(rules):
                if i >= j:
                    continue
                if has_conflict(r1, r2):
                    self.edges[i].add(j)
                    self.edges[j].add(i)

def rule_dependency_graph(rules: List[SymbolicRule]) -> Dict[int, Set[int]]:
    """Return a directed dependency graph between rules."""
    graph: Dict[int, Set[int]] = defaultdict(set)
    for i, r1 in enumerate(rules):
        targets = {s.value for s in r1.target if s.type is SymbolType.COLOR}
        for j, r2 in enumerate(rules):
            if i == j:
                continue
            sources = {s.value for s in r2.source if s.type is SymbolType.COLOR}
            if targets & sources:
                graph[i].add(j)
    return graph


def sort_rules_by_dependency(rules: List[SymbolicRule]) -> List[SymbolicRule]:
    """Return ``rules`` sorted by dependency order."""
    graph = rule_dependency_graph(rules)
    indegree: Dict[int, int] = {i: 0 for i in range(len(rules))}
    for deps in graph.values():
        for j in deps:
            indegree[j] += 1
    queue = deque([i for i, d in indegree.items() if d == 0])
    order: List[int] = []
    while queue:
        i = queue.popleft()
        order.append(i)
        for j in graph.get(i, set()):
            indegree[j] -= 1
            if indegree[j] == 0:
                queue.append(j)
    order.extend(i for i in range(len(rules)) if i not in order)
    return [rules[i] for i in order]


def _extract_color(rule: SymbolicRule) -> str | None:
    for sym in rule.target:
        if sym.type is SymbolType.COLOR:
            return sym.value
    return None


def has_conflict(r1: SymbolicRule, r2: SymbolicRule) -> bool:
    zone1 = r1.condition.get("zone") if r1.condition else None
    zone2 = r2.condition.get("zone") if r2.condition else None
    if zone1 and zone2 and zone1 != zone2:
        return False

    if (
        r1.transformation.ttype is TransformationType.REPLACE
        and r2.transformation.ttype is TransformationType.REPLACE
    ):
        src1 = [s.value for s in r1.source if s.type is SymbolType.COLOR]
        src2 = [s.value for s in r2.source if s.type is SymbolType.COLOR]
        tgt1 = _extract_color(r1)
        tgt2 = _extract_color(r2)
        if src1 and src2 and src1 == src2 and tgt1 != tgt2:
            return True
    return False


def select_independent_rules(rules: List[SymbolicRule]) -> List[SymbolicRule]:
    graph = RuleDependencyGraph(rules)
    chosen: List[SymbolicRule] = []
    for idx, rule in enumerate(rules):
        if any(j in graph.edges.get(idx, set()) for j in range(len(chosen))):
            continue
        chosen.append(rule)
    return chosen


__all__ = [
    "RuleDependencyGraph",
    "has_conflict",
    "select_independent_rules",
    "rule_dependency_graph",
    "sort_rules_by_dependency",
]
