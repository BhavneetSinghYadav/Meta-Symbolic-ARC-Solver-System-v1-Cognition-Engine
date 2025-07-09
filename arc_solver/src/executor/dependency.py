from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set

from arc_solver.src.symbolic.vocabulary import (
    SymbolType,
    SymbolicRule,
    TransformationType,
)
from arc_solver.src.symbolic.rule_language import CompositeRule


class RuleDependencyGraph:
    """Simple undirected conflict graph between rules."""

    def __init__(self, rules: List[SymbolicRule | CompositeRule]) -> None:
        self.edges: Dict[int, Set[int]] = defaultdict(set)
        self.build(rules)

    def build(self, rules: List[SymbolicRule | CompositeRule]) -> None:
        for i, r1 in enumerate(rules):
            for j, r2 in enumerate(rules):
                if i >= j:
                    continue
                if has_conflict(r1, r2):
                    self.edges[i].add(j)
                    self.edges[j].add(i)

def rule_dependency_graph(rules: List[SymbolicRule | CompositeRule]) -> Dict[int, Set[int]]:
    """Return a directed dependency graph between rules."""

    def _targets(rule: SymbolicRule | CompositeRule) -> Set[str]:
        if isinstance(rule, CompositeRule):
            t: Set[str] = set()
            for step in rule.steps:
                t.update(s.value for s in step.target if s.type is SymbolType.COLOR)
            return t
        return {s.value for s in rule.target if s.type is SymbolType.COLOR}

    def _sources(rule: SymbolicRule | CompositeRule) -> Set[str]:
        if isinstance(rule, CompositeRule):
            s: Set[str] = set()
            for step in rule.steps:
                s.update(sy.value for sy in step.source if sy.type is SymbolType.COLOR)
            return s
        return {s.value for s in rule.source if s.type is SymbolType.COLOR}

    graph: Dict[int, Set[int]] = defaultdict(set)
    for i, r1 in enumerate(rules):
        targets = _targets(r1)
        for j, r2 in enumerate(rules):
            if i == j:
                continue
            sources = _sources(r2)
            if targets & sources:
                graph[i].add(j)
    return graph


def sort_rules_by_dependency(rules: List[SymbolicRule | CompositeRule]) -> List[SymbolicRule | CompositeRule]:
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


def sort_rules_by_topology(rules: List[SymbolicRule | CompositeRule]) -> List[SymbolicRule | CompositeRule]:
    """Return ``rules`` ordered respecting zone and color dependencies."""

    zone_groups: Dict[str | None, List[SymbolicRule | CompositeRule]] = defaultdict(list)

    for rule in rules:
        zone: str | None = None
        if isinstance(rule, CompositeRule):
            zones = {
                step.condition.get("zone")
                for step in rule.steps
                if step.condition and "zone" in step.condition
            }
            if len(zones) == 1:
                zone = next(iter(zones))
        else:
            zone = rule.condition.get("zone") if rule.condition else None
        zone_groups[zone].append(rule)

    ordered: List[SymbolicRule | CompositeRule] = []
    for key in sorted(zone_groups.keys(), key=lambda z: "" if z is None else str(z)):
        ordered.extend(sort_rules_by_dependency(zone_groups[key]))
    return ordered


def _extract_color(rule: SymbolicRule) -> str | None:
    for sym in rule.target:
        if sym.type is SymbolType.COLOR:
            return sym.value
    return None


def _has_conflict_simple(r1: SymbolicRule, r2: SymbolicRule) -> bool:
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


def has_conflict(r1: SymbolicRule | CompositeRule, r2: SymbolicRule | CompositeRule) -> bool:
    if isinstance(r1, CompositeRule) or isinstance(r2, CompositeRule):
        steps1 = r1.steps if isinstance(r1, CompositeRule) else [r1]
        steps2 = r2.steps if isinstance(r2, CompositeRule) else [r2]
        for s1 in steps1:
            for s2 in steps2:
                if _has_conflict_simple(s1, s2):
                    return True
        return False
    return _has_conflict_simple(r1, r2)


def select_independent_rules(rules: List[SymbolicRule | CompositeRule]) -> List[SymbolicRule | CompositeRule]:
    graph = RuleDependencyGraph(rules)
    chosen: List[SymbolicRule | CompositeRule] = []
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
    "sort_rules_by_topology",
]
