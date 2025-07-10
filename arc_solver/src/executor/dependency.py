from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Union

from arc_solver.src.symbolic.vocabulary import (
    SymbolType,
    SymbolicRule,
    TransformationType,
)
from arc_solver.src.symbolic.rule_language import CompositeRule, final_targets


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

def rule_dependency_graph(rules: List[Union[SymbolicRule, CompositeRule]]) -> Dict[int, Set[int]]:
    """Return a directed dependency graph between rules."""

    graph: Dict[int, Set[int]] = {}
    for i, r1 in enumerate(rules):
        s1 = final_targets(r1) if isinstance(r1, CompositeRule) else r1.target
        t1_colors = {s.value for s in s1 if s.type is SymbolType.COLOR}
        deps: Set[int] = set()
        for j, r2 in enumerate(rules):
            if i == j:
                continue
            s2 = r2.get_sources() if isinstance(r2, CompositeRule) else r2.source
            s2_colors = {s.value for s in s2 if s.type is SymbolType.COLOR}
            if t1_colors & s2_colors:
                deps.add(j)
        graph[i] = deps
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
    """Return ``rules`` ordered respecting zone transition dependencies."""

    chains: List[List[Tuple[List[str], List[str]]]] = []
    for rule in rules:
        if isinstance(rule, CompositeRule):
            proxy = rule.as_symbolic_proxy()
            chain = proxy.meta.get("zone_scope_chain")
            if not chain:
                chain = [
                    ([z1] if z1 else [], [z2] if z2 else [])
                    for z1, z2 in proxy.meta.get("zone_chain", [])
                ]
            chains.append(chain)
        else:
            zone = rule.condition.get("zone") if rule.condition else None
            chains.append([([zone] if zone else [], [zone] if zone else [])])

    graph = rule_dependency_graph(rules)

    def _union(seq):
        s: Set[str] = set()
        for item in seq:
            s.update(item)
        return s

    for i, c1 in enumerate(chains):
        if not c1:
            continue
        out_union = _union([set(out) for _, out in c1])
        if not out_union:
            continue
        for j, c2 in enumerate(chains):
            if i == j or not c2:
                continue
            in_first = set(c2[0][0])
            if in_first and out_union.intersection(in_first):
                graph.setdefault(i, set()).add(j)

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


def select_independent_rules(rules: List[Union[SymbolicRule, CompositeRule]]) -> List[Union[SymbolicRule, CompositeRule]]:
    """Filter and return rule list with minimal dependency conflicts."""
    graph = rule_dependency_graph(rules)
    selected: List[Union[SymbolicRule, CompositeRule]] = []
    visited: Set[int] = set()

    def visit(idx: int) -> None:
        if idx in visited:
            return
        visited.add(idx)
        for dep in graph.get(idx, set()):
            visit(dep)
        selected.append(rules[idx])

    for idx in range(len(rules)):
        visit(idx)
    return selected[::-1]


__all__ = [
    "RuleDependencyGraph",
    "has_conflict",
    "select_independent_rules",
    "rule_dependency_graph",
    "sort_rules_by_dependency",
    "sort_rules_by_topology",
]
