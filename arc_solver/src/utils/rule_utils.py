from __future__ import annotations

from typing import List, Dict
import json
from pathlib import Path

from arc_solver.src.symbolic.vocabulary import SymbolicRule
from arc_solver.src.symbolic.rule_language import rule_to_dsl


def generalize_rules(rules: List[SymbolicRule]) -> List[SymbolicRule]:
    """Return list of rules merged by identical DSL."""

    groups: Dict[str, List[SymbolicRule]] = {}
    for r in rules:
        dsl = getattr(r, "dsl_str", None) or rule_to_dsl(r)
        r.dsl_str = dsl
        groups.setdefault(dsl, []).append(r)

    generalized: List[SymbolicRule] = []
    trace = {"original_count": len(rules), "groups": []}

    for dsl, bucket in groups.items():
        merged = bucket[0]
        for other in bucket[1:]:
            g = merged.generalize_with(other)
            if g:
                merged = g
            else:
                generalized.append(merged)
                merged = other
        generalized.append(merged)
        trace["groups"].append({"dsl": dsl, "count": len(bucket)})

    trace["final_count"] = len(generalized)
    try:
        path = Path("logs/generalization_trace.jsonl")
        path.parent.mkdir(exist_ok=True)
        path.open("a", encoding="utf-8").write(json.dumps(trace) + "\n")
    except Exception:
        pass

    return generalized


__all__ = ["generalize_rules"]
