import json
from pathlib import Path
from typing import Any, Dict, List

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.rule_language import (
    get_extended_operators,
    parse_rule,
    rule_to_dsl,
)
from arc_solver.src.executor.simulator import _apply_functional

DOC_PATH = Path("docs/generated_dsl_operators.md")
LOG_PATH = Path("logs/dsl_validation_report.json")

_SAMPLE_VALUES = {
    "axis": "horizontal",
    "repeats": "2",
    "pivot": "(0,0)",
    "angle": "90",
    "cx": "0",
    "cy": "0",
    "mapping": "{1: 2}",
    "zone": "1",
    "zone_id": "1",
}

_DEF_META = {
    "pattern_fill": {
        "mask": Grid([[1, 0], [0, 1]]),
        "pattern": Grid([[1, 1], [1, 1]]),
    }
}


def _param_in_rule(param: str, rule: Any) -> bool:
    if param in rule.transformation.params:
        return True
    if param == "pivot":
        return {"cx", "cy"}.issubset(rule.transformation.params)
    if param == "zone_id":
        return "zone" in rule.transformation.params
    if param in rule.meta:
        return True
    return False


def _build_dsl(op: str, params: Dict[str, str]) -> str:
    if params:
        inner = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{op}({inner}) [COLOR=1] -> [COLOR=1]"
    return f"{op} [COLOR=1] -> [COLOR=1]"


def validate_dsl_integrity() -> Dict[str, Any]:
    registry = get_extended_operators()
    doc_text = DOC_PATH.read_text() if DOC_PATH.exists() else ""
    results: Dict[str, Any] = {}

    for name, spec in registry.items():
        res = {
            "parse_ok": True,
            "roundtrip_ok": True,
            "simulate_ok": True,
            "documented": f"## {name}" in doc_text,
            "param_mismatch": False,
            "errors": [],
        }
        samples = []
        if spec.params:
            min_params = {spec.params[0]: _SAMPLE_VALUES.get(spec.params[0], "1")}
            max_params = {p: _SAMPLE_VALUES.get(p, "1") for p in spec.params}
            samples = [min_params, max_params]
        else:
            samples = [{}]
        for idx, prm in enumerate(samples):
            dsl = _build_dsl(name, prm)
            try:
                rule = parse_rule(dsl)
            except Exception as exc:
                res["parse_ok"] = False
                res["errors"].append(f"parse {prm}: {exc}")
                continue
            if idx == len(samples) - 1:
                if not all(_param_in_rule(p, rule) for p in spec.params):
                    res["param_mismatch"] = True
            try:
                roundtrip = rule_to_dsl(rule)
                parse_rule(roundtrip)
            except Exception as exc:
                res["roundtrip_ok"] = False
                res["errors"].append(f"roundtrip {prm}: {exc}")
            try:
                meta = {**_DEF_META.get(name, {})}
                rule.meta.update(meta)
                _apply_functional(Grid([[1, 0], [0, 1]]), rule)
            except Exception as exc:
                res["simulate_ok"] = False
                res["errors"].append(f"simulate {prm}: {exc}")
        results[name] = res

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(results, indent=2))
    summary = []
    for op, r in results.items():
        ok = all(
            r[key]
            for key in ["parse_ok", "roundtrip_ok", "simulate_ok", "documented"]
        ) and not r["param_mismatch"]
        summary.append(f"{op}: {'OK' if ok else 'FAIL'}")
    print("\n".join(summary))
    return results


def main() -> None:
    validate_dsl_integrity()


if __name__ == "__main__":
    main()
