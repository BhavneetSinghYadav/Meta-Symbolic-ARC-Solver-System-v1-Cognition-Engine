import argparse
from arc_solver.src.data.agi_loader import load_agi_tasks
from arc_solver.src.abstractions.abstractor import (
    abstract,
    extract_color_change_rules,
    extract_shape_based_rules,
)
from arc_solver.src.abstractions.rule_generator import remove_duplicate_rules
from arc_solver.src.symbolic.repeat_rule import generate_repeat_rules
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.executor.simulator import simulate_rules, simulate_composite_rule
from arc_solver.src.executor.scoring import score_rule, preferred_rule_types


def log_calls(label):
    """Return decorator logging basic call telemetry."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            shapes = []
            for a in args:
                if hasattr(a, "shape"):
                    try:
                        shapes.append(a.shape())
                    except Exception:
                        shapes.append("?")
            result = func(*args, **kwargs)
            ttype = None
            for a in args:
                if hasattr(a, "transformation"):
                    ttype = getattr(a.transformation, "ttype", None)
                    break
            length = len(result) if isinstance(result, list) else 1
            print(f"[{label}] \u2190 {shapes} / {ttype} / {length}")
            return result

        return wrapper

    return decorator

# Instrument repeat related helpers
generate_repeat_rules = log_calls("repeat_gen")(generate_repeat_rules)
import arc_solver.src.executor.simulator as simulator
simulator._apply_repeat = log_calls("apply_repeat")(simulator._apply_repeat)


def run(bundle: str, task_id: str) -> None:
    """Run repeat rule extraction on a single training pair."""
    tasks = load_agi_tasks(bundle)
    task = next((t for t in tasks if t.task_id == task_id), None)
    if not task:
        raise SystemExit(f"task {task_id} not found")
    inp, out = task.train[0]

    cc_rules = extract_color_change_rules(inp, out)
    print(f"colour-change {len(cc_rules)}")
    mid = simulate_rules(inp, cc_rules) if cc_rules else inp

    shape_rules = extract_shape_based_rules(mid, out)
    print(f"shape {len(shape_rules)}")

    repeat_rules = generate_repeat_rules(mid, out, post_process=True)
    print(f"repeat {len(repeat_rules)}")
    composite_rules = [r for r in repeat_rules if isinstance(r, CompositeRule)]
    print(f"composite {len(composite_rules)}")
    best_score = -1.0
    best_rule = None
    for comp in composite_rules:
        pred = simulate_composite_rule(mid, comp)
        score = pred.compare_to(out)
        if score > best_score:
            best_score = score
            best_rule = comp
    if best_rule is not None:
        diff_before = mid.compare_to(out)
        diff_after = simulate_composite_rule(mid, best_rule).compare_to(out)
        print(
            f"best composite {best_rule.to_string()} diff_before={diff_before:.2f} diff_after={diff_after:.2f}"
        )
    
    rules = cc_rules + shape_rules + repeat_rules
    wf_rules = [r for r in rules if r.is_well_formed()]
    print(f"is_well_formed {len(wf_rules)}")

    dedup_rules = remove_duplicate_rules(wf_rules)
    print(f"dedup {len(dedup_rules)}")

    final_rules = abstract([inp, out])
    print(f"abstract total {len(final_rules)}")

    if final_rules:
        scores = []
        for r in final_rules:
            s = score_rule(inp, out, r, prefer_composites=True)
            scores.append(s)
            try:
                if isinstance(r, CompositeRule):
                    pred = simulate_composite_rule(inp, r)
                else:
                    pred = simulate_rules(inp, [r])
                raw = pred.compare_to(out)
                before = inp.compare_to(out)
                desc = r.to_string() if isinstance(r, CompositeRule) else str(r)
                print(
                    f"rule {desc} raw={raw:.3f} adj={s:.3f} gain={raw - before:.3f}"
                )
            except Exception:
                desc = r.to_string() if isinstance(r, CompositeRule) else str(r)
                print(f"rule {desc} failed to simulate")
        print(f"top_score {max(scores):.3f}")
        hist = {}
        for r in final_rules:
            t = r.transformation.ttype.value
            hist[t] = hist.get(t, 0) + 1
        print(f"type_hist {hist}")
        delta = preferred_rule_types(inp, out)
        perf = {}
        for t in hist:
            best = max(
                (score_rule(inp, out, r) for r in final_rules if r.transformation.ttype.value == t),
                default=0.0,
            )
            perf[t] = round(best, 3)
        print(f"perf_{delta} {perf}")


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="run")
    parser.add_argument("--task_id")
    parser.add_argument("--bundle")
    args = parser.parse_args()

    if args.command == "validate-dsl-integrity":
        from arc_solver.tools.validate_dsl_integrity import validate_dsl_integrity

        validate_dsl_integrity()
        return

    if not args.task_id or not args.bundle:
        parser.error("--task_id and --bundle required for run")

    run(args.bundle, args.task_id)


if __name__ == "__main__":
    _cli()
