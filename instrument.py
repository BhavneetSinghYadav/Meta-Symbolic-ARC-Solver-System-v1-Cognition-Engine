import argparse
from arc_solver.src.data.agi_loader import load_agi_tasks
from arc_solver.src.abstractions.abstractor import (
    abstract,
    extract_color_change_rules,
    extract_shape_based_rules,
)
from arc_solver.src.abstractions.rule_generator import remove_duplicate_rules
from arc_solver.src.symbolic.repeat_rule import generate_repeat_rules
from arc_solver.src.executor.simulator import simulate_rules


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

    repeat_rules = generate_repeat_rules(mid, out)
    print(f"repeat {len(repeat_rules)}")
    for r in repeat_rules:
        if r.meta.get("replace_map"):
            print(f"composite mapping: {r.meta['replace_map']}")

    rules = cc_rules + shape_rules + repeat_rules
    wf_rules = [r for r in rules if r.is_well_formed()]
    print(f"is_well_formed {len(wf_rules)}")

    dedup_rules = remove_duplicate_rules(wf_rules)
    print(f"dedup {len(dedup_rules)}")

    final_rules = abstract([inp, out])
    print(f"abstract total {len(final_rules)}")


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    run(args.bundle, args.task_id)


if __name__ == "__main__":
    _cli()
