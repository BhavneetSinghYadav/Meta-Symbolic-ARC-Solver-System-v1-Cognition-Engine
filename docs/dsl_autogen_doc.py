import importlib
import inspect
import pkgutil
from pathlib import Path
import sys

# Ensure project root is on the module search path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

vocabulary = importlib.import_module("arc_solver.src.symbolic.vocabulary")
rule_language = importlib.import_module("arc_solver.src.symbolic.rule_language")
get_extended_operators = rule_language.get_extended_operators
OperatorSpec = rule_language.OperatorSpec
vocab_get_ops = vocabulary.get_extended_operators


def _discover_symbolic_modules():
    pkg = importlib.import_module("arc_solver.src.symbolic")
    modules = {}
    for _, name, _ in pkgutil.iter_modules(pkg.__path__):
        try:
            modules[name] = importlib.import_module(f"{pkg.__name__}.{name}")
        except Exception:
            continue
    return modules


def _find_operator_info(name: str, modules: dict):
    op_lower = name.lower()
    for module in modules.values():
        for attr_name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ != module.__name__:
                continue
            if op_lower in attr_name:
                doc = inspect.getdoc(obj)
                sig = str(inspect.signature(obj))
                return obj.__name__ + sig, (doc.splitlines()[0] if doc else None)
    return None, None


def generate_dsl_docs(output_path: str = "docs/generated_dsl_operators.md") -> None:
    """Generate Markdown documentation for available DSL operators."""

    modules = _discover_symbolic_modules()
    lines = ["# Symbolic DSL Operators", ""]

    for op in vocabulary.TransformationType:
        syntax, desc = _find_operator_info(op.name, modules)
        if syntax is None:
            syntax = f"{op.name}(...)"
        if not desc:
            desc = "TODO: add description."
        lines.append(f"## {op.name}\n")
        lines.append(f"**Syntax:** `{syntax}`\n")
        lines.append(f"**Description:** {desc}\n")
        lines.append("**Example:**\n")
        lines.append("```python\n# Example coming soon\n```\n")

    ext = get_extended_operators()
    vocab_ops = vocab_get_ops()
    sample_args = {
        "mirror_tile": ("horizontal", 2),
        "pattern_fill": (None, None),
        "draw_line": ((0, 0), (1, 1), 1),
        "dilate_zone": (1,),
        "erode_zone": (1,),
        "rotate_about_point": ((1, 1), 90),
        "zone_remap": (None, {1: 2}),
    }
    for name, spec in sorted(ext.items()):
        syntax, desc = _find_operator_info(name, modules)
        if syntax is None:
            syntax = f"{name}(...)"
        if not desc:
            desc = spec.description or "TODO: add description."
        lines.append(f"## {name}\n")
        lines.append(f"**DSL Keyword:** `{name}`\n")
        lines.append(f"**Transformation Type:** `{spec.ttype.value}`\n")
        lines.append(f"**Parameters:** {', '.join(spec.params)}\n")
        lines.append("**DSL Version:** v1\n")
        lines.append(f"**Description:** {desc}\n")
        example = "# Example coming soon"
        impl_path = None
        rule_fn = vocab_ops.get(name)
        if rule_fn:
            args = sample_args.get(name, ())
            try:
                rule = rule_fn(*args)
                example = rule_language.rule_to_dsl(rule)
                impl_path = Path(inspect.getsourcefile(rule_fn)).relative_to(ROOT)
            except Exception:
                pass
        lines.append("**Example:**\n")
        lines.append(f"```python\n{example}\n```\n")
        if impl_path:
            lines.append(f"**Implementation:** `{impl_path}`\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))


if __name__ == "__main__":
    generate_dsl_docs()
