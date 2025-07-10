import inspect
import pkgutil
import importlib
from pathlib import Path
import sys

# Ensure project root is on the module search path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_solver.src.symbolic import vocabulary


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

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))


if __name__ == "__main__":
    generate_dsl_docs()
