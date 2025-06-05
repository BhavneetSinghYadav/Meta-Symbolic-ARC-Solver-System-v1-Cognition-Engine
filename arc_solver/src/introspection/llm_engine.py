"""Offline LLM helpers using a local GGUF model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple, Any

try:  # optional dependency
    from llama_cpp import Llama
except Exception:  # pragma: no cover - fallback when library missing
    Llama = None

from arc_solver.src.symbolic.rule_language import parse_rule

_MODEL_PATH = Path("/kaggle/input/llm-models/tinyllama.gguf")

if Llama is not None and _MODEL_PATH.exists():
    try:  # pragma: no cover - external model load
        llm = Llama(model_path=str(_MODEL_PATH), n_ctx=1024)
    except Exception:  # pragma: no cover - handle load failure
        llm = None
else:  # pragma: no cover - missing model
    llm = None


def build_refine_prompt(trace: Any, feedback: Any) -> str:
    return (
        f"Given the symbolic transformation trace:\n{trace}\n\n"
        f"And the feedback highlighting mismatches:\n{feedback}\n\n"
        "Suggest an improved symbolic rule program (in DSL format)."
    )


def build_fix_prompt(entry: Any, discrepancy: Dict[Tuple[int, int], Tuple[int, int]]) -> str:
    return (
        f"Rule: {entry.rule}\n"
        f"Before: {entry.before}\n"
        f"After: {entry.after}\n"
        f"Discrepancies: {discrepancy}\n"
        "Suggest a corrected rule in DSL format."
    )


def build_narration_prompt(trace: Any) -> str:
    return f"Provide a brief explanation for the following trace:\n{trace}\n"


def _invoke(prompt: str) -> str:
    if llm is None:
        return ""
    out = llm(prompt)
    if isinstance(out, dict):
        text = out.get("choices", [{}])[0].get("text", "")
    else:
        text = str(out)
    return text.strip()


def local_refine_program(trace: Any, feedback: Any) -> List:
    text = _invoke(build_refine_prompt(trace, feedback))
    rules: List = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rules.append(parse_rule(line))
        except Exception:
            continue
    if not rules:
        rules.append(trace.rule)
    return rules


def local_suggest_rule_fix(entry: Any, discrepancy: Dict[Tuple[int, int], Tuple[int, int]]):
    text = _invoke(build_fix_prompt(entry, discrepancy))
    try:
        return parse_rule(text)
    except Exception:
        return None


def local_narrate(trace: Any) -> str:
    return _invoke(build_narration_prompt(trace))


__all__ = [
    "local_refine_program",
    "local_suggest_rule_fix",
    "local_narrate",
    "build_refine_prompt",
    "build_fix_prompt",
    "build_narration_prompt",
]
