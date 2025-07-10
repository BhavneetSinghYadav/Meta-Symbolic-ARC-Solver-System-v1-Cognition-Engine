from __future__ import annotations

"""Failure pattern mining utilities for ARC solver debugging."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List
import json


__all__ = ["mine_failure_patterns"]


def _normalise_reason(reason: str) -> str:
    """Return a simplified reason label."""
    if not reason:
        return "unknown"
    reason = reason.split("(")[0]
    reason = reason.split(":")[0]
    return reason.strip()


def mine_failure_patterns(
    log_path: str = "failure_log.jsonl",
    *,
    top_n: int = 10,
    save: bool = True,
) -> None:
    """Analyse ``log_path`` and print common failure reasons.

    Parameters
    ----------
    log_path:
        Path to the ``failure_log.jsonl`` file.
    top_n:
        Number of top reasons to display in the histogram.
    save:
        When ``True`` save cluster details to ``debug/failure_patterns.json``.
    """
    path = Path(log_path)
    if not path.is_file():
        print(f"Log file not found: {path}")
        return

    reason_counts: Counter[str] = Counter()
    prefix_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    samples: Dict[str, List[str]] = defaultdict(list)

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            reason = _normalise_reason(str(data.get("reason")))
            reason_counts[reason] += 1

            dsl = data.get("rule_dsl") or data.get("rule")
            if isinstance(dsl, str):
                prefix = dsl.split(".")[0].split()[0]
                prefix_counts[reason][prefix] += 1
                if len(samples[reason]) < 5 and dsl not in samples[reason]:
                    samples[reason].append(dsl)

    for reason, count in reason_counts.most_common(top_n):
        print(f"{reason} \u2192 {count}")

    if save:
        summary = {
            reason: {
                "count": count,
                "dsl_prefixes": dict(prefix_counts[reason]),
                "examples": samples[reason],
            }
            for reason, count in reason_counts.items()
        }
        out_path = Path("debug/failure_patterns.json")
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2)


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Mine failure patterns from logs")
    parser.add_argument("log_path", nargs="?", default="failure_log.jsonl")
    parser.add_argument("--top", type=int, default=10, help="number of top reasons")
    parser.add_argument("--no-save", action="store_true", help="do not save summary")
    args = parser.parse_args()
    mine_failure_patterns(args.log_path, top_n=args.top, save=not args.no_save)


if __name__ == "__main__":
    _cli()
