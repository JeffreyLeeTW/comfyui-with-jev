"""Append-only log of models that refused to write a prompt (runs/refusals.jsonl)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

FILENAME = "refusals.jsonl"


@dataclass(frozen=True)
class Refusal:
    model: str
    mode: str
    attempt: int
    reason: str
    idea: str
    provider: str = ""
    time: str = ""


def record_refusal(runs_dir: Path, refusal: Refusal) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / FILENAME
    entry = asdict(refusal) | {"time": refusal.time or datetime.now().isoformat(timespec="seconds")}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def refusal_counts(runs_dir: Path) -> Counter[str]:
    """How many times each model has refused, skipping unreadable lines."""
    path = runs_dir / FILENAME
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            counts[json.loads(line)["model"]] += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return counts
