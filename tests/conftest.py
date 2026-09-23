from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfy_agent.config import PROJECT_ROOT, load_settings
from comfy_agent.judge import QUESTIONS


@pytest.fixture
def settings(tmp_path: Path):
    return replace(load_settings(PROJECT_ROOT / "config.yaml"), runs_dir=tmp_path / "runs")


def score_answer(level_probs: list[float]):
    """Fake ScoreAnswer: probabilities over the question's levels."""
    score = sum(i * p for i, p in enumerate(level_probs))
    return SimpleNamespace(
        score=score,
        confidence=max(level_probs),
        probabilities={str(i): p for i, p in enumerate(level_probs)},
    )


class FakeSystemOne:
    """Returns queued per-call {dimension: level_probs} answers."""

    def __init__(self, calls: list[dict[str, list[float]]]):
        self.calls = list(calls)
        self.requests = []

    def system_one(self, state, questions, *, model=None):
        self.requests.append((state, questions, model))
        answers = self.calls.pop(0)
        assert set(answers) == set(QUESTIONS)
        return SimpleNamespace(scores={k: score_answer(v) for k, v in answers.items()})


GOOD = [0.0, 0.0, 0.1, 0.9]   # ~0.97 normalized
BAD = [0.1, 0.8, 0.1, 0.0]    # ~0.33
MID = [0.0, 0.2, 0.6, 0.2]    # ~0.67


def all_dims(probs):
    return {d: probs for d in QUESTIONS}
