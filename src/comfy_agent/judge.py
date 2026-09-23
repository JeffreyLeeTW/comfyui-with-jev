"""Jev (TypeSafe System One) grades a prompt on several dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from typesafe_sdk import Score

from .config import DIMENSIONS

QUESTIONS: dict[str, Score] = {
    "fidelity": Score(
        instructions=(
            "How faithfully does `positive_prompt` depict what the user asked for in `idea` "
            "(and in `image_description`, when present)? Ignore the fixed quality tags described in `notes`."
        ),
        criteria=[
            "Unrelated or contradicts the idea: the main subject or intent is missing or wrong",
            "Partially matches: the main subject is present but important requested elements are missing or changed",
            "Mostly matches: all key elements are present, with minor omissions or small added liberties",
            "Fully matches: every requested element is present and nothing contradicts the idea",
        ],
    ),
    "format": Score(
        instructions=(
            "How well is `positive_prompt` formatted for a danbooru-tag based anime image model? "
            "Good prompts are English, lowercase, comma-separated short tags, with no sentences and no contradictory tags."
        ),
        criteria=[
            "Not tag-based: written as sentences or paragraphs, or not in English",
            "Mixed: some tags but also sentences, long phrases, or non-English text",
            "Mostly clean tags with a few long phrases, duplicates, or slightly conflicting tags",
            "Clean English comma-separated danbooru-style tags with no conflicts",
        ],
    ),
    "completeness": Score(
        instructions=(
            "How completely does `positive_prompt` specify the picture: subject, appearance/clothing, "
            "pose or composition/framing, and background/scene or lighting?"
        ),
        criteria=[
            "Only the subject is named; almost nothing else is specified",
            "Subject plus one other aspect; several aspects are left unspecified",
            "Most aspects are specified, but one of pose/composition or scene/lighting is missing",
            "Subject, appearance, pose/composition, and scene/lighting are all specified",
        ],
    ),
    "negative": Score(
        instructions=(
            "How appropriate is `negative_prompt` for this `idea`? It should list English tags for unwanted "
            "artifacts or content and must NOT exclude anything the idea asks for."
        ),
        criteria=[
            "Harmful: it excludes something the idea explicitly wants, or it is not a tag list",
            "Weak: empty of useful content or mostly irrelevant tags",
            "Reasonable generic quality/artifact tags, but nothing specific to this idea",
            "Well suited: generic quality tags plus idea-specific exclusions, and no conflict with the idea",
        ],
    ),
}

STATE_NOTES = (
    "`positive_prompt` always starts with fixed quality tags (masterpiece, best quality, score_N, year N); "
    "they are expected and should not count against the prompt."
)


class SystemOneClient(Protocol):
    def system_one(self, state: Any, questions: Any, *, model: str | None = None) -> Any: ...


@dataclass(frozen=True)
class DimensionResult:
    name: str
    value: float  # normalized 0..1
    confidence: float
    threshold: float
    likely_level: str

    @property
    def passed(self) -> bool:
        return self.value >= self.threshold


@dataclass(frozen=True)
class Verdict:
    dimensions: dict[str, DimensionResult]

    @property
    def passed(self) -> bool:
        return all(d.passed for d in self.dimensions.values())

    @property
    def rank_value(self) -> float:
        """Weakest dimension; used to pick the best attempt when none passes."""
        return min(d.value for d in self.dimensions.values())

    def critique(self) -> str:
        lines = []
        for d in self.dimensions.values():
            if d.passed:
                continue
            top = QUESTIONS[d.name].criteria[-1]
            lines.append(
                f"- {d.name}: {d.value:.2f} (needs {d.threshold:.2f}). "
                f"Reviewer judged it as: \"{d.likely_level}\". Aim for: \"{top}\"."
            )
        return "\n".join(lines) or "All dimensions passed."


def build_state(idea: str, image_description: str, positive: str, negative: str) -> dict:
    return {
        "idea": idea.strip() or "(no text idea; see image_description)",
        "image_description": image_description or "(no reference image)",
        "positive_prompt": positive,
        "negative_prompt": negative,
        "notes": STATE_NOTES,
    }


class JevJudge:
    def __init__(self, client: SystemOneClient, model: str = "jev-latest"):
        self.client = client
        self.model = model

    def evaluate(
        self,
        idea: str,
        image_description: str,
        positive: str,
        negative: str,
        thresholds: dict[str, float],
    ) -> Verdict:
        state = build_state(idea, image_description, positive, negative)
        resp = self.client.system_one(state, QUESTIONS, model=self.model)
        dims = {}
        for name in DIMENSIONS:
            ans = resp.scores[name]
            levels = len(QUESTIONS[name].criteria) - 1
            probs = {int(k): v for k, v in ans.probabilities.items()}
            top_level = max(probs, key=probs.get)
            dims[name] = DimensionResult(
                name=name,
                value=float(ans.score) / levels,
                confidence=float(ans.confidence),
                threshold=float(thresholds[name]),
                likely_level=QUESTIONS[name].criteria[top_level],
            )
        return Verdict(dims)


def make_typesafe_client(api_key: str):
    """Build the real SDK client. Raises if the key is missing or invalid."""
    from typesafe_sdk import TypeSafeClient

    if not api_key:
        raise RuntimeError("TYPESAFE_API_KEY is not set (add it to .env)")
    return TypeSafeClient(api_key=api_key)
