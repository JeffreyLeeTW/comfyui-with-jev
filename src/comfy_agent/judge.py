"""Jev (TypeSafe System One) grades a prompt on several dimensions; format/negative wording depends on the style."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from typesafe_sdk import Score

from .config import DIMENSIONS, RATING_DESCRIPTIONS

_FIDELITY = Score(
    instructions=(
        "How faithfully does `positive_prompt` depict what the user asked for in `idea` "
        "(and in `image_description`, when present), within the limits of `content_rating`? "
        "Content that `content_rating` forbids counts as contradicting the idea. "
        "Ignore the fixed quality and rating tags described in `notes`."
    ),
    criteria=[
        "Unrelated or contradicts the idea: the main subject or intent is missing or wrong",
        "Partially matches: the main subject is present but important requested elements are missing or changed",
        "Mostly matches: all key elements are present, with minor omissions or small added liberties",
        "Fully matches: every requested element is present and nothing contradicts the idea",
    ],
)

_COMPLETENESS = Score(
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
)

# Dimensions whose wording depends on the prompt style; keyed by style, then dimension.
QUESTIONS: dict[str, dict[str, Score]] = {
    "tags": {
        "fidelity": _FIDELITY,
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
        "completeness": _COMPLETENESS,
        "negative": Score(
            instructions=(
                "How appropriate is `negative_prompt` for this `idea`? It should list English tags for unwanted "
                "artifacts or content and must NOT exclude anything the idea asks for. "
                "Excluding content that `content_rating` forbids is correct, not harmful."
            ),
            criteria=[
                "Harmful: it excludes something the idea explicitly wants, or it is not a tag list",
                "Weak: empty of useful content or mostly irrelevant tags",
                "Reasonable generic quality/artifact tags, but nothing specific to this idea",
                "Well suited: generic quality tags plus idea-specific exclusions, and no conflict with the idea",
            ],
        ),
    },
    "natural": {
        "fidelity": _FIDELITY,
        "format": Score(
            instructions=(
                "How well is the description in `positive_prompt` written as natural language for an anime image "
                "model that understands English sentences? Judge only the text after the fixed tag prefix described "
                "in `notes`. Good descriptions are fluent, concrete English sentences with no tag lists and no contradictions."
            ),
            criteria=[
                "Not natural language: only a comma-separated tag list or keywords, or not in English",
                "Mixed: mostly tags or fragments with a few sentences, or partly non-English",
                "Mostly fluent English sentences, but vague, repetitive, or with a minor contradiction",
                "Clear, fluent English sentences that concretely describe the image, with no contradictions",
            ],
        ),
        "completeness": _COMPLETENESS,
        "negative": Score(
            instructions=(
                "How appropriate is `negative_prompt` for this `idea`? It starts with fixed quality tags (see `notes`), "
                "followed by an English description of unwanted content, and must NOT exclude anything the idea asks for. "
                "Excluding content that `content_rating` forbids is correct, not harmful."
            ),
            criteria=[
                "Harmful: it excludes something the idea explicitly wants, or the added text is unrelated or not English",
                "Weak: nothing useful is added beyond the fixed tags, or the additions are mostly irrelevant",
                "Reasonable: it names generic unwanted things, but nothing specific to this idea",
                "Well suited: it names idea-specific things to avoid, with no conflict with the idea",
            ],
        ),
    },
}

STATE_NOTES = {
    "tags": (
        "`positive_prompt` always starts with fixed quality tags (masterpiece, best quality, score_N, year N); "
        "they are expected and should not count against the prompt. When `content_rating` is set, "
        "matching rating tags (such as safe, nsfw, explicit) are added automatically as well."
    ),
    "natural": (
        "`positive_prompt` always starts with a fixed comma-separated tag prefix (masterpiece, best quality, score_N, "
        "year N, and rating tags such as safe, nsfw, explicit when `content_rating` is set), followed by the "
        "natural-language description being judged. `negative_prompt` likewise starts with fixed quality tags. "
        "These fixed tags are expected and should not count against the prompt."
    ),
}


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
    style: str = "tags"

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
            top = QUESTIONS[self.style][d.name].criteria[-1]
            lines.append(
                f"- {d.name}: {d.value:.2f} (needs {d.threshold:.2f}). "
                f"Reviewer judged it as: \"{d.likely_level}\". Aim for: \"{top}\"."
            )
        return "\n".join(lines) or "All dimensions passed."


def build_state(
    idea: str, image_description: str, positive: str, negative: str, rating: str | None = None,
    style: str = "tags",
) -> dict:
    return {
        "idea": idea.strip() or "(no text idea; see image_description)",
        "image_description": image_description or "(no reference image)",
        "content_rating": RATING_DESCRIPTIONS[rating] if rating else "(not specified; follow the idea)",
        "positive_prompt": positive,
        "negative_prompt": negative,
        "notes": STATE_NOTES[style],
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
        rating: str | None = None,
        style: str = "tags",
    ) -> Verdict:
        questions = QUESTIONS[style]
        state = build_state(idea, image_description, positive, negative, rating, style)
        resp = self.client.system_one(state, questions, model=self.model)
        dims = {}
        for name in DIMENSIONS:
            ans = resp.scores[name]
            levels = len(questions[name].criteria) - 1
            probs = {int(k): v for k, v in ans.probabilities.items()}
            top_level = max(probs, key=probs.get)
            dims[name] = DimensionResult(
                name=name,
                value=float(ans.score) / levels,
                confidence=float(ans.confidence),
                threshold=float(thresholds[name]),
                likely_level=questions[name].criteria[top_level],
            )
        return Verdict(dims, style)


def make_typesafe_client(api_key: str):
    """Build the real SDK client. Raises if the key is missing or invalid."""
    from typesafe_sdk import TypeSafeClient

    if not api_key:
        raise RuntimeError("TYPESAFE_API_KEY is not set (add it to .env)")
    return TypeSafeClient(api_key=api_key)
