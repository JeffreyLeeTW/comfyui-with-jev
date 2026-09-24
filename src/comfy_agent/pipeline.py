"""Generate -> judge loop, then render the accepted (or best) prompt in ComfyUI.

`run` uses the Jev loop (or, with use_judge=False, one draft scored for reference only);
`compare` renders the Jev result and the first draft with the same seed for an A/B comparison.
"""

from __future__ import annotations

import json
import mimetypes
import random
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from .comfyui import ComfyUIClient, OutputImage, build_workflow, load_workflow
from .config import PROMPT_STYLES, RATINGS, GenParams, RatingTags, Settings
from .judge import JevJudge, Verdict
from .openrouter import (
    Feedback,
    ModelRefusal,
    PromptWriter,
    image_to_data_url,
    join_prose,
    merge_tags,
    remove_tags,
)
from .refusals import Refusal, record_refusal


@dataclass(frozen=True)
class InputImage:
    data: bytes
    filename: str

    @classmethod
    def from_path(cls, path: str | Path) -> InputImage:
        p = Path(path)
        return cls(p.read_bytes(), p.name)


@dataclass
class Attempt:
    number: int
    positive: str
    negative: str
    image_description: str
    verdict: Verdict | None  # None when reference scoring was unavailable


@dataclass
class RunResult:
    idea: str
    provider: str
    model: str
    mode: str
    judge_mode: str = "on"  # "on" = Jev loop decided; "off" = first draft, score is reference only
    rating: str | None = None
    style: str = "tags"
    attempts: list[Attempt] = field(default_factory=list)
    chosen: Attempt | None = None
    passed: bool = False
    refused: str | None = None
    seed: int | None = None
    prompt_id: str | None = None
    images: list[OutputImage] = field(default_factory=list)
    log_path: Path | None = None


@dataclass
class CompareResult:
    with_jev: RunResult
    without_jev: RunResult
    identical: bool = False  # the first draft already passed, so both arms share one render
    log_path: Path | None = None

    @property
    def refused(self) -> str | None:
        return self.with_jev.refused


Event = dict
EventHandler = Callable[[Event], None]


def _new_log_path(runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}"
    path, n = runs_dir / f"{stem}.json", 1
    while path.exists():
        n += 1
        path = runs_dir / f"{stem}-{n}.json"
    return path


def _attempt_dict(a: Attempt) -> dict:
    v = a.verdict
    return {
        "number": a.number,
        "positive": a.positive,
        "negative": a.negative,
        "image_description": a.image_description,
        "passed": v.passed if v else None,
        "scores": {
            d.name: {
                "value": round(d.value, 3), "threshold": d.threshold,
                "confidence": round(d.confidence, 3), "level": d.likely_level,
            }
            for d in v.dimensions.values()
        } if v else {},
    }


def _result_dict(r: RunResult) -> dict:
    return {
        "judge_mode": r.judge_mode,
        "passed": r.passed,
        "refused": r.refused,
        "chosen_attempt": r.chosen.number if r.chosen else None,
        "seed": r.seed,
        "prompt_id": r.prompt_id,
        "images": [asdict(i) for i in r.images],
        "attempts": [_attempt_dict(a) for a in r.attempts],
    }


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        writer: PromptWriter,
        judge: JevJudge | None,
        comfy: ComfyUIClient,
    ):
        self.settings = settings
        self.writer = writer
        self.judge = judge
        self.comfy = comfy

    # -- public entry points ------------------------------------------------

    def run(
        self,
        idea: str,
        model: str,
        image: InputImage | None = None,
        gen: GenParams | None = None,
        thresholds: dict[str, float] | None = None,
        max_attempts: int | None = None,
        render: bool = True,
        on_event: EventHandler | None = None,
        rating: str | None = None,
        use_judge: bool = True,
        style: str | None = None,
    ) -> RunResult:
        emit, gen, thresholds, max_attempts, style = self._prepare(
            idea, image, gen, thresholds, max_attempts, rating, style, on_event, need_judge=use_judge
        )
        result = self._write(
            idea, model, image, rating, style, thresholds, max_attempts if use_judge else 1, use_judge, emit
        )
        if render and not result.refused:
            self._render(result, self._upload(image, emit), gen, emit)
        result.log_path = self._save_log(result, gen, thresholds)
        emit({"type": "done", "result": result})
        return result

    def compare(
        self,
        idea: str,
        model: str,
        image: InputImage | None = None,
        gen: GenParams | None = None,
        thresholds: dict[str, float] | None = None,
        max_attempts: int | None = None,
        render: bool = True,
        on_event: EventHandler | None = None,
        rating: str | None = None,
        style: str | None = None,
    ) -> CompareResult:
        """A/B: Jev loop vs. no Jev. The no-Jev arm is the loop's first draft (what a
        no-Jev run would send), so both arms start from the same prompt and share one seed."""
        emit, gen, thresholds, max_attempts, style = self._prepare(
            idea, image, gen, thresholds, max_attempts, rating, style, on_event, need_judge=True
        )
        if gen.seed < 0:
            gen = gen.override(seed=random.randint(0, 2**50))

        a = self._write(idea, model, image, rating, style, thresholds, max_attempts, True, emit)
        b = replace(a, judge_mode="off", attempts=a.attempts[:1], images=[])
        b.chosen = a.attempts[0] if a.attempts else None
        b.passed = bool(b.chosen and b.chosen.verdict and b.chosen.verdict.passed)
        result = CompareResult(a, b, identical=b.chosen is not None and b.chosen is a.chosen)

        if render and not a.refused:
            image_name = self._upload(image, emit)
            self._render(a, image_name, gen, emit, arm="with_jev")
            if result.identical:
                b.seed, b.prompt_id, b.images = a.seed, a.prompt_id, a.images
            else:
                self._render(b, image_name, gen, emit, arm="without_jev")

        result.log_path = self._save_compare_log(result, gen, thresholds)
        emit({"type": "done", "result": result})
        return result

    # -- steps ----------------------------------------------------------------

    def _prepare(self, idea, image, gen, thresholds, max_attempts, rating, style, on_event, need_judge: bool):
        if not idea.strip() and image is None:
            raise ValueError("provide an idea text, an image, or both")
        if rating is not None and rating not in RATINGS:
            raise ValueError(f"rating must be one of {RATINGS} or None, got {rating!r}")
        style = style or self.settings.prompt_style
        if style not in PROMPT_STYLES:
            raise ValueError(f"style must be one of {PROMPT_STYLES}, got {style!r}")
        if need_judge and self.judge is None:
            raise RuntimeError("Jev is required for this mode but TYPESAFE_API_KEY is not set")
        s = self.settings
        return (
            on_event or (lambda _e: None),
            gen or s.gen,
            thresholds or s.thresholds,
            max_attempts or s.max_attempts,
            style,
        )

    def _write(
        self, idea: str, model: str, image: InputImage | None, rating: str | None, style: str,
        thresholds: dict[str, float], max_attempts: int, use_judge: bool, emit: EventHandler,
    ) -> RunResult:
        """Generate (and judge) prompts; no rendering, no log."""
        s = self.settings
        result = RunResult(
            idea=idea, provider=self.writer.name, model=model, mode="img2img" if image else "txt2img",
            judge_mode="on" if use_judge else "off", rating=rating, style=style,
        )
        tags = s.ratings[rating] if rating else RatingTags()
        image_url = (
            image_to_data_url(image.data, mimetypes.guess_type(image.filename)[0]) if image else None
        )
        history: list[Feedback] = []

        for n in range(1, max_attempts + 1):
            emit({"type": "generating", "attempt": n, "max": max_attempts})
            try:
                gp = self.writer.generate(model, idea, image_url, history, rating=rating, style=style)
            except ModelRefusal as e:
                self._refused(result, n, e.reason, emit)
                return result
            positive, negative = self._compose(gp.positive, gp.negative, tags, style)

            emit({"type": "judging", "attempt": n, "positive": positive, "negative": negative,
                  "reference": not use_judge})
            verdict = self._judge(idea, gp.image_description, positive, negative, thresholds, rating, style,
                                  use_judge, emit)
            attempt = Attempt(n, positive, negative, gp.image_description, verdict)
            result.attempts.append(attempt)
            emit({"type": "judged", "attempt": attempt, "reference": not use_judge})

            if not use_judge:
                result.chosen, result.passed = attempt, bool(verdict and verdict.passed)
                break
            if verdict.passed:
                result.chosen, result.passed = attempt, True
                break
            history.append(Feedback(gp.positive, gp.negative, verdict.critique()))

        if result.chosen is None:
            result.chosen = max(result.attempts, key=lambda a: a.verdict.rank_value)
        emit({"type": "chosen", "attempt": result.chosen, "passed": result.passed, "judge": use_judge})
        return result

    def _compose(self, positive: str, negative: str, tags: RatingTags, style: str) -> tuple[str, str]:
        """Add the fixed prefix and rating tags. Only tag bodies can have conflicting rating tags stripped."""
        s = self.settings
        if style == "tags":
            return (
                merge_tags(s.positive_prefix, tags.positive, remove_tags(positive, tags.negative)),
                merge_tags(s.negative_base, tags.negative, remove_tags(negative, tags.positive)),
            )
        return (
            join_prose(merge_tags(s.positive_prefix, tags.positive), positive),
            join_prose(merge_tags(s.negative_base, tags.negative), negative),
        )

    def _judge(self, idea, image_description, positive, negative, thresholds, rating, style,
               use_judge: bool, emit: EventHandler) -> Verdict | None:
        args = (idea, image_description, positive, negative, thresholds, rating, style)
        if use_judge:
            return self.judge.evaluate(*args)
        # Reference score only: never let Jev problems block a no-Jev run.
        if self.judge is None:
            emit({"type": "warning", "message": "TYPESAFE_API_KEY not set; skipping reference score"})
            return None
        try:
            return self.judge.evaluate(*args)
        except Exception as e:
            emit({"type": "warning", "message": f"reference score failed: {type(e).__name__}: {e}"})
            return None

    def _refused(self, result: RunResult, attempt: int, reason: str, emit: EventHandler) -> None:
        """Stop the whole run (no render) and remember which model refused."""
        result.refused = reason
        record_refusal(
            self.settings.runs_dir,
            Refusal(result.model, result.mode, attempt, reason, result.idea, result.provider),
        )
        emit({"type": "refused", "attempt": attempt, "model": result.model, "reason": reason})

    def _upload(self, image: InputImage | None, emit: EventHandler) -> str | None:
        if image is None:
            return None
        emit({"type": "uploading"})
        return self.comfy.upload_image(image.data, image.filename)

    def _render(
        self, result: RunResult, image_name: str | None, gen: GenParams, emit: EventHandler,
        arm: str | None = None,
    ) -> None:
        s = self.settings
        wf, result.seed = build_workflow(
            load_workflow(s.workflow_path), result.chosen.positive, result.chosen.negative, gen, image_name,
        )
        result.prompt_id = self.comfy.queue(wf)
        emit({"type": "rendering", "prompt_id": result.prompt_id, "seed": result.seed, "arm": arm})
        result.images = self.comfy.wait(result.prompt_id, s.comfyui_timeout_s, s.comfyui_poll_interval_s)

    # -- logs -------------------------------------------------------------------

    def _common_log(self, r: RunResult, gen: GenParams, thresholds: dict[str, float]) -> dict:
        return {
            "time": datetime.now().isoformat(timespec="seconds"),
            "idea": r.idea,
            "provider": r.provider,
            "model": r.model,
            "mode": r.mode,
            "rating": r.rating,
            "prompt_style": r.style,
            "thresholds": thresholds,
            "generation": asdict(gen),
        }

    def _save_log(self, result: RunResult, gen: GenParams, thresholds: dict[str, float]) -> Path:
        path = _new_log_path(self.settings.runs_dir)
        data = {"kind": "run"} | self._common_log(result, gen, thresholds) | _result_dict(result)
        data["generation"]["seed"] = result.seed
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _save_compare_log(self, result: CompareResult, gen: GenParams, thresholds: dict[str, float]) -> Path:
        path = _new_log_path(self.settings.runs_dir)
        data = {"kind": "compare"} | self._common_log(result.with_jev, gen, thresholds) | {
            "identical": result.identical,
            "with_jev": _result_dict(result.with_jev),
            "without_jev": _result_dict(result.without_jev),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def build_pipeline(settings: Settings, provider: str | None = None) -> Pipeline:
    """Jev is optional: without TYPESAFE_API_KEY only no-Jev runs (without reference score) work."""
    from .judge import make_typesafe_client
    from .llm import make_writer

    judge = (
        JevJudge(make_typesafe_client(settings.typesafe_api_key), settings.judge_model)
        if settings.typesafe_api_key else None
    )
    return Pipeline(settings, make_writer(settings, provider), judge, ComfyUIClient(settings.comfyui_url))
