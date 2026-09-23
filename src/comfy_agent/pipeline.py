"""Generate -> judge loop, then render the accepted (or best) prompt in ComfyUI."""

from __future__ import annotations

import json
import mimetypes
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .comfyui import ComfyUIClient, OutputImage, build_workflow, load_workflow
from .config import GenParams, Settings
from .judge import JevJudge, Verdict
from .openrouter import Feedback, OpenRouterClient, image_to_data_url, merge_tags


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
    verdict: Verdict


@dataclass
class RunResult:
    idea: str
    model: str
    mode: str
    attempts: list[Attempt] = field(default_factory=list)
    chosen: Attempt | None = None
    passed: bool = False
    seed: int | None = None
    prompt_id: str | None = None
    images: list[OutputImage] = field(default_factory=list)
    log_path: Path | None = None


Event = dict
EventHandler = Callable[[Event], None]


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        openrouter: OpenRouterClient,
        judge: JevJudge,
        comfy: ComfyUIClient,
    ):
        self.settings = settings
        self.openrouter = openrouter
        self.judge = judge
        self.comfy = comfy

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
    ) -> RunResult:
        if not idea.strip() and image is None:
            raise ValueError("provide an idea text, an image, or both")
        emit = on_event or (lambda _e: None)
        s = self.settings
        gen = gen or s.gen
        thresholds = thresholds or s.thresholds
        max_attempts = max_attempts or s.max_attempts

        result = RunResult(idea=idea, model=model, mode="img2img" if image else "txt2img")
        image_url = (
            image_to_data_url(image.data, mimetypes.guess_type(image.filename)[0]) if image else None
        )
        history: list[Feedback] = []

        for n in range(1, max_attempts + 1):
            emit({"type": "generating", "attempt": n, "max": max_attempts})
            gp = self.openrouter.generate(model, idea, image_url, history)
            positive = merge_tags(s.positive_prefix, gp.positive)
            negative = merge_tags(s.negative_base, gp.negative)

            emit({"type": "judging", "attempt": n, "positive": positive, "negative": negative})
            verdict = self.judge.evaluate(idea, gp.image_description, positive, negative, thresholds)
            attempt = Attempt(n, positive, negative, gp.image_description, verdict)
            result.attempts.append(attempt)
            emit({"type": "judged", "attempt": attempt})

            if verdict.passed:
                result.chosen, result.passed = attempt, True
                break
            history.append(Feedback(gp.positive, gp.negative, verdict.critique()))

        if result.chosen is None:
            result.chosen = max(result.attempts, key=lambda a: a.verdict.rank_value)
        emit({"type": "chosen", "attempt": result.chosen, "passed": result.passed})

        if render:
            image_name = None
            if image:
                emit({"type": "uploading"})
                image_name = self.comfy.upload_image(image.data, image.filename)
            wf, result.seed = build_workflow(
                load_workflow(s.workflow_path),
                result.chosen.positive,
                result.chosen.negative,
                gen,
                image_name,
            )
            result.prompt_id = self.comfy.queue(wf)
            emit({"type": "rendering", "prompt_id": result.prompt_id, "seed": result.seed})
            result.images = self.comfy.wait(
                result.prompt_id, s.comfyui_timeout_s, s.comfyui_poll_interval_s
            )

        result.log_path = self._save_log(result, gen, thresholds)
        emit({"type": "done", "result": result})
        return result

    def _save_log(self, result: RunResult, gen: GenParams, thresholds: dict[str, float]) -> Path:
        self.settings.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
        data = {
            "idea": result.idea,
            "model": result.model,
            "mode": result.mode,
            "passed": result.passed,
            "chosen_attempt": result.chosen.number if result.chosen else None,
            "thresholds": thresholds,
            "generation": asdict(gen) | {"seed": result.seed},
            "prompt_id": result.prompt_id,
            "images": [asdict(i) for i in result.images],
            "attempts": [
                {
                    "number": a.number,
                    "positive": a.positive,
                    "negative": a.negative,
                    "image_description": a.image_description,
                    "passed": a.verdict.passed,
                    "scores": {
                        d.name: {"value": round(d.value, 3), "confidence": round(d.confidence, 3), "level": d.likely_level}
                        for d in a.verdict.dimensions.values()
                    },
                }
                for a in result.attempts
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def build_pipeline(settings: Settings) -> Pipeline:
    from .judge import make_typesafe_client

    return Pipeline(
        settings,
        OpenRouterClient(
            settings.openrouter_api_key,
            settings.openrouter_base_url,
            settings.openrouter_temperature,
        ),
        JevJudge(make_typesafe_client(settings.typesafe_api_key), settings.judge_model),
        ComfyUIClient(settings.comfyui_url),
    )
