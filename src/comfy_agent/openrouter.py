"""OpenRouter: list free models and turn an idea (text and/or image) into SD prompts."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import RATING_DESCRIPTIONS

SYSTEM_PROMPT = """You write prompts for an anime-style text-to-image model (Anima, danbooru-tag based).

Given the user's idea (text and/or a reference image), reply with ONLY a JSON object:
{
  "positive": "<comma-separated English danbooru-style tags>",
  "negative": "<comma-separated English tags for things to avoid>",
  "image_description": "<one or two English sentences describing the reference image, or empty string if none>"
}

Rules for "positive":
- English only, lowercase danbooru tags separated by ", " (e.g. "1girl, solo, long hair, red eyes, school uniform, standing, full body").
- Order: subject count/type -> character/appearance -> clothing -> pose/action -> composition/framing -> background/scene -> lighting/mood.
- Cover subject, appearance, pose/composition and scene/lighting. No sentences, no contradictory tags.
- Do NOT include quality tags such as masterpiece/best quality/score_N; they are added automatically.

Rules for "negative":
- English tags only, specific to this idea (e.g. unwanted extra subjects, wrong styles). Generic quality tags are added automatically.
- Never exclude anything the idea asks for.

If you will not write a prompt for this idea, reply with ONLY {"refused": true, "reason": "<short English reason>"} instead.

Output the JSON object and nothing else."""


class OpenRouterError(RuntimeError):
    pass


class ModelRefusal(OpenRouterError):
    """The model declined to write a prompt; retrying the same model is pointless."""

    def __init__(self, model: str, reason: str):
        super().__init__(f"{model} refused: {reason}")
        self.model = model
        self.reason = reason


@dataclass(frozen=True)
class FreeModel:
    id: str
    name: str
    context_length: int
    vision: bool


@dataclass(frozen=True)
class GeneratedPrompt:
    positive: str
    negative: str
    image_description: str
    raw: str


@dataclass(frozen=True)
class Feedback:
    """A previous attempt and why Jev rejected it."""

    positive: str
    negative: str
    critique: str


def _is_free(model: dict) -> bool:
    pricing = model.get("pricing") or {}
    try:
        return float(pricing.get("prompt", 1)) == 0 and float(pricing.get("completion", 1)) == 0
    except (TypeError, ValueError):
        return False


def image_to_data_url(image: str | Path | bytes, mime: str | None = None) -> str:
    if isinstance(image, (str, Path)):
        path = Path(image)
        mime = mime or mimetypes.guess_type(path.name)[0] or "image/png"
        data = path.read_bytes()
    else:
        data = image
        mime = mime or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def parse_prompt_json(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates code fences / chatter)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in reply")
        candidate = text[start : end + 1]
    data = json.loads(candidate)
    if not isinstance(data, dict) or not str(data.get("positive", "")).strip():
        raise ValueError("reply JSON lacks a non-empty 'positive' field")
    return data


_REFUSAL_RE = re.compile(
    r"\bI(?:'m| am)? (?:sorry|unable to)\b"
    r"|\bI (?:can(?:'|no)t|won't|will not|must decline)"
    r" (?:help|assist|create|generate|write|provide|produce|comply|fulfill|do that|make)"
    r"|\bnot able to (?:help|assist|create|generate|write|provide)"
    r"|\bagainst (?:my|the|our) (?:guidelines|polic(?:y|ies)|content polic(?:y|ies))",
    re.I,
)


def _json_object(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def detect_refusal(text: str, finish_reason: str | None = None) -> str | None:
    """Return a refusal reason, or None if the reply is not a refusal.

    Checks, in order: provider content filter, the explicit {"refused": true} signal,
    then refusal phrases, but only when the reply has no usable prompt JSON.
    """
    if finish_reason == "content_filter":
        return "blocked by provider content filter"
    data = _json_object(text)
    if data and data.get("refused") is True:
        return str(data.get("reason") or "no reason given").strip()
    if data and str(data.get("positive", "")).strip():
        return None
    if _REFUSAL_RE.search(text):
        return " ".join(text.split())[:300]
    return None


def remove_tags(tags: str, banned: str) -> str:
    """Drop every tag listed in `banned` (case-insensitive) from a tag string."""
    drop = {t.lower() for t in merge_tags(banned).split(", ") if t}
    return ", ".join(t for t in merge_tags(tags).split(", ") if t and t.lower() not in drop)


def merge_tags(*parts: str) -> str:
    """Join comma-separated tag strings, dropping empty and duplicate (case-insensitive) tags."""
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        for tag in (t.strip() for t in (part or "").replace("\n", ",").split(",")):
            if tag and tag.lower() not in seen:
                seen.add(tag.lower())
                out.append(tag)
    return ", ".join(out)


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.8,
        http: httpx.Client | None = None,
        max_http_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.http = http or httpx.Client(timeout=120)
        self.max_http_retries = max_http_retries

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set (add it to .env)")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Title": "comfy-agent",
        }

    def list_free_models(self, need_vision: bool = False) -> list[FreeModel]:
        resp = self.http.get(f"{self.base_url}/models")
        resp.raise_for_status()
        models = []
        for m in resp.json().get("data", []):
            if not _is_free(m):
                continue
            arch = m.get("architecture") or {}
            vision = "image" in (arch.get("input_modalities") or [])
            if need_vision and not vision:
                continue
            models.append(
                FreeModel(
                    id=m["id"],
                    name=m.get("name", m["id"]),
                    context_length=int(m.get("context_length") or 0),
                    vision=vision,
                )
            )
        return sorted(models, key=lambda fm: fm.id)

    def _chat(self, model: str, messages: list[dict]) -> tuple[str, str | None]:
        body = {"model": model, "messages": messages, "temperature": self.temperature}
        for attempt in range(self.max_http_retries + 1):
            resp = self.http.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=body)
            # Free models are often rate-limited or briefly overloaded.
            if resp.status_code in (429, 502, 503) and attempt < self.max_http_retries:
                wait = float(resp.headers.get("retry-after") or 5 * (attempt + 1))
                time.sleep(min(wait, 60))
                continue
            if resp.status_code >= 400:
                raise OpenRouterError(f"OpenRouter {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            if "error" in data:
                raise OpenRouterError(f"OpenRouter error: {data['error']}")
            try:
                choice = data["choices"][0]
                return choice["message"].get("content") or "", choice.get("finish_reason")
            except (KeyError, IndexError, AttributeError) as e:
                raise OpenRouterError(f"Unexpected OpenRouter response: {str(data)[:500]}") from e
        raise OpenRouterError("OpenRouter retries exhausted")

    def generate(
        self,
        model: str,
        idea: str,
        image_data_url: str | None = None,
        history: list[Feedback] | None = None,
        max_parse_retries: int = 2,
        rating: str | None = None,
    ) -> GeneratedPrompt:
        user_text = f"Idea:\n{idea.strip() or '(no text; use the reference image)'}"
        if rating:
            user_text += f"\n\nContent rating: {RATING_DESCRIPTIONS[rating]}."
        if image_data_url:
            user_text += "\n\nA reference image is attached."
        for i, fb in enumerate(history or [], 1):
            user_text += (
                f"\n\nPrevious attempt #{i} was REJECTED by the reviewer.\n"
                f"positive: {fb.positive}\nnegative: {fb.negative}\n"
                f"Reviewer feedback: {fb.critique}"
            )
        if history:
            user_text += "\n\nWrite an improved prompt that fixes every point of feedback."

        content: list[dict] | str = user_text
        if image_data_url:
            content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        last_err: Exception | None = None
        for _ in range(max_parse_retries + 1):
            raw, finish_reason = self._chat(model, messages)
            if reason := detect_refusal(raw, finish_reason):
                raise ModelRefusal(model, reason)
            try:
                data = parse_prompt_json(raw)
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "That was not valid JSON with the required keys. Reply with ONLY the JSON object."},
                ]
                continue
            return GeneratedPrompt(
                positive=str(data.get("positive", "")).strip(),
                negative=str(data.get("negative", "")).strip(),
                image_description=str(data.get("image_description", "") or "").strip(),
                raw=raw,
            )
        raise OpenRouterError(f"Model did not return valid prompt JSON: {last_err}")
