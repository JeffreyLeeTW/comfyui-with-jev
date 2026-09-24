"""Load config.yaml + .env into typed settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv, set_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
# Endpoints that the WebUI can save to .env; they override config.yaml.
ENDPOINT_ENV = {"comfyui": "COMFYUI_URL", "ollama": "OLLAMA_URL"}
DIMENSIONS = ("fidelity", "format", "completeness", "negative")
RATINGS = ("sfw", "nsfw")
PROVIDERS = ("openrouter", "ollama")
JUDGE_MODES = ("on", "off", "ab")  # Jev loop / no Jev (score for reference only) / both, same seed
# How the writer phrases the prompt body: danbooru tags, or English sentences (fixed prefix tags stay tags).
PROMPT_STYLES = ("natural", "tags")
# Told to both the prompt writer and Jev when a rating is chosen.
RATING_DESCRIPTIONS = {
    "sfw": "SFW: the image must be safe for work, with no nudity or sexual content, even if the idea hints at it",
    "nsfw": "NSFW: the image is intended to be explicit adult content",
}


@dataclass(frozen=True)
class RatingTags:
    """Tags forced into the prompt for a content rating; in tag style the opposite side is stripped of them."""

    positive: str = ""
    negative: str = ""


@dataclass(frozen=True)
class GenParams:
    width: int = 832
    height: int = 1216
    batch_size: int = 1
    seed: int = -1
    steps: int = 35
    cfg: float = 5.5
    sampler_name: str = "er_sde"
    scheduler: str = "beta"
    denoise: float = 0.9
    filename_prefix: str = "comfy_agent"
    lora_strengths: dict[str, float] = field(default_factory=dict)

    def override(self, **changes: Any) -> GenParams:
        """Return a copy with every non-None change applied."""
        return replace(self, **{k: v for k, v in changes.items() if v is not None})


@dataclass(frozen=True)
class Settings:
    comfyui_url: str
    workflow_path: Path
    comfyui_timeout_s: float
    comfyui_poll_interval_s: float
    llm_provider: str
    openrouter_base_url: str
    openrouter_default_model: str
    openrouter_temperature: float
    ollama_url: str
    ollama_default_model: str
    ollama_temperature: float
    ollama_think: bool
    ollama_timeout_s: float
    judge_model: str
    max_attempts: int
    thresholds: dict[str, float]
    prompt_style: str
    positive_prefix: str
    negative_base: str
    ratings: dict[str, RatingTags]
    gen: GenParams
    runs_dir: Path
    language: str
    openrouter_api_key: str
    typesafe_api_key: str


def endpoint_url(host: str, port: int | str) -> str:
    """Build http://host:port from user input; tolerates a pasted scheme or trailing slash."""
    host = str(host).strip().removeprefix("http://").removeprefix("https://").strip("/")
    if not host or any(c in host for c in " /:"):
        raise ValueError(f"invalid host: {host!r}")
    try:
        port = int(port)
    except (TypeError, ValueError):
        raise ValueError(f"invalid port: {port!r}") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"port out of range: {port}")
    return f"http://{host}:{port}"


def split_endpoint(url: str) -> tuple[str, int]:
    parts = urlsplit(url if "://" in url else f"http://{url}")
    return parts.hostname or "", parts.port or 80


def save_env(values: dict[str, str], path: Path = ENV_PATH) -> None:
    """Write keys into .env (other lines, including API keys, are left untouched)."""
    path.touch(exist_ok=True)
    for key, value in values.items():
        set_key(str(path), key, value, quote_mode="never")
        os.environ[key] = value  # load_dotenv never overrides, so keep this process in sync


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_settings(config_path: str | Path | None = None) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    comfy = raw.get("comfyui", {})
    llm = raw.get("llm", {})
    orr = raw.get("openrouter", {})
    oll = raw.get("ollama", {})
    judge = raw.get("judge", {})
    prompt = raw.get("prompt", {})
    gen_raw = raw.get("generation", {})

    known = {f.name for f in fields(GenParams)}
    gen = GenParams(**{k: v for k, v in gen_raw.items() if k in known and v is not None})

    provider = str(llm.get("provider", "openrouter")).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"llm.provider must be one of {PROVIDERS}, got {provider!r}")
    style = str(prompt.get("style", "natural")).lower()
    if style not in PROMPT_STYLES:
        raise ValueError(f"prompt.style must be one of {PROMPT_STYLES}, got {style!r}")
    thresholds = {d: 0.67 for d in DIMENSIONS} | (judge.get("thresholds") or {})
    ratings_raw = prompt.get("ratings") or {}
    ratings = {
        r: RatingTags(str(t.get("positive") or ""), str(t.get("negative") or ""))
        for r in RATINGS
        for t in [ratings_raw.get(r) or {}]
    }

    return Settings(
        comfyui_url=str(os.environ.get("COMFYUI_URL") or comfy.get("url", "http://10.0.0.15:8188")).rstrip("/"),
        workflow_path=_resolve(comfy.get("workflow", "workflows/anima_flow.json")),
        comfyui_timeout_s=float(comfy.get("timeout_s", 600)),
        comfyui_poll_interval_s=float(comfy.get("poll_interval_s", 2)),
        llm_provider=provider,
        openrouter_base_url=str(orr.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/"),
        openrouter_default_model=orr.get("default_model") or "",
        openrouter_temperature=float(orr.get("temperature", 0.8)),
        ollama_url=str(os.environ.get("OLLAMA_URL") or oll.get("url", "http://10.0.0.15:11434")).rstrip("/"),
        ollama_default_model=oll.get("default_model") or "",
        ollama_temperature=float(oll.get("temperature", 0.8)),
        ollama_think=bool(oll.get("think", False)),
        ollama_timeout_s=float(oll.get("timeout_s", 600)),
        judge_model=judge.get("model", "jev-latest"),
        max_attempts=int(judge.get("max_attempts", 5)),
        thresholds={d: float(thresholds[d]) for d in DIMENSIONS},
        prompt_style=style,
        positive_prefix=prompt.get("positive_prefix", ""),
        negative_base=prompt.get("negative_base", ""),
        ratings=ratings,
        gen=gen,
        runs_dir=_resolve(raw.get("runs_dir", "runs")),
        language=str((raw.get("ui") or {}).get("language", "en")),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        typesafe_api_key=os.environ.get("TYPESAFE_API_KEY", ""),
    )
