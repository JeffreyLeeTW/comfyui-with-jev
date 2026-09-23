"""Load config.yaml + .env into typed settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = ("fidelity", "format", "completeness", "negative")


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
    openrouter_base_url: str
    openrouter_default_model: str
    openrouter_temperature: float
    judge_model: str
    max_attempts: int
    thresholds: dict[str, float]
    positive_prefix: str
    negative_base: str
    gen: GenParams
    runs_dir: Path
    openrouter_api_key: str
    typesafe_api_key: str


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_settings(config_path: str | Path | None = None) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    comfy = raw.get("comfyui", {})
    orr = raw.get("openrouter", {})
    judge = raw.get("judge", {})
    prompt = raw.get("prompt", {})
    gen_raw = raw.get("generation", {})

    known = {f.name for f in fields(GenParams)}
    gen = GenParams(**{k: v for k, v in gen_raw.items() if k in known and v is not None})

    thresholds = {d: 0.67 for d in DIMENSIONS} | (judge.get("thresholds") or {})

    return Settings(
        comfyui_url=str(comfy.get("url", "http://10.0.0.15:8188")).rstrip("/"),
        workflow_path=_resolve(comfy.get("workflow", "workflows/anima_flow.json")),
        comfyui_timeout_s=float(comfy.get("timeout_s", 600)),
        comfyui_poll_interval_s=float(comfy.get("poll_interval_s", 2)),
        openrouter_base_url=str(orr.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/"),
        openrouter_default_model=orr.get("default_model") or "",
        openrouter_temperature=float(orr.get("temperature", 0.8)),
        judge_model=judge.get("model", "jev-latest"),
        max_attempts=int(judge.get("max_attempts", 5)),
        thresholds={d: float(thresholds[d]) for d in DIMENSIONS},
        positive_prefix=prompt.get("positive_prefix", ""),
        negative_base=prompt.get("negative_base", ""),
        gen=gen,
        runs_dir=_resolve(raw.get("runs_dir", "runs")),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        typesafe_api_key=os.environ.get("TYPESAFE_API_KEY", ""),
    )
