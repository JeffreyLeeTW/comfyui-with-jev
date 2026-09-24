"""Pick the prompt-writing backend (OpenRouter or Ollama) from settings."""

from __future__ import annotations

from .config import PROVIDERS, Settings
from .ollama import OllamaClient
from .openrouter import OpenRouterClient, PromptWriter

PROVIDER_LABELS = {"openrouter": "OpenRouter（免費模型）", "ollama": "Ollama（本地）"}


def resolve_provider(settings: Settings, provider: str | None) -> str:
    provider = (provider or settings.llm_provider).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")
    return provider


def make_writer(settings: Settings, provider: str | None = None) -> PromptWriter:
    if resolve_provider(settings, provider) == "ollama":
        return OllamaClient(
            settings.ollama_url, settings.ollama_temperature, settings.ollama_think, settings.ollama_timeout_s
        )
    return OpenRouterClient(
        settings.openrouter_api_key, settings.openrouter_base_url, settings.openrouter_temperature
    )


def default_model(settings: Settings, provider: str | None = None) -> str:
    if resolve_provider(settings, provider) == "ollama":
        return settings.ollama_default_model
    return settings.openrouter_default_model
