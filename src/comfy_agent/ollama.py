"""Ollama backend: local models via the native /api/chat (so thinking can be turned off)."""

from __future__ import annotations

import httpx

from .openrouter import LLMError, ModelInfo, PromptWriter


class OllamaError(LLMError):
    pass


def _to_native(messages: list[dict]) -> list[dict]:
    """OpenAI-style content parts -> Ollama's text `content` plus base64 `images`."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        text = "\n".join(p["text"] for p in content if p.get("type") == "text")
        images = [
            p["image_url"]["url"].split(",", 1)[-1] for p in content if p.get("type") == "image_url"
        ]
        out.append({"role": m["role"], "content": text} | ({"images": images} if images else {}))
    return out


def _context_length(model_info: dict) -> int:
    for key, value in model_info.items():
        if key.endswith(".context_length"):
            return int(value)
    return 0


class OllamaClient(PromptWriter):
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://10.0.0.15:11434",
        temperature: float = 0.8,
        think: bool = False,
        timeout_s: float = 600,
        http: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.think = think
        # Local models may need to load into VRAM before the first token.
        self.http = http or httpx.Client(timeout=timeout_s)

    def _get(self, path: str) -> dict:
        try:
            resp = self.http.get(f"{self.base_url}{path}")
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama unreachable at {self.base_url}: {e}") from e
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> httpx.Response:
        try:
            return self.http.post(f"{self.base_url}{path}", json=body)
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama unreachable at {self.base_url}: {e}") from e

    def version(self) -> str:
        return self._get("/api/version").get("version", "?")

    def list_models(self, need_vision: bool = False) -> list[ModelInfo]:
        """Installed models that can chat (embedding-only models are skipped)."""
        models = []
        for m in self._get("/api/tags").get("models", []):
            resp = self._post("/api/show", {"model": m["name"]})
            show = resp.json() if resp.status_code < 400 else {}
            caps = show.get("capabilities") or ["completion"]  # older servers omit capabilities
            if "completion" not in caps:
                continue
            vision = "vision" in caps
            if need_vision and not vision:
                continue
            models.append(
                ModelInfo(
                    id=m["name"],
                    name=m["name"],
                    context_length=_context_length(show.get("model_info") or {}),
                    vision=vision,
                )
            )
        return sorted(models, key=lambda mi: mi.id)

    def _chat(self, model: str, messages: list[dict]) -> tuple[str, str | None]:
        body = {
            "model": model,
            "messages": _to_native(messages),
            "stream": False,
            "think": self.think,
            "options": {"temperature": self.temperature},
        }
        resp = self._post("/api/chat", body)
        # Non-thinking models reject the `think` field; retry without it.
        if resp.status_code == 400 and "think" in resp.text.lower():
            body.pop("think")
            resp = self._post("/api/chat", body)
        if resp.status_code >= 400:
            raise OllamaError(f"Ollama {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        if "error" in data:
            raise OllamaError(f"Ollama error: {data['error']}")
        try:
            return data["message"].get("content") or "", data.get("done_reason")
        except (KeyError, AttributeError) as e:
            raise OllamaError(f"Unexpected Ollama response: {str(data)[:500]}") from e
