import json
from dataclasses import replace

import pytest
import respx

from comfy_agent.llm import default_model, make_writer
from comfy_agent.ollama import OllamaClient, OllamaError
from comfy_agent.openrouter import ModelRefusal, OpenRouterClient

BASE = "http://ollama.test:11434"


def _show(caps, ctx=8192):
    return {"capabilities": caps, "model_info": {"qwen3.context_length": ctx}}


@respx.mock
def test_list_models_skips_embedding_and_filters_vision():
    respx.get(f"{BASE}/api/tags").respond(json={"models": [
        {"name": "gemma4:e4b"}, {"name": "qwen3-embedding:latest"}, {"name": "qwen3:27b"},
    ]})
    shows = {
        "gemma4:e4b": _show(["completion", "vision", "thinking"]),
        "qwen3-embedding:latest": _show(["embedding"]),
        "qwen3:27b": _show(["completion", "thinking"], 32768),
    }
    respx.post(f"{BASE}/api/show").mock(
        side_effect=lambda req: respx.MockResponse(200, json=shows[json.loads(req.content)["model"]])
    )
    c = OllamaClient(BASE)
    found = c.list_models()
    assert [m.id for m in found] == ["gemma4:e4b", "qwen3:27b"]
    assert found[1].context_length == 32768 and found[0].vision
    assert [m.id for m in c.list_models(need_vision=True)] == ["gemma4:e4b"]


def _reply(content, done_reason="stop"):
    return {"message": {"role": "assistant", "content": content}, "done_reason": done_reason}


@respx.mock
def test_generate_uses_native_chat_with_images_and_think_off():
    route = respx.post(f"{BASE}/api/chat").respond(json=_reply('{"positive": "1girl", "negative": "2girls"}'))
    gp = OllamaClient(BASE, temperature=0.5).generate("gemma4:e4b", "a girl", "data:image/png;base64,AAAA", rating="nsfw")
    assert gp.positive == "1girl"

    body = json.loads(route.calls[0].request.content)
    assert body["think"] is False and body["stream"] is False and body["options"] == {"temperature": 0.5}
    user = body["messages"][1]
    assert user["images"] == ["AAAA"] and "Content rating: NSFW" in user["content"]


@respx.mock
def test_think_field_dropped_for_models_that_reject_it():
    route = respx.post(f"{BASE}/api/chat")
    route.side_effect = [
        respx.MockResponse(400, json={"error": "\"m\" does not support thinking"}),
        respx.MockResponse(200, json=_reply('{"positive": "1girl"}')),
    ]
    OllamaClient(BASE).generate("m", "idea")
    assert "think" not in json.loads(route.calls[1].request.content)


@respx.mock
def test_ollama_refusal_and_errors():
    respx.post(f"{BASE}/api/chat").respond(json=_reply("I'm sorry, but I can't help with that."))
    with pytest.raises(ModelRefusal):
        OllamaClient(BASE).generate("m", "idea")

    respx.post(f"{BASE}/api/chat").respond(404, json={"error": "model 'x' not found"})
    with pytest.raises(OllamaError, match="404"):
        OllamaClient(BASE).generate("x", "idea")


def test_make_writer_picks_provider(settings):
    s = replace(settings, ollama_url=BASE, ollama_default_model="qwen3:27b", openrouter_default_model="or/m")
    assert isinstance(make_writer(s, "ollama"), OllamaClient)
    assert isinstance(make_writer(s, "openrouter"), OpenRouterClient)
    assert default_model(s, "ollama") == "qwen3:27b"
    assert isinstance(make_writer(replace(s, llm_provider="ollama")), OllamaClient)
    with pytest.raises(ValueError):
        make_writer(s, "gpt")
