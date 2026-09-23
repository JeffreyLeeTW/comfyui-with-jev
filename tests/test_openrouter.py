import json

import pytest
import respx

from comfy_agent.openrouter import (
    Feedback,
    OpenRouterClient,
    OpenRouterError,
    merge_tags,
    parse_prompt_json,
)

BASE = "https://openrouter.ai/api/v1"


def test_parse_prompt_json_tolerates_fences_and_chatter():
    assert parse_prompt_json('```json\n{"positive": "a, b", "negative": ""}\n```')["positive"] == "a, b"
    assert parse_prompt_json('Sure! {"positive": "x"} hope this helps')["positive"] == "x"
    with pytest.raises(ValueError):
        parse_prompt_json("no json here")
    with pytest.raises(ValueError):
        parse_prompt_json('{"positive": ""}')


def test_merge_tags_dedupes_case_insensitively():
    assert merge_tags("masterpiece, Best Quality", "best quality, 1girl,\nsolo, ") == "masterpiece, Best Quality, 1girl, solo"


@respx.mock
def test_list_free_models_filters_price_and_vision():
    respx.get(f"{BASE}/models").respond(json={"data": [
        {"id": "a:free", "pricing": {"prompt": "0", "completion": "0"},
         "architecture": {"input_modalities": ["text", "image"]}},
        {"id": "b:free", "pricing": {"prompt": "0", "completion": "0"},
         "architecture": {"input_modalities": ["text"]}},
        {"id": "paid", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
    ]})
    c = OpenRouterClient("k")
    assert [m.id for m in c.list_free_models()] == ["a:free", "b:free"]
    assert [m.id for m in c.list_free_models(need_vision=True)] == ["a:free"]


def _reply(content):
    return {"choices": [{"message": {"content": content}}]}


@respx.mock
def test_generate_sends_image_and_feedback_and_retries_bad_json():
    route = respx.post(f"{BASE}/chat/completions")
    route.side_effect = [
        respx.MockResponse(200, json=_reply("I cannot format that")),
        respx.MockResponse(200, json=_reply('{"positive": "1girl", "negative": "2girls", "image_description": "a girl"}')),
    ]
    c = OpenRouterClient("k")
    gp = c.generate(
        "m:free", "a girl", "data:image/png;base64,AAAA",
        [Feedback("old", "oldneg", "- fidelity: too vague")],
    )
    assert (gp.positive, gp.negative, gp.image_description) == ("1girl", "2girls", "a girl")

    first = json.loads(route.calls[0].request.content)
    user = first["messages"][1]["content"]
    assert user[1]["image_url"]["url"].startswith("data:image/png")
    assert "fidelity: too vague" in user[0]["text"]
    assert route.calls[0].request.headers["authorization"] == "Bearer k"
    assert len(json.loads(route.calls[1].request.content)["messages"]) == 4


@respx.mock
def test_generate_raises_on_http_error():
    respx.post(f"{BASE}/chat/completions").respond(400, json={"error": "bad model"})
    with pytest.raises(OpenRouterError, match="400"):
        OpenRouterClient("k").generate("m", "idea")


def test_missing_key_raises():
    with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY"):
        OpenRouterClient("").generate("m", "idea")
