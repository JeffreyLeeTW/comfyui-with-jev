import json

import pytest

from conftest import BAD, GOOD, MID, FakeSystemOne, all_dims

from comfy_agent.comfyui import OutputImage
from comfy_agent.judge import JevJudge
from comfy_agent.openrouter import GeneratedPrompt, ModelRefusal
from comfy_agent.pipeline import InputImage, Pipeline


class FakeOpenRouter:
    name = "fake"

    def __init__(self, refuse_on: int | None = None):
        self.calls = []
        self.refuse_on = refuse_on
        self.ratings = []
        self.styles = []

    def generate(self, model, idea, image_url, history, rating=None, style="tags"):
        self.calls.append((model, idea, image_url, list(history)))
        self.ratings.append(rating)
        self.styles.append(style)
        n = len(self.calls)
        if n == self.refuse_on:
            raise ModelRefusal(model, "explicit content")
        return GeneratedPrompt(f"tag{n}, 1girl, Nude", f"neg{n}, safe", "desc" if image_url else "", "{}")


class FakeComfy:
    def __init__(self):
        self.queued = []
        self.uploaded = []

    def upload_image(self, data, filename):
        self.uploaded.append(filename)
        return filename

    def queue(self, wf):
        self.queued.append(wf)
        return "pid"

    def wait(self, prompt_id, timeout_s, poll_s):
        return [OutputImage("out.png", "", "output", "http://x/view")]


def make(settings, calls):
    orc, comfy = FakeOpenRouter(), FakeComfy()
    return Pipeline(settings, orc, JevJudge(FakeSystemOne(calls)), comfy), orc, comfy


def test_passes_on_second_attempt_and_feeds_back_critique(settings):
    p, orc, comfy = make(settings, [all_dims(BAD), all_dims(GOOD)])
    r = p.run("a girl", "m:free")

    assert r.passed and r.chosen.number == 2 and len(r.attempts) == 2
    assert orc.calls[0][3] == [] and "fidelity" in orc.calls[1][3][0].critique
    wf = comfy.queued[0]
    assert wf["6"]["inputs"]["text"].startswith("masterpiece, best quality")
    assert "tag2" in wf["6"]["inputs"]["text"]
    assert wf["4"]["inputs"]["latent_image"] == ["3", 0]  # txt2img
    log = json.loads(r.log_path.read_text())
    assert log["passed"] and log["chosen_attempt"] == 2 and len(log["attempts"]) == 2
    assert log["provider"] == "fake"


def test_after_max_attempts_sends_best_marked_not_passed(settings):
    calls = [all_dims(BAD), all_dims(MID) | {"negative": BAD}, all_dims(MID) | {"format": [0.1, 0.4, 0.5, 0.0]},
             all_dims(BAD), all_dims(BAD)]
    p, orc, comfy = make(settings, calls)
    events = []
    r = p.run("a girl", "m:free", max_attempts=5, on_event=events.append)

    assert not r.passed and len(r.attempts) == 5 and len(orc.calls) == 5
    assert r.chosen.number == 3  # weakest dimension highest among attempts
    assert len(comfy.queued) == 1
    chosen = [e for e in events if e["type"] == "chosen"][0]
    assert chosen["passed"] is False


def test_image_input_uses_img2img(settings):
    p, orc, comfy = make(settings, [all_dims(GOOD)])
    r = p.run("", "m:free", image=InputImage(b"\x89PNG", "ref.png"))
    assert r.mode == "img2img" and comfy.uploaded == ["ref.png"]
    assert orc.calls[0][2].startswith("data:image/png;base64,")
    assert comfy.queued[0]["4"]["inputs"]["latent_image"] == ["16", 0]


def test_no_render(settings):
    p, _, comfy = make(settings, [all_dims(GOOD)])
    r = p.run("x", "m", render=False)
    assert r.passed and not comfy.queued and r.images == []


def test_refusal_stops_run_without_render_and_is_recorded(settings):
    from comfy_agent.refusals import refusal_counts

    orc, comfy = FakeOpenRouter(refuse_on=2), FakeComfy()
    p = Pipeline(settings, orc, JevJudge(FakeSystemOne([all_dims(BAD)])), comfy)
    events = []
    r = p.run("a girl", "m:free", max_attempts=5, on_event=events.append)

    assert r.refused == "explicit content" and not r.passed and r.chosen is None
    assert len(orc.calls) == 2 and len(r.attempts) == 1 and comfy.queued == []
    assert [e["type"] for e in events][-2:] == ["refused", "done"]
    assert json.loads(r.log_path.read_text())["refused"] == "explicit content"

    entry = json.loads((settings.runs_dir / "refusals.jsonl").read_text().splitlines()[0])
    assert entry["model"] == "m:free" and entry["attempt"] == 2 and entry["mode"] == "txt2img"
    assert entry["provider"] == "fake"
    assert refusal_counts(settings.runs_dir) == {"m:free": 1}


def test_sfw_rating_forces_tags_and_strips_conflicts(settings):
    fake = FakeSystemOne([all_dims(GOOD)])
    orc, comfy = FakeOpenRouter(), FakeComfy()
    r = Pipeline(settings, orc, JevJudge(fake), comfy).run("a girl", "m:free", rating="sfw", style="tags")

    a = r.chosen
    assert orc.ratings == ["sfw"] and r.rating == "sfw"
    assert "safe" in a.positive.split(", ") and "nude" not in a.positive.lower()
    assert {"nsfw", "explicit", "nude"} <= set(a.negative.split(", ")) and "safe" not in a.negative.split(", ")
    assert fake.requests[0][0]["content_rating"].startswith("SFW")
    assert json.loads(r.log_path.read_text())["rating"] == "sfw"


def test_nsfw_rating_and_unset_rating(settings):
    fake = FakeSystemOne([all_dims(GOOD), all_dims(GOOD)])
    orc, comfy = FakeOpenRouter(), FakeComfy()
    p = Pipeline(settings, orc, JevJudge(fake), comfy)

    r = p.run("a girl", "m:free", rating="nsfw")
    assert set(r.chosen.positive.split(", ")) >= {"nsfw", "explicit", "Nude"}

    r = p.run("a girl", "m:free")
    assert "nsfw" not in r.chosen.positive and "safe" in r.chosen.negative  # untouched
    assert "not specified" in fake.requests[1][0]["content_rating"]


class ProseWriter(FakeOpenRouter):
    def generate(self, model, idea, image_url, history, rating=None, style="tags"):
        super().generate(model, idea, image_url, history, rating, style)
        return GeneratedPrompt("A  girl stands\nin the rain, nude.", "Extra people.", "", "{}")


def test_natural_style_adds_prefix_tags_and_keeps_prose(settings):
    fake = FakeSystemOne([all_dims(GOOD)])
    orc, comfy = ProseWriter(), FakeComfy()
    r = Pipeline(settings, orc, JevJudge(fake), comfy).run("a girl", "m:free", rating="sfw", style="natural")

    a = r.chosen
    assert orc.styles == ["natural"] and r.style == "natural"
    assert a.positive.startswith("masterpiece, best quality") and a.positive.endswith(", safe, A girl stands in the rain, nude.")
    assert a.negative.startswith("worst quality") and a.negative.endswith("sex, Extra people.")
    state, questions, _ = fake.requests[0]
    assert "natural-language" in state["notes"] and "fluent" in questions["format"].instructions
    assert r.chosen.verdict.style == "natural"
    assert json.loads(r.log_path.read_text())["prompt_style"] == "natural"


def test_style_defaults_to_config_and_is_validated(settings):
    from dataclasses import replace

    p, orc, _ = make(replace(settings, prompt_style="tags"), [all_dims(GOOD)])
    assert p.run("a girl", "m").style == "tags" and orc.styles == ["tags"]
    with pytest.raises(ValueError, match="style"):
        p.run("a girl", "m", style="haiku")


def test_without_jev_sends_first_draft_and_scores_for_reference(settings):
    p, orc, comfy = make(settings, [all_dims(BAD)])
    events = []
    r = p.run("a girl", "m:free", max_attempts=5, use_judge=False, on_event=events.append)

    assert len(orc.calls) == 1 and r.judge_mode == "off" and r.chosen.number == 1
    assert r.chosen.verdict is not None and not r.passed and len(comfy.queued) == 1
    assert [e["reference"] for e in events if e["type"] == "judged"] == [True]
    log = json.loads(r.log_path.read_text())
    assert log["kind"] == "run" and log["judge_mode"] == "off" and log["attempts"][0]["scores"]


def test_without_jev_survives_missing_or_failing_judge(settings):
    orc, comfy = FakeOpenRouter(), FakeComfy()
    events = []
    r = Pipeline(settings, orc, None, comfy).run("a girl", "m:free", use_judge=False, on_event=events.append)
    assert r.chosen.verdict is None and len(comfy.queued) == 1
    assert any(e["type"] == "warning" for e in events)

    class Broken:
        def evaluate(self, *a, **k):
            raise RuntimeError("quota")

    r = Pipeline(settings, FakeOpenRouter(), Broken(), FakeComfy()).run("a girl", "m:free", use_judge=False)
    assert r.chosen.verdict is None and r.images


def test_jev_modes_require_a_judge(settings):
    p = Pipeline(settings, FakeOpenRouter(), None, FakeComfy())
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        p.run("a girl", "m:free")
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        p.compare("a girl", "m:free")


def test_compare_renders_both_arms_with_same_seed(settings):
    p, orc, comfy = make(settings, [all_dims(BAD), all_dims(GOOD)])
    events = []
    r = p.compare("a girl", "m:free", on_event=events.append)

    assert len(orc.calls) == 2 and not r.identical
    assert r.with_jev.chosen.number == 2 and r.without_jev.chosen.number == 1
    assert r.without_jev.judge_mode == "off" and r.without_jev.chosen.verdict is r.with_jev.attempts[0].verdict
    assert len(comfy.queued) == 2
    seeds = {wf["4"]["inputs"]["seed"] for wf in comfy.queued}
    assert len(seeds) == 1 and r.with_jev.seed == r.without_jev.seed
    assert "tag2" in comfy.queued[0]["6"]["inputs"]["text"] and "tag1" in comfy.queued[1]["6"]["inputs"]["text"]
    assert [e["arm"] for e in events if e["type"] == "rendering"] == ["with_jev", "without_jev"]
    log = json.loads(r.log_path.read_text())
    assert log["kind"] == "compare" and log["with_jev"]["chosen_attempt"] == 2
    assert log["without_jev"]["attempts"][0]["number"] == 1


def test_compare_identical_when_first_draft_passes(settings):
    p, orc, comfy = make(settings, [all_dims(GOOD)])
    r = p.compare("a girl", "m:free")
    assert r.identical and len(comfy.queued) == 1
    assert r.without_jev.images == r.with_jev.images and r.without_jev.passed


def test_compare_refusal_renders_nothing(settings):
    orc, comfy = FakeOpenRouter(refuse_on=2), FakeComfy()
    p = Pipeline(settings, orc, JevJudge(FakeSystemOne([all_dims(BAD)])), comfy)
    r = p.compare("a girl", "m:free")
    assert r.refused and comfy.queued == []
    assert json.loads(r.log_path.read_text())["with_jev"]["refused"]
