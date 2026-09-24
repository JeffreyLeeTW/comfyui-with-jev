import json

from conftest import BAD, GOOD, MID, FakeSystemOne, all_dims

from comfy_agent.comfyui import OutputImage
from comfy_agent.judge import JevJudge
from comfy_agent.openrouter import GeneratedPrompt, ModelRefusal
from comfy_agent.pipeline import InputImage, Pipeline


class FakeOpenRouter:
    def __init__(self, refuse_on: int | None = None):
        self.calls = []
        self.refuse_on = refuse_on
        self.ratings = []

    def generate(self, model, idea, image_url, history, rating=None):
        self.calls.append((model, idea, image_url, list(history)))
        self.ratings.append(rating)
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
    assert refusal_counts(settings.runs_dir) == {"m:free": 1}


def test_sfw_rating_forces_tags_and_strips_conflicts(settings):
    fake = FakeSystemOne([all_dims(GOOD)])
    orc, comfy = FakeOpenRouter(), FakeComfy()
    r = Pipeline(settings, orc, JevJudge(fake), comfy).run("a girl", "m:free", rating="sfw")

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
