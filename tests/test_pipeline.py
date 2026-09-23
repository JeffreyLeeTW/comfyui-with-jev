import json

from conftest import BAD, GOOD, MID, FakeSystemOne, all_dims

from comfy_agent.comfyui import OutputImage
from comfy_agent.judge import JevJudge
from comfy_agent.openrouter import GeneratedPrompt
from comfy_agent.pipeline import InputImage, Pipeline


class FakeOpenRouter:
    def __init__(self):
        self.calls = []

    def generate(self, model, idea, image_url, history):
        self.calls.append((model, idea, image_url, list(history)))
        n = len(self.calls)
        return GeneratedPrompt(f"tag{n}, 1girl", f"neg{n}", "desc" if image_url else "", "{}")


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
