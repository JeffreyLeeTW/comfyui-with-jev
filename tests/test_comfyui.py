import httpx
import pytest
import respx

from comfy_agent.comfyui import ComfyUIClient, ComfyUIError, build_workflow, load_workflow
from comfy_agent.config import GenParams


@pytest.fixture
def template(settings):
    return load_workflow(settings.workflow_path)


def test_txt2img_uses_empty_latent_and_prunes_image_chain(template):
    params = GenParams(width=512, height=768, batch_size=2, seed=42, steps=20, cfg=4.0)
    wf, seed = build_workflow(template, "1girl, solo", "lowres", params)

    assert seed == 42
    ks = wf["4"]["inputs"]
    assert ks["latent_image"] == ["3", 0]
    assert ks["denoise"] == 1.0
    assert (ks["seed"], ks["steps"], ks["cfg"]) == (42, 20, 4.0)
    assert wf["3"]["inputs"] == {"width": 512, "height": 768, "batch_size": 2}
    assert wf["6"]["inputs"]["text"] == "1girl, solo"
    assert wf["5"]["inputs"]["text"] == "lowres"
    for removed in ("14", "15", "16"):
        assert removed not in wf


def test_img2img_uses_uploaded_image_and_upscale(template):
    params = GenParams(width=640, height=960, seed=1, denoise=0.55)
    wf, _ = build_workflow(template, "p", "n", params, image_name="ref.png")

    assert wf["14"]["inputs"]["image"] == "ref.png"
    assert wf["4"]["inputs"]["latent_image"] == ["16", 0]
    assert wf["4"]["inputs"]["denoise"] == 0.55
    assert wf["16"]["inputs"]["width"] == 640 and wf["16"]["inputs"]["height"] == 960
    assert "3" not in wf  # unused EmptyLatentImage pruned


def test_preview_replaced_with_save_image(template):
    wf, _ = build_workflow(template, "p", "n", GenParams(filename_prefix="xyz"))
    assert wf["2"]["class_type"] == "SaveImage"
    assert wf["2"]["inputs"] == {"images": ["1", 0], "filename_prefix": "xyz"}
    assert not any(n["class_type"] == "PreviewImage" for n in wf.values())


def test_random_seed_and_lora_strength(template):
    params = GenParams(seed=-1, lora_strengths={"deadpussy_epoch24.safetensors": 0.3})
    wf, seed = build_workflow(template, "p", "n", params)
    assert seed >= 0 and wf["4"]["inputs"]["seed"] == seed
    assert wf["13"]["inputs"]["strength_model"] == 0.3
    assert wf["12"]["inputs"]["strength_model"] == 0.9  # untouched


def test_template_not_mutated(template):
    before = repr(template)
    build_workflow(template, "p", "n", GenParams())
    assert repr(template) == before


@respx.mock
def test_client_queue_and_wait():
    base = "http://comfy:8188"
    respx.post(f"{base}/prompt").respond(json={"prompt_id": "abc"})
    respx.get(f"{base}/history/abc").mock(side_effect=[
        httpx.Response(200, json={}),
        httpx.Response(200, json={"abc": {
            "status": {"status_str": "success", "completed": True},
            "outputs": {"2": {"images": [{"filename": "a_00001_.png", "subfolder": "", "type": "output"}]}},
        }}),
    ])
    c = ComfyUIClient(base)
    assert c.queue({"x": {}}) == "abc"
    imgs = c.wait("abc", timeout_s=5, poll_s=0)
    assert [i.filename for i in imgs] == ["a_00001_.png"]
    assert imgs[0].url == f"{base}/view?filename=a_00001_.png&subfolder=&type=output"


@respx.mock
def test_client_reports_errors():
    base = "http://comfy:8188"
    respx.post(f"{base}/prompt").respond(400, json={"error": "bad node"})
    respx.get(f"{base}/history/p").respond(json={"p": {"status": {"status_str": "error", "messages": []}}})
    c = ComfyUIClient(base)
    with pytest.raises(ComfyUIError, match="rejected"):
        c.queue({})
    with pytest.raises(ComfyUIError, match="failed"):
        c.wait("p", timeout_s=5, poll_s=0)


@respx.mock
def test_upload_image_returns_subfolder_path():
    base = "http://comfy:8188"
    respx.post(f"{base}/upload/image").respond(json={"name": "r.png", "subfolder": "s", "type": "input"})
    assert ComfyUIClient(base).upload_image(b"x", "r.png") == "s/r.png"
