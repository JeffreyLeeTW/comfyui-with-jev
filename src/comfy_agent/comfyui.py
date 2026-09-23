"""ComfyUI HTTP client and API-format workflow patching."""

from __future__ import annotations

import copy
import json
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import httpx

from .config import GenParams

Workflow = dict[str, dict]


class ComfyUIError(RuntimeError):
    pass


@dataclass(frozen=True)
class OutputImage:
    filename: str
    subfolder: str
    type: str
    url: str


def load_workflow(path: str | Path) -> Workflow:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nodes_of(wf: Workflow, class_type: str) -> list[str]:
    return [nid for nid, node in wf.items() if node.get("class_type") == class_type]


def _single(wf: Workflow, class_type: str) -> str:
    ids = _nodes_of(wf, class_type)
    if len(ids) != 1:
        raise ComfyUIError(f"workflow must contain exactly one {class_type} node (found {len(ids)})")
    return ids[0]


def _link_source(value) -> str | None:
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        return value[0]
    return None


def _prune_unreachable(wf: Workflow, output_ids: list[str]) -> None:
    """Drop nodes that no output depends on (ComfyUI validates every node it receives)."""
    keep: set[str] = set()
    stack = list(output_ids)
    while stack:
        nid = stack.pop()
        if nid in keep or nid not in wf:
            continue
        keep.add(nid)
        stack.extend(src for v in wf[nid]["inputs"].values() if (src := _link_source(v)))
    for nid in list(wf):
        if nid not in keep:
            del wf[nid]


def build_workflow(
    template: Workflow,
    positive: str,
    negative: str,
    params: GenParams,
    image_name: str | None = None,
) -> tuple[Workflow, int]:
    """Return (patched workflow, seed used).

    image_name set  -> img2img through LoadImage -> VAEEncode -> (LatentUpscale) -> KSampler.
    image_name None -> txt2img from EmptyLatentImage with denoise 1.0.
    """
    wf = copy.deepcopy(template)
    ks_id = _single(wf, "KSampler")
    ks = wf[ks_id]["inputs"]

    pos_id, neg_id = _link_source(ks["positive"]), _link_source(ks["negative"])
    if not pos_id or not neg_id:
        raise ComfyUIError("KSampler positive/negative must be linked to text encoder nodes")
    wf[pos_id]["inputs"]["text"] = positive
    wf[neg_id]["inputs"]["text"] = negative

    seed = params.seed if params.seed >= 0 else random.randint(0, 2**50)
    ks.update(
        seed=seed,
        steps=params.steps,
        cfg=params.cfg,
        sampler_name=params.sampler_name,
        scheduler=params.scheduler,
    )

    if image_name:
        load_id = _single(wf, "LoadImage")
        wf[load_id]["inputs"]["image"] = image_name
        upscale_ids = _nodes_of(wf, "LatentUpscale")
        if upscale_ids:
            latent_id = upscale_ids[0]
            wf[latent_id]["inputs"].update(width=params.width, height=params.height)
        else:
            latent_id = _single(wf, "VAEEncode")
        ks["latent_image"] = [latent_id, 0]
        ks["denoise"] = params.denoise
    else:
        empty_id = _single(wf, "EmptyLatentImage")
        wf[empty_id]["inputs"].update(
            width=params.width, height=params.height, batch_size=params.batch_size
        )
        ks["latent_image"] = [empty_id, 0]
        ks["denoise"] = 1.0

    for nid in _nodes_of(wf, "LoraLoaderModelOnly") + _nodes_of(wf, "LoraLoader"):
        inputs = wf[nid]["inputs"]
        strength = params.lora_strengths.get(inputs.get("lora_name", ""))
        if strength is not None:
            inputs["strength_model"] = strength
            if "strength_clip" in inputs:
                inputs["strength_clip"] = strength

    for nid in _nodes_of(wf, "PreviewImage"):
        wf[nid] = {
            "class_type": "SaveImage",
            "inputs": {"images": wf[nid]["inputs"]["images"], "filename_prefix": params.filename_prefix},
            "_meta": {"title": "Save Image"},
        }
    for nid in _nodes_of(wf, "SaveImage"):
        wf[nid]["inputs"]["filename_prefix"] = params.filename_prefix

    outputs = _nodes_of(wf, "SaveImage")
    if not outputs:
        raise ComfyUIError("workflow has no PreviewImage/SaveImage output node")
    _prune_unreachable(wf, outputs)
    return wf, seed


class ComfyUIClient:
    def __init__(self, base_url: str, http: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=60)
        self.client_id = uuid.uuid4().hex

    def system_stats(self) -> dict:
        resp = self.http.get(f"{self.base_url}/system_stats")
        resp.raise_for_status()
        return resp.json()

    def upload_image(self, data: bytes, filename: str) -> str:
        """Upload to ComfyUI's input folder; return the value LoadImage expects."""
        resp = self.http.post(
            f"{self.base_url}/upload/image",
            files={"image": (filename, data)},
            data={"overwrite": "true", "type": "input"},
        )
        if resp.status_code >= 400:
            raise ComfyUIError(f"image upload failed {resp.status_code}: {resp.text[:300]}")
        info = resp.json()
        return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]

    def queue(self, workflow: Workflow) -> str:
        resp = self.http.post(
            f"{self.base_url}/prompt", json={"prompt": workflow, "client_id": self.client_id}
        )
        if resp.status_code >= 400:
            raise ComfyUIError(f"ComfyUI rejected the workflow {resp.status_code}: {resp.text[:1000]}")
        return resp.json()["prompt_id"]

    def view_url(self, filename: str, subfolder: str = "", type_: str = "output") -> str:
        query = urlencode({"filename": filename, "subfolder": subfolder, "type": type_})
        return f"{self.base_url}/view?{query}"

    def wait(self, prompt_id: str, timeout_s: float = 600, poll_s: float = 2) -> list[OutputImage]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            resp = self.http.get(f"{self.base_url}/history/{prompt_id}")
            resp.raise_for_status()
            entry = resp.json().get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise ComfyUIError(f"generation failed: {json.dumps(status.get('messages', []))[:1000]}")
                if status.get("completed", True):
                    return [
                        OutputImage(
                            filename=img["filename"],
                            subfolder=img.get("subfolder", ""),
                            type=img.get("type", "output"),
                            url=self.view_url(img["filename"], img.get("subfolder", ""), img.get("type", "output")),
                        )
                        for out in entry.get("outputs", {}).values()
                        for img in out.get("images", [])
                        if img.get("type", "output") == "output"
                    ]
            time.sleep(poll_s)
        raise ComfyUIError(f"timed out after {timeout_s:.0f}s waiting for prompt {prompt_id}")
