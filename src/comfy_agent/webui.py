"""Gradio WebUI for the same pipeline the CLI runs."""

from __future__ import annotations

import html
import queue
import threading

import gradio as gr

from .comfyui import load_workflow
from .config import DIMENSIONS, PROVIDERS, Settings
from .llm import PROVIDER_LABELS, default_model, make_writer
from .pipeline import InputImage, build_pipeline
from .refusals import refusal_counts

SAMPLERS = ["er_sde", "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "uni_pc"]
SCHEDULERS = ["beta", "normal", "karras", "exponential", "sgm_uniform", "simple"]
TABLE_HEADERS = ["#", *DIMENSIONS, "result", "positive"]
RATING_CHOICES = [("不指定", ""), ("SFW", "sfw"), ("NSFW", "nsfw")]


def _workflow_loras(settings: Settings) -> str:
    """Prefill 'name=strength' lines from the workflow, overlaid with config values."""
    lines = []
    for node in load_workflow(settings.workflow_path).values():
        if node.get("class_type") in ("LoraLoaderModelOnly", "LoraLoader"):
            name = node["inputs"]["lora_name"]
            strength = settings.gen.lora_strengths.get(name, node["inputs"].get("strength_model", 1.0))
            lines.append(f"{name}={strength}")
    return "\n".join(lines)


def _parse_loras(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        name, sep, strength = line.strip().rpartition("=")
        if sep and name:
            out[name.strip()] = float(strength)
    return out


def _images_html(images) -> str:
    if not images:
        return ""
    tags = "".join(
        f'<a href="{html.escape(i.url)}" target="_blank">'
        f'<img src="{html.escape(i.url)}" style="max-width:100%;max-height:640px;margin:4px;border-radius:6px"></a>'
        for i in images
    )
    return f"<div>{tags}</div>"


def build_app(settings: Settings) -> gr.Blocks:
    writers = {p: make_writer(settings, p) for p in PROVIDERS}

    def model_choices(provider: str, need_vision: bool) -> list[tuple[str, str]]:
        try:
            found = writers[provider].list_models(need_vision=need_vision)
        except Exception as e:
            gr.Warning(f"Could not list {provider} models: {e}")
            return []
        refused = refusal_counts(settings.runs_dir)
        return [(f"{m.id}  ⚠ 拒絕過 {refused[m.id]} 次" if refused[m.id] else m.id, m.id) for m in found]

    def refresh_models(provider, image_path, current):
        choices = model_choices(provider, image_path is not None)
        ids = [v for _, v in choices]
        preferred = [m for m in (current, default_model(settings, provider)) if m in ids]
        value = preferred[0] if preferred else (ids[0] if ids else None)
        return gr.Dropdown(choices=choices, value=value, label=f"{PROVIDER_LABELS[provider]} 模型")

    def run(idea, image_path, provider, model, rating, max_attempts, t_fid, t_fmt, t_comp, t_neg,
            width, height, batch_size, seed, steps, cfg, sampler, scheduler, denoise, loras_text):
        if not (idea or "").strip() and not image_path:
            raise gr.Error("請輸入構想文字或上傳圖片（至少一項）")
        if not model:
            raise gr.Error("請選擇模型")

        gen = settings.gen.override(
            width=int(width), height=int(height), batch_size=int(batch_size), seed=int(seed),
            steps=int(steps), cfg=float(cfg), sampler_name=sampler, scheduler=scheduler,
            denoise=float(denoise), lora_strengths=_parse_loras(loras_text),
        )
        thresholds = dict(zip(DIMENSIONS, (t_fid, t_fmt, t_comp, t_neg)))
        image = InputImage.from_path(image_path) if image_path else None

        events: queue.Queue = queue.Queue()
        outcome: dict = {}

        def worker():
            try:
                pipeline = build_pipeline(settings, provider)
                outcome["result"] = pipeline.run(
                    idea or "", model, image, gen, thresholds, int(max_attempts),
                    on_event=events.put, rating=rating or None,
                )
            except Exception as e:
                outcome["error"] = e
            finally:
                events.put(None)

        threading.Thread(target=worker, daemon=True).start()

        rows: list[list] = []
        status = "開始…"
        pos = neg = ""
        while (e := events.get()) is not None:
            match e["type"]:
                case "generating":
                    status = f"第 {e['attempt']}/{e['max']} 次：{provider} 產生 prompt 中…"
                case "judging":
                    status = f"第 {e['attempt']} 次：Jev 評分中…"
                case "judged":
                    a = e["attempt"]
                    rows.append(
                        [a.number]
                        + [f"{a.verdict.dimensions[d].value:.2f}" for d in DIMENSIONS]
                        + ["PASS" if a.verdict.passed else "FAIL", a.positive]
                    )
                case "refused":
                    status = (
                        f"第 {e['attempt']} 次：{e['model']} 拒絕產生 prompt ⛔ 已停止，未生圖\n"
                        f"原因：{e['reason']}"
                    )
                case "chosen":
                    a = e["attempt"]
                    pos, neg = a.positive, a.negative
                    status = (
                        f"第 {a.number} 次通過 ✅" if e["passed"]
                        else f"全部未通過 ⚠️ 使用分數最高的第 {a.number} 次（未達標）"
                    )
                case "uploading":
                    status += "\n上傳參考圖到 ComfyUI…"
                case "rendering":
                    status += f"\nComfyUI 生圖中… prompt_id={e['prompt_id']} seed={e['seed']}"
            yield status, rows, pos, neg, ""

        if "error" in outcome:
            raise gr.Error(f"{type(outcome['error']).__name__}: {outcome['error']}")
        result = outcome["result"]
        if result.refused:
            status += f"\n已記錄到 {settings.runs_dir / 'refusals.jsonl'}\n紀錄：{result.log_path}"
            yield status, rows, pos, neg, ""
            return
        files = ", ".join(i.filename for i in result.images)
        status += f"\n完成：圖片存在 server output/（{files}）\n紀錄：{result.log_path}"
        yield status, rows, pos, neg, _images_html(result.images)

    g = settings.gen
    with gr.Blocks(title="ComfyUI Prompt Agent") as app:
        gr.Markdown("## ComfyUI Prompt Agent\n構想 → OpenRouter／Ollama 產生 prompt → Jev 評審（不及格退回重寫）→ ComfyUI 生圖")
        with gr.Row():
            with gr.Column(scale=1):
                idea = gr.Textbox(label="構想", lines=4, placeholder="描述你想要的畫面（可搭配參考圖）")
                image = gr.Image(label="參考圖（有圖 → img2img）", type="filepath")
                provider = gr.Radio(
                    [(PROVIDER_LABELS[p], p) for p in PROVIDERS], value=settings.llm_provider, label="Prompt 產生器",
                )
                with gr.Row():
                    model = gr.Dropdown(
                        label=f"{PROVIDER_LABELS[settings.llm_provider]} 模型", choices=[],
                        value=default_model(settings) or None, allow_custom_value=True, scale=4,
                    )
                    refresh = gr.Button("↻", scale=1, min_width=40)
                rating = gr.Radio(RATING_CHOICES, value="", label="內容分級（加入對應的 positive／negative tag）")
                with gr.Accordion("Jev 門檻", open=False):
                    max_attempts = gr.Slider(1, 10, value=settings.max_attempts, step=1, label="最多嘗試次數")
                    t = [gr.Slider(0, 1, value=settings.thresholds[d], step=0.01, label=d) for d in DIMENSIONS]
                with gr.Accordion("生圖參數", open=False):
                    with gr.Row():
                        width = gr.Number(value=g.width, label="width", precision=0)
                        height = gr.Number(value=g.height, label="height", precision=0)
                        batch_size = gr.Number(value=g.batch_size, label="batch (txt2img)", precision=0)
                    with gr.Row():
                        seed = gr.Number(value=g.seed, label="seed (-1 隨機)", precision=0)
                        steps = gr.Number(value=g.steps, label="steps", precision=0)
                        cfg = gr.Number(value=g.cfg, label="cfg")
                    with gr.Row():
                        sampler = gr.Dropdown(sorted({g.sampler_name, *SAMPLERS}), value=g.sampler_name, label="sampler", allow_custom_value=True)
                        scheduler = gr.Dropdown(sorted({g.scheduler, *SCHEDULERS}), value=g.scheduler, label="scheduler", allow_custom_value=True)
                    denoise = gr.Slider(0, 1, value=g.denoise, step=0.01, label="denoise（僅 img2img）")
                    loras = gr.Textbox(value=_workflow_loras(settings), lines=3, label="LoRA 強度（name=strength，每行一個）")
                go = gr.Button("開始", variant="primary")
            with gr.Column(scale=2):
                status = gr.Textbox(label="狀態", lines=4, interactive=False)
                table = gr.Dataframe(headers=TABLE_HEADERS, label="每一輪 Jev 評分", wrap=True, interactive=False)
                final_pos = gr.Textbox(label="送出的 positive prompt", lines=3, interactive=False)
                final_neg = gr.Textbox(label="送出的 negative prompt", lines=2, interactive=False)
                images = gr.HTML(label="結果（從 server 讀取）")

        refresh.click(refresh_models, [provider, image, model], model)
        image.change(refresh_models, [provider, image, model], model)
        provider.change(refresh_models, [provider, image, model], model)
        app.load(refresh_models, [provider, image, model], model)
        go.click(
            run,
            [idea, image, provider, model, rating, max_attempts, *t, width, height, batch_size, seed, steps, cfg,
             sampler, scheduler, denoise, loras],
            [status, table, final_pos, final_neg, images],
        )
    return app
