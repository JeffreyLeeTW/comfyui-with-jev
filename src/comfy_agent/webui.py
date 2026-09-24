"""Gradio WebUI for the same pipeline the CLI runs."""

from __future__ import annotations

import html
import queue
import threading
from dataclasses import replace

import gradio as gr
import httpx

from .comfyui import ComfyUIClient, load_workflow
from .config import (
    DIMENSIONS,
    ENDPOINT_ENV,
    PROVIDERS,
    Settings,
    endpoint_url,
    save_env,
    split_endpoint,
)
from .llm import PROVIDER_LABELS, default_model, make_writer
from .ollama import OllamaClient
from .pipeline import CompareResult, InputImage, build_pipeline
from .refusals import refusal_counts
from .report import export_report

SAMPLERS = ["er_sde", "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "uni_pc"]
SCHEDULERS = ["beta", "normal", "karras", "exponential", "sgm_uniform", "simple"]
TABLE_HEADERS = ["#", *DIMENSIONS, "result", "positive"]
RATING_CHOICES = [("不指定", ""), ("SFW", "sfw"), ("NSFW", "nsfw")]
JUDGE_CHOICES = [("使用 Jev", "on"), ("不用 Jev（評一次分，僅供參考）", "off"), ("A-B 對照（同 seed 各生一組）", "ab")]


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


def _images_html(images, title: str | None = None) -> str:
    if not images:
        return ""
    tags = "".join(
        f'<a href="{html.escape(i.url)}" target="_blank">'
        f'<img src="{html.escape(i.url)}" style="max-width:100%;max-height:640px;margin:4px;border-radius:6px"></a>'
        for i in images
    )
    heading = f"<h3 style='margin:4px'>{html.escape(title)}</h3>" if title else ""
    return f"<div>{heading}{tags}</div>"


def _compare_html(result: CompareResult) -> str:
    a, b = result.with_jev, result.without_jev
    note = "<p>第 1 版就是最後送出的版本，兩組相同，只生了一次圖。</p>" if result.identical else ""
    cols = "".join(
        f"<div style='flex:1 1 320px;min-width:0'>{_images_html(r.images, title)}</div>"
        for title, r in (("有 Jev", a), ("無 Jev（第 1 版）", b))
    )
    return f"{note}<div style='display:flex;flex-wrap:wrap;gap:12px'>{cols}</div>"


def _row(a, reference: bool) -> list:
    v = a.verdict
    if v is None:
        return [a.number, *["–"] * len(DIMENSIONS), "未評分", a.positive]
    verdict = ("參考 " if reference else "") + ("PASS" if v.passed else "FAIL")
    return [a.number, *[f"{v.dimensions[d].value:.2f}" for d in DIMENSIONS], verdict, a.positive]


def _check_endpoints(comfyui_url: str, ollama_url: str) -> str:
    lines = []
    try:
        stats = ComfyUIClient(comfyui_url, http=httpx.Client(timeout=5)).system_stats()
        devices = ", ".join(d.get("name", "?") for d in stats.get("devices", []))
        lines.append(f"✅ ComfyUI {comfyui_url}（{devices or 'no device info'}）")
    except Exception as e:
        lines.append(f"❌ ComfyUI {comfyui_url}：{type(e).__name__}: {e}")
    try:
        lines.append(f"✅ Ollama {ollama_url}（v{OllamaClient(ollama_url, timeout_s=5).version()}）")
    except Exception as e:
        lines.append(f"❌ Ollama {ollama_url}：{type(e).__name__}: {e}")
    return "\n\n".join(lines)


def build_app(settings: Settings) -> gr.Blocks:
    # Endpoints can change from the UI, so settings and clients live in a mutable holder.
    state = {"settings": settings, "writers": {p: make_writer(settings, p) for p in PROVIDERS}}

    def model_choices(provider: str, need_vision: bool) -> list[tuple[str, str]]:
        try:
            found = state["writers"][provider].list_models(need_vision=need_vision)
        except Exception as e:
            gr.Warning(f"Could not list {provider} models: {e}")
            return []
        refused = refusal_counts(state["settings"].runs_dir)
        return [(f"{m.id}  ⚠ 拒絕過 {refused[m.id]} 次" if refused[m.id] else m.id, m.id) for m in found]

    def refresh_models(provider, image_path, current):
        choices = model_choices(provider, image_path is not None)
        ids = [v for _, v in choices]
        preferred = [m for m in (current, default_model(state["settings"], provider)) if m in ids]
        value = preferred[0] if preferred else (ids[0] if ids else None)
        return gr.Dropdown(choices=choices, value=value, label=f"{PROVIDER_LABELS[provider]} 模型")

    def save_endpoints(c_host, c_port, o_host, o_port, provider, image_path, current):
        try:
            urls = {"comfyui": endpoint_url(c_host, c_port), "ollama": endpoint_url(o_host, o_port)}
        except ValueError as e:
            raise gr.Error(f"位址格式錯誤：{e}") from e
        save_env({ENDPOINT_ENV[k]: v for k, v in urls.items()})
        s = replace(state["settings"], comfyui_url=urls["comfyui"], ollama_url=urls["ollama"])
        state["settings"] = s
        state["writers"]["ollama"] = make_writer(s, "ollama")
        msg = f"已儲存到 .env，下一次執行就會使用新位址。\n\n{_check_endpoints(urls['comfyui'], urls['ollama'])}"
        return msg, refresh_models(provider, image_path, current)

    def run(idea, image_path, provider, model, rating, judge_mode, max_attempts, t_fid, t_fmt, t_comp, t_neg,
            width, height, batch_size, seed, steps, cfg, sampler, scheduler, denoise, loras_text):
        if not (idea or "").strip() and not image_path:
            raise gr.Error("請輸入構想文字或上傳圖片（至少一項）")
        if not model:
            raise gr.Error("請選擇模型")

        s = state["settings"]
        gen = s.gen.override(
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
                pipeline = build_pipeline(s, provider)
                args = (idea or "", model, image, gen, thresholds, int(max_attempts))
                kwargs = dict(on_event=events.put, rating=rating or None)
                outcome["result"] = (
                    pipeline.compare(*args, **kwargs) if judge_mode == "ab"
                    else pipeline.run(*args, **kwargs, use_judge=judge_mode == "on")
                )
            except Exception as e:
                outcome["error"] = e
            finally:
                events.put(None)

        threading.Thread(target=worker, daemon=True).start()

        arm_labels = {"with_jev": "（有 Jev）", "without_jev": "（無 Jev）"}
        rows: list[list] = []
        status = "開始…"
        pos = neg = ""
        while (e := events.get()) is not None:
            match e["type"]:
                case "generating":
                    status = f"第 {e['attempt']}/{e['max']} 次：{provider} 產生 prompt 中…"
                case "judging":
                    status = f"第 {e['attempt']} 次：Jev 評分中…" + ("（僅供參考）" if e["reference"] else "")
                case "judged":
                    rows.append(_row(e["attempt"], e["reference"]))
                case "warning":
                    status += f"\n⚠️ {e['message']}"
                case "refused":
                    status = (
                        f"第 {e['attempt']} 次：{e['model']} 拒絕產生 prompt ⛔ 已停止，未生圖\n"
                        f"原因：{e['reason']}"
                    )
                case "chosen":
                    a = e["attempt"]
                    pos, neg = a.positive, a.negative
                    if not e["judge"]:
                        status = "不用 Jev：直接送出第 1 版"
                    elif e["passed"]:
                        status = f"第 {a.number} 次通過 ✅"
                    else:
                        status = f"全部未通過 ⚠️ 使用分數最高的第 {a.number} 次（未達標）"
                case "uploading":
                    status += "\n上傳參考圖到 ComfyUI…"
                case "rendering":
                    arm = arm_labels.get(e["arm"], "")
                    status += f"\nComfyUI 生圖中{arm}… prompt_id={e['prompt_id']} seed={e['seed']}"
            yield status, rows, pos, neg, "", None

        if "error" in outcome:
            raise gr.Error(f"{type(outcome['error']).__name__}: {outcome['error']}")
        result = outcome["result"]
        log = str(result.log_path)
        if result.refused:
            status += f"\n已記錄到 {s.runs_dir / 'refusals.jsonl'}\n紀錄：{log}"
            yield status, rows, pos, neg, "", log
            return
        if isinstance(result, CompareResult):
            a, b = result.with_jev.chosen, result.without_jev.chosen
            pos = f"【有 Jev】\n{a.positive}\n\n【無 Jev（第 1 版）】\n{b.positive}"
            neg = f"【有 Jev】\n{a.negative}\n\n【無 Jev（第 1 版）】\n{b.negative}"
            images_html = _compare_html(result)
            all_images = result.with_jev.images + ([] if result.identical else result.without_jev.images)
        else:
            images_html = _images_html(result.images)
            all_images = result.images
        files = ", ".join(i.filename for i in all_images)
        status += f"\n完成：圖片存在 server output/（{files}）\n紀錄：{log}"
        yield status, rows, pos, neg, images_html, log

    def export(log_path):
        if not log_path:
            raise gr.Error("還沒有可以匯出的結果，先執行一次")
        return str(export_report(log_path))

    g = settings.gen
    c_host, c_port = split_endpoint(settings.comfyui_url)
    o_host, o_port = split_endpoint(settings.ollama_url)
    with gr.Blocks(title="ComfyUI Prompt Agent") as app:
        gr.Markdown("## ComfyUI Prompt Agent\n構想 → OpenRouter／Ollama 產生 prompt → Jev 評審（可選，不及格退回重寫）→ ComfyUI 生圖")
        last_log = gr.State(None)
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
                judge_mode = gr.Radio(JUDGE_CHOICES, value="on", label="Jev 評審")
                with gr.Accordion("Jev 門檻", open=False):
                    max_attempts = gr.Slider(1, 10, value=settings.max_attempts, step=1, label="最多嘗試次數（不用 Jev 時固定 1 次）")
                    t = [gr.Slider(0, 1, value=settings.thresholds[d], step=0.01, label=d) for d in DIMENSIONS]
                with gr.Accordion("生圖參數", open=False):
                    with gr.Row():
                        width = gr.Number(value=g.width, label="width", precision=0)
                        height = gr.Number(value=g.height, label="height", precision=0)
                        batch_size = gr.Number(value=g.batch_size, label="batch (txt2img)", precision=0)
                    with gr.Row():
                        seed = gr.Number(value=g.seed, label="seed (-1 隨機；A-B 對照時兩組共用)", precision=0)
                        steps = gr.Number(value=g.steps, label="steps", precision=0)
                        cfg = gr.Number(value=g.cfg, label="cfg")
                    with gr.Row():
                        sampler = gr.Dropdown(sorted({g.sampler_name, *SAMPLERS}), value=g.sampler_name, label="sampler", allow_custom_value=True)
                        scheduler = gr.Dropdown(sorted({g.scheduler, *SCHEDULERS}), value=g.scheduler, label="scheduler", allow_custom_value=True)
                    denoise = gr.Slider(0, 1, value=g.denoise, step=0.01, label="denoise（僅 img2img）")
                    loras = gr.Textbox(value=_workflow_loras(settings), lines=3, label="LoRA 強度（name=strength，每行一個）")
                with gr.Accordion("連線設定", open=False):
                    with gr.Row():
                        comfy_host = gr.Textbox(value=c_host, label="ComfyUI IP／主機", scale=3)
                        comfy_port = gr.Number(value=c_port, label="port", precision=0, scale=1)
                    with gr.Row():
                        ollama_host = gr.Textbox(value=o_host, label="Ollama IP／主機", scale=3)
                        ollama_port = gr.Number(value=o_port, label="port", precision=0, scale=1)
                    save_conn = gr.Button("儲存並測試連線")
                    conn_status = gr.Markdown()
                go = gr.Button("開始", variant="primary")
            with gr.Column(scale=2):
                status = gr.Textbox(label="狀態", lines=4, interactive=False)
                table = gr.Dataframe(headers=TABLE_HEADERS, label="每一輪 Jev 評分", wrap=True, interactive=False)
                final_pos = gr.Textbox(label="送出的 positive prompt", lines=3, interactive=False)
                final_neg = gr.Textbox(label="送出的 negative prompt", lines=2, interactive=False)
                images = gr.HTML(label="結果（從 server 讀取）")
                with gr.Row():
                    export_btn = gr.Button("匯出報告（HTML）")
                    report_file = gr.File(label="報告", interactive=False)

        model_inputs = [provider, image, model]
        refresh.click(refresh_models, model_inputs, model)
        image.change(refresh_models, model_inputs, model)
        provider.change(refresh_models, model_inputs, model)
        app.load(refresh_models, model_inputs, model)
        save_conn.click(
            save_endpoints, [comfy_host, comfy_port, ollama_host, ollama_port, *model_inputs], [conn_status, model]
        )
        go.click(
            run,
            [idea, image, provider, model, rating, judge_mode, max_attempts, *t, width, height, batch_size, seed,
             steps, cfg, sampler, scheduler, denoise, loras],
            [status, table, final_pos, final_neg, images, last_log],
        )
        export_btn.click(export, last_log, report_file)
    return app
