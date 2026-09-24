"""Gradio WebUI for the same pipeline the CLI runs. Labels and messages come from i18n."""

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
    JUDGE_MODES,
    PROVIDERS,
    Settings,
    endpoint_url,
    save_env,
    split_endpoint,
)
from .i18n import LANG_LABELS, LANGS, normalize_lang, t
from .llm import default_model, make_writer
from .ollama import OllamaClient
from .pipeline import CompareResult, InputImage, build_pipeline
from .refusals import refusal_counts
from .report import export_report

SAMPLERS = ["er_sde", "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "uni_pc"]
SCHEDULERS = ["beta", "normal", "karras", "exponential", "sgm_uniform", "simple"]
TABLE_HEADERS = ["#", *DIMENSIONS, "result", "positive"]


def _provider_choices(lang: str) -> list[tuple[str, str]]:
    return [(t(f"provider.{p}", lang), p) for p in PROVIDERS]


def _rating_choices(lang: str) -> list[tuple[str, str]]:
    return [(t("rating.none", lang), ""), ("SFW", "sfw"), ("NSFW", "nsfw")]


def _judge_choices(lang: str) -> list[tuple[str, str]]:
    return [(t(f"judge.{m}", lang), m) for m in JUDGE_MODES]


def _model_label(provider: str, lang: str) -> str:
    return t("ui.model", lang, provider=t(f"provider.{provider}", lang))


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


def _compare_html(result: CompareResult, lang: str) -> str:
    note = f"<p>{html.escape(t('msg.identical', lang))}</p>" if result.identical else ""
    cols = "".join(
        f"<div style='flex:1 1 320px;min-width:0'>{_images_html(r.images, t(key, lang))}</div>"
        for key, r in (("arm.with_jev", result.with_jev), ("arm.without_jev", result.without_jev))
    )
    return f"{note}<div style='display:flex;flex-wrap:wrap;gap:12px'>{cols}</div>"


def _row(a, reference: bool, lang: str) -> list:
    v = a.verdict
    if v is None:
        return [a.number, *["–"] * len(DIMENSIONS), t("msg.unscored", lang), a.positive]
    verdict = (t("msg.reference_prefix", lang) if reference else "") + ("PASS" if v.passed else "FAIL")
    return [a.number, *[f"{v.dimensions[d].value:.2f}" for d in DIMENSIONS], verdict, a.positive]


def _check_endpoints(comfyui_url: str, ollama_url: str, lang: str) -> str:
    lines = []
    try:
        stats = ComfyUIClient(comfyui_url, http=httpx.Client(timeout=5)).system_stats()
        devices = ", ".join(d.get("name", "?") for d in stats.get("devices", []))
        lines.append(f"✅ ComfyUI {comfyui_url} ({devices or t('msg.no_device', lang)})")
    except Exception as e:
        lines.append(f"❌ ComfyUI {comfyui_url}: {type(e).__name__}: {e}")
    try:
        lines.append(f"✅ Ollama {ollama_url} (v{OllamaClient(ollama_url, timeout_s=5).version()})")
    except Exception as e:
        lines.append(f"❌ Ollama {ollama_url}: {type(e).__name__}: {e}")
    return "\n\n".join(lines)


def build_app(settings: Settings) -> gr.Blocks:
    # Endpoints can change from the UI, so settings and clients live in a mutable holder.
    state = {"settings": settings, "writers": {p: make_writer(settings, p) for p in PROVIDERS}}
    lang0 = normalize_lang(settings.language)

    def model_choices(provider: str, need_vision: bool, lang: str) -> list[tuple[str, str]]:
        try:
            found = state["writers"][provider].list_models(need_vision=need_vision)
        except Exception as e:
            gr.Warning(t("msg.list_failed", lang, provider=provider, error=e))
            return []
        refused = refusal_counts(state["settings"].runs_dir)
        return [
            (f"{m.id}  {t('msg.refused_count', lang, n=refused[m.id])}" if refused[m.id] else m.id, m.id)
            for m in found
        ]

    def refresh_models(provider, image_path, current, lang):
        choices = model_choices(provider, image_path is not None, lang)
        ids = [v for _, v in choices]
        preferred = [m for m in (current, default_model(state["settings"], provider)) if m in ids]
        value = preferred[0] if preferred else (ids[0] if ids else None)
        return gr.Dropdown(choices=choices, value=value, label=_model_label(provider, lang))

    def save_endpoints(c_host, c_port, o_host, o_port, provider, image_path, current, lang):
        try:
            urls = {"comfyui": endpoint_url(c_host, c_port), "ollama": endpoint_url(o_host, o_port)}
        except ValueError as e:
            raise gr.Error(t("msg.bad_address", lang, error=e)) from e
        save_env({ENDPOINT_ENV[k]: v for k, v in urls.items()})
        s = replace(state["settings"], comfyui_url=urls["comfyui"], ollama_url=urls["ollama"])
        state["settings"] = s
        state["writers"]["ollama"] = make_writer(s, "ollama")
        msg = f"{t('msg.saved_conn', lang)}\n\n{_check_endpoints(urls['comfyui'], urls['ollama'], lang)}"
        return msg, refresh_models(provider, image_path, current, lang)

    def run(lang, idea, image_path, provider, model, rating, judge_mode, max_attempts, t_fid, t_fmt, t_comp, t_neg,
            width, height, batch_size, seed, steps, cfg, sampler, scheduler, denoise, loras_text):
        if not (idea or "").strip() and not image_path:
            raise gr.Error(t("msg.need_input", lang))
        if not model:
            raise gr.Error(t("msg.need_model", lang))

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

        rows: list[list] = []
        status = t("msg.starting", lang)
        pos = neg = ""
        while (e := events.get()) is not None:
            match e["type"]:
                case "generating":
                    status = t("msg.generating", lang, n=e["attempt"], max=e["max"], provider=provider)
                case "judging":
                    status = t("msg.judging", lang, n=e["attempt"]) + (
                        t("msg.reference_suffix", lang) if e["reference"] else ""
                    )
                case "judged":
                    rows.append(_row(e["attempt"], e["reference"], lang))
                case "warning":
                    status += f"\n⚠️ {e['message']}"
                case "refused":
                    status = t("msg.refused", lang, n=e["attempt"], model=e["model"], reason=e["reason"])
                case "chosen":
                    a = e["attempt"]
                    pos, neg = a.positive, a.negative
                    if not e["judge"]:
                        status = t("msg.no_jev_chosen", lang)
                    else:
                        status = t("msg.passed" if e["passed"] else "msg.not_passed", lang, n=a.number)
                case "uploading":
                    status += "\n" + t("msg.uploading", lang)
                case "rendering":
                    arm = t(f"msg.arm.{e['arm']}", lang) if e["arm"] else ""
                    status += "\n" + t("msg.rendering", lang, arm=arm, prompt_id=e["prompt_id"], seed=e["seed"])
            yield status, rows, pos, neg, "", None

        if "error" in outcome:
            raise gr.Error(f"{type(outcome['error']).__name__}: {outcome['error']}")
        result = outcome["result"]
        log = str(result.log_path)
        if result.refused:
            status += "\n" + t("msg.refusal_logged", lang, path=s.runs_dir / "refusals.jsonl", log=log)
            yield status, rows, pos, neg, "", log
            return
        if isinstance(result, CompareResult):
            a, b = result.with_jev.chosen, result.without_jev.chosen
            wa, wb = t("arm.with_jev", lang), t("arm.without_jev", lang)
            pos = f"[{wa}]\n{a.positive}\n\n[{wb}]\n{b.positive}"
            neg = f"[{wa}]\n{a.negative}\n\n[{wb}]\n{b.negative}"
            images_html = _compare_html(result, lang)
            all_images = result.with_jev.images + ([] if result.identical else result.without_jev.images)
        else:
            images_html = _images_html(result.images)
            all_images = result.images
        files = ", ".join(i.filename for i in all_images)
        status += "\n" + t("msg.done", lang, files=files, log=log)
        yield status, rows, pos, neg, images_html, log

    def export(log_path, lang):
        if not log_path:
            raise gr.Error(t("msg.no_export", lang))
        return str(export_report(log_path, lang=lang))

    g = settings.gen
    c_host, c_port = split_endpoint(settings.comfyui_url)
    o_host, o_port = split_endpoint(settings.ollama_url)
    L = lambda key: t(key, lang0)  # noqa: E731 - initial labels
    with gr.Blocks(title="ComfyUI Prompt Agent") as app:
        last_log = gr.State(None)
        with gr.Row():
            intro = gr.Markdown(L("ui.intro"))
            lang = gr.Radio(
                [(LANG_LABELS[x], x) for x in LANGS], value=lang0, label=L("ui.language"), scale=0, min_width=200,
            )
        with gr.Row():
            with gr.Column(scale=1):
                idea = gr.Textbox(label=L("ui.idea"), lines=4, placeholder=L("ui.idea_placeholder"))
                image = gr.Image(label=L("ui.image"), type="filepath")
                provider = gr.Radio(_provider_choices(lang0), value=settings.llm_provider, label=L("ui.provider"))
                with gr.Row():
                    model = gr.Dropdown(
                        label=_model_label(settings.llm_provider, lang0), choices=[],
                        value=default_model(settings) or None, allow_custom_value=True, scale=4,
                    )
                    refresh = gr.Button("↻", scale=1, min_width=40)
                rating = gr.Radio(_rating_choices(lang0), value="", label=L("ui.rating"))
                judge_mode = gr.Radio(_judge_choices(lang0), value="on", label=L("ui.judge"))
                with gr.Accordion(L("ui.thresholds"), open=False) as acc_thr:
                    max_attempts = gr.Slider(1, 10, value=settings.max_attempts, step=1, label=L("ui.max_attempts"))
                    th = [gr.Slider(0, 1, value=settings.thresholds[d], step=0.01, label=d) for d in DIMENSIONS]
                with gr.Accordion(L("ui.generation"), open=False) as acc_gen:
                    with gr.Row():
                        width = gr.Number(value=g.width, label="width", precision=0)
                        height = gr.Number(value=g.height, label="height", precision=0)
                        batch_size = gr.Number(value=g.batch_size, label="batch (txt2img)", precision=0)
                    with gr.Row():
                        seed = gr.Number(value=g.seed, label=L("ui.seed"), precision=0)
                        steps = gr.Number(value=g.steps, label="steps", precision=0)
                        cfg = gr.Number(value=g.cfg, label="cfg")
                    with gr.Row():
                        sampler = gr.Dropdown(sorted({g.sampler_name, *SAMPLERS}), value=g.sampler_name, label="sampler", allow_custom_value=True)
                        scheduler = gr.Dropdown(sorted({g.scheduler, *SCHEDULERS}), value=g.scheduler, label="scheduler", allow_custom_value=True)
                    denoise = gr.Slider(0, 1, value=g.denoise, step=0.01, label=L("ui.denoise"))
                    loras = gr.Textbox(value=_workflow_loras(settings), lines=3, label=L("ui.loras"))
                with gr.Accordion(L("ui.connection"), open=False) as acc_conn:
                    with gr.Row():
                        comfy_host = gr.Textbox(value=c_host, label=L("ui.comfy_host"), scale=3)
                        comfy_port = gr.Number(value=c_port, label="port", precision=0, scale=1)
                    with gr.Row():
                        ollama_host = gr.Textbox(value=o_host, label=L("ui.ollama_host"), scale=3)
                        ollama_port = gr.Number(value=o_port, label="port", precision=0, scale=1)
                    save_conn = gr.Button(L("ui.save_conn"))
                    conn_status = gr.Markdown()
                go = gr.Button(L("ui.start"), variant="primary")
            with gr.Column(scale=2):
                status = gr.Textbox(label=L("ui.status"), lines=4, interactive=False)
                table = gr.Dataframe(headers=TABLE_HEADERS, label=L("ui.table"), wrap=True, interactive=False)
                final_pos = gr.Textbox(label=L("ui.final_pos"), lines=3, interactive=False)
                final_neg = gr.Textbox(label=L("ui.final_neg"), lines=2, interactive=False)
                images = gr.HTML(label=L("ui.images"))
                with gr.Row():
                    export_btn = gr.Button(L("ui.export"))
                    report_file = gr.File(label=L("ui.report_file"), interactive=False)

        # (component, update builder) pairs re-rendered when the language changes.
        localized = [
            (intro, lambda x, p: gr.update(value=t("ui.intro", x))),
            (lang, lambda x, p: gr.update(label=t("ui.language", x))),
            (idea, lambda x, p: gr.update(label=t("ui.idea", x), placeholder=t("ui.idea_placeholder", x))),
            (image, lambda x, p: gr.update(label=t("ui.image", x))),
            (provider, lambda x, p: gr.update(label=t("ui.provider", x), choices=_provider_choices(x))),
            (model, lambda x, p: gr.update(label=_model_label(p, x))),
            (rating, lambda x, p: gr.update(label=t("ui.rating", x), choices=_rating_choices(x))),
            (judge_mode, lambda x, p: gr.update(label=t("ui.judge", x), choices=_judge_choices(x))),
            (acc_thr, lambda x, p: gr.update(label=t("ui.thresholds", x))),
            (max_attempts, lambda x, p: gr.update(label=t("ui.max_attempts", x))),
            (acc_gen, lambda x, p: gr.update(label=t("ui.generation", x))),
            (seed, lambda x, p: gr.update(label=t("ui.seed", x))),
            (denoise, lambda x, p: gr.update(label=t("ui.denoise", x))),
            (loras, lambda x, p: gr.update(label=t("ui.loras", x))),
            (acc_conn, lambda x, p: gr.update(label=t("ui.connection", x))),
            (comfy_host, lambda x, p: gr.update(label=t("ui.comfy_host", x))),
            (ollama_host, lambda x, p: gr.update(label=t("ui.ollama_host", x))),
            (save_conn, lambda x, p: gr.update(value=t("ui.save_conn", x))),
            (go, lambda x, p: gr.update(value=t("ui.start", x))),
            (status, lambda x, p: gr.update(label=t("ui.status", x))),
            (table, lambda x, p: gr.update(label=t("ui.table", x))),
            (final_pos, lambda x, p: gr.update(label=t("ui.final_pos", x))),
            (final_neg, lambda x, p: gr.update(label=t("ui.final_neg", x))),
            (images, lambda x, p: gr.update(label=t("ui.images", x))),
            (export_btn, lambda x, p: gr.update(value=t("ui.export", x))),
            (report_file, lambda x, p: gr.update(label=t("ui.report_file", x))),
        ]

        def apply_lang(x, p):
            return [build(x, p) for _, build in localized]

        lang.change(apply_lang, [lang, provider], [c for c, _ in localized])

        model_inputs = [provider, image, model, lang]
        refresh.click(refresh_models, model_inputs, model)
        image.change(refresh_models, model_inputs, model)
        provider.change(refresh_models, model_inputs, model)
        app.load(refresh_models, model_inputs, model)
        save_conn.click(
            save_endpoints, [comfy_host, comfy_port, ollama_host, ollama_port, *model_inputs], [conn_status, model]
        )
        go.click(
            run,
            [lang, idea, image, provider, model, rating, judge_mode, max_attempts, *th, width, height, batch_size,
             seed, steps, cfg, sampler, scheduler, denoise, loras],
            [status, table, final_pos, final_neg, images, last_log],
        )
        export_btn.click(export, [last_log, lang], report_file)
    return app
