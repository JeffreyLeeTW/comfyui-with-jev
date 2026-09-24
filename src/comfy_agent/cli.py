"""Command line interface: comfy-agent run | models | check | report | webui."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from .config import JUDGE_MODES, PROVIDERS, RATINGS, load_settings
from .i18n import LANGS
from .llm import default_model, make_writer, resolve_provider
from .openrouter import ModelInfo
from .refusals import refusal_counts

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

ConfigOpt = Annotated[Optional[Path], typer.Option("--config", help="Path to config.yaml")]
LangOpt = Annotated[Optional[str], typer.Option("--lang", help=f"Report language: {' | '.join(LANGS)}; default ui.language")]
ProviderOpt = Annotated[
    Optional[str], typer.Option("--provider", "-p", help=f"{' | '.join(PROVIDERS)}; default from config.yaml")
]


def _provider(settings, provider: str | None) -> str:
    try:
        return resolve_provider(settings, provider)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


def _lang(settings, lang: str | None) -> str:
    lang = lang or settings.language
    if lang not in LANGS:
        raise typer.BadParameter(f"--lang must be one of: {', '.join(LANGS)}")
    return lang


def _print_models(models: list[ModelInfo], refused: dict[str, int]) -> None:
    for i, m in enumerate(models, 1):
        tag = " [vision]" if m.vision else ""
        if n := refused.get(m.id):
            tag += f" [refused {n}x]"
        typer.echo(f"{i:3d}. {m.id}{tag}  ({m.context_length:,} ctx)")


@app.command()
def models(
    vision: Annotated[bool, typer.Option("--vision", help="Only models that accept images")] = False,
    provider: ProviderOpt = None,
    config: ConfigOpt = None,
) -> None:
    """List models: free ones on OpenRouter, or those installed in Ollama."""
    s = load_settings(config)
    provider = _provider(s, provider)
    found = make_writer(s, provider).list_models(need_vision=vision)
    _print_models(found, refusal_counts(s.runs_dir))
    typer.echo(f"\n{len(found)} {provider} model(s)")


@app.command()
def check(config: ConfigOpt = None) -> None:
    """Check connectivity to ComfyUI, OpenRouter, Ollama and TypeSafe."""
    from .comfyui import ComfyUIClient
    from .judge import make_typesafe_client

    s = load_settings(config)
    ok = True

    def report(name: str, fn) -> None:
        nonlocal ok
        try:
            typer.secho(f"[ok]   {name}: {fn()}", fg="green")
        except Exception as e:  # report every service, don't stop at the first failure
            ok = False
            typer.secho(f"[fail] {name}: {type(e).__name__}: {e}", fg="red")

    def comfy() -> str:
        stats = ComfyUIClient(s.comfyui_url).system_stats()
        devices = ", ".join(d.get("name", "?") for d in stats.get("devices", []))
        return f"{s.comfyui_url} ({devices or 'no device info'})"

    def openrouter() -> str:
        if not s.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        return f"{len(make_writer(s, 'openrouter').list_models())} free models listed"

    def ollama() -> str:
        from .ollama import OllamaClient

        client = OllamaClient(s.ollama_url, timeout_s=10)
        return f"{s.ollama_url} v{client.version()} ({len(client.list_models())} chat models)"

    def typesafe() -> str:
        from typesafe_sdk import Noul

        client = make_typesafe_client(s.typesafe_api_key)
        r = client.system_one("The sky is blue.", {"q": Noul(instructions="Is this about the sky?")}, model=s.judge_model)
        return f"{s.judge_model} answered (noul={r.nouls['q'].noul:.2f})"

    report("ComfyUI", comfy)
    report("OpenRouter", openrouter)
    report("Ollama", ollama)
    report("TypeSafe/Jev", typesafe)
    raise typer.Exit(0 if ok else 1)


def _choose_model(s, provider: str, need_vision: bool) -> str:
    found = make_writer(s, provider).list_models(need_vision=need_vision)
    if not found:
        raise typer.BadParameter(f"no {provider} models available" + (" with vision" if need_vision else ""))
    _print_models(found, refusal_counts(s.runs_dir))
    idx = typer.prompt("Pick a model number", type=int)
    if not 1 <= idx <= len(found):
        raise typer.BadParameter(f"choose 1..{len(found)}")
    return found[idx - 1].id


def _parse_loras(values: list[str] | None) -> dict[str, float] | None:
    if not values:
        return None
    out = {}
    for v in values:
        name, _, strength = v.rpartition("=")
        if not name:
            raise typer.BadParameter(f"--lora expects name=strength, got {v!r}")
        out[name] = float(strength)
    return out


@app.command()
def run(
    idea: Annotated[str, typer.Option("--idea", "-i", help="Idea text (English or any language)")] = "",
    image: Annotated[Optional[Path], typer.Option("--image", help="Reference image -> img2img", exists=True, dir_okay=False)] = None,
    provider: ProviderOpt = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model id for the provider; omit to pick interactively")] = None,
    rating: Annotated[Optional[str], typer.Option("--rating", help="sfw | nsfw; omit to add no rating tags")] = None,
    judge: Annotated[str, typer.Option("--judge", help="on = Jev loop | off = one draft, scored for reference | ab = both, same seed")] = "on",
    report: Annotated[bool, typer.Option("--report", help="Also export an HTML report")] = False,
    lang: LangOpt = None,
    max_attempts: Annotated[Optional[int], typer.Option("--max-attempts")] = None,
    t_fidelity: Annotated[Optional[float], typer.Option("--t-fidelity", help="Threshold 0..1")] = None,
    t_format: Annotated[Optional[float], typer.Option("--t-format")] = None,
    t_completeness: Annotated[Optional[float], typer.Option("--t-completeness")] = None,
    t_negative: Annotated[Optional[float], typer.Option("--t-negative")] = None,
    width: Annotated[Optional[int], typer.Option()] = None,
    height: Annotated[Optional[int], typer.Option()] = None,
    batch_size: Annotated[Optional[int], typer.Option()] = None,
    seed: Annotated[Optional[int], typer.Option(help="-1 = random")] = None,
    steps: Annotated[Optional[int], typer.Option()] = None,
    cfg: Annotated[Optional[float], typer.Option()] = None,
    sampler: Annotated[Optional[str], typer.Option()] = None,
    scheduler: Annotated[Optional[str], typer.Option()] = None,
    denoise: Annotated[Optional[float], typer.Option(help="img2img only")] = None,
    lora: Annotated[Optional[list[str]], typer.Option(help="lora_name=strength, repeatable")] = None,
    no_render: Annotated[bool, typer.Option("--no-render", help="Only generate and judge prompts")] = False,
    config: ConfigOpt = None,
) -> None:
    """Idea -> LLM prompt (OpenRouter/Ollama) -> Jev judge (retry, optional) -> ComfyUI."""
    from .pipeline import InputImage, build_pipeline

    if not idea.strip() and image is None:
        raise typer.BadParameter("give --idea, --image, or both")
    rating = rating.lower() if rating else None
    if rating is not None and rating not in RATINGS:
        raise typer.BadParameter(f"--rating must be one of: {', '.join(RATINGS)}")
    judge = judge.lower()
    if judge not in JUDGE_MODES:
        raise typer.BadParameter(f"--judge must be one of: {', '.join(JUDGE_MODES)}")
    s = load_settings(config)
    provider = _provider(s, provider)
    model = model or default_model(s, provider) or _choose_model(s, provider, need_vision=image is not None)

    loras = _parse_loras(lora)
    gen = s.gen.override(
        width=width, height=height, batch_size=batch_size, seed=seed, steps=steps, cfg=cfg,
        sampler_name=sampler, scheduler=scheduler, denoise=denoise,
        lora_strengths=(s.gen.lora_strengths | loras) if loras else None,
    )
    overrides = {"fidelity": t_fidelity, "format": t_format, "completeness": t_completeness, "negative": t_negative}
    thresholds = s.thresholds | {k: v for k, v in overrides.items() if v is not None}

    def on_event(e: dict) -> None:
        match e["type"]:
            case "generating":
                typer.secho(f"\n== Attempt {e['attempt']}/{e['max']}: generating with {provider}/{model}", bold=True)
            case "judging":
                typer.echo(f"positive: {e['positive']}\nnegative: {e['negative']}")
            case "judged":
                v = e["attempt"].verdict
                if v is None:
                    return
                for d in v.dimensions.values():
                    typer.secho(
                        f"  {d.name:<13} {d.value:.2f} / {d.threshold:.2f}  conf {d.confidence:.2f}",
                        fg="green" if d.passed else "red",
                    )
                if e["reference"]:
                    typer.secho(f"  reference only ({'would pass' if v.passed else 'would fail'}); no retry", fg="cyan")
                else:
                    typer.secho("  PASS" if v.passed else "  FAIL -> feedback sent back", fg="green" if v.passed else "yellow")
            case "warning":
                typer.secho(f"  warning: {e['message']}", fg="yellow")
            case "refused":
                typer.secho(f"\n{e['model']} refused on attempt {e['attempt']}; stopping (nothing rendered).", fg="red", bold=True)
                typer.secho(f"  reason: {e['reason']}", fg="red")
            case "chosen":
                if e["judge"] and not e["passed"]:
                    typer.secho(
                        f"\nNo attempt passed; using best attempt #{e['attempt'].number} (marked NOT PASSED)",
                        fg="yellow",
                    )
            case "rendering":
                arm = {"with_jev": " [with Jev]", "without_jev": " [without Jev]"}.get(e["arm"], "")
                typer.echo(f"\nQueued in ComfyUI{arm}: prompt_id={e['prompt_id']} seed={e['seed']} - waiting...")

    pipeline = build_pipeline(s, provider)
    args = (idea, model, InputImage.from_path(image) if image else None, gen, thresholds, max_attempts)
    kwargs = dict(render=not no_render, on_event=on_event, rating=rating)
    if judge == "ab":
        result = pipeline.compare(*args, **kwargs)
        arms = [("with Jev", result.with_jev), ("without Jev", result.without_jev)]
        if result.identical:
            typer.secho("\nFirst draft was the final prompt: both arms are identical (rendered once).", fg="cyan")
    else:
        result = pipeline.run(*args, **kwargs, use_judge=judge == "on")
        arms = [("", result)]
    for label, arm in arms:
        for img in arm.images:
            prefix = f"[{label}] " if label else ""
            typer.secho(f"{prefix}image on server: output/{img.subfolder + '/' if img.subfolder else ''}{img.filename}", fg="cyan")
            typer.echo(f"  view: {img.url}")
    typer.echo(f"log: {result.log_path}")
    if report:
        from .report import export_report

        typer.echo(f"report: {export_report(result.log_path, lang=_lang(s, lang))}")
    if result.refused:
        typer.echo(f"refusal recorded in {s.runs_dir / 'refusals.jsonl'}")
        raise typer.Exit(1)


@app.command("report")
def report_cmd(
    log: Annotated[Path, typer.Argument(help="Run log, e.g. runs/20260924-120000.json", exists=True, dir_okay=False)],
    out_dir: Annotated[Optional[Path], typer.Option("--out", help="Output folder (default: <runs>/reports)")] = None,
    lang: LangOpt = None,
    config: ConfigOpt = None,
) -> None:
    """Export a run log as an HTML report."""
    from .report import export_report

    typer.echo(export_report(log, out_dir, _lang(load_settings(config), lang)))


@app.command()
def webui(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 7860,
    config: ConfigOpt = None,
) -> None:
    """Launch the Gradio WebUI."""
    from .webui import build_app

    build_app(load_settings(config)).launch(server_name=host, server_port=port)
