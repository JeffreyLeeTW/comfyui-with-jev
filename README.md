# comfy-agent

[![English](https://img.shields.io/badge/lang-English-blue)](README.md) [![繁體中文](https://img.shields.io/badge/lang-%E7%B9%81%E9%AB%94%E4%B8%AD%E6%96%87-lightgrey)](README.zh-TW.md)

Idea (text and/or image) → a free **OpenRouter** model or a local **Ollama** model writes the prompt → **Jev** (TypeSafe) reviews it → **ComfyUI** renders the image.

https://github.com/user-attachments/assets/52bff2b1-0903-404e-8181-81ee8bba78a5

```
Idea (text / image)
      │
      ▼
LLM writes positive / negative prompt ◄──────────┐
      │                                          │ failed: rewrite using Jev's critique
      ▼                                          │ (up to max_attempts times)
Jev scores 4 dimensions ── any below threshold ──┘
      │ every dimension meets its threshold
      │ (or attempts run out → send the best-scoring version, marked "NOT PASSED")
      ▼
ComfyUI: img2img with an image, txt2img without → images saved to the server's output/
```

## Requirements

- Python ≥ 3.12, managed with [uv](https://docs.astral.sh/uv/)
- Network access to the ComfyUI server (default `http://10.0.0.15:8188`, VPN required)
- API keys: [OpenRouter](https://openrouter.ai/keys), [TypeSafe](https://console.typesafe.ai/)
- Optional: an [Ollama](https://ollama.com/) server (default `http://10.0.0.15:11434`) as an alternative to OpenRouter; no API key needed

## Installation

```bash
uv sync
cp .env.example .env   # fill in OPENROUTER_API_KEY and TYPESAFE_API_KEY
uv run comfy-agent check
```

`check` tests the connections to ComfyUI, OpenRouter, Ollama and TypeSafe. Every service you actually use should show `[ok]`; if you only use one provider, a failure on the other is fine.

## Usage

### CLI

```bash
# List models (--vision: only models that accept images; --provider/-p: openrouter or ollama, default from config.yaml)
uv run comfy-agent models
uv run comfy-agent models --vision
uv run comfy-agent models -p ollama

# Write prompts with local Ollama instead
uv run comfy-agent run --idea "..." -p ollama --model gemma4:e4b

# Text only → txt2img (without --model you pick from a list)
uv run comfy-agent run --idea "a girl reading under cherry blossoms"

# Text + reference image → img2img
uv run comfy-agent run --idea "same girl at night" --image ref.png --model qwen/qwen3.8-27b:free

# Only generate and score prompts; do not render
uv run comfy-agent run --idea "..." --no-render

# Content rating (the "Content rating" option in the WebUI): sfw or nsfw; omit for unspecified
uv run comfy-agent run --idea "..." --rating sfw

# Prompt style (the "Prompt style" option in the WebUI): natural (English sentences, default) or tags (danbooru tags)
uv run comfy-agent run --idea "..." --style tags

# Jev mode: on (default) / off (one draft rendered directly; Jev scores it for reference only) / ab (both, same seed)
uv run comfy-agent run --idea "..." --judge ab --report

# Export any run log as an HTML report (saved to runs/reports/; --lang en or zh-TW)
uv run comfy-agent report runs/20260924-120000.json --lang en
```

Options for `run` (anything not given falls back to `config.yaml`):

| Category | Options |
| --- | --- |
| Model / flow | `--provider/-p` (`openrouter` / `ollama`), `--model/-m`, `--max-attempts`, `--rating` (`sfw` / `nsfw`), `--style` (`natural` / `tags`), `--judge` (`on` / `off` / `ab`), `--report` (also export an HTML report), `--lang` (report language: `en` / `zh-TW`) |
| Jev thresholds (0–1) | `--t-fidelity`, `--t-format`, `--t-completeness`, `--t-negative` |
| Generation | `--width`, `--height`, `--batch-size`, `--seed` (-1 = random), `--steps`, `--cfg`, `--sampler`, `--scheduler`, `--denoise` (img2img only) |
| LoRA | `--lora name=strength` (repeatable) |
| Other | `--config path/to/config.yaml` |

### WebUI

```bash
uv run comfy-agent webui            # http://127.0.0.1:7860
uv run comfy-agent webui --port 8000
```

- Switch the interface language (English / 繁體中文) at the top right. The default comes from `ui.language` in `config.yaml`, and exported reports use the current language.
- "Prompt writer" switches between OpenRouter and Ollama; the model list follows.
- The model dropdown only lists vision models when an image is uploaded; ↻ refreshes the list.
- "Jev review" chooses Use Jev / No Jev / A/B compare; see [Jev modes](#jev-modes).
- The "Jev thresholds" and "Generation parameters" panels adjust the settings for each run.
- "Connection" changes the ComfyUI and Ollama IP and port. "Save and test connection" writes `COMFYUI_URL` / `OLLAMA_URL` to `.env` (these override `config.yaml` and are read by the CLI too) and tests both connections right away; the next run uses the new addresses without a restart.
- The right side shows each attempt's scores, the submitted prompts and the result images loaded from the server, live.
- After a run, "Export report (HTML)" downloads the report for that run.

## Configuration: `config.yaml`

| Section | Highlights |
| --- | --- |
| `comfyui` | `url`, `workflow` (workflow template in API format), `timeout_s` |
| `llm` | `provider`: default prompt writer (`openrouter` / `ollama`); can be changed on every run |
| `openrouter` | `default_model` (empty = choose on each run), `temperature` |
| `ollama` | `url`, `default_model`, `temperature`, `think` (whether thinking models reason first; off by default for speed), `timeout_s` (includes loading the model into VRAM) |
| `judge` | `model` (`jev-latest`), `max_attempts` (default 5), `thresholds` (per dimension) |
| `prompt` | `style`: default prompt style, `natural` (anima_baseV10 understands English sentences) or `tags`. `positive_prefix` / `negative_base`: fixed tags always added in front, in both styles, such as quality tags and LoRA triggers. `ratings.sfw` / `ratings.nsfw`: positive / negative tags forced in when a content rating is chosen; the chosen rating is also passed to the prompt writer and to Jev. In tag style, rating tags the model puts on the opposite side are removed; in natural style the text is left as written and Jev's `fidelity` / `negative` checks catch conflicts |
| `generation` | Default generation parameters; `lora_strengths` is keyed by `lora_name` |
| `runs_dir` | Local folder for run logs |
| `ui` | `language`: default language of the WebUI and HTML reports (`en` / `zh-TW`, default `en`) |

## Jev scoring

Each prompt version is sent to Jev in one request that scores 4 Score dimensions. Each dimension has 4 levels, normalized to 0–1:

| Dimension | What it checks |
| --- | --- |
| `fidelity` | Does the positive prompt faithfully depict the idea (and the image description, when there is a reference image)? |
| `format` | Tag style: English, comma-separated danbooru tags, with no sentences and no contradicting tags. Natural style: fluent, concrete English sentences after the fixed tag prefix, with no tag lists and no contradictions |
| `completeness` | Are subject, appearance, pose / composition and scene / lighting all specified? |
| `negative` | Is the negative prompt reasonable, without excluding anything the idea asks for? |

- **Pass**: all 4 dimensions meet their thresholds (0.67 by default).
- **Fail**: the failing dimensions, their scores and the level Jev chose are turned into a critique and sent back to the LLM together with the previous prompt for a rewrite.
- **Out of attempts**: the version whose weakest dimension scores highest is sent, marked "NOT PASSED".
- Jev only reads text. With a reference image, the LLM first writes a short image description that Jev uses as context.
- The questions and level descriptions live in `QUESTIONS` in `src/comfy_agent/judge.py`.

> The default thresholds are only a starting point. Run a few times and tune them from the actual score distribution in `runs/`.

### Jev modes

| Mode | Flow | Jev cost |
| --- | --- | --- |
| Use Jev (`on`, default) | The scoring / rewrite loop described above | 1 per version |
| No Jev (`off`) | Generates once and renders. Jev still scores it once for the record only; skipped when `TYPESAFE_API_KEY` is missing or scoring fails | 1 |
| A/B compare (`ab`) | After the Jev loop, renders "the version Jev sent" and "the first draft" (what a no-Jev run would send) with **the same seed**. If the first draft already passed, both arms are identical and rendered once | Same as `on` |

A/B compare uses the Jev loop's first draft as the "without Jev" arm, so both arms start from the same point and differ only in whether Jev's feedback rewrote the prompt, at no extra LLM or Jev cost.

### Reports

`runs/reports/<log name>.html` is a single HTML file with the idea, parameters, every attempt's scores and prompts, and the final images; A/B reports show both arms side by side with per-dimension score differences. Images are loaded from ComfyUI's `/view` endpoint and never downloaded, so viewing them requires the VPN.

## ComfyUI workflow

The template is `workflows/anima_flow.json`. Nodes are located by class type and connections, never by hard-coded IDs. On every run:

- **txt2img** (no reference image): KSampler is connected to `EmptyLatentImage` with denoise 1.0; the unused LoadImage chain is removed.
- **img2img** (with a reference image): the image is uploaded to ComfyUI's `input/` and goes through `LoadImage → VAEEncode → LatentUpscale → KSampler` with your denoise.
- `PreviewImage` is replaced with `SaveImage`, so images are saved on the server as `ComfyUI/output/<filename_prefix>_xxxxx_.png`.
- Prompts, seed, steps, cfg, sampler, scheduler, size and LoRA strengths are written in.

To use another workflow, export it from ComfyUI with **Export (API)** and point `comfyui.workflow` in `config.yaml` at it. It needs a `KSampler` (with a text-encode node on positive and negative), `EmptyLatentImage` (txt2img), `LoadImage` plus `VAEEncode` (img2img), and an output node (PreviewImage or SaveImage).

## Output

- **Images**: stored only in the server's `output/`, never downloaded. The CLI prints `/view` links and the WebUI shows them directly.
- **Logs**: `runs/<time>.json` (`kind` is `run` or `compare`; a `compare` log holds the arms in `with_jev` / `without_jev`), with every attempt's prompts, per-dimension score / confidence / level, which attempt was chosen, whether it passed, generation parameters, seed and the ComfyUI prompt_id. A refused run also has a `refused` field with the reason.
- **Refusal log**: `runs/refusals.jsonl`, one line per refusal (time, provider, model, txt2img / img2img, attempt, reason, idea). The `models` command and the WebUI model dropdown show how often each model has refused.

## Project layout

```
config.yaml               default settings
workflows/anima_flow.json ComfyUI API workflow template
src/comfy_agent/
  config.py      reads config.yaml and .env
  i18n.py        English / Traditional Chinese strings for the WebUI and reports
  openrouter.py  shared prompt-writing logic (PromptWriter) and the OpenRouter backend
  ollama.py      Ollama backend (native /api/chat, thinking can be turned off)
  llm.py         builds the backend for a provider
  refusals.py    model refusal log
  judge.py       Jev questions, pass/fail, critiques
  comfyui.py     patches the workflow, uploads images, queues and waits for renders
  pipeline.py    generate → judge → render flow, writes run logs
  cli.py         CLI (typer)
  report.py      turns run logs into HTML reports
  webui.py       WebUI (Gradio)
tests/           unit tests (all HTTP mocked)
```

## Tests

```bash
uv run pytest
```

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `check` reports ComfyUI failed | Make sure the VPN is connected and the address is right (`COMFYUI_URL` in `.env` wins over `comfyui.url` in `config.yaml`) |
| OpenRouter returns 429 or 503 | Free models are often rate-limited; the client retries a few times. If it keeps failing, switch models |
| `Model did not return valid prompt JSON` | This model struggles with the JSON format; switch models |
| `refused ... stopping (nothing rendered)` | The model refused to write the prompt (it reported a refusal, the provider's content filter blocked it, or the reply matched common refusal phrases). The run stops immediately without rendering and the refusal is logged in `runs/refusals.jsonl`. Switch models |
| ComfyUI rejected the workflow | The error includes node_errors; usually a model or LoRA file is missing on the server, or a sampler / scheduler name is misspelled |
| Nothing ever passes | Check which dimension is low in `runs/`, then adjust the thresholds or the criteria in `judge.py` |
