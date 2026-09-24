"""Turn a run log (runs/*.json) into a self-contained HTML report.

Images are referenced by their ComfyUI /view URL (never downloaded), so viewing them needs the VPN.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from .config import DIMENSIONS
from .i18n import normalize_lang, t

GEN_KEYS = ("width", "height", "batch_size", "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise")

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2dc;--accent:#3d5afe;
--ok:#1b7f3b;--ok-bg:#e3f4e8;--bad:#b3261e;--bad-bg:#fbe7e5;--warn:#8a5a00;--warn-bg:#fdf1d8;--bar:#e9e8e2}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--ink:#ecebe6;--muted:#a3a29b;
--line:#34332f;--accent:#8c9eff;--ok:#7fd69a;--ok-bg:#1d3325;--bad:#f2a49c;--bad-bg:#3a2220;
--warn:#f1c76b;--warn-bg:#3a2f16;--bar:#34332f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"PingFang TC","Noto Sans TC",sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 12px}h3{font-size:16px;margin:0 0 8px}
.muted{color:var(--muted)}.idea{font-size:17px;margin:12px 0 16px;white-space:pre-wrap}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{border:1px solid var(--line);border-radius:999px;padding:2px 10px;font-size:13px;background:var(--card)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0}
.badge{display:inline-block;border-radius:6px;padding:1px 8px;font-size:13px;font-weight:600;margin-left:6px}
.ok{color:var(--ok);background:var(--ok-bg)}.bad{color:var(--bad);background:var(--bad-bg)}.warn{color:var(--warn);background:var(--warn-bg)}
.imgs{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.imgs img{max-width:100%;max-height:560px;border-radius:8px;border:1px solid var(--line)}
.noimg{padding:24px;text-align:center;border:1px dashed var(--line);border-radius:8px;color:var(--muted);margin:12px 0}
.score{display:grid;grid-template-columns:110px 1fr 90px;align-items:center;gap:8px;font-size:13px;margin:4px 0}
.track{position:relative;height:8px;background:var(--bar);border-radius:4px}
.fill{position:absolute;inset:0 auto 0 0;border-radius:4px}.fill.ok{background:var(--ok)}.fill.bad{background:var(--bad)}
.tick{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink);opacity:.6}
.prompt{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bg);border:1px solid var(--line);
border-radius:8px;padding:8px 10px;white-space:pre-wrap;word-break:break-word;margin:4px 0 10px}
.label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{background:var(--bg)}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pos{color:var(--ok)}.neg{color:var(--bad)}
details summary{cursor:pointer;color:var(--accent)}
.banner{border-radius:10px;padding:12px 16px;margin:16px 0}
"""


def _chips(items: list[tuple[str, object]]) -> str:
    return "".join(
        f'<span class="chip"><span class="muted">{escape(k)}</span> {escape(str(v))}</span>'
        for k, v in items if v not in (None, "")
    )


def _badge(text: str, cls: str) -> str:
    return f'<span class="badge {cls}">{escape(text)}</span>'


def _images(images: list[dict], lang: str) -> str:
    if not images:
        return f'<div class="noimg">{escape(t("report.no_images", lang))}</div>'
    tags = "".join(
        f'<a href="{escape(i["url"])}" target="_blank" rel="noopener">'
        f'<img src="{escape(i["url"])}" alt="{escape(i["filename"])}" loading="lazy"></a>'
        for i in images
    )
    return f'<div class="imgs">{tags}</div>'


def _scores(scores: dict, lang: str) -> str:
    if not scores:
        return f'<p class="muted">{escape(t("report.no_scores", lang))}</p>'
    rows = []
    for d in DIMENSIONS:
        sc = scores.get(d)
        if not sc:
            continue
        v, th = sc["value"], sc.get("threshold")
        ok = th is None or v >= th
        tick = f'<span class="tick" style="left:{th * 100:.1f}%"></span>' if th is not None else ""
        rows.append(
            f'<div class="score" title="{escape(sc.get("level", ""))}"><span>{escape(d)}</span>'
            f'<span class="track"><span class="fill {"ok" if ok else "bad"}" style="width:{v * 100:.1f}%"></span>{tick}</span>'
            f'<span class="{"pos" if ok else "neg"}">{v:.2f}'
            + (f' <span class="muted">/ {th:.2f}</span>' if th is not None else "")
            + "</span></div>"
        )
    return "".join(rows)


def _chosen(arm: dict) -> dict | None:
    n = arm.get("chosen_attempt")
    return next((a for a in arm.get("attempts", []) if a["number"] == n), None)


def _status(arm: dict, lang: str) -> str:
    if arm.get("refused"):
        return _badge(t("report.refused", lang), "bad")
    chosen = _chosen(arm)
    if arm.get("judge_mode") == "off":
        if not chosen or chosen.get("passed") is None:
            return _badge(t("report.unscored", lang), "warn")
        return _badge(t("report.ref_pass" if chosen["passed"] else "report.ref_fail", lang),
                      "ok" if chosen["passed"] else "warn")
    if arm.get("passed"):
        return _badge(t("report.passed_at", lang, n=arm["chosen_attempt"]), "ok")
    return _badge(t("report.not_passed", lang), "warn")


def _arm_card(title: str, arm: dict, lang: str) -> str:
    chosen = _chosen(arm)
    chips = _chips([("seed", arm.get("seed")), ("prompt_id", arm.get("prompt_id")),
                    (t("report.attempt_count", lang), len(arm.get("attempts", [])))])
    body = [f"<h3>{escape(title)}{_status(arm, lang)}</h3>", f'<div class="chips">{chips}</div>',
            _images(arm.get("images", []), lang)]
    if chosen:
        body += [
            _scores(chosen.get("scores", {}), lang),
            '<div class="label" style="margin-top:12px">positive</div>',
            f'<div class="prompt">{escape(chosen["positive"])}</div>',
            '<div class="label">negative</div>',
            f'<div class="prompt">{escape(chosen["negative"])}</div>',
        ]
    if arm.get("refused"):
        body.append(f'<div class="banner bad">{escape(t("report.refused_reason", lang, reason=arm["refused"]))}</div>')
    return f'<section class="card">{"".join(body)}</section>'


def _delta_cls(delta: float) -> str:
    return "pos" if delta > 0 else "neg" if delta < 0 else "muted"


def _delta_table(a: dict, b: dict, lang: str) -> str:
    sa, sb = (_chosen(a) or {}).get("scores", {}), (_chosen(b) or {}).get("scores", {})
    if not sa or not sb:
        return ""
    rows = []
    for d in DIMENSIONS:
        if d not in sa or d not in sb:
            continue
        delta = sa[d]["value"] - sb[d]["value"]
        rows.append(
            f'<tr><td>{escape(d)}</td><td class="num">{sa[d]["value"]:.2f}</td>'
            f'<td class="num">{sb[d]["value"]:.2f}</td><td class="num {_delta_cls(delta)}">{delta:+.2f}</td></tr>'
        )
    wa, wb = min(v["value"] for v in sa.values()), min(v["value"] for v in sb.values())
    rows.append(
        f'<tr><th>{escape(t("report.weakest", lang))}</th><td class="num">{wa:.2f}</td><td class="num">{wb:.2f}</td>'
        f'<td class="num {_delta_cls(wa - wb)}">{wa - wb:+.2f}</td></tr>'
    )
    head = "".join(f"<th>{escape(x)}</th>" for x in (
        t("report.dimension", lang), t("arm.with_jev", lang), t("arm.without_jev", lang), t("report.difference", lang),
    ))
    return (
        f"<h2>{escape(t('report.delta_title', lang))}</h2>"
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _attempts_table(attempts: list[dict], chosen_n: int | None, lang: str) -> str:
    if not attempts:
        return ""
    head = "".join(f"<th>{escape(d)}</th>" for d in DIMENSIONS)
    rows = []
    for a in attempts:
        sc = a.get("scores", {})
        cells = "".join(
            f'<td class="num">{sc[d]["value"]:.2f}</td>' if d in sc else '<td class="num muted">–</td>'
            for d in DIMENSIONS
        )
        result = "–" if a.get("passed") is None else ("PASS" if a["passed"] else "FAIL")
        mark = " ★" if a["number"] == chosen_n else ""
        rows.append(
            f'<tr><td class="num">{a["number"]}{mark}</td>{cells}<td>{result}</td>'
            f'<td><details><summary>prompt</summary><div class="label">positive</div>'
            f'<div class="prompt">{escape(a["positive"])}</div><div class="label">negative</div>'
            f'<div class="prompt">{escape(a["negative"])}</div>'
            + (f'<div class="label">image description</div><div class="prompt">{escape(a["image_description"])}</div>'
               if a.get("image_description") else "")
            + "</details></td></tr>"
        )
    return (
        f"<h2>{escape(t('report.attempts_title', lang))}</h2>"
        f'<p class="muted">{escape(t("report.star_note", lang))}</p>'
        f'<div class="table-wrap"><table><thead><tr><th>#</th>{head}<th>{escape(t("report.verdict", lang))}</th>'
        f"<th>prompt</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def build_report(log: dict, lang: str = "en") -> str:
    lang = normalize_lang(lang)
    compare = log.get("kind") == "compare"
    judge_label = t("report.judge_off" if log.get("judge_mode") == "off" else "report.judge_on", lang)
    gen = log.get("generation", {})
    meta = _chips([
        (t("report.time", lang), log.get("time")),
        (t("report.mode", lang), t("report.ab", lang) if compare else judge_label),
        ("provider", log.get("provider")),
        (t("report.model", lang), log.get("model")),
        (t("report.render", lang), log.get("mode")),
        (t("report.rating", lang), (log.get("rating") or "").upper() or t("rating.none", lang)),
        (t("report.style", lang), t(f"style.{log.get('prompt_style') or 'tags'}", lang)),  # old logs: tags
        *[(k, gen.get(k)) for k in GEN_KEYS],
    ])
    thresholds = _chips(list((log.get("thresholds") or {}).items()))
    title = t("report.title", lang)
    parts = [
        f"<h1>{escape(title)}</h1>",
        f'<p class="muted">{escape(t("report.vpn_note", lang))}</p>',
        f'<div class="idea">{escape(log.get("idea") or t("report.no_idea", lang))}</div>',
        f'<div class="chips">{meta}</div>',
        f'<div class="chips" style="margin-top:6px"><span class="muted" style="font-size:13px">'
        f'{escape(t("report.thresholds", lang))}</span>{thresholds}</div>',
    ]
    if compare:
        a, b = log["with_jev"], log["without_jev"]
        if a.get("refused"):
            parts.append(f'<div class="banner bad">{escape(t("report.refused_banner", lang, reason=a["refused"]))}</div>')
        if log.get("identical"):
            parts.append(f'<div class="banner warn">{escape(t("msg.identical", lang))}</div>')
        parts += [
            f"<h2>{escape(t('report.compare', lang))}</h2>",
            f'<div class="grid">{_arm_card(t("arm.with_jev", lang), a, lang)}'
            f'{_arm_card(t("arm.without_jev", lang), b, lang)}</div>',
            _delta_table(a, b, lang),
            _attempts_table(a.get("attempts", []), a.get("chosen_attempt"), lang),
        ]
    else:
        parts += [
            f"<h2>{escape(t('report.result', lang))}</h2>",
            f'<div class="grid">{_arm_card(judge_label, log, lang)}</div>',
            _attempts_table(log.get("attempts", []), log.get("chosen_attempt"), lang),
        ]
    html_lang = "zh-Hant" if lang == "zh-TW" else "en"
    return (
        f'<!doctype html><html lang="{html_lang}"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{CSS}</style></head>"
        f"<body><main>{''.join(parts)}</main></body></html>"
    )


def export_report(log_path: str | Path, out_dir: str | Path | None = None, lang: str = "en") -> Path:
    """Write <runs>/reports/<log name>.html next to the log and return its path."""
    log_path = Path(log_path)
    log = json.loads(log_path.read_text(encoding="utf-8"))
    out = Path(out_dir) if out_dir else log_path.parent / "reports"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{log_path.stem}.html"
    path.write_text(build_report(log, lang), encoding="utf-8")
    return path
