"""Turn a run log (runs/*.json) into a self-contained HTML report.

Images are referenced by their ComfyUI /view URL (never downloaded), so viewing them needs the VPN.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from .config import DIMENSIONS

JUDGE_LABELS = {"on": "使用 Jev", "off": "不用 Jev（分數僅供參考）"}
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


def _images(images: list[dict]) -> str:
    if not images:
        return '<div class="noimg">沒有圖片（未生圖或被拒絕）</div>'
    tags = "".join(
        f'<a href="{escape(i["url"])}" target="_blank" rel="noopener">'
        f'<img src="{escape(i["url"])}" alt="{escape(i["filename"])}" loading="lazy"></a>'
        for i in images
    )
    return f'<div class="imgs">{tags}</div>'


def _scores(scores: dict) -> str:
    if not scores:
        return '<p class="muted">沒有 Jev 分數</p>'
    rows = []
    for d in DIMENSIONS:
        sc = scores.get(d)
        if not sc:
            continue
        v, t = sc["value"], sc.get("threshold")
        ok = t is None or v >= t
        tick = f'<span class="tick" style="left:{t * 100:.1f}%"></span>' if t is not None else ""
        rows.append(
            f'<div class="score" title="{escape(sc.get("level", ""))}"><span>{escape(d)}</span>'
            f'<span class="track"><span class="fill {"ok" if ok else "bad"}" style="width:{v * 100:.1f}%"></span>{tick}</span>'
            f'<span class="{"pos" if ok else "neg"}">{v:.2f}'
            + (f' <span class="muted">/ {t:.2f}</span>' if t is not None else "")
            + "</span></div>"
        )
    return "".join(rows)


def _chosen(arm: dict) -> dict | None:
    n = arm.get("chosen_attempt")
    return next((a for a in arm.get("attempts", []) if a["number"] == n), None)


def _status(arm: dict) -> str:
    if arm.get("refused"):
        return _badge("模型拒絕", "bad")
    chosen = _chosen(arm)
    if arm.get("judge_mode") == "off":
        if not chosen or chosen.get("passed") is None:
            return _badge("未評分", "warn")
        return _badge("參考：達標" if chosen["passed"] else "參考：未達標", "ok" if chosen["passed"] else "warn")
    return _badge(f"第 {arm['chosen_attempt']} 輪通過", "ok") if arm.get("passed") else _badge("未達標（送最高分）", "warn")


def _arm_card(title: str, arm: dict) -> str:
    chosen = _chosen(arm)
    body = [f"<h3>{escape(title)}{_status(arm)}</h3>",
            f'<div class="chips">{_chips([("seed", arm.get("seed")), ("prompt_id", arm.get("prompt_id")), ("嘗試次數", len(arm.get("attempts", [])))])}</div>',
            _images(arm.get("images", []))]
    if chosen:
        body += [
            _scores(chosen.get("scores", {})),
            '<div class="label" style="margin-top:12px">positive</div>',
            f'<div class="prompt">{escape(chosen["positive"])}</div>',
            '<div class="label">negative</div>',
            f'<div class="prompt">{escape(chosen["negative"])}</div>',
        ]
    if arm.get("refused"):
        body.append(f'<div class="banner bad">拒絕原因：{escape(arm["refused"])}</div>')
    return f'<section class="card">{"".join(body)}</section>'


def _delta_table(a: dict, b: dict) -> str:
    sa, sb = (_chosen(a) or {}).get("scores", {}), (_chosen(b) or {}).get("scores", {})
    if not sa or not sb:
        return ""
    rows = []
    for d in DIMENSIONS:
        if d not in sa or d not in sb:
            continue
        delta = sa[d]["value"] - sb[d]["value"]
        cls = "pos" if delta > 0 else "neg" if delta < 0 else "muted"
        rows.append(
            f'<tr><td>{escape(d)}</td><td class="num">{sa[d]["value"]:.2f}</td>'
            f'<td class="num">{sb[d]["value"]:.2f}</td><td class="num {cls}">{delta:+.2f}</td></tr>'
        )
    wa, wb = min(sa[d]["value"] for d in sa), min(sb[d]["value"] for d in sb)
    rows.append(
        f'<tr><th>最弱面向</th><td class="num">{wa:.2f}</td><td class="num">{wb:.2f}</td>'
        f'<td class="num {"pos" if wa > wb else "neg" if wa < wb else "muted"}">{wa - wb:+.2f}</td></tr>'
    )
    return (
        "<h2>分數差異（有 Jev − 無 Jev）</h2>"
        '<div class="table-wrap"><table><thead><tr><th>面向</th><th>有 Jev</th><th>無 Jev</th><th>差異</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _attempts_table(attempts: list[dict], chosen_n: int | None) -> str:
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
        "<h2>每一輪 Jev 評分</h2><p class=\"muted\">★ = 最後送出的版本</p>"
        f'<div class="table-wrap"><table><thead><tr><th>#</th>{head}<th>結果</th><th>prompt</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def build_report(log: dict) -> str:
    compare = log.get("kind") == "compare"
    gen = log.get("generation", {})
    meta = _chips([
        ("時間", log.get("time")),
        ("模式", "A-B 對照" if compare else JUDGE_LABELS.get(log.get("judge_mode", "on"))),
        ("provider", log.get("provider")),
        ("模型", log.get("model")),
        ("生圖", log.get("mode")),
        ("分級", (log.get("rating") or "不指定").upper()),
        *[(k, gen.get(k)) for k in GEN_KEYS],
    ])
    thresholds = _chips(list((log.get("thresholds") or {}).items()))
    parts = [
        '<h1>comfy-agent 報告</h1>',
        f'<p class="muted">圖片從 ComfyUI server 讀取（需連 VPN）</p>',
        f'<div class="idea">{escape(log.get("idea") or "（沒有文字構想，使用參考圖）")}</div>',
        f'<div class="chips">{meta}</div>',
        f'<div class="chips" style="margin-top:6px"><span class="muted" style="font-size:13px">門檻</span>{thresholds}</div>',
    ]
    if compare:
        a, b = log["with_jev"], log["without_jev"]
        if a.get("refused"):
            parts.append(f'<div class="banner bad">模型拒絕，流程已停止：{escape(a["refused"])}</div>')
        if log.get("identical"):
            parts.append('<div class="banner warn">第 1 版就是最後送出的版本，兩組 prompt 相同，只生了一次圖。</div>')
        parts += [
            "<h2>對照結果</h2>",
            f'<div class="grid">{_arm_card("有 Jev", a)}{_arm_card("無 Jev（第 1 版）", b)}</div>',
            _delta_table(a, b),
            _attempts_table(a.get("attempts", []), a.get("chosen_attempt")),
        ]
    else:
        title = JUDGE_LABELS.get(log.get("judge_mode", "on"), "結果")
        parts += [
            "<h2>結果</h2>",
            f'<div class="grid">{_arm_card(title, log)}</div>',
            _attempts_table(log.get("attempts", []), log.get("chosen_attempt")),
        ]
    return (
        '<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>comfy-agent 報告</title><style>{CSS}</style></head>"
        f"<body><main>{''.join(parts)}</main></body></html>"
    )


def export_report(log_path: str | Path, out_dir: str | Path | None = None) -> Path:
    """Write <runs>/reports/<log name>.html next to the log and return its path."""
    log_path = Path(log_path)
    log = json.loads(log_path.read_text(encoding="utf-8"))
    out = Path(out_dir) if out_dir else log_path.parent / "reports"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{log_path.stem}.html"
    path.write_text(build_report(log), encoding="utf-8")
    return path
