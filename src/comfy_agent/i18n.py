"""UI strings for the WebUI and HTML reports. English is the default; zh-TW is the alternative."""

from __future__ import annotations

LANGS = ("en", "zh-TW")
LANG_LABELS = {"en": "English", "zh-TW": "繁體中文"}

# key -> (English, Traditional Chinese). Placeholders use str.format fields.
_T: dict[str, tuple[str, str]] = {
    # common
    "provider.openrouter": ("OpenRouter (free models)", "OpenRouter（免費模型）"),
    "provider.ollama": ("Ollama (local)", "Ollama（本地）"),
    "rating.none": ("Unspecified", "不指定"),
    "style.natural": ("Natural language", "自然語言"),
    "style.tags": ("Danbooru tags", "Danbooru tag"),
    "judge.on": ("Use Jev", "使用 Jev"),
    "judge.off": ("No Jev (scored once, reference only)", "不用 Jev（評一次分，僅供參考）"),
    "judge.ab": ("A/B compare (same seed for both)", "A-B 對照（同 seed 各生一組）"),
    "arm.with_jev": ("With Jev", "有 Jev"),
    "arm.without_jev": ("Without Jev (first draft)", "無 Jev（第 1 版）"),
    # WebUI layout
    "ui.language": ("Language", "語言"),
    "ui.intro": (
        "## ComfyUI Prompt Agent\nIdea → OpenRouter / Ollama writes the prompt → Jev reviews "
        "(optional; failing prompts are rewritten) → ComfyUI renders",
        "## ComfyUI Prompt Agent\n構想 → OpenRouter／Ollama 產生 prompt → Jev 評審"
        "（可選，不及格退回重寫）→ ComfyUI 生圖",
    ),
    "ui.idea": ("Idea", "構想"),
    "ui.idea_placeholder": ("Describe the picture you want (a reference image is optional)", "描述你想要的畫面（可搭配參考圖）"),
    "ui.image": ("Reference image (image → img2img)", "參考圖（有圖 → img2img）"),
    "ui.provider": ("Prompt writer", "Prompt 產生器"),
    "ui.model": ("{provider} model", "{provider} 模型"),
    "ui.rating": ("Content rating (adds matching positive / negative tags)", "內容分級（加入對應的 positive／negative tag）"),
    "ui.style": ("Prompt style (fixed quality / rating tags are always added)", "Prompt 風格（固定的品質／分級 tag 一律加入）"),
    "ui.judge": ("Jev review", "Jev 評審"),
    "ui.thresholds": ("Jev thresholds", "Jev 門檻"),
    "ui.max_attempts": ("Max attempts (always 1 without Jev)", "最多嘗試次數（不用 Jev 時固定 1 次）"),
    "ui.generation": ("Generation parameters", "生圖參數"),
    "ui.seed": ("seed (-1 = random; shared by both A/B arms)", "seed (-1 隨機；A-B 對照時兩組共用)"),
    "ui.denoise": ("denoise (img2img only)", "denoise（僅 img2img）"),
    "ui.loras": ("LoRA strengths (name=strength, one per line)", "LoRA 強度（name=strength，每行一個）"),
    "ui.connection": ("Connection", "連線設定"),
    "ui.comfy_host": ("ComfyUI IP / host", "ComfyUI IP／主機"),
    "ui.ollama_host": ("Ollama IP / host", "Ollama IP／主機"),
    "ui.save_conn": ("Save and test connection", "儲存並測試連線"),
    "ui.start": ("Start", "開始"),
    "ui.status": ("Status", "狀態"),
    "ui.table": ("Jev scores per attempt", "每一輪 Jev 評分"),
    "ui.final_pos": ("Submitted positive prompt", "送出的 positive prompt"),
    "ui.final_neg": ("Submitted negative prompt", "送出的 negative prompt"),
    "ui.images": ("Result (loaded from the server)", "結果（從 server 讀取）"),
    "ui.export": ("Export report (HTML)", "匯出報告（HTML）"),
    "ui.report_file": ("Report", "報告"),
    # WebUI messages
    "msg.need_input": ("Enter an idea or upload an image (at least one)", "請輸入構想文字或上傳圖片（至少一項）"),
    "msg.need_model": ("Choose a model", "請選擇模型"),
    "msg.no_export": ("Nothing to export yet; run once first", "還沒有可以匯出的結果，先執行一次"),
    "msg.bad_address": ("Invalid address: {error}", "位址格式錯誤：{error}"),
    "msg.list_failed": ("Could not list {provider} models: {error}", "無法列出 {provider} 模型：{error}"),
    "msg.refused_count": ("⚠ refused {n}x", "⚠ 拒絕過 {n} 次"),
    "msg.saved_conn": ("Saved to .env; the next run uses the new addresses.", "已儲存到 .env，下一次執行就會使用新位址。"),
    "msg.no_device": ("no device info", "沒有裝置資訊"),
    "msg.starting": ("Starting…", "開始…"),
    "msg.generating": ("Attempt {n}/{max}: {provider} is writing the prompt…", "第 {n}/{max} 次：{provider} 產生 prompt 中…"),
    "msg.judging": ("Attempt {n}: Jev is scoring…", "第 {n} 次：Jev 評分中…"),
    "msg.reference_suffix": (" (reference only)", "（僅供參考）"),
    "msg.refused": (
        "Attempt {n}: {model} refused to write a prompt ⛔ stopped, nothing rendered\nReason: {reason}",
        "第 {n} 次：{model} 拒絕產生 prompt ⛔ 已停止，未生圖\n原因：{reason}",
    ),
    "msg.no_jev_chosen": ("No Jev: sending the first draft as-is", "不用 Jev：直接送出第 1 版"),
    "msg.passed": ("Attempt {n} passed ✅", "第 {n} 次通過 ✅"),
    "msg.not_passed": ("Nothing passed ⚠️ using the best attempt #{n} (NOT PASSED)", "全部未通過 ⚠️ 使用分數最高的第 {n} 次（未達標）"),
    "msg.uploading": ("Uploading the reference image to ComfyUI…", "上傳參考圖到 ComfyUI…"),
    "msg.rendering": ("ComfyUI is rendering{arm}… prompt_id={prompt_id} seed={seed}", "ComfyUI 生圖中{arm}… prompt_id={prompt_id} seed={seed}"),
    "msg.arm.with_jev": (" (with Jev)", "（有 Jev）"),
    "msg.arm.without_jev": (" (without Jev)", "（無 Jev）"),
    "msg.refusal_logged": ("Recorded in {path}\nLog: {log}", "已記錄到 {path}\n紀錄：{log}"),
    "msg.done": ("Done: images saved on the server in output/ ({files})\nLog: {log}", "完成：圖片存在 server output/（{files}）\n紀錄：{log}"),
    "msg.identical": ("The first draft was the final prompt, so both arms are identical and were rendered once.",
                      "第 1 版就是最後送出的版本，兩組相同，只生了一次圖。"),
    "msg.unscored": ("unscored", "未評分"),
    "msg.reference_prefix": ("ref ", "參考 "),
    # report
    "report.title": ("comfy-agent report", "comfy-agent 報告"),
    "report.vpn_note": ("Images are loaded from the ComfyUI server (VPN required)", "圖片從 ComfyUI server 讀取（需連 VPN）"),
    "report.no_idea": ("(no text idea; reference image used)", "（沒有文字構想，使用參考圖）"),
    "report.time": ("time", "時間"),
    "report.mode": ("mode", "模式"),
    "report.model": ("model", "模型"),
    "report.render": ("render", "生圖"),
    "report.rating": ("rating", "分級"),
    "report.style": ("prompt style", "prompt 風格"),
    "report.thresholds": ("thresholds", "門檻"),
    "report.attempt_count": ("attempts", "嘗試次數"),
    "report.ab": ("A/B compare", "A-B 對照"),
    "report.judge_on": ("Jev on", "使用 Jev"),
    "report.judge_off": ("No Jev (scores are reference only)", "不用 Jev（分數僅供參考）"),
    "report.result": ("Result", "結果"),
    "report.compare": ("Comparison", "對照結果"),
    "report.no_images": ("No images (not rendered or refused)", "沒有圖片（未生圖或被拒絕）"),
    "report.no_scores": ("No Jev scores", "沒有 Jev 分數"),
    "report.refused": ("Model refused", "模型拒絕"),
    "report.refused_reason": ("Refusal reason: {reason}", "拒絕原因：{reason}"),
    "report.refused_banner": ("The model refused; the run stopped: {reason}", "模型拒絕，流程已停止：{reason}"),
    "report.unscored": ("Unscored", "未評分"),
    "report.ref_pass": ("Reference: passes", "參考：達標"),
    "report.ref_fail": ("Reference: fails", "參考：未達標"),
    "report.passed_at": ("Passed on attempt {n}", "第 {n} 輪通過"),
    "report.not_passed": ("Not passed (best sent)", "未達標（送最高分）"),
    "report.delta_title": ("Score difference (with Jev − without Jev)", "分數差異（有 Jev − 無 Jev）"),
    "report.dimension": ("dimension", "面向"),
    "report.difference": ("difference", "差異"),
    "report.weakest": ("weakest dimension", "最弱面向"),
    "report.attempts_title": ("Jev scores per attempt", "每一輪 Jev 評分"),
    "report.star_note": ("★ = the version that was sent", "★ = 最後送出的版本"),
    "report.verdict": ("result", "結果"),
}


def normalize_lang(lang: str | None) -> str:
    return lang if lang in LANGS else "en"


def t(key: str, lang: str | None = "en", **fields: object) -> str:
    """Translate `key`; unknown keys fall back to the key itself so a typo is visible, not fatal."""
    en, zh = _T.get(key, (key, key))
    text = zh if normalize_lang(lang) == "zh-TW" else en
    return text.format(**fields) if fields else text
