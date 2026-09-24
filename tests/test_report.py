import json

from conftest import BAD, GOOD, FakeSystemOne, all_dims
from test_pipeline import FakeComfy, FakeOpenRouter

from comfy_agent.judge import JevJudge
from comfy_agent.pipeline import Pipeline
from comfy_agent.report import build_report, export_report


def _pipeline(settings, calls):
    return Pipeline(settings, FakeOpenRouter(), JevJudge(FakeSystemOne(calls)), FakeComfy())


def test_run_report(settings):
    r = _pipeline(settings, [all_dims(BAD), all_dims(GOOD)]).run("<b>girl</b>", "m:free", rating="sfw")
    path = export_report(r.log_path)
    html = path.read_text()
    assert path.parent == settings.runs_dir / "reports" and path.suffix == ".html"
    assert "&lt;b&gt;girl&lt;/b&gt;" in html and "<b>girl</b>" not in html  # escaped
    assert "http://x/view" in html and "Passed on attempt 2" in html and "SFW" in html
    assert '<html lang="en">' in html and "Natural language" in html  # config default style


def test_compare_report_has_both_arms_and_delta(settings):
    r = _pipeline(settings, [all_dims(BAD), all_dims(GOOD)]).compare("a girl", "m:free")
    html = export_report(r.log_path).read_text()
    assert "With Jev" in html and "Without Jev (first draft)" in html and "Score difference" in html
    assert "+0.6" in html  # GOOD (~0.97) - BAD (~0.33)


def test_report_handles_refusal_and_old_logs(settings, tmp_path):
    orc = FakeOpenRouter(refuse_on=1)
    r = Pipeline(settings, orc, JevJudge(FakeSystemOne([])), FakeComfy()).run("x", "m:free")
    assert "Model refused" in export_report(r.log_path).read_text()

    old = {"idea": "old", "model": "m", "mode": "txt2img", "passed": True, "chosen_attempt": 1,
           "thresholds": {"fidelity": 0.67}, "generation": {"seed": 1}, "images": [],
           "attempts": [{"number": 1, "positive": "p", "negative": "n", "image_description": "",
                         "passed": True, "scores": {"fidelity": {"value": 0.9, "confidence": 1, "level": "x"}}}]}
    assert "0.90" in build_report(old)


def test_report_in_traditional_chinese(settings):
    r = _pipeline(settings, [all_dims(BAD), all_dims(GOOD)]).compare("a girl", "m:free")
    html = export_report(r.log_path, lang="zh-TW").read_text()
    assert '<html lang="zh-Hant">' in html and "有 Jev" in html and "分數差異" in html
    assert "Score difference" not in html


def test_manual_report(settings):
    r = _pipeline(settings, []).manual("<i>1girl</i>", "blurry")
    html = export_report(r.log_path).read_text()
    assert "Manual prompt" in html and "&lt;i&gt;1girl&lt;/i&gt;" in html and "http://x/view" in html
    assert "No Jev scores" not in html and "thresholds" not in html
    assert "手動 prompt" in export_report(r.log_path, lang="zh-TW").read_text()
