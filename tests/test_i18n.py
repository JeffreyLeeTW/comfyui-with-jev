import string

from comfy_agent.i18n import _T, normalize_lang, t


def _fields(text):
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


def test_every_key_has_both_languages_with_same_placeholders():
    for key, (en, zh) in _T.items():
        assert en and zh, key
        assert _fields(en) == _fields(zh), key


def test_t_formats_and_falls_back():
    assert t("msg.passed", "en", n=2) == "Attempt 2 passed ✅"
    assert t("msg.passed", "zh-TW", n=2) == "第 2 次通過 ✅"
    assert t("ui.idea", "fr") == "Idea"  # unknown language -> English
    assert t("no.such.key") == "no.such.key"
    assert normalize_lang(None) == "en"
