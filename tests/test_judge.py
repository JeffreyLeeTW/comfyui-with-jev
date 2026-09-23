from conftest import BAD, GOOD, MID, FakeSystemOne, all_dims

from comfy_agent.judge import JevJudge

TH = {"fidelity": 0.67, "format": 0.67, "completeness": 0.67, "negative": 0.67}


def test_all_good_passes_and_state_is_structured():
    fake = FakeSystemOne([all_dims(GOOD)])
    v = JevJudge(fake, "jev-latest").evaluate("idea", "", "1girl", "lowres", TH)
    assert v.passed
    state, _, model = fake.requests[0]
    assert model == "jev-latest"
    assert state["positive_prompt"] == "1girl" and state["image_description"] == "(no reference image)"


def test_one_failing_dimension_fails_with_targeted_critique():
    fake = FakeSystemOne([all_dims(GOOD) | {"completeness": BAD}])
    v = JevJudge(fake).evaluate("idea", "", "p", "n", TH)
    assert not v.passed
    assert not v.dimensions["completeness"].passed
    crit = v.critique()
    assert "completeness" in crit and "fidelity" not in crit
    assert "Subject plus one other aspect" in crit  # most likely level is quoted


def test_normalization_and_rank_value():
    fake = FakeSystemOne([all_dims(MID) | {"format": BAD}])
    v = JevJudge(fake).evaluate("i", "", "p", "n", TH)
    assert abs(v.dimensions["fidelity"].value - 2 / 3) < 1e-9
    assert abs(v.rank_value - 1 / 3) < 1e-9


def test_threshold_is_respected():
    fake = FakeSystemOne([all_dims(MID), all_dims(MID)])
    j = JevJudge(fake)
    assert j.evaluate("i", "", "p", "n", {k: 0.6 for k in TH}).passed
    assert not j.evaluate("i", "", "p", "n", {k: 0.7 for k in TH}).passed
