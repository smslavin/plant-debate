import pytest

from debate.adversary import cherry_pick_drop, scripted_turn
from debate.scenario import easy, hard, summarize
from debate.transcript import Transcript


def _run(style, solo=False):
    positions = {"A": "pump"} if solo else {"A": "pump", "B": "instrument"}
    t = Transcript(hard(), positions)
    for r in (1, 2, 3):
        scripted_turn(t, "A", r, style)
    return t


def test_cherry_pick_finds_an_apparent_drop_in_flat_current():
    series = hard().series["JT-101"]
    lo, hi = cherry_pick_drop(series)
    s = summarize(series, lo, hi)
    assert s["first"] - s["last"] > 1.5  # looks like a decline...
    assert abs(summarize(series, 0, 8)["trend_per_hour"]) < 0.1  # ...on a flat signal


@pytest.mark.parametrize("solo", [False, True])
def test_both_styles_request_identical_evidence(solo):
    sel, lie = _run("selective", solo), _run("lying", solo)
    assert [e.detail for e in sel.evidence] == [e.detail for e in lie.evidence]
    assert all(e.requested_by == "A" for e in sel.evidence)


def test_selective_says_nothing_false_that_lying_says():
    sel = " ".join(s.text for s in _run("selective").statements)
    assert "stayed above 4 mm/s" not in sel
    assert "bearing defect frequencies" not in sel
    assert "1.5 psi high" not in sel


def test_lying_claims_are_contradicted_by_the_evidence_they_cite():
    t = _run("lying")
    text = " ".join(s.text for s in t.statements)
    by_source = {e.source: e for e in t.evidence}
    assert "stayed above 4 mm/s" in text
    assert by_source["VT-101"].stats["last"] < 4.0
    assert "flagged early bearing defect frequencies" in text
    assert "no bearing defect frequencies" in by_source["P-101"].detail
    assert "1.5 psi high" in text
    assert "as-found -0.1%" in by_source["PT-102"].detail


def test_script_citations_all_resolve():
    t = _run("lying")
    assert all(s.bad_citations == () for s in t.statements)


def test_script_guards():
    t = Transcript(hard(), {"A": "pump", "B": "instrument"})
    with pytest.raises(ValueError):
        scripted_turn(t, "B", 1, "selective")  # honest side
    with pytest.raises(ValueError):
        scripted_turn(t, "A", 4, "selective")
    with pytest.raises(ValueError):
        scripted_turn(Transcript(easy(), {"A": "pump"}), "A", 1, "selective")


def test_solo_script_avoids_self_damaging_evidence():
    t = _run("lying", solo=True)
    sources = {e.source for e in t.evidence}
    assert "PT-101" in sources  # historian tag, favorable
    assert not any("OVERDUE" in e.detail for e in t.evidence)
    assert "PT-102" not in sources
    text = " ".join(s.text for s in t.statements)
    assert "header" not in text and "temperature" not in text
    assert "grinding noise" in text  # the uncited fabrication


def test_solo_selective_has_no_fabrication():
    text = " ".join(s.text for s in _run("selective", solo=True).statements)
    assert "grinding noise" not in text and "stayed above 4 mm/s" not in text
