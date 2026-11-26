from types import SimpleNamespace as NS

import pytest

from debate.agents import MAX_QUERIES_PER_TURN, Usage, Verdict, debater_turn, model_judge
from debate.scenario import booster_station, summarize
from debate.transcript import Transcript


@pytest.fixture
def transcript():
    return Transcript(booster_station(), {"A": "pump", "B": "instrument"})


# --- the scenario has the physics we claim -----------------------------------

def test_scenario_is_deterministic():
    assert booster_station(seed=3).series == booster_station(seed=3).series


def test_pressure_falls_while_flow_and_current_stay_flat():
    s = booster_station().series
    assert summarize(s["PT-101"], 3, 8)["trend_per_hour"] < -2.0
    assert abs(summarize(s["FT-101"], 3, 8)["trend_per_hour"]) < 2.0
    assert abs(summarize(s["JT-101"], 3, 8)["trend_per_hour"]) < 0.1


def test_red_herring_vibration_spike_is_present_but_brief():
    s = booster_station().series
    assert summarize(s["VT-101"], 2.4, 2.7)["max"] > 4.0
    assert summarize(s["VT-101"], 4, 8)["max"] < 2.6


def test_summarize_rejects_empty_window():
    with pytest.raises(ValueError):
        summarize(booster_station().series["PT-101"], 5, 5)


# --- evidence ledger ----------------------------------------------------------

def test_harness_assigns_evidence_ids(transcript):
    e1 = transcript.query("A", "PT-101", 0, 8)
    e2 = transcript.query("B", "FT-101", 0, 8)
    assert (e1.id, e2.id) == ("E1", "E2")
    assert "requested by B" in e2.render()


def test_unknown_tag_raises(transcript):
    with pytest.raises(KeyError):
        transcript.query("A", "XX-999", 0, 8)


def test_fabricated_citation_is_flagged(transcript):
    transcript.query("A", "PT-101", 0, 8)
    st = transcript.add_statement("A", 1, "Pressure fell [E1] and vibration doubled [E7].")
    assert st.bad_citations == ("E7",)
    assert "E7 cited by Debater A does not exist" in transcript.render()


def test_unverified_render_hides_ledger_and_warnings(transcript):
    transcript.query("A", "PT-101", 0, 8)
    transcript.add_statement("A", 1, "See [E9].")
    text = transcript.render(show_evidence=False)
    assert "VERIFIED EVIDENCE" not in text
    assert "does not exist" not in text
    assert "See [E9]." in text


# --- debater loop, against a scripted fake client ------------------------------

def _tool_use(i, **inp):
    return NS(type="tool_use", id=f"tu{i}", name="query_historian", input=inp)


def _resp(stop, *blocks):
    return NS(stop_reason=stop, content=list(blocks), model="claude-opus-5-5",
              usage=NS(input_tokens=100, output_tokens=50), stop_details=None)


class FakeClient:
    def __init__(self, responses, parsed=None):
        self._responses = iter(responses)
        self.calls = []
        self._parsed = parsed
        self.beta = NS(messages=NS(create=self._create))
        self.messages = NS(parse=self._parse)

    def _create(self, **kw):
        self.calls.append(kw)
        return next(self._responses)

    def _parse(self, **kw):
        self.calls.append(kw)
        return NS(stop_reason="end_turn", parsed_output=self._parsed, model="claude-haiku-5-5",
                  usage=NS(input_tokens=10, output_tokens=5), stop_details=None)


def test_debater_turn_queries_then_states(transcript):
    client = FakeClient([
        _resp("tool_use", _tool_use(1, tag="FT-101", start_hour=3, end_hour=8)),
        _resp("end_turn", NS(type="text", text="Flow is flat [E1], so the pump is fine.")),
    ])
    text = debater_turn(client, transcript, "B", 1, 3, True, Usage())
    assert text.startswith("Flow is flat")
    assert [e.tag for e in transcript.evidence] == ["FT-101"]
    assert transcript.statements[-1].bad_citations == ()
    # The tool result sent back is the harness-rendered evidence, not model text.
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["content"].startswith("E1 (requested by B)")


def test_query_budget_is_enforced(transcript):
    many = [_tool_use(i, tag="PT-101", start_hour=0, end_hour=8) for i in range(MAX_QUERIES_PER_TURN + 2)]
    client = FakeClient([
        _resp("tool_use", *many),
        _resp("end_turn", NS(type="text", text="Done.")),
    ])
    debater_turn(client, transcript, "A", 1, 3, True, Usage())
    assert len(transcript.evidence) == MAX_QUERIES_PER_TURN
    results = client.calls[1]["messages"][-1]["content"]
    assert sum(1 for r in results if r.get("is_error")) == 2


def test_bad_tool_input_returns_error_result(transcript):
    client = FakeClient([
        _resp("tool_use", _tool_use(1, tag="PT-101", start_hour=6, end_hour=2)),
        _resp("end_turn", NS(type="text", text="ok")),
    ])
    debater_turn(client, transcript, "A", 1, 3, True, Usage())
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True
    assert transcript.evidence == []


def test_judge_gets_no_tools_and_no_ledger_when_unverified(transcript):
    transcript.query("A", "PT-101", 0, 8)
    v = Verdict(decision="instrument", crux="flow", deciding_evidence=[], reasoning="r", confidence="high")
    client = FakeClient([], parsed=v)
    assert model_judge(client, transcript, verified=False, usage=Usage()).decision == "instrument"
    call = client.calls[0]
    assert "tools" not in call
    assert "VERIFIED EVIDENCE" not in call["messages"][0]["content"]


def test_judge_rejects_unknown_decision(transcript):
    v = Verdict(decision="gremlins", crux="", deciding_evidence=[], reasoning="", confidence="low")
    with pytest.raises(ValueError):
        model_judge(FakeClient([], parsed=v), transcript, verified=True, usage=Usage())


def test_usage_cost():
    u = Usage()
    u.add("claude-opus-5-5", NS(input_tokens=1_000_000, output_tokens=0))
    u.add("claude-haiku-5-5", NS(input_tokens=0, output_tokens=1_000_000))
    assert u.cost() == pytest.approx(4.50)
