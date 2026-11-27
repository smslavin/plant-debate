"""Offline tests for the Gemini debater loop (fake client, no API calls)."""

from types import SimpleNamespace

from google.genai import types

from debate.gemini import GeminiUsage, gemini_debater_turn
from debate.scenario import hard
from debate.transcript import Transcript


def _resp(parts):
    content = types.Content(role="model", parts=parts)
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=content, finish_reason="STOP")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=100, candidates_token_count=20),
    )


def _call(name, **args):
    return types.Part(function_call=types.FunctionCall(id=f"c-{name}", name=name, args=args))


class FakeGemini:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.requests.append(list(contents))
        return self.responses.pop(0)


def test_gemini_turn_queries_ledger_then_states():
    t = Transcript(hard(), {"A": "pump", "B": "instrument"})
    client = FakeGemini([
        _resp([_call("query_historian", tag="FCV-101", start_hour=0, end_hour=8)]),
        _resp([types.Part(text="The valve opened [E1].")]),
    ])
    usage = GeminiUsage()
    text = gemini_debater_turn(client, t, "A", 1, 3, True, usage)
    assert text == "The valve opened [E1]."
    assert t.evidence[0].requested_by == "A" and t.evidence[0].source == "FCV-101"
    assert t.statements[-1].bad_citations == ()
    # second request carries the model turn plus a function response with the same id
    reply = client.requests[1][-1].parts[0].function_response
    assert reply.id == "c-query_historian" and "result" in reply.response
    assert usage.tokens["gemini-4-neon"] == [200, 40]


def test_gemini_turn_enforces_query_budget_and_reports_bad_input():
    t = Transcript(hard(), {"A": "pump", "B": "instrument"})
    calls = [_call("query_historian", tag="PT-101", start_hour=0, end_hour=8) for _ in range(4)]
    calls.append(_call("read_maintenance_log", asset="NOPE"))
    client = FakeGemini([_resp(calls), _resp([types.Part(text="done")])])
    gemini_debater_turn(client, t, "A", 1, 3, True, GeminiUsage())
    replies = [p.function_response.response for p in client.requests[1][-1].parts]
    assert len(t.evidence) == 3
    assert all("error" in r for r in replies[3:])
