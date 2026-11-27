"""Gemini as a judge or a debater, to check for same-family effects.

The debaters in most runs are Claude, and so is the default judge. LLM
judges tend to favor output from their own model family, and models from
one lab can share blind spots. A judge from another provider, given the
identical system prompt, user message and output schema, is the control.

As a debater, Gemini gets the same system prompt, turn prompt, tools and
query budget as Claude, so a difference in how hard it argues the false
side comes from the model.

Kept in its own module so the Claude code path never imports a second SDK.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from debate.agents import (
    JUDGE_SYSTEM,
    MAX_QUERIES_PER_TURN,
    RefusedError,
    Verdict,
    build_tools,
    check_decision,
    debater_prompt,
    debater_system,
    extract_statement,
    judge_prompt,
    run_tool,
)
from debate.transcript import Transcript

GEMINI_JUDGE_MODEL = "gemini-3.8-flash"
GEMINI_DEBATER_MODEL = "gemini-4-neon"
MAX_CALLS_PER_TURN = 8  # model round trips, not tool calls; a guard against a loop that never writes

# USD per 1M tokens (input, output incl. thinking), paid tier, standard, as of
# 2026-11-27 (aistudio.google.com/docs/pricing). Several are promotional and
# change soon: 3.8 Flash doubles on 2027-01-01, Carbon and Argon on 2026-12-01.
GEMINI_PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-4-neon": (3.00, 15.00),
    "gemini-4-hydrogen": (1.50, 7.50),
    "gemini-4-carbon": (2.00, 10.00),
    "gemini-4-argon": (2.00, 10.00),
}


class GeminiUsage:
    """Gemini token counts and cost. Output includes thinking tokens, as Google bills them."""

    def __init__(self) -> None:
        self.tokens: dict[str, list[int]] = {}

    def add(self, model: str, meta) -> None:
        row = self.tokens.setdefault(model, [0, 0])
        row[0] += meta.prompt_token_count or 0
        row[1] += (meta.candidates_token_count or 0) + (getattr(meta, "thoughts_token_count", 0) or 0)

    def cost(self) -> float:
        total = 0.0
        for model, (i, o) in self.tokens.items():
            pi, po = GEMINI_PRICES.get(model, (0.0, 0.0))
            total += i / 1e6 * pi + o / 1e6 * po
        return total

    def summary(self) -> str:
        rows = [f"{m}: {i:,} in / {o:,} out" for m, (i, o) in self.tokens.items()]
        return "; ".join(rows) + f" (~${self.cost():.2f})"


def gemini_judge_text(
    client: genai.Client, judge_view: str, hypotheses, usage: GeminiUsage, model: str = GEMINI_JUDGE_MODEL
) -> Verdict:
    response = client.models.generate_content(
        model=model,
        contents=judge_prompt(judge_view, hypotheses),
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM,
            response_mime_type="application/json",
            response_schema=Verdict,
        ),
    )
    if response.usage_metadata:
        usage.add(model, response.usage_metadata)
    verdict = response.parsed
    if verdict is None:
        reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
        raise RefusedError(f"Gemini judge returned no parsable verdict ({reason})")
    if not isinstance(verdict, Verdict):
        verdict = Verdict.model_validate(verdict)
    return check_decision(verdict, hypotheses)


def make_client() -> genai.Client:
    """A Gemini client that retries, like the Anthropic SDK does by default.

    Without retry options a single dropped connection ends the whole batch.
    """
    return genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=4)))


def _tools(scenario) -> list[types.Tool]:
    decls = [
        types.FunctionDeclaration(
            name=t["name"], description=t["description"], parameters_json_schema=t["input_schema"]
        )
        for t in build_tools(scenario)
    ]
    return [types.Tool(function_declarations=decls)]


def _no_answer(response) -> str:
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        return f"prompt blocked ({response.prompt_feedback.block_reason})"
    if not response.candidates:
        return "no candidates"
    return f"finish reason {response.candidates[0].finish_reason}"


def is_gemini(model: str) -> bool:
    return model.startswith("gemini")


def gemini_debater_turn(
    client: genai.Client,
    transcript: Transcript,
    speaker: str,
    round_: int,
    total_rounds: int,
    verified: bool,
    usage: GeminiUsage,
    model: str = GEMINI_DEBATER_MODEL,
    solo: bool = False,
    closing: bool = False,
) -> str:
    """The Gemini counterpart of agents.debater_turn: same prompts, tools and budget."""
    config = types.GenerateContentConfig(
        system_instruction=debater_system(transcript.scenario, speaker, verified, solo),
        tools=_tools(transcript.scenario),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents: list = [
        types.Content(role="user", parts=[types.Part(text=debater_prompt(transcript, speaker, round_, total_rounds, closing))])
    ]
    queries = 0
    truncations = 0

    for _ in range(MAX_CALLS_PER_TURN):
        response = client.models.generate_content(model=model, contents=contents, config=config)
        if response.usage_metadata:
            usage.add(model, response.usage_metadata)
        if not response.candidates or response.candidates[0].content is None:
            raise RefusedError(f"Debater {speaker} ({model}) gave no answer: {_no_answer(response)}")

        if response.candidates[0].finish_reason == types.FinishReason.MAX_TOKENS:
            truncations += 1
            if truncations > 1:
                raise RefusedError(f"Debater {speaker} ({model}) hit max_tokens twice; skipping the trial")
            continue

        calls = response.function_calls or []
        if not calls:
            text = extract_statement(response.text or "")
            if not text.strip():
                raise RefusedError(f"Debater {speaker} ({model}) returned an empty statement: {_no_answer(response)}")
            transcript.add_statement(speaker, round_, text, closing=closing)
            return text

        # Append the model's own turn unchanged: it carries the thought signatures
        # Gemini needs to continue a function-calling exchange.
        contents.append(response.candidates[0].content)
        parts = []
        for call in calls:
            if queries >= MAX_QUERIES_PER_TURN:
                content, is_error = "Query budget for this turn is used up. Write your statement now.", True
            else:
                queries += 1
                content, is_error = run_tool(transcript, speaker, call.name, dict(call.args or {}))
            part = types.Part.from_function_response(
                name=call.name, response={"error": content} if is_error else {"result": content}
            )
            if call.id:
                part.function_response.id = call.id
            parts.append(part)
        contents.append(types.Content(role="user", parts=parts))

    raise RefusedError(f"Debater {speaker} ({model}) never wrote a statement in {MAX_CALLS_PER_TURN} calls")


__all__ = ["make_client", "GEMINI_DEBATER_MODEL", "GEMINI_JUDGE_MODEL", "GEMINI_PRICES", "gemini_debater_turn", "is_gemini", "GeminiUsage", "gemini_judge_text"]
