"""Gemini as a judge, to check for same-family bias.

The debaters in most runs are Claude, and so is the default judge. LLM
judges tend to favor output from their own model family, and models from
one lab can share blind spots. A judge from another provider, given the
identical system prompt, user message and output schema, is the control.

Kept in its own module so the Claude code path never imports a second SDK.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from debate.agents import JUDGE_SYSTEM, RefusedError, Verdict, check_decision, judge_prompt

GEMINI_JUDGE_MODEL = "gemini-3.8-flash"

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


__all__ = ["GEMINI_JUDGE_MODEL", "GEMINI_PRICES", "GeminiUsage", "gemini_judge_text"]
