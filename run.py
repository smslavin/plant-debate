"""Run a debate (or a single-advocate consultancy baseline) over the booster-pump alarm.

Examples:
    uv run run.py                          # one debate, honest side chosen at random, Haiku judge
    uv run run.py --judge human            # you are the judge
    uv run run.py --mode consultancy --honest none   # one advocate argues the false side
    uv run run.py --unverified             # judge sees prose only, no evidence ledger
    uv run run.py --trials 6               # small tally across honest-side assignments
    uv run run.py --false-model gemini-4-neon        # Gemini argues the false side against Claude
    uv run run.py --judge-model gemini-3.8-flash     # Gemini judges
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import httpx2
from dotenv import load_dotenv
from google.genai import errors as genai_errors

from debate.agents import (
    DEBATER_MODEL,
    JUDGE_MODEL,
    RefusedError,
    Usage,
    debater_turn,
    human_judge,
    model_judge,
    verdict_json,
)
from debate.gemini import GeminiUsage, make_client, gemini_debater_turn, gemini_judge_text, is_gemini
from debate.adversary import STYLES, scripted_turn
from debate.scenario import SCENARIOS
from debate.transcript import Transcript

RUNS = Path(__file__).parent / "runs"


def assign_positions(mode: str, honest: str, rng: random.Random, truth: str, false: str) -> dict[str, str]:
    if mode == "consultancy":
        if honest == "random":
            honest = rng.choice(["A", "none"])
        return {"A": truth if honest == "A" else false}
    if honest == "random":
        honest = rng.choice(["A", "B"])
    return {"A": truth, "B": false} if honest == "A" else {"A": false, "B": truth}


class Clients:
    """API clients and usage per provider. The Gemini client is made only if a Gemini model is used."""

    def __init__(self) -> None:
        self.claude = anthropic.Anthropic()
        self.usage = Usage()
        self.gusage = GeminiUsage()
        self._gemini = None

    @property
    def gemini(self):
        if self._gemini is None:
            self._gemini = make_client()
        return self._gemini

    def turn(self, model: str, *args, **kwargs) -> str:
        if is_gemini(model):
            return gemini_debater_turn(self.gemini, *args, usage=self.gusage, model=model, **kwargs)
        return debater_turn(self.claude, *args, usage=self.usage, model=model, **kwargs)

    def judge(self, model: str, transcript: Transcript, verified: bool):
        if is_gemini(model):
            view = transcript.render(show_evidence=verified)
            return gemini_judge_text(self.gemini, view, transcript.scenario.hypotheses, self.gusage, model)
        return model_judge(self.claude, transcript, verified, self.usage, model=model)

    def summary(self) -> str:
        parts = [u.summary() for u in (self.usage, self.gusage) if u.tokens]
        return "; ".join(parts) or "no model calls"


def run_once(clients: Clients, args, rng: random.Random) -> tuple[Transcript, object]:
    scenario = SCENARIOS[args.scenario](args.seed)
    truth = scenario.truth
    false = next(h.key for h in scenario.hypotheses if h.key != truth)
    positions = assign_positions(args.mode, args.honest, rng, truth, false)
    transcript = Transcript(scenario, positions)
    verified = not args.unverified
    solo = args.mode == "consultancy"

    for r in range(1, args.rounds + 1):
        for speaker in sorted(positions):
            scripted = args.adversary != "none" and positions[speaker] != truth
            print(f"  round {r}: Debater {speaker} ({positions[speaker]}{', scripted' if scripted else ''})...",
                  file=sys.stderr)
            if scripted:
                scripted_turn(transcript, speaker, r, args.adversary)
                continue
            model = args.debater_model if positions[speaker] == truth else args.false_model
            print(f"    ({model})", file=sys.stderr)
            clients.turn(model, transcript, speaker, r, args.rounds, verified, solo=solo)

    if not solo and not args.no_closing:
        # Each round the second speaker talks last. Give the first speaker one
        # short reply so neither side ends with an unanswered statement.
        first = sorted(positions)[0]
        r = args.rounds + 1
        scripted = args.adversary != "none" and positions[first] != truth
        print(f"  closing: Debater {first} ({positions[first]}{', scripted' if scripted else ''})...", file=sys.stderr)
        if scripted:
            scripted_turn(transcript, first, r, args.adversary, closing=True)
        else:
            model = args.debater_model if positions[first] == truth else args.false_model
            clients.turn(model, transcript, first, r, args.rounds, verified, closing=True)

    if args.judge == "human":
        verdict = human_judge(transcript, verified)
    else:
        print("  judging...", file=sys.stderr)
        verdict = clients.judge(args.judge_model, transcript, verified)
    return transcript, verdict


def write_run(args, transcript: Transcript, verdict, trial: int) -> Path:
    RUNS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    adv = f"-{args.adversary}" if args.adversary != "none" else ""
    fm = f"-{args.false_model}" if args.adversary == "none" and args.false_model != args.debater_model else ""
    jm = f"-judge-{args.judge_model}" if args.judge == "model" and args.judge_model != JUDGE_MODEL else ""
    tag = f"{args.scenario}-{args.mode}{adv}{fm}{'-unverified' if args.unverified else ''}{jm}"
    path = RUNS / f"{stamp}-{tag}-{trial}.md"
    truth = transcript.scenario.truth
    honest = [s for s, k in transcript.positions.items() if k == truth] or ["none"]
    path.write_text(
        f"# {args.mode.title()} run {trial}\n\n"
        f"- Scenario: {args.scenario}\n"
        f"- False side: {'scripted adversary (' + args.adversary + ')' if args.adversary != 'none' else args.false_model}\n"
        f"- Honest side model: {args.debater_model}, judge: {args.judge if args.judge == 'human' else args.judge_model}\n"
        f"- Closing rebuttal: {'no' if args.no_closing or args.mode == 'consultancy' else 'yes'}\n"
        f"- Evidence shown to judge: {'no' if args.unverified else 'yes'}\n"
        f"- Honest side: {', '.join(honest)} · truth: `{truth}` · judge decided: `{verdict.decision}` "
        f"→ **{'correct' if verdict.decision == truth else 'WRONG'}**\n\n"
        f"## What the judge saw\n\n```text\n{transcript.render(show_evidence=not args.unverified)}\n```\n\n"
        f"## Verdict\n\n```json\n{verdict_json(verdict)}\n```\n"
    )
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=list(SCENARIOS), default="hard")
    p.add_argument("--mode", choices=["debate", "consultancy"], default="debate")
    p.add_argument("--honest", default="random",
                   help="debate: A|B|random. consultancy: A (advocate tells the truth)|none|random")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--adversary", choices=("none",) + STYLES, default="none",
                   help="replace the false-side debater with a script (hard scenario only)")
    p.add_argument("--judge", choices=["model", "human"], default="model")
    p.add_argument("--unverified", action="store_true", help="hide the evidence ledger from the judge")
    p.add_argument("--no-closing", action="store_true",
                   help="skip the first speaker's closing rebuttal (the original format)")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=7, help="plant data seed")
    p.add_argument("--debater-model", default=DEBATER_MODEL, help="the honest side (and the false side, unless --false-model)")
    p.add_argument("--false-model", default=None, help="model for the false side, e.g. gemini-4-neon")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    args = p.parse_args()

    valid = {"debate": {"A", "B", "random"}, "consultancy": {"A", "none", "random"}}[args.mode]
    if args.honest not in valid:
        p.error(f"--honest for {args.mode} must be one of {sorted(valid)}")

    if args.adversary != "none":
        if args.scenario != "hard":
            p.error("--adversary needs --scenario hard")
        if args.rounds > 3:
            p.error("the scripted adversary has 3 rounds")
        if args.mode == "consultancy" and args.honest != "none":
            p.error("with --adversary, consultancy needs --honest none (the script only argues the false side)")

    if args.false_model is None:
        args.false_model = args.debater_model
    elif args.adversary != "none":
        p.error("--false-model and --adversary both set the false side; pick one")

    load_dotenv(Path(__file__).parent / ".env")
    clients = Clients()
    rng = random.Random()
    correct = 0

    for trial in range(1, args.trials + 1):
        print(f"trial {trial}/{args.trials}", file=sys.stderr)
        try:
            transcript, verdict = run_once(clients, args, rng)
        except (RefusedError, genai_errors.APIError, anthropic.APIError, httpx2.TransportError) as exc:
            print(f"  skipped: {exc}", file=sys.stderr)
            continue
        ok = verdict.decision == transcript.scenario.truth
        correct += ok
        path = write_run(args, transcript, verdict, trial)
        print(f"  judge decided '{verdict.decision}' ({'correct' if ok else 'WRONG'}): {verdict.crux}")
        print(f"  saved {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")

    print(f"\n{correct}/{args.trials} correct · {clients.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
