"""Run a debate (or a single-advocate consultancy baseline) over the booster-pump alarm.

Examples:
    uv run run.py                          # one debate, honest side chosen at random, Haiku judge
    uv run run.py --judge human            # you are the judge
    uv run run.py --mode consultancy --honest none   # one advocate argues the false side
    uv run run.py --unverified             # judge sees prose only, no evidence ledger
    uv run run.py --trials 6               # small tally across honest-side assignments
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

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


def run_once(client, args, rng: random.Random, usage: Usage) -> tuple[Transcript, object]:
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
            debater_turn(
                client, transcript, speaker, r, args.rounds, verified, usage,
                model=args.debater_model, solo=solo,
            )

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
            debater_turn(
                client, transcript, first, r, args.rounds, verified, usage,
                model=args.debater_model, closing=True,
            )

    if args.judge == "human":
        verdict = human_judge(transcript, verified)
    else:
        print("  judging...", file=sys.stderr)
        verdict = model_judge(client, transcript, verified, usage, model=args.judge_model)
    return transcript, verdict


def write_run(args, transcript: Transcript, verdict, trial: int) -> Path:
    RUNS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    adv = f"-{args.adversary}" if args.adversary != "none" else ""
    tag = f"{args.scenario}-{args.mode}{adv}{'-unverified' if args.unverified else ''}"
    path = RUNS / f"{stamp}-{tag}-{trial}.md"
    truth = transcript.scenario.truth
    honest = [s for s, k in transcript.positions.items() if k == truth] or ["none"]
    path.write_text(
        f"# {args.mode.title()} run {trial}\n\n"
        f"- Scenario: {args.scenario}\n"
        f"- False side: {'scripted adversary (' + args.adversary + ')' if args.adversary != 'none' else args.debater_model}\n"
        f"- Debaters: {args.debater_model}, judge: {args.judge if args.judge == 'human' else args.judge_model}\n"
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
    p.add_argument("--debater-model", default=DEBATER_MODEL)
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

    load_dotenv()
    client = anthropic.Anthropic()
    rng = random.Random()
    usage = Usage()
    correct = 0

    for trial in range(1, args.trials + 1):
        print(f"trial {trial}/{args.trials}", file=sys.stderr)
        try:
            transcript, verdict = run_once(client, args, rng, usage)
        except RefusedError as exc:
            print(f"  skipped: {exc}", file=sys.stderr)
            continue
        ok = verdict.decision == transcript.scenario.truth
        correct += ok
        path = write_run(args, transcript, verdict, trial)
        print(f"  judge decided '{verdict.decision}' ({'correct' if ok else 'WRONG'}): {verdict.crux}")
        print(f"  saved {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")

    print(f"\n{correct}/{args.trials} correct · {usage.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
