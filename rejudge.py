"""Re-judge saved transcripts with a Claude judge and a Gemini judge.

Each runs/*.md file holds exactly what the original judge saw. This replays
that text through both judges with the *current* judge prompt, so the two
are compared on identical input and identical instructions. Debaters aren't
re-run, so the cost is two small judge calls per transcript.

    uv run rejudge.py                    # every transcript in runs/
    uv run rejudge.py --limit 5          # quick check
    uv run rejudge.py --only lying       # transcripts whose condition contains "lying"

Writes runs/rejudge-<timestamp>.md (per-transcript verdicts and a summary
by condition) and prints the summary.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from google import genai

from debate.agents import JUDGE_MODEL, RefusedError, Usage, claude_judge_text
from debate.gemini import GEMINI_JUDGE_MODEL, GeminiUsage, gemini_judge_text
from debate.scenario import HYPOTHESES

RUNS = Path(__file__).parent / "runs"
NAME = re.compile(r"^\d{8}-\d{6}-(?P<cond>.+)-(?P<trial>\d+)\.md$")
VIEW = re.compile(r"## What the judge saw\n\n```text\n(?P<view>.*?)\n```", re.S)
TRUTH = re.compile(r"truth: `(?P<truth>\w+)` · judge decided: `(?P<orig>\w+)`")


@dataclass
class Case:
    path: Path
    condition: str
    truth: str
    original: str
    view: str


def load_cases() -> list[Case]:
    cases = []
    for path in sorted(RUNS.glob("*.md")):
        m = NAME.match(path.name)
        if not m:
            continue  # rejudge reports and anything else
        text = path.read_text()
        v, t = VIEW.search(text), TRUTH.search(text)
        if not (v and t):
            print(f"skipping {path.name}: unrecognized format", file=sys.stderr)
            continue
        cond = m["cond"]
        if not cond.startswith(("easy-", "hard-")):
            cond = f"easy-{cond}"  # runs from before the --scenario flag
        cases.append(Case(path, cond, t["truth"], t["orig"], v["view"]))
    return cases


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", default="", help="substring filter on the condition name")
    p.add_argument("--claude-model", default=JUDGE_MODEL)
    p.add_argument("--gemini-model", default=GEMINI_JUDGE_MODEL)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    load_dotenv(Path(__file__).parent / ".env")
    cases = [c for c in load_cases() if args.only in c.condition]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("no transcripts found", file=sys.stderr)
        return 1

    claude, gemini = anthropic.Anthropic(), genai.Client()
    cu, gu = Usage(), GeminiUsage()

    def judge(case: Case) -> tuple[Case, str, str]:
        out = []
        for fn, client, usage, model in (
            (claude_judge_text, claude, cu, args.claude_model),
            (gemini_judge_text, gemini, gu, args.gemini_model),
        ):
            try:
                out.append(fn(client, case.view, HYPOTHESES, usage, model).decision)
            except (RefusedError, ValueError, anthropic.APIError, genai.errors.APIError) as exc:
                print(f"  {case.path.name}: {model} failed: {exc}", file=sys.stderr)
                out.append("error")
        print(f"  {case.path.name}: claude={out[0]} gemini={out[1]} (truth {case.truth})", file=sys.stderr)
        return case, out[0], out[1]

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(judge, cases))

    by_cond: dict[str, list] = defaultdict(list)
    for case, c, g in results:
        by_cond[case.condition].append((case, c, g))

    def score(rows, i):
        valid = [r for r in rows if r[i] != "error"]
        return f"{sum(r[i] == r[0].truth for r in valid)}/{len(valid)}"

    header = f"| Condition | n | Original | {args.claude_model} | {args.gemini_model} | Judges disagree |\n|---|---|---|---|---|---|"
    lines = [header]
    for cond in sorted(by_cond):
        rows = [(case, case.original, c, g) for case, c, g in by_cond[cond]]
        orig = f"{sum(r[1] == r[0].truth for r in rows)}/{len(rows)}"
        cl = score([(r[0], r[2]) for r in rows], 1)
        ge = score([(r[0], r[3]) for r in rows], 1)
        dis = sum(r[2] != r[3] and "error" not in (r[2], r[3]) for r in rows)
        lines.append(f"| {cond} | {len(rows)} | {orig} | {cl} | {ge} | {dis} |")
    summary = "\n".join(lines)

    detail = "\n".join(
        f"| {case.path.name} | {case.truth} | {case.original} | {c} | {g} |" for case, c, g in results
    )
    report = RUNS / f"rejudge-{datetime.now():%Y%m%d-%H%M%S}.md"
    report.write_text(
        f"# Re-judge: {args.claude_model} vs {args.gemini_model}\n\n"
        "Both judges got the identical system prompt, transcript and schema (current judge prompt). "
        "'Original' is the verdict recorded when the run happened, possibly under an older judge prompt.\n\n"
        f"{summary}\n\n## Per transcript\n\n| File | Truth | Original | Claude | Gemini |\n|---|---|---|---|---|\n{detail}\n\n"
        f"Usage: {cu.summary()}; {gu.summary()}\n"
    )
    print(summary)
    print(f"\n{cu.summary()}; {gu.summary()}\nsaved {report.relative_to(Path.cwd()) if report.is_relative_to(Path.cwd()) else report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
