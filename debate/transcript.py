"""The debate record: an evidence ledger plus the statements that cite it.

The key mechanism in this demo: debaters never type numbers into the
evidence ledger. They ask the harness for data, and the harness writes
the result into the ledger under an ID (E1, E2, ...). A debater can
misread evidence, but cannot forge it. This is the industrial version of
the paper's MNIST experiment, where debaters reveal real pixels to the
judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from debate.scenario import TAGS, Scenario, hhmm, summarize

CITATION = re.compile(r"\[(E\d+)\]")


@dataclass(frozen=True)
class Evidence:
    id: str
    requested_by: str
    tag: str
    start_hour: float
    end_hour: float
    stats: dict[str, float]

    def render(self) -> str:
        t = TAGS[self.tag]
        s = self.stats
        return (
            f"{self.id} (requested by {self.requested_by}) · {self.tag} {t.description} [{t.unit}] · "
            f"{hhmm(self.start_hour)}-{hhmm(self.end_hour)} · "
            f"first {s['first']:.1f}, last {s['last']:.1f}, min {s['min']:.1f}, "
            f"max {s['max']:.1f}, mean {s['mean']:.1f}, trend {s['trend_per_hour']:+.2f}/h"
        )


@dataclass(frozen=True)
class Statement:
    speaker: str
    round: int
    text: str
    bad_citations: tuple[str, ...] = ()


@dataclass
class Transcript:
    scenario: Scenario
    positions: dict[str, str]  # speaker -> hypothesis key
    evidence: list[Evidence] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)

    def query(self, speaker: str, tag: str, start_hour: float, end_hour: float) -> Evidence:
        if tag not in TAGS:
            raise KeyError(f"Unknown tag {tag!r}. Known tags: {', '.join(TAGS)}")
        stats = summarize(self.scenario.series[tag], start_hour, end_hour)
        ev = Evidence(f"E{len(self.evidence) + 1}", speaker, tag, start_hour, end_hour, stats)
        self.evidence.append(ev)
        return ev

    def add_statement(self, speaker: str, round_: int, text: str) -> Statement:
        known = {e.id for e in self.evidence}
        bad = tuple(sorted({c for c in CITATION.findall(text) if c not in known}))
        st = Statement(speaker, round_, text.strip(), bad)
        self.statements.append(st)
        return st

    def hypothesis(self, speaker: str):
        key = self.positions[speaker]
        return next(h for h in self.scenario.hypotheses if h.key == key)

    def case_brief(self) -> str:
        sc = self.scenario
        lines = [f"ALARM: {sc.alarm}", sc.question, "", "Positions:"]
        for speaker in sorted(self.positions):
            h = self.hypothesis(speaker)
            lines.append(f"  Debater {speaker}: {h.claim} Recommended action: {h.action}")
        return "\n".join(lines)

    def render(self, show_evidence: bool = True) -> str:
        """What the judge sees.

        With show_evidence=False the judge gets only the debaters' prose:
        any number a debater quotes is unverifiable. Comparing the two modes
        shows how much of debate's value comes from verified evidence.
        """
        parts = [self.case_brief(), ""]
        if show_evidence:
            parts.append("VERIFIED EVIDENCE (retrieved by the harness from the plant historian):")
            parts.extend(f"  {e.render()}" for e in self.evidence)
            if not self.evidence:
                parts.append("  (none)")
            parts.append("")
        parts.append("DEBATE:")
        for st in self.statements:
            parts.append(f"[Round {st.round}] Debater {st.speaker}: {st.text}")
            if show_evidence and st.bad_citations:
                parts.append(
                    f"  ⚠ harness: {', '.join(st.bad_citations)} cited by Debater {st.speaker} "
                    "does not exist in the evidence ledger."
                )
        return "\n".join(parts)
