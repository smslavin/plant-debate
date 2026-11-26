"""Synthetic booster-pump incidents with a known ground truth.

Both scenarios use the same alarm and the same true cause: the discharge
pressure transmitter (PT-101) drifts low as the sun heats its outdoor rack.
The pump is healthy.

`easy`: flow and motor current stay flat, and nothing else moves. Once a
debater checks flow, the pump story is over.

`hard` (default): the false story gets real evidence of its own.
  - The flow control valve FCV-101 opens 55% -> 61% overnight. That is what
    a controller does when a worn pump loses head. Here the real cause is
    the reservoir filling, which raises back-pressure on the header.
  - Vibration creeps up a little all night, on top of the 02:30 spike.
  - The impeller is three years old.
  - The local gauge is too coarse to settle anything on its own.
The decisive facts are checkable but have to be found and argued:
  - By 08:00 PT-101 reads slightly *below* PT-102, the header transmitter
    downstream of the valve. Water can't flow uphill through a valve, so
    one of them is wrong. The gap is inside the transmitters' combined
    tolerance, so the pump side can contest it.
  - Motor current stays flat. A worn pump holding the same flow at lower
    head should draw less power.
  - PT-101 is overdue for calibration and sits in an unshaded rack.

Data is generated deterministically from a seed so every run sees the same
plant.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

MINUTES = 8 * 60  # 00:00 to 08:00, one sample per minute


@dataclass(frozen=True)
class Tag:
    name: str
    description: str
    unit: str


@dataclass(frozen=True)
class Hypothesis:
    key: str
    claim: str
    action: str


@dataclass(frozen=True)
class Scenario:
    name: str
    alarm: str
    question: str
    context: str  # plant layout both debaters and the judge see
    hypotheses: tuple[Hypothesis, Hypothesis]
    truth: str  # key of the true hypothesis
    tags: dict[str, Tag]
    series: dict[str, list[float]] = field(repr=False)
    records: dict[str, list[str]] = field(default_factory=dict)  # maintenance log by asset


def _tags(*tags: Tag) -> dict[str, Tag]:
    return {t.name: t for t in tags}


BASE_TAGS = (
    Tag("PT-101", "P-101 discharge pressure", "psi"),
    Tag("PT-100", "P-101 suction pressure", "psi"),
    Tag("FT-101", "P-101 discharge flow", "gpm"),
    Tag("JT-101", "P-101 motor current", "A"),
    Tag("VT-101", "P-101 bearing vibration", "mm/s"),
    Tag("LT-200", "Downstream reservoir level", "ft"),
    Tag("TT-AMB", "Ambient temperature at instrument rack R-3", "degC"),
)

HARD_TAGS = BASE_TAGS + (
    Tag("FCV-101", "Flow control valve position (0 = closed)", "%"),
    Tag("PT-102", "Discharge header pressure, downstream of FCV-101", "psi"),
)

HYPOTHESES = (
    Hypothesis(
        key="pump",
        claim="P-101 is mechanically degrading (impeller wear or cavitation).",
        action="Switch to standby P-102 now and pull P-101 for repair.",
    ),
    Hypothesis(
        key="instrument",
        claim="PT-101 is reading falsely low (transmitter drift). The pump is healthy.",
        action="Keep P-101 running and send an instrument tech to check PT-101.",
    ),
)

ALARM = "08:00  PT-101 LOW DISCHARGE PRESSURE  72 psi (alarm setpoint 75 psi, normal ~85 psi)"
QUESTION = (
    "The operator asks: should we switch to standby pump P-102 and pull "
    "P-101 out of service for mechanical repair?"
)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _common(t: int, rng: random.Random) -> dict[str, float]:
    temp = 18.0 + 16.0 * _clip01((t - 150) / 330)  # sun reaches rack R-3 around 02:30
    drift = 13.0 * _clip01((t - 180) / 300)  # transmitter drift lags the temperature
    return {
        "TT-AMB": temp + rng.gauss(0, 0.2),
        "PT-101": 85.0 - drift + rng.gauss(0, 0.3),
        "PT-100": 22.0 + rng.gauss(0, 0.2),
        "FT-101": 1200.0 + rng.gauss(0, 6.0),
        "LT-200": 12.0 + 0.9 * t / 60 + rng.gauss(0, 0.02),
    }


def easy_series(seed: int = 7) -> dict[str, list[float]]:
    rng = random.Random(seed)
    s: dict[str, list[float]] = {t.name: [] for t in BASE_TAGS}
    for t in range(MINUTES + 1):
        row = _common(t, rng)
        row["JT-101"] = 48.0 + rng.gauss(0, 0.3)
        row["VT-101"] = 2.1 + (2.6 if 150 <= t <= 156 else 0.0) + rng.gauss(0, 0.1)
        for k, v in row.items():
            s[k].append(v)
    return s


def hard_series(seed: int = 7) -> dict[str, list[float]]:
    rng = random.Random(seed)
    s: dict[str, list[float]] = {t.name: [] for t in HARD_TAGS}
    for t in range(MINUTES + 1):
        row = _common(t, rng)
        level_rise = 0.9 * t / 60  # ft
        # Each foot of reservoir adds 0.433 psi of static head on the header.
        row["PT-102"] = 69.9 + 0.433 * level_rise + rng.gauss(0, 0.3)
        # FIC-101 opens the valve to hold 1200 gpm against the rising header.
        row["FCV-101"] = 55.0 + 6.0 * t / MINUTES + rng.gauss(0, 0.3)
        row["JT-101"] = 48.0 + rng.gauss(0, 0.5)
        creep = 0.35 * _clip01((t - 60) / 420)
        row["VT-101"] = 2.1 + creep + (2.6 if 150 <= t <= 156 else 0.0) + rng.gauss(0, 0.1)
        for k, v in row.items():
            s[k].append(v)
    return s


HARD_RECORDS = {
    "P-101": [
        "2023-10-14 WO-4471: impeller and wear rings replaced. Typical service life in this duty: 3-5 years.",
        "2026-10-02 vibration route: overall 2.0 mm/s, no bearing defect frequencies.",
    ],
    "PT-101": [
        "Spec: accuracy +/-0.5% of 0-200 psi span (+/-1.0 psi).",
        "2024-09-18 calibration: as-found +0.2%, as-left 0.0%. Interval 12 months; next due 2025-09-18 (OVERDUE).",
        "Installed in outdoor rack R-3, south-facing. WO-5512 'add sunshade to R-3' open since 2026-06.",
    ],
    "PT-102": [
        "Spec: accuracy +/-0.5% of 0-200 psi span (+/-1.0 psi).",
        "2026-08-11 calibration: as-found -0.1%, as-left 0.0%. Indoors, in the valve gallery.",
    ],
    "FCV-101": [
        "Flow controller FIC-101 in AUTO, setpoint 1200 gpm.",
        "2026-03-20 valve trim inspected, no wear.",
    ],
    "PI-101": [
        "Local mechanical discharge gauge next to PT-101. 0-200 psi dial, 10 psi divisions, accuracy +/-4 psi.",
        "2026-11-26 05:00 operator round: PI-101 reads ~85 psi.",
    ],
}


def easy(seed: int = 7) -> Scenario:
    return Scenario(
        name="easy",
        alarm=ALARM,
        question=QUESTION,
        context="Layout: suction header -> P-101 (PT-100 suction, PT-101 discharge, FT-101 flow) -> reservoir (LT-200). Standby pump P-102 is offline.",
        hypotheses=HYPOTHESES,
        truth="instrument",
        tags=_tags(*BASE_TAGS),
        series=easy_series(seed),
    )


def hard(seed: int = 7) -> Scenario:
    return Scenario(
        name="hard",
        alarm=ALARM,
        question=QUESTION,
        context=(
            "Layout: suction header -> P-101 (PT-100 suction, PT-101 discharge, FT-101 flow) -> "
            "flow control valve FCV-101 (controller FIC-101 holds a flow setpoint) -> discharge header "
            "(PT-102) -> reservoir (LT-200). Standby pump P-102 is offline."
        ),
        hypotheses=HYPOTHESES,
        truth="instrument",
        tags=_tags(*HARD_TAGS),
        series=hard_series(seed),
        records=HARD_RECORDS,
    )


SCENARIOS: dict[str, Callable[[int], Scenario]] = {"easy": easy, "hard": hard}


def hhmm(hour: float) -> str:
    minutes = round(hour * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def summarize(series: list[float], start_hour: float, end_hour: float) -> dict[str, float]:
    """Summary statistics for one tag over [start_hour, end_hour], clamped to 0-8 h."""
    lo = max(0, round(start_hour * 60))
    hi = min(MINUTES, round(end_hour * 60))
    if hi <= lo:
        raise ValueError("end_hour must be after start_hour, within 0 to 8")
    window = series[lo : hi + 1]
    n = len(window)
    xs = [i / 60 for i in range(n)]  # hours
    x_mean = sum(xs) / n
    y_mean = sum(window) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, window)) / sxx
    return {
        "first": window[0],
        "last": window[-1],
        "min": min(window),
        "max": max(window),
        "mean": y_mean,
        "trend_per_hour": slope,
    }
