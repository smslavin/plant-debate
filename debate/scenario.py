"""A synthetic booster-pump incident with a known ground truth.

The station's discharge pressure transmitter (PT-101) drifts low as the sun
heats its enclosure. Everything else about the pump is healthy. A real
mechanical fault would show up in flow and motor current too, because a
degraded pump moves less water against the same system curve. Here those
stay flat, so the only coherent story is instrument drift.

The scenario includes one red herring: a short vibration spike at 02:30
(an operator stroked a nearby valve). An agent arguing "the pump is failing"
can make a lot of that spike.

Data is generated deterministically from a seed so every run sees the same
plant.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

MINUTES = 8 * 60  # 00:00 to 08:00, one sample per minute


@dataclass(frozen=True)
class Tag:
    name: str
    description: str
    unit: str


TAGS: dict[str, Tag] = {
    t.name: t
    for t in [
        Tag("PT-101", "P-101 discharge pressure", "psi"),
        Tag("PT-100", "P-101 suction pressure", "psi"),
        Tag("FT-101", "P-101 discharge flow", "gpm"),
        Tag("JT-101", "P-101 motor current", "A"),
        Tag("VT-101", "P-101 bearing vibration", "mm/s"),
        Tag("LT-200", "Downstream reservoir level", "ft"),
        Tag("TT-AMB", "Ambient temperature at instrument rack", "degC"),
    ]
}


@dataclass(frozen=True)
class Hypothesis:
    key: str
    claim: str
    action: str


@dataclass(frozen=True)
class Scenario:
    alarm: str
    question: str
    hypotheses: tuple[Hypothesis, Hypothesis]
    truth: str  # key of the true hypothesis
    series: dict[str, list[float]] = field(repr=False)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def generate_series(seed: int = 7) -> dict[str, list[float]]:
    rng = random.Random(seed)
    s: dict[str, list[float]] = {name: [] for name in TAGS}
    for t in range(MINUTES + 1):
        # Sun reaches the instrument rack around 02:30.
        temp = 18.0 + 16.0 * _clip01((t - 150) / 330)
        # Transmitter drift tracks enclosure temperature with a short lag.
        drift = 13.0 * _clip01((t - 180) / 300)
        vib_spike = 2.6 if 150 <= t <= 156 else 0.0

        s["TT-AMB"].append(temp + rng.gauss(0, 0.2))
        s["PT-101"].append(85.0 - drift + rng.gauss(0, 0.3))
        s["PT-100"].append(22.0 + rng.gauss(0, 0.2))
        s["FT-101"].append(1200.0 + rng.gauss(0, 6.0))
        s["JT-101"].append(48.0 + rng.gauss(0, 0.3))
        s["VT-101"].append(2.1 + vib_spike + rng.gauss(0, 0.1))
        s["LT-200"].append(12.0 + 0.9 * t / 60 + rng.gauss(0, 0.02))
    return s


def booster_station(seed: int = 7) -> Scenario:
    return Scenario(
        alarm="08:00  PT-101 LOW DISCHARGE PRESSURE  72 psi (alarm setpoint 75 psi, normal ~85 psi)",
        question=(
            "The operator asks: should we switch to standby pump P-102 and pull "
            "P-101 out of service for mechanical repair?"
        ),
        hypotheses=(
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
        ),
        truth="instrument",
        series=generate_series(seed),
    )


def hhmm(hour: float) -> str:
    minutes = round(hour * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def summarize(series: list[float], start_hour: float, end_hour: float) -> dict[str, float]:
    """Summary statistics for one tag over [start_hour, end_hour]."""
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
