"""Scenario parameter space.

A scenario is a plain dict mapping parameter name to a physical value, plus an integer
"seed" that fixes every remaining random choice (offset directions, distractor placement,
image noise). Keeping scenarios as flat JSON-able dicts makes them easy to send to an LLM,
to a Serverless Job, and to store in JSONL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Param:
    name: str
    low: float
    high: float
    nominal: tuple[float, float]  # the range the policy was trained/tuned on
    unit: str
    description: str
    integer: bool = False
    short: str = ""

    @property
    def label(self) -> str:
        return self.short or self.name


PARAMS: tuple[Param, ...] = (
    Param("obj_x", -0.10, 0.10, (-0.08, 0.08), "m", "object x position on the table", short="object x"),
    Param("obj_y", -0.10, 0.10, (-0.08, 0.08), "m", "object y position on the table", short="object y"),
    Param("obj_yaw", 0.0, 45.0, (0.0, 10.0), "deg", "object rotation about the vertical axis", short="object rotation"),
    Param("obj_size", 0.012, 0.040, (0.018, 0.028), "m", "object half-width (cube)", short="object size"),
    Param("obj_mass", 0.05, 1.50, (0.05, 0.40), "kg", "object mass", short="object mass"),
    Param("obj_friction", 0.15, 1.20, (0.60, 1.00), "", "object sliding friction coefficient", short="object friction"),
    Param("hue_shift", 0.0, 0.50, (0.0, 0.03), "", "object hue distance from the trained red (0.5 = opposite hue)", short="object hue shift"),
    Param("obj_value", 0.15, 1.00, (0.75, 1.00), "", "object colour brightness", short="object brightness"),
    Param("light", 0.08, 1.00, (0.60, 1.00), "", "scene light intensity", short="scene light"),
    Param("distractors", 0, 3, (0, 0), "count", "number of distractor objects on the table", integer=True, short="distractor count"),
    Param("distractor_hue_gap", 0.0, 0.50, (0.30, 0.50), "", "hue distance between distractors and the target", short="distractor hue gap"),
    Param("camera_offset", 0.0, 0.030, (0.0, 0.002), "m", "camera extrinsic calibration error", short="camera offset"),
    Param("image_noise", 0.0, 0.20, (0.0, 0.02), "", "sensor noise standard deviation", short="sensor noise"),
    Param("disturbance", 0.0, 0.050, (0.0, 0.0), "m", "object displacement by an external bump during approach", short="bump during approach"),
)

NAMES: tuple[str, ...] = tuple(p.name for p in PARAMS)
BY_NAME: dict[str, Param] = {p.name: p for p in PARAMS}


def clip(scenario: dict) -> dict:
    """Clamp every parameter into bounds and round integer parameters. Keeps the seed."""
    out = {"seed": int(scenario.get("seed", 0))}
    for p in PARAMS:
        v = float(scenario.get(p.name, (p.nominal[0] + p.nominal[1]) / 2))
        if not np.isfinite(v):
            v = (p.nominal[0] + p.nominal[1]) / 2
        v = min(max(v, p.low), p.high)
        out[p.name] = int(round(v)) if p.integer else round(v, 5)
    return out


def to_unit(scenario: dict) -> np.ndarray:
    """Map a scenario to [0, 1]^d in PARAMS order."""
    return np.array([(scenario[p.name] - p.low) / (p.high - p.low) for p in PARAMS], dtype=float)


def from_unit(u: np.ndarray, seed: int) -> dict:
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return clip({"seed": seed, **{p.name: p.low + ui * (p.high - p.low) for p, ui in zip(PARAMS, u)}})


def sample_uniform(rng: np.random.Generator) -> dict:
    return from_unit(rng.random(len(PARAMS)), int(rng.integers(0, 2**31 - 1)))


def sample_nominal(rng: np.random.Generator) -> dict:
    s = {"seed": int(rng.integers(0, 2**31 - 1))}
    for p in PARAMS:
        lo, hi = p.nominal
        s[p.name] = rng.integers(lo, hi + 1) if p.integer else rng.uniform(lo, hi)
    return clip(s)


def is_nominal(scenario: dict) -> bool:
    return all(p.nominal[0] - 1e-9 <= scenario[p.name] <= p.nominal[1] + 1e-9 for p in PARAMS)


def off_nominal(scenario: dict) -> list[str]:
    """Names of parameters outside the nominal range, ordered by how far out they are."""
    dist = []
    for p in PARAMS:
        lo, hi = p.nominal
        v = scenario[p.name]
        d = (lo - v) if v < lo else (v - hi) if v > hi else 0.0
        if d > 1e-9:
            dist.append((d / (p.high - p.low), p.name))
    return [n for _, n in sorted(dist, reverse=True)]


def schema_text() -> str:
    """Human/LLM-readable description of the parameter space."""
    lines = []
    for p in PARAMS:
        kind = "integer" if p.integer else "float"
        lines.append(
            f"- {p.name} ({kind}, {p.unit or 'unitless'}): {p.description}. "
            f"Allowed [{p.low}, {p.high}]; policy trained on [{p.nominal[0]}, {p.nominal[1]}]."
        )
    return "\n".join(lines)
