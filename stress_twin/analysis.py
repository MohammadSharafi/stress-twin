"""Campaign analysis: failure clusters, per-factor boundaries, and the safety envelope.

The envelope is found with PRIM-style box peeling (Friedman & Fisher's Patient Rule
Induction Method, widely used for scenario discovery): starting from the full tested
space, repeatedly peel a thin slice off whichever side of whichever factor most raises
the pass rate of what remains, never peeling into the policy's nominal range. The result
is an axis-aligned operating envelope a safety reviewer can read as a list of limits.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import params
from .strategies import STRESSABLE, stressed_sample

ENVELOPE_FACTORS = tuple(n for n in STRESSABLE if n != "distractor_hue_gap")


def load_history(run: Path) -> list[dict]:
    return [json.loads(line) for line in (Path(run) / "results.jsonl").read_text().splitlines() if line.strip()]


def fmt_value(name: str, v: float) -> str:
    p = params.BY_NAME[name]
    if p.integer:
        return str(int(round(v)))
    if p.unit == "m":
        return f"{v * 1000:.0f} mm" if name != "obj_size" else f"{v * 1000:.1f} mm"
    if p.unit == "kg":
        return f"{v:.2f} kg"
    if p.unit == "deg":
        return f"{v:.0f}°"
    return f"{v:.2f}"


def fmt_nominal(name: str) -> str:
    lo, hi = params.BY_NAME[name].nominal
    return f"{fmt_value(name, lo)}–{fmt_value(name, hi)}"


def stressors(scenario: dict, limit: int = 3) -> list[dict]:
    out = []
    for name in params.off_nominal(scenario)[:limit]:
        out.append({"name": name, "value": scenario[name], "text": fmt_value(name, scenario[name]),
                    "nominal": fmt_nominal(name), "description": params.BY_NAME[name].description,
                    "label": params.BY_NAME[name].label})
    return out


def clusters(history: list[dict]) -> list[dict]:
    """Group failures by mode (cause x primary stressor), most frequent first."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for h in history:
        if not h["success"]:
            groups[h["mode"]].append(h)
    out = []
    for mode, recs in groups.items():
        cause, factor = mode.split("|")
        rep = min(recs, key=lambda r: r["stress"])
        factor_counts = Counter(n for r in recs for n in params.off_nominal(r["scenario"])[:3])  # co-occurring
        out.append({
            "mode": mode, "cause": cause, "factor": factor, "count": len(recs),
            "min_stress": rep["stress"], "representative": rep,
            "co_factors": [n for n, _ in factor_counts.most_common(4) if n != factor][:3],
        })
    return sorted(out, key=lambda c: (-c["count"], c["min_stress"]))


def factor_boundaries(history: list[dict]) -> dict[str, dict]:
    """For each factor, on each side of its nominal range: the mildest value at which a failure
    was attributed to it, and the most extreme value at which the policy still passed with that
    factor pushed. Failures are credited to one factor via cause-aware attribution."""
    out = {}
    for name in STRESSABLE:
        p = params.BY_NAME[name]
        lo, hi = p.nominal
        pushed = [h for h in history if name in params.off_nominal(h["scenario"])]
        fails = [h for h in history if not h["success"] and h["mode"].split("|")[1] == name]
        passes = [h for h in pushed if h["success"]]
        info = {"tested": len(pushed), "failures": len(fails)}
        for side in ("low", "high"):
            below = (lambda v: v < lo) if side == "low" else (lambda v: v > hi)
            fv = [h["scenario"][name] for h in fails if below(h["scenario"][name])]
            pv = [h["scenario"][name] for h in passes if below(h["scenario"][name])]
            info[side] = {
                "n": sum(1 for h in pushed if below(h["scenario"][name])),
                "mildest_failure": (max(fv) if side == "low" else min(fv)) if fv else None,
                "furthest_pass": (min(pv) if side == "low" else max(pv)) if pv else None,
            }
        out[name] = info
    return out


def _inside(scenario: dict, box: dict) -> bool:
    return all(box[k][0] - 1e-9 <= scenario[k] <= box[k][1] + 1e-9 for k in box)


def prim_envelope(history: list[dict], alpha: float = 0.06, target: float = 0.97, min_support: int = 25) -> dict:
    box = {k: [params.BY_NAME[k].low, params.BY_NAME[k].high] for k in ENVELOPE_FACTORS}
    data = [h for h in history]
    inside = data[:]
    steps = []

    def purity(rs):
        return sum(r["success"] for r in rs) / len(rs) if rs else 0.0

    while inside and purity(inside) < target and len(inside) > min_support:
        best = None
        base = purity(inside)
        for k in ENVELOPE_FACTORS:
            p = params.BY_NAME[k]
            vals = np.array([r["scenario"][k] for r in inside])
            for side in ("low", "high"):
                if side == "low":
                    cut = float(np.quantile(vals, alpha))
                    cut = min(cut, p.nominal[0])
                    if p.integer:
                        cut = math.floor(cut) + (1 if cut > math.floor(cut) else 0)
                        cut = min(cut, p.nominal[0])
                    if cut <= box[k][0] + 1e-12:
                        continue
                    new = [r for r in inside if r["scenario"][k] >= cut - 1e-12]
                else:
                    cut = float(np.quantile(vals, 1 - alpha))
                    cut = max(cut, p.nominal[1])
                    if p.integer:
                        cut = max(math.floor(cut), p.nominal[1])
                    if cut >= box[k][1] - 1e-12:
                        continue
                    new = [r for r in inside if r["scenario"][k] <= cut + 1e-12]
                removed = len(inside) - len(new)
                if removed <= 0 or len(new) < min_support:
                    continue
                gain = (purity(new) - base) / removed
                if best is None or gain > best[0]:
                    best = (gain, k, side, cut, new)
        if best is None or best[0] <= 0:
            break
        _, k, side, cut, new = best
        box[k][0 if side == "low" else 1] = cut
        inside = new
        steps.append({"factor": k, "side": side, "bound": cut, "support": len(inside), "pass_rate": purity(inside)})

    limits = []
    for k in ENVELOPE_FACTORS:
        p = params.BY_NAME[k]
        lo, hi = box[k]
        if lo > p.low + 1e-12:
            limits.append({"factor": k, "op": ">=", "bound": lo, "text": f"{p.label} ≥ {fmt_value(k, lo)}"})
        if hi < p.high - 1e-12:
            limits.append({"factor": k, "op": "<=", "bound": hi, "text": f"{p.label} ≤ {fmt_value(k, hi)}"})
    return {"box": box, "limits": limits, "train_support": len(inside), "train_pass_rate": purity(inside),
            "train_total": len(data), "steps": steps}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def validate_envelope(run: Path, runner, n: int = 150, seed: int = 99) -> dict:
    """Run fresh held-out scenarios inside and outside the envelope and measure pass rates."""
    history = load_history(run)
    env = prim_envelope(history)
    # Held-out: an independent random stream, and no scenario seed the campaign ever used.
    rng = np.random.default_rng([seed, 0x5EED, 2026])
    seen = {h["scenario"]["seed"] for h in history}
    inside, outside = [], []
    while len(inside) < n or len(outside) < n // 3:
        s = stressed_sample(rng)
        if s["seed"] in seen:
            continue
        if _inside(s, env["box"]):
            if len(inside) < n:
                inside.append(s)
        elif len(outside) < n // 3:
            outside.append(s)
    res_in = runner.run(inside)
    res_out = runner.run(outside)
    k_in = sum(r["success"] for r in res_in)
    k_out = sum(r["success"] for r in res_out)
    lo, hi = wilson(k_in, len(res_in))
    result = {
        "n_inside": len(res_in), "pass_inside": k_in, "pass_rate_inside": round(k_in / len(res_in), 4),
        "ci95_inside": [round(lo, 4), round(hi, 4)],
        "n_outside": len(res_out), "pass_rate_outside": round(k_out / len(res_out), 4),
        "failures_inside": [{"cause": r["cause"], "stressors": [s["label"] + " " + s["text"] for s in stressors(r["scenario"])]}
                            for r in res_in if not r["success"]],
        "limits": [l["text"] for l in env["limits"]],
        "seed": seed,
        "overlap_with_campaign": len({x["seed"] for x in inside + outside} & seen),
    }
    (Path(run) / "validation.json").write_text(json.dumps(result, indent=2))
    return result
