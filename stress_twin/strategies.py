"""Scenario proposal strategies.

- RandomStress: the baseline a test engineer would write. Start from nominal conditions,
  push 1-3 random parameters anywhere in their full range.
- BoundarySearch: offline active learning. A random-forest surrogate predicts failure
  probability; candidates are scored for uncertainty (near the success/failure boundary),
  subtlety (small departure from nominal) and novelty, then picked greedily with diversity.
- LLMAdversary (llm_strategy.py): Nemotron proposes hypothesis-driven stress scenarios.

Every proposal is a dict: {"scenario", "source", "hypothesis", "expected_cause"}.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import params
from .diagnose import CAUSE_FACTORS

STRESSABLE = tuple(p.name for p in params.PARAMS if p.name not in ("obj_x", "obj_y"))


def stress(scenario: dict) -> float:
    """L1 distance outside the nominal box, in unit (normalised) coordinates."""
    total = 0.0
    for p in params.PARAMS:
        lo, hi = p.nominal
        v = scenario[p.name]
        d = (lo - v) if v < lo else (v - hi) if v > hi else 0.0
        total += max(d, 0.0) / (p.high - p.low)
    return round(total, 4)


def attributed_factor(scenario: dict, cause: str) -> str:
    """The most out-of-nominal condition among those that can physically cause `cause`."""
    off = params.off_nominal(scenario)
    allowed = CAUSE_FACTORS.get(cause, ())
    for name in off:
        if name in allowed:
            return name
    return off[0] if off else "nominal"


def failure_mode(record: dict) -> str | None:
    if record["success"]:
        return None
    return f"{record['cause']}|{attributed_factor(record['scenario'], record['cause'])}"


def proposal(scenario: dict, source: str, hypothesis: str = "", expected_cause: str = "") -> dict:
    return {"scenario": params.clip(scenario), "source": source, "hypothesis": hypothesis,
            "expected_cause": expected_cause}


def stressed_sample(rng: np.random.Generator, k: int | None = None, force: str | None = None) -> dict:
    s = params.sample_nominal(rng)
    k = k or int(rng.integers(1, 4))
    names = list(rng.choice(STRESSABLE, size=k, replace=False))
    if force and force not in names:
        names[0] = force
    for name in names:
        p = params.BY_NAME[name]
        s[name] = rng.integers(p.low, p.high + 1) if p.integer else rng.uniform(p.low, p.high)
    return params.clip(s)


class RandomStress:
    name = "random"

    def propose(self, history: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
        return [proposal(stressed_sample(rng), self.name) for _ in range(n)]


class BoundarySearch:
    name = "boundary"

    def __init__(self, warmup: int = 40, pool: int = 3000, subtlety: float = 0.60) -> None:
        self.warmup = warmup
        self.pool = pool
        self.subtlety = subtlety

    def _candidates(self, history: list[dict], rng: np.random.Generator) -> list[dict]:
        cands = [stressed_sample(rng) for _ in range(self.pool)]
        # Single- and double-stressor candidates for every factor, so each boundary gets mapped.
        for name in STRESSABLE:
            cands += [stressed_sample(rng, k=int(rng.integers(1, 3)), force=name) for _ in range(80)]
        # Bisection between near pairs with opposite outcomes: the boundary lies between them.
        ok = [h["scenario"] for h in history if h["success"]]
        bad = [h["scenario"] for h in history if not h["success"]]
        if ok and bad:
            U_ok = np.array([params.to_unit(s) for s in ok])
            for s in bad[-150:]:
                u = params.to_unit(s)
                j = int(np.argmin(np.linalg.norm(U_ok - u, axis=1)))
                for a in (0.25, 0.5, 0.75):
                    m = (1 - a) * u + a * U_ok[j] + rng.normal(0, 0.02, size=u.shape)
                    cands.append(params.from_unit(m, int(rng.integers(0, 2**31 - 1))))
        # Shrink known failures toward nominal to find the minimal breaking perturbation.
        for s in bad[-150:]:
            base = params.sample_nominal(rng)
            a = rng.uniform(0.3, 0.9)
            mix = {k: (a * s[k] + (1 - a) * base[k]) for k in params.NAMES}
            cands.append(params.clip({"seed": int(rng.integers(0, 2**31 - 1)), **mix}))
        return cands

    def propose(self, history: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
        labels = [h["cause"] for h in history]
        if len(history) < self.warmup or len(set(labels)) < 2:
            return [proposal(stressed_sample(rng), self.name, "warm-up exploration") for _ in range(n)]

        X = np.array([params.to_unit(h["scenario"]) for h in history])
        y = np.array([h["cause"] for h in history])
        rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=int(rng.integers(1e9)), n_jobs=1)
        rf.fit(X, y)

        cands = self._candidates(history, rng)
        U = np.array([params.to_unit(c) for c in cands])
        proba = rf.predict_proba(U)
        classes = list(rf.classes_)
        p_fail = 1.0 - (proba[:, classes.index("none")] if "none" in classes else 0.0)
        primary = np.array([(params.off_nominal(c) or ["nominal"])[0] for c in cands])
        # The factor a candidate would be credited with, per cause it might produce.
        cand_off = [params.off_nominal(c) for c in cands]

        # Probability that a candidate reveals a (cause, factor) pair not seen yet.
        found_modes = {h["mode"] for h in history if not h["success"]}
        p_new = np.zeros(len(cands))
        for ci, cause in enumerate(classes):
            if cause == "none":
                continue
            allowed = CAUSE_FACTORS.get(cause, ())
            credit = [next((n for n in off if n in allowed), off[0] if off else "nominal") for off in cand_off]
            unseen = np.array([f"{cause}|{k}" not in found_modes for k in credit])
            p_new += proba[:, ci] * unseen
        # Causes the surrogate has never seen get a small prior so exploration never stops.
        p_new += 0.05 * np.array([k != "nominal" for k in primary])

        uncertainty = 1.0 - np.abs(2 * p_fail - 1)
        subtle = np.exp(-np.array([stress(c) for c in cands]) / self.subtlety)
        dmin = np.min(np.linalg.norm(U[:, None, :] - X[None, :, :], axis=2), axis=1)
        novelty = dmin / (dmin.max() + 1e-9)
        score = (0.35 * uncertainty + 0.15 * p_fail + 0.60 * p_new) * subtle + 0.10 * novelty

        # Stratify by primary stressor: factors with the fewest discovered modes go first,
        # factors that keep passing get de-prioritised as evidence of robustness accumulates.
        modes_per = {k: 0 for k in STRESSABLE}
        tested = {k: 0 for k in STRESSABLE}
        for m in found_modes:
            k = m.split("|")[1]
            if k in modes_per:
                modes_per[k] += 1
        for h in history:
            off = params.off_nominal(h["scenario"])
            if off and off[0] in tested:
                tested[off[0]] += 1
        order = sorted(STRESSABLE, key=lambda k: (modes_per[k] + 0.04 * tested[k], rng.random()))
        ranked = {k: [int(i) for i in np.argsort(-score) if primary[i] == k] for k in order}
        chosen: list[int] = []
        while len(chosen) < n and any(ranked.values()):
            for k in order:
                while ranked[k]:
                    i = ranked[k].pop(0)
                    if all(np.linalg.norm(U[i] - U[j]) > 0.10 for j in chosen):
                        chosen.append(i)
                        break
                if len(chosen) == n:
                    break
        return [proposal(cands[i], self.name, f"surrogate p_fail={p_fail[i]:.2f}") for i in chosen]


def make(name: str, **kw):
    if name == "random":
        return RandomStress()
    if name == "boundary":
        return BoundarySearch()
    if name == "llm":
        from .llm_strategy import LLMAdversary
        return LLMAdversary(**kw)
    raise ValueError(f"unknown strategy {name!r}")
