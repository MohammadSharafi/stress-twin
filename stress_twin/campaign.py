"""Campaign loop: propose -> simulate -> record, round after round."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .strategies import failure_mode, stress


def make_record(prop: dict, result: dict, round_idx: int) -> dict:
    rec = {
        "round": round_idx,
        "source": prop["source"],
        "hypothesis": prop.get("hypothesis", ""),
        "expected_cause": prop.get("expected_cause", ""),
        "scenario": result["scenario"],
        "success": result["success"],
        "cause": result["cause"],
        "est_xy": result["est_xy"],
        "true_xy": result["true_xy"],
        "telemetry": result["telemetry"],
        "stress": stress(result["scenario"]),
    }
    rec["mode"] = failure_mode(rec)
    if rec["expected_cause"]:
        rec["confirmed"] = (not rec["success"]) and rec["cause"] == rec["expected_cause"]
    return rec


def summarize(history: list[dict]) -> dict:
    fails = [h for h in history if not h["success"]]
    return {
        "episodes": len(history),
        "failures": len(fails),
        "subtle_failures": sum(1 for h in fails if h["stress"] <= 0.30),
        "modes": len({h["mode"] for h in fails}),
        "causes": sorted({h["cause"] for h in fails}),
        "median_failure_stress": float(np.median([h["stress"] for h in fails])) if fails else None,
    }


def run_campaign(strategy, runner, budget: int, rounds: int, seed: int, out: Path | None = None,
                 log=print) -> list[dict]:
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    per_round = [budget // rounds + (1 if i < budget % rounds else 0) for i in range(rounds)]
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.jsonl").write_text("")
    for r, n in enumerate(per_round):
        t0 = time.time()
        props = strategy.propose(history, n, rng)
        results = runner.run([p["scenario"] for p in props])
        new = [make_record(p, res, r) for p, res in zip(props, results)]
        history.extend(new)
        if out is not None:
            with open(out / "results.jsonl", "a") as f:
                for rec in new:
                    f.write(json.dumps(rec) + "\n")
        s = summarize(history)
        log(f"round {r + 1}/{rounds} [{strategy.name}] +{len(new)} episodes in {time.time() - t0:.1f}s | "
            f"failures {s['failures']}, subtle {s['subtle_failures']}, modes {s['modes']}")
    return history
