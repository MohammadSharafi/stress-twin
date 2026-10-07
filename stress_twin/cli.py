"""Command line interface.

    stress-twin run --budget 240 --rounds 4 --strategy boundary --out runs/demo
    stress-twin validate --run runs/demo --n 150
    stress-twin benchmark --budget 240 --seeds 3
    stress-twin selftest | nominal | surface
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")


def _runner(args):
    if getattr(args, "runner", "local") == "nebius":
        from .nebius_jobs import NebiusJobRunner

        return NebiusJobRunner(image=args.image, batch_size=args.batch_size, dry_run=args.dry_run)
    from .runners import LocalRunner

    return LocalRunner(workers=args.workers)


def cmd_selftest(args) -> int:
    import mujoco

    from . import params, scene

    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        s = params.sample_uniform(rng) if i % 2 else _extreme(rng)
        model = mujoco.MjModel.from_xml_string(scene.build_xml(s))
        data = mujoco.MjData(model)
        for _ in range(50):
            mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            print(f"non-finite state for scenario {s}")
            return 1
    print(f"built and stepped {args.n} scenarios across full bounds")
    print("selftest ok")
    return 0


def _extreme(rng):
    from . import params

    u = rng.integers(0, 2, size=len(params.PARAMS)).astype(float)
    return params.from_unit(u, int(rng.integers(0, 2**31 - 1)))


def cmd_nominal(args) -> int:
    from . import params

    rng = np.random.default_rng(args.seed)
    scen = [params.sample_nominal(rng) for _ in range(args.n)]
    with _runner(args) as runner:
        res = runner.run(scen)
    ok = sum(r["success"] for r in res)
    rate = ok / len(res)
    print(f"nominal success {ok}/{len(res)} = {rate:.1%}; causes {dict(collections.Counter(r['cause'] for r in res))}")
    print("NOMINAL_OK" if rate >= 0.85 else "NOMINAL_FAIL")
    return 0 if rate >= 0.85 else 1


def cmd_surface(args) -> int:
    from .strategies import stressed_sample

    rng = np.random.default_rng(args.seed)
    scen = [stressed_sample(rng) for _ in range(args.n)]
    with _runner(args) as runner:
        res = runner.run(scen)
    causes = collections.Counter(r["cause"] for r in res if not r["success"])
    print(f"stressed scenarios {len(res)}: success {sum(r['success'] for r in res)}, failure causes {dict(causes)}")
    print("SURFACE_OK" if len(causes) >= 3 else "SURFACE_FAIL")
    return 0 if len(causes) >= 3 else 1


def cmd_benchmark(args) -> int:
    from . import strategies
    from .campaign import run_campaign, summarize

    rows = collections.defaultdict(list)
    t0 = time.time()
    with _runner(args) as runner:
        for seed in range(args.seeds):
            for name in ("random", "boundary"):
                hist = run_campaign(strategies.make(name), runner, args.budget, args.rounds, seed=1000 + seed,
                                    log=lambda *_: None)
                s = summarize(hist)
                rows[name].append(s)
                print(f"seed {seed} {name:8s} failures {s['failures']:4d}  subtle {s['subtle_failures']:4d}  "
                      f"modes {s['modes']:3d}  median failure stress {s['median_failure_stress']}")
    mean = {k: {m: float(np.mean([r[m] for r in v])) for m in ("failures", "subtle_failures", "modes")}
            for k, v in rows.items()}
    print(f"mean over {args.seeds} seeds, budget {args.budget}: {json.dumps(mean)} ({time.time() - t0:.0f}s)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"budget": args.budget, "seeds": args.seeds, "per_seed": rows,
                                              "mean": mean}, indent=2))
    ok = (mean["boundary"]["subtle_failures"] > mean["random"]["subtle_failures"]
          and mean["boundary"]["modes"] > mean["random"]["modes"])
    print("BENCHMARK_OK" if ok else "BENCHMARK_FAIL")
    return 0 if ok else 1


def cmd_run(args) -> int:
    from . import strategies
    from .campaign import run_campaign, summarize

    out = Path(args.out)
    for stale in ("validation.json", "llm_calls.jsonl", "planner_notes.json", "captions.json"):
        (out / stale).unlink(missing_ok=True)  # a new campaign invalidates earlier artefacts
    kw = {}
    if args.strategy == "llm":
        from .llm import TokenFactory

        kw["client"] = TokenFactory.from_env(log_path=out / "llm_calls.jsonl")
    strategy = strategies.make(args.strategy, **kw)
    t0 = time.time()
    with _runner(args) as runner:
        hist = run_campaign(strategy, runner, args.budget, args.rounds, seed=args.seed, out=out)
    if hasattr(strategy, "notes"):
        (out / "planner_notes.json").write_text(json.dumps(strategy.notes(), indent=2))
    meta = {"strategy": args.strategy, "budget": args.budget, "rounds": args.rounds, "seed": args.seed,
            "runner": args.runner, "summary": summarize(hist), "seconds": round(time.time() - t0, 1)}
    (out / "campaign.json").write_text(json.dumps(meta, indent=2))
    from .report import build_report

    path = build_report(out, analyst_mode=args.analyst, log=print)
    print(f"report written to {path}")
    return 0


def cmd_report(args) -> int:
    from .report import build_report

    print(f"report written to {build_report(Path(args.run), analyst_mode=args.analyst, log=print)}")
    return 0


def cmd_validate(args) -> int:
    from .analysis import validate_envelope

    run = Path(args.run)
    with _runner(args) as runner:
        res = validate_envelope(run, runner, n=args.n, seed=args.seed)
    print(json.dumps(res, indent=2))
    ok = res["pass_rate_inside"] >= 0.90
    print("ENVELOPE_OK" if ok else "ENVELOPE_FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stress-twin", description="Adversarial simulation testing for robot policies")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--workers", type=int, default=None)
        p.add_argument("--runner", choices=("local", "nebius"), default="local")
        p.add_argument("--image", default=os.environ.get("STRESS_TWIN_IMAGE", ""))
        p.add_argument("--batch-size", type=int, default=40)
        p.add_argument("--dry-run", action="store_true", help="nebius runner: print job commands, run locally")
        return p

    p = common(sub.add_parser("selftest", help="build and step scenarios across the full bounds"))
    p.add_argument("--n", type=int, default=200)
    p.set_defaults(fn=cmd_selftest)
    p = common(sub.add_parser("nominal", help="policy success rate in its nominal domain"))
    p.add_argument("--n", type=int, default=200)
    p.set_defaults(fn=cmd_nominal)
    p = common(sub.add_parser("surface", help="failure causes under random stress"))
    p.add_argument("--n", type=int, default=300)
    p.set_defaults(fn=cmd_surface)
    p = common(sub.add_parser("benchmark", help="boundary search vs random stress testing"))
    p.add_argument("--budget", type=int, default=240)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="")
    p.set_defaults(fn=cmd_benchmark)
    p = common(sub.add_parser("run", help="run a stress campaign and write a report"))
    p.add_argument("--budget", type=int, default=240)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--strategy", choices=("random", "boundary", "llm"), default="boundary")
    p.add_argument("--analyst", choices=("auto", "offline", "llm"), default="auto")
    p.add_argument("--out", default="runs/latest")
    p.set_defaults(fn=cmd_run)
    p = common(sub.add_parser("report", help="rebuild the report for an existing run"))
    p.add_argument("--run", required=True)
    p.add_argument("--analyst", choices=("auto", "offline", "llm"), default="auto")
    p.set_defaults(fn=cmd_report)
    p = common(sub.add_parser("validate", help="check the safety envelope on held-out scenarios"))
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=150)
    p.set_defaults(fn=cmd_validate)

    args = ap.parse_args(argv)
    from .llm import LLMUnavailable

    try:
        return args.fn(args)
    except LLMUnavailable as err:
        print(f"stress-twin: {err}", file=sys.stderr)
        print("To add a key: copy .env.example to .env and paste the key after NEBIUS_API_KEY=", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
