"""Estimate the universe of failure modes with a large random campaign, then report which
modes each strategy finds at a normal budget. Used to check for ceiling effects."""

import collections
import json
import sys

from stress_twin import strategies
from stress_twin.campaign import run_campaign
from stress_twin.runners import LocalRunner


def main(budget=240, seeds=3, ref_budget=3000):
    found = collections.defaultdict(collections.Counter)
    with LocalRunner() as r:
        for seed in range(seeds):
            for name in ("random", "boundary"):
                h = run_campaign(strategies.make(name), r, budget, 6, seed=1000 + seed, log=lambda *_: None)
                for m in {x["mode"] for x in h if not x["success"]}:
                    found[m][name] += 1
        ref = run_campaign(strategies.make("random"), r, ref_budget, 1, seed=77, log=lambda *_: None)
    universe = collections.Counter(x["mode"] for x in ref if not x["success"])
    print(f"mode universe from {ref_budget} random episodes: {len(universe)}")
    for m, c in universe.most_common():
        print(f"  {m:34s} ref={c:4d}  random {found[m]['random']}/{seeds}  boundary {found[m]['boundary']}/{seeds}")
    print("found but not in reference:", [m for m in found if m not in universe])
    json.dump({"universe": universe, "found": found}, open("results/mode_universe.json", "w"), indent=2)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
