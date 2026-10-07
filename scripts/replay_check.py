"""Replay recorded episodes from a run and check they reproduce the stored outcome."""

import json
import sys

from stress_twin.runners import LocalRunner


def main(run="runs/demo", n=60):
    recs = [json.loads(l) for l in open(f"{run}/results.jsonl")][: int(n)]
    with LocalRunner() as r:
        res = r.run([x["scenario"] for x in recs])
    same = sum(a["success"] == b["success"] and a["cause"] == b["cause"] and a["est_xy"] == b["est_xy"]
               for a, b in zip(recs, res))
    print(f"replayed {len(recs)}: identical outcome, cause and perception estimate in {same}/{len(recs)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
