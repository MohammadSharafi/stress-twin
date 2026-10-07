"""Run the same scenarios locally and inside the worker container; compare outcomes."""

import json
import subprocess
import sys

import numpy as np

from stress_twin.nebius_jobs import parse_results
from stress_twin.runners import LocalRunner
from stress_twin.strategies import stressed_sample
from stress_twin.worker import encode_payload


def main(image="stress-twin:local", n=24):
    rng = np.random.default_rng(2026)
    scen = [stressed_sample(rng) for _ in range(n)]
    payload = encode_payload(scen)
    out = subprocess.run(["docker", "run", "--rm", image, "python", "-m", "stress_twin.worker", "--payload", payload],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr[-2000:])
        sys.exit(1)
    remote = parse_results(out.stdout, n)
    with LocalRunner() as r:
        local = r.run(scen)
    same_outcome = sum(a["success"] == b["success"] and a["cause"] == b["cause"] for a, b in zip(remote, local))
    est_diff = max(float(np.max(np.abs(np.array(a["est_xy"]) - np.array(b["est_xy"])))) for a, b in zip(remote, local))
    identical = sum(json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) for a, b in zip(remote, local))
    print(f"container episodes: {len(remote)}; same outcome and cause as local: {same_outcome}/{n}; "
          f"bit-identical: {identical}/{n}; max perception estimate difference: {est_diff * 1000:.2f} mm")


if __name__ == "__main__":
    main(*sys.argv[1:])
