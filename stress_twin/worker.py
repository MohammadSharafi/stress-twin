"""Serverless Job entrypoint: run a batch of scenarios and print results to stdout.

    python -m stress_twin.worker --payload <base64(gzip(json list of scenarios))>

Results come back through the job's logs, one line per episode, so the job needs no
storage bucket. Clips are re-rendered locally from the recorded seed when needed. Within one
environment episodes are bit-for-bit deterministic; across renderers (OSMesa in the container
vs a desktop GPU) outcomes and causes agree but perception estimates can differ by ~1 mm, and
the report flags any replay whose cause changed.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import sys

RESULT_PREFIX = "STRESS_TWIN_RESULT "
DONE_PREFIX = "STRESS_TWIN_DONE "


def encode_payload(scenarios: list[dict]) -> str:
    return base64.b64encode(gzip.compress(json.dumps(scenarios, separators=(",", ":")).encode())).decode()


def decode_payload(payload: str) -> list[dict]:
    return json.loads(gzip.decompress(base64.b64decode(payload)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args(argv)
    scenarios = decode_payload(args.payload)

    from .runners import LocalRunner

    workers = args.workers or max(1, (os.cpu_count() or 2))
    with LocalRunner(workers=workers) as runner:
        results = runner.run(scenarios)
    for i, r in enumerate(results):
        sys.stdout.write(RESULT_PREFIX + json.dumps({"i": i, "result": r}, separators=(",", ":")) + "\n")
    sys.stdout.write(f"{DONE_PREFIX}{len(results)}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
