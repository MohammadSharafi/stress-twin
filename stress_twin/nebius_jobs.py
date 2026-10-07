"""Run episode batches on Nebius Serverless Jobs.

Each batch becomes one job created with the documented CLI:

    nebius ai create --type job --name <name> --image <image> \
      --container-command python --args "-m stress_twin.worker --payload <b64>" \
      --platform <platform> --preset <preset> --timeout <t>

The runner then polls `nebius ai job get-by-name` for the job id and `nebius ai job logs`
for result lines until the worker prints its completion marker. Jobs are submitted in
parallel, so a 240-episode round is a handful of concurrent containers.

dry_run=True prints the exact commands and executes the same worker locally, which
exercises payload encoding, the worker and result parsing without cloud credentials.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import uuid

from .worker import DONE_PREFIX, RESULT_PREFIX, encode_payload


def parse_results(log_text: str, expected: int) -> list[dict] | None:
    results: dict[int, dict] = {}
    done = False
    for line in log_text.splitlines():
        line = line.strip()
        idx = line.find(RESULT_PREFIX)
        if idx >= 0:
            obj = json.loads(line[idx + len(RESULT_PREFIX):])
            results[obj["i"]] = obj["result"]
        elif DONE_PREFIX in line:
            done = True
    if done and len(results) == expected:
        return [results[i] for i in range(expected)]
    return None


class NebiusJobRunner:
    name = "nebius"

    def __init__(self, image: str, batch_size: int = 40, platform: str | None = None, preset: str | None = None,
                 timeout: str = "30m", dry_run: bool = False, cli: str | None = None, poll_s: float = 10.0,
                 max_wait_s: float = 3600.0, log=print) -> None:
        if not image and not dry_run:
            raise ValueError("NebiusJobRunner needs a container image (--image or STRESS_TWIN_IMAGE)")
        self.image = image or "stress-twin:local"
        self.batch_size = batch_size
        self.platform = platform or os.environ.get("STRESS_TWIN_PLATFORM", "cpu-d3")
        self.preset = preset or os.environ.get("STRESS_TWIN_PRESET", "16vcpu-64gb")
        self.timeout = timeout
        self.dry_run = dry_run
        self.cli = cli or os.environ.get("NEBIUS_CLI", "nebius")
        self.poll_s = poll_s
        self.max_wait_s = max_wait_s
        self.log = log
        self.run_id = uuid.uuid4().hex[:8]
        self.submitted: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def create_command(self, name: str, payload: str) -> list[str]:
        return [self.cli, "ai", "create", "--type", "job", "--name", name, "--image", self.image,
                "--container-command", "python", "--args", f"-m stress_twin.worker --payload {payload}",
                "--platform", self.platform, "--preset", self.preset, "--timeout", self.timeout]

    def _sh(self, cmd: list[str]) -> str:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{shlex.join(cmd[:4])} ... failed ({proc.returncode}): {proc.stderr.strip()[:500]}")
        return proc.stdout

    def run(self, scenarios: list[dict]) -> list[dict]:
        batches = [scenarios[i:i + self.batch_size] for i in range(0, len(scenarios), self.batch_size)]
        jobs = []
        for b, batch in enumerate(batches):
            name = f"stress-twin-{self.run_id}-{len(self.submitted)}"
            payload = encode_payload(batch)
            cmd = self.create_command(name, payload)
            self.submitted.append({"name": name, "n": len(batch)})
            if self.dry_run:
                self.log(f"[dry-run] {shlex.join(cmd[:11])} --args '-m stress_twin.worker --payload <{len(payload)} chars>' "
                         f"{shlex.join(cmd[13:])}")
                out = subprocess.run([sys.executable, "-m", "stress_twin.worker", "--payload", payload],
                                     capture_output=True, text=True, check=True).stdout
                jobs.append({"name": name, "n": len(batch), "results": parse_results(out, len(batch))})
            else:
                self._sh(cmd)
                jobs.append({"name": name, "n": len(batch), "id": None, "results": None})
        start = time.time()
        while any(j["results"] is None for j in jobs):
            if time.time() - start > self.max_wait_s:
                raise TimeoutError(f"Serverless Jobs did not finish within {self.max_wait_s}s: "
                                   f"{[j['name'] for j in jobs if j['results'] is None]}")
            for j in jobs:
                if j["results"] is not None:
                    continue
                if j["id"] is None:
                    try:
                        j["id"] = self._sh([self.cli, "ai", "job", "get-by-name", "--name", j["name"],
                                            "--format", "jsonpath={.metadata.id}"]).strip().strip("'\"")
                    except RuntimeError:
                        continue
                try:
                    j["results"] = parse_results(self._sh([self.cli, "ai", "job", "logs", j["id"]]), j["n"])
                except RuntimeError:
                    pass
            if any(j["results"] is None for j in jobs):
                time.sleep(self.poll_s)
        return [r for j in jobs for r in j["results"]]
