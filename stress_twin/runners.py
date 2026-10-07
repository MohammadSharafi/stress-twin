"""Episode runners.

LocalRunner fans episodes out over local CPU processes. NebiusJobRunner (nebius_jobs.py)
fans batches out to Nebius Serverless Jobs. Both return plain JSON results in input order.
"""

from __future__ import annotations

import multiprocessing as mp
import os


def _init_worker() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .perception import Perception

    Perception.shared()


def _run_one(scenario: dict) -> dict:
    from .episode import run_episode

    return run_episode(scenario).to_json()


class LocalRunner:
    name = "local"

    def __init__(self, workers: int | None = None) -> None:
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        self._pool = None

    def __enter__(self) -> "LocalRunner":
        self._pool = mp.get_context("spawn").Pool(self.workers, initializer=_init_worker)
        return self

    def __exit__(self, *exc) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def run(self, scenarios: list[dict]) -> list[dict]:
        if self._pool is None:
            with self:
                return self.run(scenarios)
        return self._pool.map(_run_one, scenarios, chunksize=max(1, len(scenarios) // (self.workers * 4) or 1))
