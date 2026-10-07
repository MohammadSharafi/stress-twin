import numpy as np

from stress_twin import params
from stress_twin.analysis import clusters, factor_boundaries, prim_envelope, wilson
from stress_twin.campaign import make_record
from stress_twin.strategies import stressed_sample


def _synthetic(n=600, seed=0):
    """Ground truth: the policy fails whenever light < 0.45 or mass > 0.7."""
    rng = np.random.default_rng(seed)
    hist = []
    for _ in range(n):
        s = stressed_sample(rng)
        fail_light, fail_mass = s["light"] < 0.45, s["obj_mass"] > 0.7
        cause = "perception_miss" if fail_light else "slip" if fail_mass else "none"
        res = {"scenario": s, "success": cause == "none", "cause": cause, "est_xy": [0, 0], "true_xy": [0, 0],
               "telemetry": {}}
        hist.append(make_record({"source": "test", "hypothesis": "", "expected_cause": ""}, res, 0))
    return hist


def test_prim_recovers_true_limits():
    env = prim_envelope(_synthetic())
    assert 0.35 <= env["box"]["light"][0] <= 0.60
    assert 0.40 <= env["box"]["obj_mass"][1] <= 0.80
    assert env["train_pass_rate"] >= 0.97
    for k, (lo, hi) in env["box"].items():  # never peels into the trained range
        assert lo <= params.BY_NAME[k].nominal[0] + 1e-9 and hi >= params.BY_NAME[k].nominal[1] - 1e-9


def test_boundaries_and_clusters():
    hist = _synthetic()
    b = factor_boundaries(hist)
    assert 0.30 <= b["light"]["low"]["mildest_failure"] < 0.45
    assert 0.70 < b["obj_mass"]["high"]["mildest_failure"] <= 0.85
    modes = {c["mode"] for c in clusters(hist)}
    assert {"perception_miss|light", "slip|obj_mass"} <= modes


def test_wilson():
    lo, hi = wilson(95, 100)
    assert 0.88 < lo < 0.95 < hi < 0.99
