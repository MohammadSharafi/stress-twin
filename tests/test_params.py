import numpy as np

from stress_twin import params
from stress_twin.strategies import attributed_factor, stress, stressed_sample


def test_clip_bounds_and_integers():
    s = params.clip({"seed": 3, "obj_mass": 99, "light": -5, "distractors": 2.6, "obj_x": float("nan")})
    assert s["obj_mass"] == params.BY_NAME["obj_mass"].high
    assert s["light"] == params.BY_NAME["light"].low
    assert s["distractors"] == 3 and isinstance(s["distractors"], int)
    assert np.isfinite(s["obj_x"])


def test_unit_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = params.sample_uniform(rng)
        back = params.from_unit(params.to_unit(s), s["seed"])
        for p in params.PARAMS:
            assert abs(back[p.name] - s[p.name]) < 1e-3 * (p.high - p.low) + (0.5 if p.integer else 0)


def test_nominal_has_zero_stress():
    rng = np.random.default_rng(1)
    for _ in range(50):
        s = params.sample_nominal(rng)
        assert params.is_nominal(s)
        assert stress(s) == 0.0


def test_stressed_sample_pushes_some_factor():
    rng = np.random.default_rng(2)
    pushed = [len(params.off_nominal(stressed_sample(rng))) for _ in range(200)]
    assert max(pushed) <= 3 and np.mean(pushed) > 0.8


def test_attribution_is_cause_aware():
    s = params.sample_nominal(np.random.default_rng(3))
    s.update({"camera_offset": 0.006, "obj_mass": 0.57})
    assert attributed_factor(s, "slip") == "obj_mass"
    assert attributed_factor(s, "perception_miss") == "camera_offset"
