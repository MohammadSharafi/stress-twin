import numpy as np

from stress_twin import params
from stress_twin.episode import run_episode


def _nominal(seed=0, **kw):
    s = params.sample_nominal(np.random.default_rng(seed))
    s.update(kw)
    return params.clip(s)


def test_deterministic_replay():
    s = _nominal(4, light=0.4, obj_mass=0.7)
    a, b = run_episode(s), run_episode(s)
    assert a.digest() == b.digest()


def test_recording_does_not_change_outcome():
    s = _nominal(5, disturbance=0.03)
    plain, rec = run_episode(s), run_episode(s, record=True)
    assert plain.digest() == rec.digest()
    assert len(rec.frames) > 30 and rec.overhead.shape == (64, 64, 3)


def test_nominal_success():
    assert all(run_episode(_nominal(i)).success for i in range(6))


def test_known_failure_causes():
    assert run_episode(_nominal(6, disturbance=0.045)).cause == "disturbance"
    assert run_episode(_nominal(7, obj_mass=1.4, obj_friction=0.3)).cause == "slip"
    assert run_episode(_nominal(8, light=0.1)).cause == "perception_miss"
