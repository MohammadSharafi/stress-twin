"""Run one episode of the policy under test in one scenario.

Deterministic within one environment: the result depends only on the scenario dict
(including its seed). Different OpenGL backends can shift the perception image slightly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from . import scene
from .diagnose import diagnose
from .perception import IMG, Perception, add_noise, render_overhead

DT = 0.002
SAMPLE_EVERY = 10          # telemetry at 50 Hz
FRAME_EVERY = 0.08         # seconds between clip frames
SPEED = 0.35               # m/s cartesian speed limit
APPROACH_Z = 0.20
GRASP_Z = 0.075
LIFT_Z = 0.22
SUCCESS_Z = 0.12
PHASES = (("approach", 1.10), ("descend", 0.70), ("close", 0.45), ("lift", 0.75), ("hold", 0.80))


@dataclass
class EpisodeResult:
    scenario: dict
    success: bool
    cause: str
    est_xy: list
    true_xy: list
    telemetry: dict
    trace: dict = field(default_factory=dict)       # sampled time series (compact)
    frames: list = field(default_factory=list, repr=False)   # side-view frames (record=True)
    overhead: object = field(default=None, repr=False)      # perception image (record=True)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("frames")
        d.pop("overhead")
        return d

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()[:16]


def _contacts(model, data, gid_obj, gids) -> dict[str, bool]:
    hit = {k: False for k in gids}
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if gid_obj in pair:
            for k, g in gids.items():
                if g in pair:
                    hit[k] = True
    return hit


def run_episode(scenario: dict, record: bool = False, perception: Perception | None = None,
                frame_size: tuple[int, int] = (240, 320)) -> EpisodeResult:
    """frame_size is (height, width) of recorded side-view frames."""
    perception = perception or Perception.shared()
    model = mujoco.MjModel.from_xml_string(scene.build_xml(scenario))
    data = mujoco.MjData(model)
    jx, jy, jz = (model.joint(n).qposadr[0] for n in ("gx", "gy", "gz"))
    data.qpos[[jx, jy, jz]] = scene.HAND_HOME
    data.ctrl[:3] = scene.HAND_HOME
    mujoco.mj_forward(model, data)

    obj_body = model.body("object").id
    obj_qvel = model.joint("object").dofadr[0]
    gid_obj = model.geom("object").id
    gids = {"l": model.geom("finger_l").id, "r": model.geom("finger_r").id, "palm": model.geom("palm").id}

    # --- perception (once, at t = 0, open loop like many deployed pick pipelines) ---
    eye = mujoco.Renderer(model, IMG, IMG)
    img = add_noise(render_overhead(model, data, eye), scenario["image_noise"], scenario["seed"])
    eye.close()
    est = perception.predict(img)
    true0 = data.xpos[obj_body][:2].copy()

    side = mujoco.Renderer(model, *frame_size) if record else None
    frames: list[np.ndarray] = []
    next_frame = 0.0

    # --- disturbance: a bump that slides the object while the robot is approaching ---
    bump_rng = np.random.default_rng(int(scenario["seed"]) + 31337)
    bump_dir = bump_rng.uniform(0, 2 * np.pi)
    mu = max(scenario["obj_friction"], scene.TABLE_FRICTION)
    bump_speed = np.sqrt(2 * mu * 9.81 * scenario["disturbance"])
    bump_t = 0.15

    trace = {k: [] for k in ("t", "hand_z", "obj_z", "grip", "contact")}
    target = np.array(scene.HAND_HOME, dtype=float)
    grip_cmd = 0.0
    marks: dict[str, np.ndarray] = {}
    tel = {"palm_contact_descent": False, "contact_l_at_close": False, "contact_r_at_close": False,
           "max_obj_z": 0.0, "bumped": False}

    t = 0.0
    step = 0
    phase_end = 0.0
    for phase, duration in PHASES:
        phase_end += duration
        if phase == "approach":
            goal = np.array([est[0], est[1], APPROACH_Z])
        elif phase == "descend":
            marks["pre_descent"] = data.xpos[obj_body][:2].copy()
            goal = np.array([est[0], est[1], GRASP_Z])
        elif phase == "close":
            marks["pre_close"] = data.xpos[obj_body][:2].copy()
            grip_cmd = 0.046
        elif phase == "lift":
            hit = _contacts(model, data, gid_obj, gids)
            tel["contact_l_at_close"], tel["contact_r_at_close"] = hit["l"], hit["r"]
            goal = np.array([est[0], est[1], LIFT_Z])

        while t < phase_end - 1e-9:
            if not tel["bumped"] and scenario["disturbance"] > 0 and t >= bump_t:
                data.qvel[obj_qvel:obj_qvel + 2] = bump_speed * np.array([np.cos(bump_dir), np.sin(bump_dir)])
                tel["bumped"] = True
            delta = goal - target
            dist = np.linalg.norm(delta)
            if dist > 1e-9:
                target = target + delta * min(1.0, SPEED * DT / dist)
            data.ctrl[:3] = target
            data.ctrl[3] = grip_cmd
            mujoco.mj_step(model, data)
            t += DT
            step += 1

            if phase == "descend":
                if _contacts(model, data, gid_obj, gids)["palm"]:
                    tel["palm_contact_descent"] = True
            tel["max_obj_z"] = max(tel["max_obj_z"], float(data.xpos[obj_body][2]))
            if step % SAMPLE_EVERY == 0:
                hit = _contacts(model, data, gid_obj, gids)
                trace["t"].append(round(t, 3))
                trace["hand_z"].append(round(float(data.qpos[jz]), 4))
                trace["obj_z"].append(round(float(data.xpos[obj_body][2]), 4))
                trace["grip"].append(round(float(data.qpos[model.joint("fl").qposadr[0]]), 4))
                trace["contact"].append(int(hit["l"]) + int(hit["r"]))
            if record and t >= next_frame:
                side.update_scene(data, camera="side")
                frames.append(side.render().copy())
                next_frame += FRAME_EVERY

    if side is not None:
        side.close()

    final = _contacts(model, data, gid_obj, gids)
    obj_z = float(data.xpos[obj_body][2])
    success = bool(obj_z > SUCCESS_Z and final["l"] and final["r"])
    tel.update({
        "success": success,
        "est_err": round(float(np.linalg.norm(est - true0)), 5),
        "bump_disp": round(float(np.linalg.norm(marks["pre_descent"] - true0)), 5),
        "descent_disp": round(float(np.linalg.norm(marks["pre_close"] - marks["pre_descent"])), 5),
        "final_obj_z": round(obj_z, 4),
        "max_obj_z": round(tel["max_obj_z"], 4),
        "final_contact": int(final["l"]) + int(final["r"]),
    })
    tel.pop("bumped")
    cause = diagnose(tel)
    return EpisodeResult(
        scenario=scenario, success=success, cause=cause,
        est_xy=[round(float(v), 5) for v in est], true_xy=[round(float(v), 5) for v in true0],
        telemetry=tel, trace=trace, frames=frames, overhead=img if record else None,
    )
