"""MuJoCo scene construction.

The robot is a Cartesian gantry with a force-limited parallel gripper. Every scenario
parameter that affects physics or appearance is applied here; parameters that act on the
policy's inputs (image noise) or on the episode timeline (disturbance) are applied in
episode.py.
"""

from __future__ import annotations

import colorsys

import numpy as np

TABLE_FRICTION = 0.5
TRAINED_HUE = 0.0  # red
HAND_HOME = (-0.21, 0.0, 0.25)  # outside the overhead camera's view
FINGER_OPEN_Y = 0.055
FINGER_HALF = (0.012, 0.006, 0.030)
GRIP_FORCE = 9.0  # N per finger, the actuator force limit
CAM_HEIGHT = 0.60
CAM_FOVY = 30.0
SIDE_CAM = (0.50, -0.20, 0.34)


def _rgb(hue: float, value: float, sat: float = 0.85) -> str:
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, sat, value)
    return f"{r:.4f} {g:.4f} {b:.4f} 1"


def _look_at(pos, target=(0.0, 0.0, 0.03)) -> str:
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return " ".join(f"{v:.5f}" for v in (*right, *up))


def scenario_rng(scenario: dict) -> np.random.Generator:
    return np.random.default_rng(int(scenario["seed"]))


def target_hue(scenario: dict) -> float:
    sign = 1.0 if scenario_rng(scenario).random() < 0.5 else -1.0
    return TRAINED_HUE + sign * scenario["hue_shift"]


def camera_shift(scenario: dict) -> np.ndarray:
    rng = scenario_rng(scenario)
    rng.random()  # consume the hue-sign draw so streams stay independent
    ang = rng.uniform(0, 2 * np.pi)
    return scenario["camera_offset"] * np.array([np.cos(ang), np.sin(ang)])


def distractor_layout(scenario: dict) -> list[tuple[float, float, float]]:
    """Positions and hues of distractor cubes, kept clear of the target and each other."""
    rng = np.random.default_rng(int(scenario["seed"]) + 7919)
    target = np.array([scenario["obj_x"], scenario["obj_y"]])
    hue0 = target_hue(scenario)
    placed: list[np.ndarray] = []
    out = []
    for i in range(int(scenario["distractors"])):
        for _ in range(200):
            p = rng.uniform(-0.12, 0.12, size=2)
            if np.linalg.norm(p - target) > 0.07 and all(np.linalg.norm(p - q) > 0.05 for q in placed):
                break
        placed.append(p)
        sign = 1.0 if i % 2 == 0 else -1.0
        out.append((float(p[0]), float(p[1]), hue0 + sign * scenario["distractor_hue_gap"]))
    return out


def build_xml(scenario: dict) -> str:
    s = scenario
    size = s["obj_size"]
    light = s["light"]
    yaw = np.deg2rad(s["obj_yaw"])
    quat = f"{np.cos(yaw / 2):.6f} 0 0 {np.sin(yaw / 2):.6f}"
    cam = camera_shift(s)
    fx, fy, fz = FINGER_HALF
    hx, hy, hz = HAND_HOME

    distractors = "\n".join(
        f'''    <body name="distractor{i}" pos="{x:.4f} {y:.4f} 0.0181">
      <freejoint/>
      <geom type="box" size="0.018 0.018 0.018" mass="0.08" rgba="{_rgb(h, 0.9)}" friction="0.8 0.005 0.0001"/>
    </body>'''
        for i, (x, y, h) in enumerate(distractor_layout(s))
    )

    return f"""<mujoco model="stress_twin_gantry">
  <compiler angle="radian"/>
  <option timestep="0.002" cone="elliptic" impratio="10" noslip_iterations="3"/>
  <visual>
    <headlight ambient="{0.10 * light:.3f} {0.10 * light:.3f} {0.10 * light:.3f}" diffuse="{0.25 * light:.3f} {0.25 * light:.3f} {0.25 * light:.3f}" specular="0 0 0"/>
    <quality shadowsize="2048"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.32 0.38 0.50" rgb2="0.06 0.08 0.14" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.56 0.53 0.49" rgb2="0.52 0.49 0.45" width="256" height="256"/>
    <material name="table" texture="grid" texrepeat="6 6" reflectance="0.02"/>
    <material name="metal" rgba="0.78 0.81 0.86 1" specular="0.4"/>
    <material name="dark" rgba="0.22 0.27 0.38 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.15 -0.2 1.2" dir="-0.12 0.16 -1" directional="true" castshadow="true"
           diffuse="{0.75 * light:.3f} {0.75 * light:.3f} {0.75 * light:.3f}" specular="0.1 0.1 0.1"/>
    <geom name="table" type="box" pos="0 0 -0.02" size="0.35 0.35 0.02" material="table"
          friction="{TABLE_FRICTION} 0.005 0.0001"/>
    <!-- gantry frame: visual only -->
    <geom type="box" pos="0 0.30 0.30" size="0.30 0.012 0.012" material="dark" contype="0" conaffinity="0"/>
    <geom type="box" pos="0 -0.30 0.30" size="0.30 0.012 0.012" material="dark" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.30 0.30 0.15" size="0.012 0.012 0.15" material="dark" contype="0" conaffinity="0"/>
    <geom type="box" pos="-0.30 0.30 0.15" size="0.012 0.012 0.15" material="dark" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.30 -0.30 0.15" size="0.012 0.012 0.15" material="dark" contype="0" conaffinity="0"/>
    <geom type="box" pos="-0.30 -0.30 0.15" size="0.012 0.012 0.15" material="dark" contype="0" conaffinity="0"/>

    <camera name="overhead" pos="{cam[0]:.5f} {cam[1]:.5f} {CAM_HEIGHT}" xyaxes="1 0 0 0 1 0" fovy="{CAM_FOVY}"/>
    <camera name="side" pos="{SIDE_CAM[0]} {SIDE_CAM[1]} {SIDE_CAM[2]}" xyaxes="{_look_at(SIDE_CAM)}" fovy="42"/>

    <body name="hand" pos="0 0 0" gravcomp="1">
      <joint name="gx" type="slide" axis="1 0 0" range="-0.24 0.16" damping="40"/>
      <joint name="gy" type="slide" axis="0 1 0" range="-0.16 0.16" damping="40"/>
      <joint name="gz" type="slide" axis="0 0 1" range="0.0 0.40" damping="40"/>
      <geom name="carriage" type="box" pos="0 0 0.05" size="0.025 0.025 0.04" material="dark" mass="0.4" contype="0" conaffinity="0"/>
      <geom name="palm" type="box" pos="0 0 -0.005" size="0.02 0.065 0.008" material="metal" mass="0.2"/>
      <body name="finger_l" pos="0 {FINGER_OPEN_Y} -0.04" gravcomp="1">
        <joint name="fl" type="slide" axis="0 -1 0" range="0 0.046" damping="2"/>
        <geom name="finger_l" type="box" size="{fx} {fy} {fz}" material="metal" mass="0.05" friction="0.1 0.005 0.0001" condim="4"/>
      </body>
      <body name="finger_r" pos="0 -{FINGER_OPEN_Y} -0.04" gravcomp="1">
        <joint name="fr" type="slide" axis="0 1 0" range="0 0.046" damping="2"/>
        <geom name="finger_r" type="box" size="{fx} {fy} {fz}" material="metal" mass="0.05" friction="0.1 0.005 0.0001" condim="4"/>
      </body>
    </body>

    <body name="object" pos="{s['obj_x']:.5f} {s['obj_y']:.5f} {size + 0.0005:.5f}" quat="{quat}">
      <freejoint name="object"/>
      <geom name="object" type="box" size="{size:.5f} {size:.5f} {size:.5f}" mass="{s['obj_mass']:.4f}"
            rgba="{_rgb(target_hue(s), s['obj_value'])}" friction="{s['obj_friction']:.4f} 0.005 0.0001" condim="4"/>
    </body>
{distractors}
  </worldbody>
  <equality>
    <joint joint1="fl" joint2="fr" polycoef="0 1 0 0 0" solref="0.005 1"/>
  </equality>
  <actuator>
    <position name="ax" joint="gx" kp="1500" ctrlrange="-0.24 0.16"/>
    <position name="ay" joint="gy" kp="1500" ctrlrange="-0.16 0.16"/>
    <position name="az" joint="gz" kp="1500" ctrlrange="0.0 0.40"/>
    <position name="grip" joint="fl" kp="600" ctrlrange="0 0.046" forcerange="-{GRIP_FORCE} {GRIP_FORCE}"/>
  </actuator>
</mujoco>
"""
