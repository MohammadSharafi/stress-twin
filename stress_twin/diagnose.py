"""Deterministic failure diagnosis from episode telemetry.

This is the ground truth that every model-written explanation is checked against. A
caption that names a different cause than the telemetry supports is flagged, never
silently accepted.
"""

from __future__ import annotations

CAUSES: dict[str, str] = {
    "none": "Task succeeded.",
    "perception_miss": "The vision model mislocated the object, so the gripper closed on empty space.",
    "disturbance": "The object was moved after perception; the policy acted on a stale estimate.",
    "collision": "The hand struck the object while descending (object too tall, too wide or rotated for the grasp).",
    "grasp_miss": "Perception was accurate but the fingers failed to secure the object.",
    "slip": "The object was grasped but slid out of the fingers during lift or hold.",
}

# Which conditions can physically produce each cause. Used to attribute a failure to the
# right factor when several conditions are pushed at once.
CAUSE_FACTORS: dict[str, tuple[str, ...]] = {
    "perception_miss": ("light", "hue_shift", "obj_value", "distractors", "distractor_hue_gap", "camera_offset",
                        "image_noise", "obj_size", "obj_yaw", "obj_x", "obj_y"),
    "disturbance": ("disturbance",),
    "collision": ("obj_size", "obj_yaw", "camera_offset", "light", "hue_shift", "image_noise"),
    "grasp_miss": ("obj_size", "obj_yaw", "obj_friction", "camera_offset", "image_noise", "light", "hue_shift"),
    "slip": ("obj_mass", "obj_friction", "obj_size", "obj_yaw"),
}

PERCEPTION_TOL = 0.015  # m, beyond this the grasp is outside the finger capture region
BUMP_TOL = 0.012        # m, object displacement before descent that invalidates perception
COLLISION_TOL = 0.010   # m, object displacement during descent caused by the hand


def diagnose(t: dict) -> str:
    """t is the telemetry summary produced by episode.run_episode."""
    if t["success"]:
        return "none"
    grasped = t["contact_l_at_close"] and t["contact_r_at_close"]
    if not grasped and t["bump_disp"] > BUMP_TOL:
        return "disturbance"
    if t["palm_contact_descent"] or t["descent_disp"] > COLLISION_TOL:
        return "collision"
    if not grasped:
        return "perception_miss" if t["est_err"] > PERCEPTION_TOL else "grasp_miss"
    return "slip"
