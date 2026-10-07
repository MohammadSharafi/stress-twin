"""Failure analyst: explains each failure cluster from keyframes and telemetry.

With Token Factory available, the vision role (Nemotron 3 Nano Omni or Cosmos3 Reasoner)
looks at the perception image and side-view keyframes. Its stated cause is then checked
against the deterministic telemetry diagnosis; a disagreement is flagged in the report and
the telemetry label stays authoritative. Without a key, an offline analyst writes the
explanation from telemetry alone and says so.
"""

from __future__ import annotations

import numpy as np

from .analysis import fmt_value, stressors
from .diagnose import CAUSES

SYSTEM = """You are a robotics failure analyst. You get the robot's overhead perception image
(green circle = true object position, red X = where the vision model thought it was),
four side-view keyframes (end of approach, end of descent, end of grasp, end of episode),
the scenario conditions that differ from training, and telemetry. Explain in plain language
what went wrong and why, citing what is visible in the frames. Be specific and brief."""

SCHEMA = {
    "type": "object",
    "properties": {
        "cause": {"type": "string", "enum": [c for c in CAUSES if c != "none"]},
        "explanation": {"type": "string"},
        "visual_evidence": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["cause", "explanation", "visual_evidence", "confidence"],
}


def _conditions(rec: dict) -> str:
    st = stressors(rec["scenario"])
    if not st:
        return "all conditions inside the training range"
    return "; ".join(f"{s['label']} {s['text']} (trained {s['nominal']})" for s in st)


def offline_caption(rec: dict) -> dict:
    t = rec["telemetry"]
    cause = rec["cause"]
    cond = _conditions(rec)
    err_mm = t["est_err"] * 1000
    if cause == "perception_miss":
        why = f"The vision model placed the object {err_mm:.0f} mm from its true position, outside the gripper's capture range."
    elif cause == "disturbance":
        why = f"The object was bumped {t['bump_disp'] * 1000:.0f} mm after the single perception pass, so the gripper went to where it used to be."
    elif cause == "collision":
        why = f"The hand struck the object while descending and pushed it {t['descent_disp'] * 1000:.0f} mm before the fingers closed."
    elif cause == "slip":
        why = (f"The fingers closed on the object but it slid out during lift (peak height {t['max_obj_z'] * 100:.1f} cm); "
               f"grip force cannot carry {fmt_value('obj_mass', rec['scenario']['obj_mass'])} at friction {rec['scenario']['obj_friction']:.2f}.")
    else:
        why = f"Perception was within {err_mm:.0f} mm but the fingers did not secure the object."
    return {"cause": cause, "explanation": f"{why} Conditions: {cond}.", "visual_evidence": "telemetry only (offline analyst)",
            "confidence": 1.0, "grounded": True, "analyst": "offline"}


def keyframes(frames: list) -> list:
    if not frames:
        return []
    idx = np.linspace(len(frames) * 0.28, len(frames) - 1, 4).astype(int)
    return [frames[i] for i in idx]


def llm_caption(client, rec: dict, overhead_marked, frames: list) -> dict:
    t = rec["telemetry"]
    user = (f"Conditions that differ from training: {_conditions(rec)}.\n"
            f"Telemetry: perception error {t['est_err'] * 1000:.0f} mm; object moved {t['bump_disp'] * 1000:.0f} mm before descent "
            f"and {t['descent_disp'] * 1000:.0f} mm during descent; finger contact at grasp: left={t['contact_l_at_close']}, "
            f"right={t['contact_r_at_close']}; peak object height {t['max_obj_z'] * 100:.1f} cm; final height "
            f"{t['final_obj_z'] * 100:.1f} cm.\nCause definitions: " + "; ".join(f"{k}: {v}" for k, v in CAUSES.items() if k != "none"))
    data = client.chat_json("vision", SYSTEM, user, SCHEMA, images=[overhead_marked, *keyframes(frames)], max_tokens=1200)
    cause = data.get("cause", "")
    return {"cause": cause, "explanation": str(data.get("explanation", ""))[:600],
            "visual_evidence": str(data.get("visual_evidence", ""))[:400],
            "confidence": float(data.get("confidence", 0) or 0), "grounded": cause == rec["cause"],
            "analyst": client.models().get("vision", "?")}
