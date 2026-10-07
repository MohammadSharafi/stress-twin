"""Professional demo video for Stress Twin (1920x1080, 30 fps, about 2:20).

    python scripts/make_pro_video.py runs/demo ../submission/stress-twin-demo-pro.mp4
    python scripts/make_pro_video.py --check-claims runs/demo

Footage is deterministic MuJoCo replays of episodes from the run (cinematic free-camera
orbits and the fixed side camera); every number on screen and in the narration is read from,
or checked against, the run's files. Narration: Kokoro neural TTS (Apache-2.0, run locally
through scripts/tts_kokoro.py). Music and sound effects are synthesized here, so the video
contains no third-party audio.
"""

from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import make_gifs as mg  # noqa: E402  (perception sweep)
from stress_twin import params  # noqa: E402
from stress_twin.analysis import clusters, factor_boundaries, fmt_value, load_history, prim_envelope, stressors  # noqa: E402
from stress_twin.diagnose import PERCEPTION_TOL  # noqa: E402
from stress_twin.episode import run_episode  # noqa: E402
from stress_twin.perception import world_to_overhead_px  # noqa: E402
from stress_twin.report import CAUSE_TITLE  # noqa: E402
from stress_twin import scene as scn  # noqa: E402
from stress_twin.strategies import STRESSABLE  # noqa: E402

W, H, FPS, SR = 1920, 1080, 30, 48000
XFADE = 0.5
BG = (7, 11, 26)
INK = (238, 242, 250)
MUTED = (142, 156, 186)
RED = (255, 70, 70)
GREEN = (46, 222, 122)
AMBER = (251, 191, 36)
BLUE = (99, 160, 255)
NV = (118, 185, 0)
CARD = (17, 26, 50)
LINE = (38, 52, 92)

# ---------------- narration (numbers are verified by --check-claims) ----------------

NARRATION = {
    "cold_ok": ["This robot passes every test in the conditions it was trained on."],
    "cold_fail": ["Dim the lights, and it reaches for empty air."],
    "title": ["This is Stress Twin. A crash-test lab for robot brains."],
    "problem": ["Robot policies learn from narrow data, but the real world keeps changing: light, colour, weight, calibration.",
                "Before a robot ships, someone should find out exactly where it breaks. Today, almost nobody does."],
    "how": ["Stress Twin builds a digital twin of the task, and lets an adversary attack the policy.",
            "In live mode, the adversary is NVIDIA Nemotron 3 Ultra on Nebius Token Factory. It reads every result so far, and proposes new, testable hypotheses.",
            "Each failure is diagnosed from telemetry, and Nemotron Nano Omni explains it from the video, checked against that diagnosis.",
            "Then the evidence becomes a safety case."],
    "search": ["Every round targets the mildest change most likely to break the robot.",
               "In this campaign, two hundred and forty simulated episodes uncovered seventy-two failures, across fourteen distinct failure modes."],
    "failures": ["A heavier load slips out of the gripper.", "A bump makes it act on a stale estimate.",
                 "An unfamiliar colour fools the vision model.", "And a larger cube collides on the way down."],
    "robot_view": ["Here is what the robot's own camera sees as the light falls. Its estimate drifts more than a hundred millimetres from the cube."],
    "safety": ["Stress Twin then writes an operating envelope: limits a safety reviewer can actually read.",
               "On one hundred and fifty scenarios the search never saw, ninety-five percent pass inside the envelope. Only thirty-four percent pass outside it."],
    "results": ["Against random stress testing with the same budget, its search finds nearly four times as many subtle failures.",
                "This demo ran offline. With a Token Factory key, the same campaign runs with Nemotron as the adversary."],
    "cta": ["Stress Twin. Find where your robot breaks, before your customers do."],
}


def check_claims(run: Path) -> bool:
    camp = json.loads((run / "campaign.json").read_text())["summary"]
    val = json.loads((run / "validation.json").read_text())
    bench = json.loads(Path("results/benchmark_240.json").read_text())["mean"]
    ratio = bench["boundary"]["subtle_failures"] / bench["random"]["subtle_failures"]
    hist = load_history(run)
    cl = {c["mode"]: c for c in clusters(hist)}
    sweep = mg.perception_sweep(params.sample_nominal(np.random.default_rng(3)), [0.1])
    s0 = params.sample_nominal(np.random.default_rng(3))
    sweep_err = float(np.linalg.norm(sweep[0][2] - np.array([s0["obj_x"], s0["obj_y"]])))
    text = " ".join(" ".join(v) for v in NARRATION.values())
    checks = [
        ("two hundred and forty simulated episodes", camp["episodes"] == 240),
        ("seventy-two failures", camp["failures"] == 72),
        ("fourteen distinct failure modes", camp["modes"] == 14),
        ("one hundred and fifty scenarios", val["n_inside"] == 150),
        ("ninety-five percent pass inside", round(val["pass_rate_inside"] * 100) == 95),
        ("thirty-four percent pass outside", round(val["pass_rate_outside"] * 100) == 34),
        ("nearly four times as many subtle failures", 3.5 <= ratio < 4.0),
        ("more than a hundred millimetres", sweep_err > 0.100),
        ("heavier load slips", "slip|obj_mass" in cl),
        ("bump makes it act on a stale estimate", "disturbance|disturbance" in cl),
        ("unfamiliar colour fools the vision model", "perception_miss|hue_shift" in cl),
        ("larger cube collides", "collision|obj_size" in cl),
        ("Dim the lights, and it reaches for empty air", "perception_miss|light" in cl),
        ("This demo ran offline", True),
    ]
    ok = True
    for phrase, holds in checks:
        present = phrase in text
        print(f"{'ok ' if present and holds else 'BAD'} {phrase!r}: in narration={present}, matches data={holds}")
        ok &= present and holds
    print(f"benchmark ratio {ratio:.2f}, sweep error at light 0.10: {sweep_err * 1000:.0f} mm")
    print("CLAIMS_OK" if ok else "CLAIMS_BAD")
    return ok


# ---------------- primitives ----------------

def F(size, bold=True):
    return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
                              else "/System/Library/Fonts/Supplemental/Arial.ttf", size)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return 3 * x * x - 2 * x * x * x


def make_backdrop():
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]
    base = np.stack([np.full((H, W), BG[c], float) + 22 * (1 - y) for c in range(3)], axis=2)
    glow = np.exp(-(((x - 0.82) ** 2) / 0.06 + ((y - 0.95) ** 2) / 0.1))[..., None] * np.array([80, 12, 26])
    glow2 = np.exp(-(((x - 0.1) ** 2) / 0.05 + ((y - 0.05) ** 2) / 0.08))[..., None] * np.array([10, 24, 70])
    img = Image.fromarray(np.clip(base + glow + glow2, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for gx in range(0, W, 48):
        d.line([(gx, 0), (gx, H)], fill=(17, 25, 50))
    for gy in range(0, H, 48):
        d.line([(0, gy), (W, gy)], fill=(17, 25, 50))
    return img


def make_vignette():
    y = np.linspace(-1, 1, H)[:, None]
    x = np.linspace(-1, 1, W)[None, :]
    r = np.sqrt(x ** 2 * 0.8 + y ** 2)
    a = np.clip((r - 0.75) / 0.6, 0, 1) ** 1.6 * 150
    return Image.fromarray(a.astype(np.uint8), "L")


BACKDROP = None
VIGNETTE = None


def canvas():
    return BACKDROP.copy()


def brand_bar(img, label=""):
    d = ImageDraw.Draw(img)
    d.text((64, 44), "STRESS", font=F(34), fill=INK)
    d.text((64 + d.textlength("STRESS ", font=F(34)), 44), "TWIN", font=F(34), fill=RED)
    if label:
        d.text((W - 64 - d.textlength(label.upper(), font=F(20)), 54), label.upper(), font=F(20), fill=MUTED)


def glow_box(img, box, color, radius=16, strength=140, blur=16):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    x0, y0, x1, y1 = box
    ImageDraw.Draw(layer).rounded_rectangle([x0 - 8, y0 - 8, x1 + 8, y1 + 8], radius + 6, fill=color + (strength,))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    img.paste(layer, (0, 0), layer)


def badge(img, xy, text, color, size=36, glow=True, scale=1.0):
    f = F(int(size * scale))
    d = ImageDraw.Draw(img)
    w = d.textlength(text, font=f)
    x, y = xy
    box = [x, y, x + w + 40 * scale, y + f.size + 26 * scale]
    if glow:
        glow_box(img, box, color, 14)
        d = ImageDraw.Draw(img)
    d.rounded_rectangle(box, int(12 * scale), fill=color)
    d.text((x + 20 * scale, y + 12 * scale), text, font=f, fill=(255, 255, 255))
    return box


def chip(d, xy, text, color, size=24):
    f = F(size)
    w = d.textlength(text, font=f)
    x, y = xy
    d.rounded_rectangle([x, y, x + w + 36, y + size + 22], (size + 22) // 2, outline=color, width=3)
    d.text((x + 18, y + 10), text, font=f, fill=color)
    return x + w + 36


def caption(img, text, alpha=1.0):
    if not text or alpha <= 0:
        return
    d = ImageDraw.Draw(img)
    f = F(40)
    words, lines, cur = text.split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if d.textlength(trial, font=f) > 1500 and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    lines.append(cur)
    lines = lines[:2]
    lh = 54
    widths = [d.textlength(l, font=f) for l in lines]
    bw = max(widths) + 64
    bh = lh * len(lines) + 30
    x0, y0 = (W - bw) / 2, H - 70 - bh
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], 18, fill=(4, 7, 18, int(205 * alpha)))
    for i, l in enumerate(lines):
        ld.text(((W - widths[i]) / 2, y0 + 15 + i * lh), l, font=f, fill=(255, 255, 255, int(255 * alpha)))
    img.paste(layer, (0, 0), layer)


class Clip:
    """HD frames kept as JPEG bytes so long recordings stay small in memory."""

    def __init__(self):
        self.frames: list[bytes] = []
        self.result = None

    def add(self, arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=92)
        self.frames.append(buf.getvalue())

    def get(self, i) -> Image.Image:
        i = int(min(max(i, 0), len(self.frames) - 1))
        return Image.open(io.BytesIO(self.frames[i])).convert("RGB")

    def __len__(self):
        return len(self.frames)


def orbit(scenario, az0=150.0, rate=8.0, dist=0.42):
    def cam(t):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        rise = ease_io((t - 2.4) / 1.4)
        c.lookat[:] = [scenario["obj_x"] * 0.6, scenario["obj_y"] * 0.6, 0.085 + 0.06 * rise]
        c.distance = dist + 0.03 * rise
        c.azimuth = az0 + rate * t
        c.elevation = -15 - 3 * t
        return c
    return cam


def record(scenario, camera, size=(1080, 1920), frame_every=1 / 30) -> Clip:
    clip = Clip()
    clip.result = run_episode(scenario, record=True, frame_size=size, frame_every=frame_every, camera=camera,
                              on_frame=clip.add)
    return clip


# ---------------- scenes ----------------

class Scene:
    def __init__(self, name, draw, min_dur=0.0, module=False, sfx=None, pad=(0.55, 0.7)):
        self.name, self.draw, self.min_dur, self.module = name, draw, min_dur, module
        self.sentences = NARRATION.get(name, [])
        self.sfx = sfx or []  # (local_time or 'end-x', kind)
        self.pad = pad
        self.local = []  # (start, end, text) filled by the timeline
        self.start = 0.0
        self.dur = 0.0


def play(clip, t, dur, speed_min=0.45, speed_max=1.0, hold=0.9):
    """Map scene time to a clip frame: fit the clip into the scene (leaving `hold` seconds
    on the final frame), at a speed between speed_min and speed_max."""
    clip_secs = len(clip) / FPS
    speed = min(speed_max, max(speed_min, clip_secs / max(dur - hold, 0.5)))
    return clip.get(t * speed * FPS), t * speed * FPS >= len(clip) - 1


def scene_cold(clip, ok: bool, label: str):
    def draw(t, dur):
        img, done = play(clip, t, dur)
        d = ImageDraw.Draw(img)
        a = ease(t / 0.6)
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).rounded_rectangle([56, H - 300, 56 + 720, H - 216], 14, fill=(4, 7, 18, int(190 * a)))
        img.paste(layer, (0, 0), layer)
        d.text((80, H - 290), "CONDITIONS", font=F(20), fill=MUTED)
        d.text((80, H - 262), label, font=F(32), fill=GREEN if ok else AMBER)
        if done or t > dur - 0.9:
            k = ease((t - (dur - 0.9)) / 0.25) if not done else 1.0
            s = 1.6 - 0.6 * k
            badge(img, (64, 70), "SUCCESS" if ok else "FAILURE", GREEN if ok else RED, 56, scale=s)
        return img
    return draw


def scene_title(t, dur):
    img = canvas()
    d = ImageDraw.Draw(img)
    f = F(170)
    w1 = d.textlength("STRESS ", font=f)
    w = w1 + d.textlength("TWIN", font=f)
    x = (W - w) / 2
    a = ease(t / 0.8)
    off = (1 - a) * 60
    col = tuple(int(BG[i] + (INK[i] - BG[i]) * a) for i in range(3))
    rcol = tuple(int(BG[i] + (RED[i] - BG[i]) * ease((t - 0.25) / 0.8)) for i in range(3))
    d.text((x - off, 360), "STRESS ", font=f, fill=col)
    d.text((x + w1 + off, 360), "TWIN", font=f, fill=rcol)
    lw = ease((t - 0.6) / 0.8) * w
    d.rounded_rectangle([x, 572, x + lw, 580], 4, fill=RED)
    tag = "A crash-test lab for robot brains"
    ta = ease((t - 1.0) / 0.8)
    d.text(((W - d.textlength(tag, font=F(48, False))) / 2, 612), tag, font=F(48, False),
           fill=tuple(int(BG[i] + (MUTED[i] - BG[i]) * ta) for i in range(3)))
    return img


def scene_problem(bg_frame):
    blurred = bg_frame.filter(ImageFilter.GaussianBlur(14))
    blurred = Image.blend(blurred, Image.new("RGB", (W, H), BG), 0.72)
    left = ["bright, even light", "one red part", "a calibrated camera", "a known weight"]
    right = ["dim corners and glare", "new colours and lookalikes", "calibration drift", "heavier loads and bumps"]

    def draw(t, dur):
        img = blurred.copy()
        brand_bar(img, "the gap")
        d = ImageDraw.Draw(img)
        d.text((140, 190), "Trained on", font=F(66), fill=GREEN)
        d.text((1020, 190), "Deployed into", font=F(66), fill=RED)
        for i, s in enumerate(left):
            a = ease((t - 0.3 - i * 0.35) / 0.5)
            if a > 0:
                d.text((140 - (1 - a) * 40, 310 + i * 110), "•  " + s, font=F(50, False),
                       fill=tuple(int(BG[c] + (INK[c] - BG[c]) * a) for c in range(3)))
        t2 = (self_start2[0] if self_start2 else dur * 0.45)
        for i, s in enumerate(right):
            a = ease((t - t2 - i * 0.3) / 0.5)
            if a > 0:
                d.text((1020 + (1 - a) * 40, 310 + i * 110), "•  " + s, font=F(50, False),
                       fill=tuple(int(BG[c] + (INK[c] - BG[c]) * a) for c in range(3)))
        d.line([(960, 200), (960, 760)], fill=LINE, width=3)
        return img
    self_start2 = []
    draw.set_second = lambda v: self_start2.append(v)
    return draw


NODES = [
    ("Policy under test", "CNN perception|+ scripted grasp", (70, 330)),
    ("Digital twin", "MuJoCo|simulation", (436, 330)),
    ("Adversary", "NVIDIA Nemotron 3 Ultra|live mode", (802, 330)),
    ("Diagnosis", "deterministic,|from telemetry", (1168, 330)),
    ("Analyst", "NVIDIA Nemotron Nano Omni|live mode", (1534, 330)),
    ("Safety case", "envelope|+ held-out validation", (1168, 660)),
]
NW, NH = 316, 170
REVEAL = [[0, 1], [2], [3, 4], [5]]  # nodes revealed per narration sentence


def scene_how(scene_ref):
    def node_center(i):
        x, y = NODES[i][2]
        return x + NW / 2, y + NH / 2

    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    def draw(t, dur):
        img = canvas()
        brand_bar(img, "how it works")
        d = ImageDraw.Draw(img)
        starts = [a for a, _, _ in scene_ref[0].local] or [0.0]
        cur = max([k for k, a in enumerate(starts) if t >= a - 0.2] or [0])
        shown = {n for k in range(cur + 1) for n in REVEAL[min(k, 3)]}
        active = set(REVEAL[min(cur, 3)])
        # edges
        for a_i, b_i in edges:
            if a_i in shown and b_i in shown:
                (ax, ay), (bx, by) = node_center(a_i), node_center(b_i)
                if b_i == 5:
                    pts = [(ax, ay + NH / 2), (ax, by - 40), (bx + NW / 2 + 30, by - 40), (bx + NW / 2 + 30, by)]
                    pts = [(ax, ay + NH / 2), (ax, by), (bx + NW / 2, by)]
                else:
                    pts = [(ax + NW / 2, ay), (bx - NW / 2, by)]
                d.line(pts, fill=LINE, width=6, joint="curve")
                # moving pulse
                seg = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
                lens = [math.dist(p, q) for p, q in seg]
                pos = ((t * 260) % sum(lens))
                for (p, q), L in zip(seg, lens):
                    if pos <= L:
                        px, py = p[0] + (q[0] - p[0]) * pos / L, p[1] + (q[1] - p[1]) * pos / L
                        d.ellipse([px - 9, py - 9, px + 9, py + 9], fill=RED if a_i >= 2 else BLUE)
                        break
                    pos -= L
        if {2, 3} <= shown:  # loop back: next round
            (ax, ay), (bx, by) = node_center(3), node_center(2)
            top = ay - NH / 2 - 70
            d.line([(ax, ay - NH / 2), (ax, top), (bx, top), (bx, by - NH / 2)], fill=(120, 40, 50), width=5)
            d.text(((ax + bx) / 2 - d.textlength("next round", font=F(24)) / 2, top - 36), "next round", font=F(24), fill=RED)
        # nodes
        for i, (title, sub, (x, y)) in enumerate(NODES):
            if i not in shown:
                continue
            k = min([n for n, grp in enumerate(REVEAL) if i in grp])
            a = ease((t - (starts[k] if k < len(starts) else 0) + 0.2) / 0.5)
            yy = y + (1 - a) * 30
            if i in active:
                glow_box(img, (x, yy, x + NW, yy + NH), RED if i in (2, 4) else BLUE, 18, 110)
                d = ImageDraw.Draw(img)
            d.rounded_rectangle([x, yy, x + NW, yy + NH], 18, fill=CARD, outline=RED if i in active else LINE, width=3)
            d.text((x + 22, yy + 26), title, font=F(32), fill=INK)
            for li, part in enumerate(sub.split("|")):
                col = (NV if "Nemotron" in part else (AMBER if part == "live mode" else MUTED))
                f_sub = F(21, False)
                while d.textlength(part, font=f_sub) > NW - 36:
                    f_sub = F(f_sub.size - 1, False)
                d.text((x + 22, yy + 82 + li * 32), part, font=f_sub, fill=col)
        return img
    return draw


def scene_search(history, log_lines):
    rows = [n for n in STRESSABLE if n != "distractor_hue_gap"]
    bounds = factor_boundaries(history)
    x0, x1, y0, rh = 1010, 1720, 220, 50

    def draw(t, dur):
        img = canvas()
        brand_bar(img, "adversarial campaign")
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([64, 150, 860, 760], 18, fill=(5, 9, 20), outline=LINE, width=3)
        for k, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse([92 + k * 30, 174, 108 + k * 30, 190], fill=c)
        shown = min(len(log_lines), int(len(log_lines) * (t / (dur * 0.75))) + 1)
        for i, line in enumerate(log_lines[:shown]):
            d.text((92, 222 + i * 58), line, font=ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 26),
                   fill=GREEN if line.startswith("$") else INK)
        n = int(len(history) * ease_io(t / (dur * 0.85)))
        d.text((x0 - 230, 160), f"{n} / {len(history)} episodes", font=F(26), fill=MUTED)
        fails = sum(not h["success"] for h in history[:n])
        d.text((x1 - 260, 160), f"{fails} failures", font=F(30), fill=RED)
        for r_i, name in enumerate(rows):
            p = params.BY_NAME[name]
            y = y0 + r_i * rh
            d.text((x0 - 14 - d.textlength(p.label, font=F(20, False)), y - 12), p.label, font=F(20, False), fill=INK)
            d.rounded_rectangle([x0, y - 9, x1, y + 9], 9, fill=CARD)
            lo = x0 + (p.nominal[0] - p.low) / (p.high - p.low) * (x1 - x0)
            hi = x0 + (p.nominal[1] - p.low) / (p.high - p.low) * (x1 - x0)
            d.rounded_rectangle([lo, y - 9, max(hi, lo + 4), y + 9], 5, fill=(22, 101, 52))
        for h in history[:n]:
            off = params.off_nominal(h["scenario"])
            if not off:
                continue
            name = h["mode"].split("|")[1] if not h["success"] else off[0]
            if name not in rows:
                continue
            p = params.BY_NAME[name]
            y = y0 + rows.index(name) * rh + ((h["scenario"]["seed"] % 100) / 100 - 0.5) * 10
            x = x0 + (h["scenario"][name] - p.low) / (p.high - p.low) * (x1 - x0)
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=GREEN if h["success"] else RED)
        if t > dur * 0.85:
            a = ease((t - dur * 0.85) / 0.6)
            for r_i, name in enumerate(rows):
                b = bounds[name]
                txt = " ".join(f"{'≤' if s == 'low' else '≥'}{fmt_value(name, b[s]['mildest_failure'])}"
                               for s in ("low", "high") if b[s]["mildest_failure"] is not None)
                if txt:
                    d.text((x1 + 16, y0 + r_i * rh - 12), txt, font=F(20),
                           fill=tuple(int(BG[c] + (RED[c] - BG[c]) * a) for c in range(3)))
        return img
    return draw


def scene_failures(clips, scene_ref):
    tw, th = 800, 450
    pos = [(140, 110), (W - 140 - tw, 110), (140, 110 + th + 30), (W - 140 - tw, 110 + th + 30)]

    def draw(t, dur):
        img = canvas()
        starts = [a for a, _, _ in scene_ref[0].local] or [0.0]
        cur = max([k for k, a in enumerate(starts) if t >= a - 0.15] or [0])
        for k, ((x, y), c) in enumerate(zip(pos, clips)):
            local = t - (starts[k] if k < len(starts) else 0) + 0.4
            frame_i = max(0.0, local) * FPS * 0.9
            tile = c.get(frame_i).resize((tw, th), Image.LANCZOS)
            if k != cur:
                tile = Image.blend(tile, Image.new("RGB", tile.size, BG), 0.55)
            else:
                glow_box(img, (x, y, x + tw, y + th), RED, 14, 160, 20)
            img.paste(tile, (x, y))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle([x, y, x + tw, y + th], 10, outline=RED if k == cur else LINE, width=4)
            title = CAUSE_TITLE.get(c.result.cause, c.result.cause)
            st = stressors(c.result.scenario, 1)
            detail = f"{st[0]['label']} {st[0]['text']} · trained {st[0]['nominal']}" if st else ""
            d.rounded_rectangle([x + 16, y + 16, x + 16 + 28 + max(d.textlength(title, font=F(30)), d.textlength(detail, font=F(20, False))), y + 76],
                                10, fill=(4, 7, 18))
            d.text((x + 30, y + 22), title, font=F(30), fill=RED if k == cur else INK)
            d.text((x + 30, y + 52), detail, font=F(20, False), fill=MUTED)
        return img
    return draw


def scene_robot_view(sweep, scenario):
    true_xy = np.array([scenario["obj_x"], scenario["obj_y"]])
    cam = scn.camera_shift(scenario)

    def draw(t, dur):
        img = canvas()
        brand_bar(img, "the policy's own camera · live CNN output")
        k = int(min(len(sweep) - 1, (t / (dur * 0.8)) * (len(sweep) - 1)))
        lv, im, est = sweep[k]
        view = Image.fromarray(im).resize((660, 660), Image.NEAREST)
        dv = ImageDraw.Draw(view)
        z = scenario["obj_size"] * 2
        tx, ty = world_to_overhead_px(true_xy, cam, z, 660)
        ex, ey = world_to_overhead_px(est, cam, z, 660)
        err = float(np.linalg.norm(est - true_xy))
        bad = err > PERCEPTION_TOL
        dv.ellipse([tx - 34, ty - 34, tx + 34, ty + 34], outline=GREEN, width=7)
        col = RED if bad else AMBER
        dv.line([ex - 28, ey - 28, ex + 28, ey + 28], fill=col, width=10)
        dv.line([ex - 28, ey + 28, ex + 28, ey - 28], fill=col, width=10)
        if bad:
            glow_box(img, (130, 140, 790, 800), RED, 10, 120, 18)
        img.paste(view, (130, 140))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([130, 140, 790, 800], 6, outline=RED if bad else LINE, width=4)
        x0 = 920
        d.text((x0, 170), "scene light", font=F(30, False), fill=MUTED)
        d.text((x0, 206), f"{lv:.2f}", font=F(110), fill=INK)
        lo, hi = params.BY_NAME["light"].nominal
        fx = lambda v: x0 + (v - 0.08) / 0.92 * (1780 - x0)
        d.rounded_rectangle([x0, 350, 1780, 372], 11, fill=CARD)
        d.rounded_rectangle([fx(lo), 350, fx(hi), 372], 11, fill=(22, 101, 52))
        d.ellipse([fx(lv) - 17, 344, fx(lv) + 17, 378], fill=INK)
        d.text((fx(lo), 384), "trained range", font=F(22, False), fill=GREEN)
        d.text((x0, 450), "perception error", font=F(30, False), fill=MUTED)
        d.text((x0, 486), f"{err * 1000:.0f} mm", font=F(110), fill=RED if bad else INK)
        d.text((x0, 620), f"grasp tolerance {PERCEPTION_TOL * 1000:.0f} mm", font=F(24, False), fill=MUTED)
        if bad:
            badge(img, (x0, 676), "GRASP WILL MISS", RED, 44)
        else:
            badge(img, (x0, 676), "ON TARGET", GREEN, 44, glow=False)
        return img
    return draw


def scene_safety(env, val):
    limits = [l["text"] for l in env["limits"]]

    def draw(t, dur):
        img = canvas()
        brand_bar(img, f"validated on {val['n_inside']} held-out scenarios")
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([64, 140, 900, 860], 20, fill=CARD)
        d.text((100, 170), "Safe operating envelope", font=F(42), fill=INK)
        for j, l in enumerate(limits):
            a = ease((t - 0.3 - j * 0.28) / 0.4)
            if a <= 0:
                continue
            y = 250 + j * 72
            d.line([(102, y + 20), (114, y + 32), (136, y + 6)], fill=GREEN, width=6)
            d.text((158 - (1 - a) * 30, y), l, font=F(36, False), fill=tuple(int(CARD[c] + (INK[c] - CARD[c]) * a) for c in range(3)))
        starts = [a for a, _, _ in safety_ref[0].local] if safety_ref else [0, dur * 0.5]
        t2 = starts[1] if len(starts) > 1 else dur * 0.5
        p = ease((t - t2) / 2.0)
        d.rounded_rectangle([960, 140, W - 64, 490], 20, fill=CARD)
        d.text((1000, 170), "pass rate INSIDE the envelope", font=F(30, False), fill=MUTED)
        d.text((1000, 220), f"{val['pass_rate_inside'] * 100 * p:.0f}%", font=F(180), fill=GREEN)
        lo, hi = val["ci95_inside"]
        d.text((1000, 432), f"95% CI {lo:.0%}–{hi:.0%} · n = {val['n_inside']}", font=F(26, False), fill=MUTED)
        d.rounded_rectangle([960, 510, W - 64, 860], 20, fill=CARD)
        d.text((1000, 540), "pass rate OUTSIDE it", font=F(30, False), fill=MUTED)
        d.text((1000, 590), f"{val['pass_rate_outside'] * 100 * p:.0f}%", font=F(180), fill=RED)
        d.text((1000, 802), f"n = {val['n_outside']} · none were seen during the search", font=F(26, False), fill=MUTED)
        return img
    safety_ref: list = []
    draw.ref = safety_ref
    return draw


def scene_results(bench):
    rows = sorted(bench.items())

    def draw(t, dur):
        img = canvas()
        brand_bar(img, "same budget · 3 seeds")
        d = ImageDraw.Draw(img)
        b240 = bench[240]
        ratio = b240["boundary"]["subtle_failures"] / b240["random"]["subtle_failures"]
        p = ease(t / 1.6)
        d.text((100, 170), f"{ratio * p:.1f}×", font=F(220), fill=RED)
        d.text((100, 420), "more subtle failures than random stress testing", font=F(40), fill=INK)
        d.text((100, 476), "(240 episodes: 27.0 vs 7.0, smallest changes that break the robot)", font=F(28, False), fill=MUTED)
        top = max(r["boundary"]["subtle_failures"] for _, r in rows)
        bx = 1180
        for i, (budget, r) in enumerate(rows):
            y = 200 + i * 150
            d.text((bx, y + 18), f"{budget}", font=F(30), fill=MUTED)
            for j, (k, col) in enumerate((("random", MUTED), ("boundary", RED))):
                v = r[k]["subtle_failures"]
                L = 520 * v / top * ease((t - 0.4 - i * 0.2) / 1.2)
                d.rounded_rectangle([bx + 90, y + j * 44, bx + 90 + max(L, 6), y + j * 44 + 32], 8, fill=col)
                d.text((bx + 100 + max(L, 6), y + j * 44 + 2), f"{v:.1f}", font=F(24), fill=col)
        d.text((bx + 90, 660), "■ random   ", font=F(24), fill=MUTED)
        d.text((bx + 250, 660), "■ Stress Twin", font=F(24), fill=RED)
        starts = [a for a, _, _ in res_ref[0].local] if res_ref else [0, dur * 0.55]
        if len(starts) > 1 and t > starts[1] - 0.2:
            a = ease((t - starts[1] + 0.2) / 0.6)
            box = [100, 730, 1820, 830]
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(layer).rounded_rectangle(box, 18, outline=NV + (int(255 * a),), width=3, fill=(10, 18, 8, int(200 * a)))
            img.paste(layer, (0, 0), layer)
            d = ImageDraw.Draw(img)
            d.text((136, 758), "Offline demo · live mode: NVIDIA Nemotron on Nebius Token Factory (add a key in .env)",
                   font=F(32), fill=tuple(int(BG[c] + ((185, 240, 90)[c] - BG[c]) * a) for c in range(3)))
        return img
    res_ref: list = []
    draw.ref = res_ref
    return draw


def scene_cta(t, dur):
    img = scene_title(10, 10)
    d = ImageDraw.Draw(img)
    a = ease((t - 0.3) / 0.8)
    url = "github.com/MohammadSharafi/stress-twin"
    d.text(((W - d.textlength(url, font=F(46))) / 2, 700), url,
           font=F(46), fill=tuple(int(BG[c] + (INK[c] - BG[c]) * a) for c in range(3)))
    x = (W - 1180) / 2
    for text, col in (("NVIDIA Nemotron", NV), ("Nebius Token Factory", BLUE), ("Serverless Jobs", BLUE),
                      ("MuJoCo", MUTED), ("Apache-2.0", MUTED)):
        if a > 0.5:
            x = chip(d, (x, 800), text, col, 24) + 16
    return img


# ---------------- audio ----------------

def synth_music(total: float) -> np.ndarray:
    n = int(total * SR)
    t = np.arange(n) / SR
    bpm = 92
    beat = 60 / bpm
    bar = 4 * beat
    prog = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]  # Am F C G (MIDI)
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    out = np.zeros((n, 2))
    span = 2 * bar
    for c_i in range(int(total / span) + 2):
        notes = prog[c_i % 4]
        s0 = c_i * span - 0.5
        i0, i1 = max(0, int(s0 * SR)), min(n, int((s0 + span + 1.0) * SR))
        if i0 >= i1:
            continue
        tt = t[i0:i1] - s0
        env = np.clip(tt / 1.2, 0, 1) * np.clip((span + 1.0 - tt) / 1.0, 0, 1)
        for ch, det in ((0, -0.004), (1, 0.004)):
            sig = np.zeros_like(tt)
            for m in notes:
                f = hz(m) * (1 + det)
                for h_i in range(1, 7):
                    sig += np.sin(2 * np.pi * f * h_i * tt + h_i * 0.3) / (h_i ** 1.6)
            out[i0:i1, ch] += 0.05 * sig * env
        bass = np.sin(2 * np.pi * hz(notes[0] - 24) * tt) * env
        out[i0:i1] += 0.10 * bass[:, None]
    # gentle arpeggio (8th notes) and soft pulse
    step = beat / 2
    for k in range(int(total / step)):
        st = k * step
        notes = prog[int(st / span) % 4]
        m = notes[k % 3] + 12 + (12 if k % 8 >= 6 else 0)
        i0, i1 = int(st * SR), min(n, int((st + 0.6) * SR))
        tt = t[i0:i1] - st
        pl = np.sin(2 * np.pi * hz(m) * tt) * np.exp(-tt / 0.18) * 0.035
        pan = 0.3 if k % 2 else 0.7
        out[i0:i1, 0] += pl * (1 - pan)
        out[i0:i1, 1] += pl * pan
    for k in range(int(total / beat)):
        st = k * beat
        i0, i1 = int(st * SR), min(n, int((st + 0.25) * SR))
        tt = t[i0:i1] - st
        kick = np.sin(2 * np.pi * (48 + 60 * np.exp(-tt / 0.03)) * tt) * np.exp(-tt / 0.09) * 0.07
        out[i0:i1] += kick[:, None]
    # simple one-pole low-pass to soften
    a = math.exp(-2 * math.pi * 3500 / SR)
    from scipy.signal import lfilter
    out = lfilter([1 - a], [1, -a], out, axis=0)
    fade = np.clip(t / 2.5, 0, 1) * np.clip((total - t) / 3.0, 0, 1)
    return out * fade[:, None]


def sfx_whoosh(dur=0.7):
    n = int(dur * SR)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    y = 0.0
    for i in range(n):
        fc = 250 + 5200 * (i / n) ** 1.5
        a = math.exp(-2 * math.pi * fc / SR)
        y = (1 - a) * noise[i] + a * y
        out[i] = y
    env = np.sin(np.pi * np.arange(n) / n) ** 2
    return out * env * 0.5


def sfx_impact(dur=0.9):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    body = np.sin(2 * np.pi * (45 + 70 * np.exp(-tt / 0.05)) * tt) * np.exp(-tt / 0.22)
    click = np.random.default_rng(2).standard_normal(n) * np.exp(-tt / 0.012) * 0.4
    return (body + click) * 0.8


def sfx_tick(dur=0.12):
    tt = np.arange(int(dur * SR)) / SR
    return np.sin(2 * np.pi * 1320 * tt) * np.exp(-tt / 0.03) * 0.35


def place(track, clip, at):
    i0 = int(at * SR)
    if i0 >= len(track):
        return
    i1 = min(len(track), i0 + len(clip))
    if clip.ndim == 1:
        track[i0:i1] += clip[: i1 - i0, None]
    else:
        track[i0:i1] += clip[: i1 - i0]


def gated_rms_db(x):
    m = x.mean(axis=1)
    win = 4800
    nb = len(m) // win
    blocks = np.sqrt(np.mean(m[: nb * win].reshape(nb, win) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(blocks)
    act = db[db > -50]
    return float(10 * np.log10(np.mean(10 ** (act / 10)))) if len(act) else -120.0


# ---------------- build ----------------

def synthesize(scenes, tmp: Path) -> dict:
    job = {"voice": os.environ.get("STRESS_TWIN_VOICE", "af_heart"), "speed": 1.0, "out_dir": str(tmp / "tts"),
           "sentences": {f"{s.name}_{i}": txt for s in scenes for i, txt in enumerate(s.sentences)}}
    (tmp / "job.json").write_text(json.dumps(job))
    py = os.environ["KOKORO_PYTHON"]
    subprocess.run([py, str(Path(__file__).parent / "tts_kokoro.py"), str(tmp / "job.json")], check=True)
    return json.loads((tmp / "tts" / "durations.json").read_text())


def main(run="runs/demo", out="../submission/stress-twin-demo-pro.mp4"):
    global BACKDROP, VIGNETTE
    BACKDROP, VIGNETTE = make_backdrop(), make_vignette()
    run_p, out_p = Path(run), Path(out).resolve()
    history = load_history(run_p)
    val = json.loads((run_p / "validation.json").read_text())
    env = prim_envelope(history)
    cl = {c["mode"]: c for c in clusters(history)}
    bench = {b: json.loads(Path(f"results/benchmark_{b}.json").read_text())["mean"] for b in (120, 240, 480)}
    camp = json.loads((run_p / "campaign.json").read_text())
    log_lines = ["$ stress-twin run --budget 240 --rounds 4", *[l.split(" [")[0] + " ·" + l.split("|")[1]
                                                              for l in (run_p / "campaign.log").read_text().splitlines()],
                 "$ stress-twin validate --n 150",
                 f"inside {val['pass_rate_inside']:.1%} · outside {val['pass_rate_outside']:.0%}", "ENVELOPE_OK"]

    print("recording footage ...")
    nominal_s = params.sample_nominal(np.random.default_rng(3))
    c_ok = record(nominal_s, orbit(nominal_s))
    dim_s = cl["perception_miss|light"]["representative"]["scenario"]
    c_fail = record(dim_s, orbit(dim_s, az0=150, rate=6))
    grid = [record(cl[m]["representative"]["scenario"], "side", size=(540, 960))
            for m in ("slip|obj_mass", "disturbance|disturbance", "perception_miss|hue_shift", "collision|obj_size")]
    sweep = mg.perception_sweep(nominal_s, np.concatenate([np.full(4, 1.0), np.linspace(1.0, 0.1, 60), np.full(6, 0.1)]))
    st_l = stressors(dim_s, 1)[0]

    refs = {k: [] for k in ("how", "failures")}
    problem_draw = scene_problem(c_ok.get(len(c_ok) // 2))
    safety_draw = scene_safety(env, val)
    results_draw = scene_results(bench)
    scenes = [
        Scene("cold_ok", scene_cold(c_ok, True, "trained conditions"), module=True, pad=(0.4, 1.0)),
        Scene("cold_fail", scene_cold(c_fail, False, f"{st_l['label']} {st_l['text']} · trained {st_l['nominal']}"),
              module=True, sfx=[("end-0.9", "impact")], pad=(0.3, 1.3)),
        Scene("title", scene_title, min_dur=4.6, sfx=[(0.0, "whoosh")]),
        Scene("problem", problem_draw, sfx=[(0.0, "whoosh")]),
        Scene("how", scene_how(refs["how"]), module=True, sfx=[(0.0, "whoosh")], pad=(0.5, 1.2)),
        Scene("search", scene_search(history, log_lines), module=True, min_dur=12, sfx=[(0.0, "whoosh")], pad=(0.5, 1.4)),
        Scene("failures", scene_failures(grid, refs["failures"]), module=True, sfx=[(0.0, "whoosh")], pad=(0.4, 1.0)),
        Scene("robot_view", scene_robot_view(sweep, nominal_s), module=True, min_dur=9, sfx=[(0.0, "whoosh")], pad=(0.5, 1.5)),
        Scene("safety", safety_draw, module=True, sfx=[(0.0, "whoosh")], pad=(0.5, 1.6)),
        Scene("results", results_draw, sfx=[(0.0, "whoosh")], pad=(0.5, 1.4)),
        Scene("cta", scene_cta, min_dur=6.5, sfx=[(0.0, "whoosh")], pad=(0.6, 2.5)),
    ]
    refs["how"].append(scenes[4])
    refs["failures"].append(scenes[6])
    safety_draw.ref.append(scenes[8])
    results_draw.ref.append(scenes[9])

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("synthesizing narration ...")
        durs = synthesize(scenes, tmp)
        gap = 0.35
        t_cursor = 0.0
        sentences_abs = []
        for s in scenes:
            lead, tail = s.pad
            local_t = lead
            s.local = []
            for i, txt in enumerate(s.sentences):
                d_s = durs[f"{s.name}_{i}"]
                s.local.append((local_t, local_t + d_s, txt))
                local_t += d_s + gap
            s.dur = max(s.min_dur, local_t - gap + tail)
            s.start = t_cursor
            t_cursor += s.dur - XFADE
            for i, (a, b, txt) in enumerate(s.local):
                sentences_abs.append((s.start + a, s.start + b, txt, f"{s.name}_{i}"))
        if hasattr(problem_draw, "set_second") and len(scenes[3].local) > 1:
            problem_draw.set_second(scenes[3].local[1][0])
        total = scenes[-1].start + scenes[-1].dur
        print(f"timeline {total:.1f}s")

        # ---- video ----
        silent = tmp / "video.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(silent)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        nframes = int(round(total * FPS))
        black = Image.new("RGB", (W, H), (0, 0, 0))
        for f in range(nframes):
            T = f / FPS
            act = [s for s in scenes if s.start <= T < s.start + s.dur]
            if not act:
                act = [scenes[-1]]
            img = act[0].draw(T - act[0].start, act[0].dur)
            if len(act) > 1:
                nxt = act[1].draw(T - act[1].start, act[1].dur)
                img = Image.blend(img, nxt, ease_io((T - act[1].start) / XFADE))
            img.paste((0, 0, 0), (0, 0, W, H), VIGNETTE)
            cur = next(((a, b, txt) for a, b, txt, _ in sentences_abs if a - 0.05 <= T < b + 0.35), None)
            if cur:
                a, b, txt = cur
                caption(img, txt, min(1.0, (T - a + 0.05) / 0.2, (b + 0.35 - T) / 0.2))
            fade = min(1.0, T / 0.6, (total - T) / 1.0)
            if fade < 1:
                img = Image.blend(black, img, max(fade, 0))
            proc.stdin.write(img.tobytes())
            if f % 300 == 0:
                print(f"  frame {f}/{nframes}")
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg video encode failed")

        # ---- audio ----
        n = int(total * SR) + SR
        voice = np.zeros((n, 2))
        for a, b, txt, sid in sentences_abs:
            x, sr = sf.read(tmp / "tts" / f"{sid}.wav")
            x = resample_poly(x, SR, sr) if sr != SR else x
            place(voice, x, a)
        voice *= 0.72 / (np.max(np.abs(voice)) + 1e-9)
        music = synth_music(n / SR)
        env_v = np.abs(voice.mean(axis=1))
        from scipy.signal import lfilter
        att = math.exp(-1 / (0.05 * SR))
        env_s = lfilter([1 - att], [1, -att], env_v)
        env_s = env_s / (np.max(env_s) + 1e-9)
        duck = 1 - 0.55 * np.clip(env_s * 3, 0, 1)
        music *= duck[:, None]
        target = gated_rms_db(voice) - 15.0
        music *= 10 ** ((target - gated_rms_db(music)) / 20)
        fx = np.zeros((n, 2))
        for s in scenes:
            for when, kind in s.sfx:
                at = s.start + (s.dur + float(when[3:]) if isinstance(when, str) else when)
                place(fx, {"whoosh": sfx_whoosh, "impact": sfx_impact, "tick": sfx_tick}[kind](), max(0.0, at - (0.25 if kind == "whoosh" else 0)))
        fx *= 10 ** ((gated_rms_db(voice) - 14 - gated_rms_db(fx)) / 20) if np.any(fx) else 1
        mix = voice + music + fx
        mix = np.tanh(mix * 1.1) / np.tanh(1.1)
        mix *= 0.89 / (np.max(np.abs(mix)) + 1e-9)
        mix_path = out_p.with_name(out_p.stem + "-mix.wav")
        sf.write(mix_path, mix.astype(np.float32), SR, subtype="PCM_16")
        sf.write(out_p.with_name(out_p.stem + "-mix-voice.wav"), voice.astype(np.float32), SR, subtype="PCM_16")
        sf.write(out_p.with_name(out_p.stem + "-mix-music.wav"), music.astype(np.float32), SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(mix_path), "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out_p)], check=True)

    log = [{"scene": s.name, "start": round(s.start, 2), "seconds": round(s.dur, 2), "module_footage": s.module}
           for s in scenes]
    module = sum(s.dur - XFADE for s in scenes if s.module)
    summary = {"video": str(out_p), "total_seconds": round(total, 1), "module_footage_seconds": round(module, 1),
               "voice": "Kokoro af_heart (Apache-2.0)", "music": "synthesized in make_pro_video.py", "scenes": log,
               "sentences": [{"start": round(a, 2), "end": round(b, 2), "text": t} for a, b, t, _ in sentences_abs]}
    out_p.with_name(out_p.stem + "-scenes.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_p}: {total:.1f}s, {module:.1f}s of modules in action")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check-claims":
        sys.exit(0 if check_claims(Path(sys.argv[2] if len(sys.argv) > 2 else "runs/demo")) else 1)
    main(*sys.argv[1:])
