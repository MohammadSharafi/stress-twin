"""Animated 3:2 gallery GIFs for Devpost (each <= 5 MB), built from real Stress Twin output.

    python scripts/make_gifs.py runs/demo ../submission/gallery-gif
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from stress_twin import params, scene
from stress_twin.analysis import clusters, factor_boundaries, fmt_value, load_history, prim_envelope, stressors
from stress_twin.diagnose import PERCEPTION_TOL
from stress_twin.episode import FRAME_EVERY, run_episode
from stress_twin.perception import IMG, Perception, add_noise, render_overhead, world_to_overhead_px
from stress_twin.report import CAUSE_TITLE
from stress_twin.strategies import STRESSABLE

W, H = 900, 600
BG = (8, 12, 28)
INK = (236, 241, 250)
MUTED = (140, 155, 185)
RED = (255, 72, 72)
GREEN = (46, 220, 120)
BLUE = (96, 165, 250)
AMBER = (251, 191, 36)
CARD = (18, 27, 52)
MAX_BYTES = 5 * 1024 * 1024


def font(size, bold=True):
    return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
                              else "/System/Library/Fonts/Supplemental/Arial.ttf", size)


def backdrop() -> Image.Image:
    """Navy gradient with a faint engineering grid and a red glow, shared by every GIF."""
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]
    base = np.stack([BG[c] + 18 * (1 - y) + 0 * x for c in range(3)], axis=2)
    glow = np.exp(-(((x - 0.85) ** 2) / 0.05 + ((y - 0.9) ** 2) / 0.08))[..., None] * np.array([70, 10, 20])
    img = Image.fromarray(np.clip(base + glow, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for gx in range(0, W, 30):
        d.line([(gx, 0), (gx, H)], fill=(20, 28, 54))
    for gy in range(0, H, 30):
        d.line([(0, gy), (W, gy)], fill=(20, 28, 54))
    return img


BACK = None


def frame() -> Image.Image:
    global BACK
    if BACK is None:
        BACK = backdrop()
    return BACK.copy()


def brand(img, subtitle=""):
    d = ImageDraw.Draw(img)
    d.text((24, 18), "STRESS", font=font(26), fill=INK)
    d.text((24 + d.textlength("STRESS ", font=font(26)), 18), "TWIN", font=font(26), fill=RED)
    if subtitle:
        d.text((W - 24 - d.textlength(subtitle, font=font(17, False)), 25), subtitle, font=font(17, False), fill=MUTED)


def headline(img, text, y=58, size=34, color=INK):
    d = ImageDraw.Draw(img)
    d.text((24, y), text, font=font(size), fill=color)


def badge(img, xy, text, color, size=24, glow=True):
    d = ImageDraw.Draw(img)
    f = font(size)
    w = d.textlength(text, font=f)
    x, y = xy
    box = [x, y, x + w + 28, y + size + 18]
    if glow:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).rounded_rectangle([box[0] - 6, box[1] - 6, box[2] + 6, box[3] + 6], 14, fill=color + (120,))
        img.paste(layer.filter(ImageFilter.GaussianBlur(8)), (0, 0), layer.filter(ImageFilter.GaussianBlur(8)))
        d = ImageDraw.Draw(img)
    d.rounded_rectangle(box, 10, fill=color)
    d.text((x + 14, y + 8), text, font=f, fill=(255, 255, 255))
    return box


def framed(img, tile: Image.Image, xy, border=(40, 56, 96), width=3):
    x, y = xy
    ImageDraw.Draw(img).rounded_rectangle([x - width, y - width, x + tile.width + width, y + tile.height + width], 8,
                                          outline=border, width=width)
    img.paste(tile, (x, y))


def encode_gif(frames: list[Image.Image], path: Path, fps: int = 12, colors: int = 160) -> int:
    """Two-pass palette encode with ffmpeg; shrink colours, then size, until under 5 MB.
    The final state is held at the start too, so a static preview never shows a half-built frame."""
    frames = [frames[-1]] * int(fps * 0.9) + frames + [frames[-1]] * int(fps * 1.2)
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            f.save(Path(td) / f"f{i:04d}.png")
        scale = 1.0
        while True:
            vf = f"fps={fps}" + (f",scale={int(W * scale)}:-1:flags=lanczos" if scale < 1 else "")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{td}/f%04d.png",
                            "-vf", f"{vf},palettegen=max_colors={colors}:stats_mode=full", f"{td}/pal.png"], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{td}/f%04d.png",
                            "-i", f"{td}/pal.png", "-lavfi",
                            f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                            "-loop", "0", str(path)], check=True)
            size = path.stat().st_size
            if size <= MAX_BYTES * 0.95:
                return size
            if colors > 96:
                colors -= 32
            else:
                scale -= 0.1


# ---------------- data helpers ----------------

def replay(scenario, size=(405, 720)):
    r = run_episode(scenario, record=True, frame_size=size)
    return r


def clip_frames(r, n_out: int, crop_w: int, crop_h: int):
    """Resample a replay to n_out frames, centre-cropped to the given aspect."""
    idx = np.linspace(0, len(r.frames) - 1, n_out).astype(int)
    out = []
    for i in idx:
        im = Image.fromarray(r.frames[i])
        sw, sh = im.size
        target = crop_w / crop_h
        if sw / sh > target:
            nw = int(sh * target)
            im = im.crop(((sw - nw) // 2, 0, (sw + nw) // 2, sh))
        else:
            nh = int(sw / target)
            im = im.crop((0, (sh - nh) // 3, sw, (sh - nh) // 3 + nh))
        out.append(im.resize((crop_w, crop_h), Image.LANCZOS))
    return out


def verdict(r):
    if r.success:
        return "SUCCESS", GREEN
    return "FAILURE · " + CAUSE_TITLE.get(r.cause, r.cause), RED


def cond_text(r):
    st = stressors(r.scenario, 1)
    return f"{st[0]['label']} {st[0]['text']}  (trained {st[0]['nominal']})" if st else "trained conditions"


# ---------------- the six GIFs ----------------

def gif_cover(ctx, out):
    r = ctx["hero"]
    tiles = clip_frames(r, 36, 520, 330)
    s = ctx["summary"]
    val = ctx["val"]
    frames = []
    for i in range(54):
        img = frame()
        d = ImageDraw.Draw(img)
        d.text((24, 22), "STRESS", font=font(58), fill=INK)
        d.text((24 + d.textlength("STRESS ", font=font(58)), 22), "TWIN", font=font(58), fill=RED)
        d.text((26, 92), "A crash-test lab for robot brains", font=font(24, False), fill=MUTED)
        k = min(i, len(tiles) - 1)
        framed(img, tiles[k], (24, 150), border=RED if i >= len(tiles) - 4 else (40, 56, 96))
        if i >= len(tiles) - 4:
            if (i // 3) % 2 == 0 or i > len(tiles) + 6:
                badge(img, (40, 166), verdict(r)[0], RED, 22)
        # live counters, eased in
        ease = 1.0
        stats = [(f"{int(s['episodes'] * ease)}", "simulated episodes", INK),
                 (f"{int(s['failures'] * ease)}", "failures found", RED),
                 (f"{int(s['modes'] * ease)}", "failure modes", AMBER),
                 (f"{int(round(val['pass_rate_inside'] * 100 * ease))}%", "safe inside envelope", GREEN)]
        for j, (num, lab, col) in enumerate(stats):
            y = 150 + j * 84
            d.rounded_rectangle([572, y, W - 24, y + 72], 12, fill=CARD)
            d.text((590, y + 8), num, font=font(38), fill=col)
            d.text((590, y + 50), lab, font=font(16, False), fill=MUTED)
        chips = ["NVIDIA Nemotron", "Nebius Token Factory", "Serverless Jobs", "MuJoCo"]
        x = 24
        for c in chips:
            w = d.textlength(c, font=font(16))
            d.rounded_rectangle([x, 506, x + w + 24, 538], 16, outline=(118, 185, 0) if "NVIDIA" in c else BLUE, width=2)
            d.text((x + 12, 512), c, font=font(16), fill=(185, 240, 90) if "NVIDIA" in c else (160, 190, 255))
            x += w + 36
        d.text((24, 556), "Finds the mildest conditions that break a robot policy, then proves where it is safe.",
               font=font(18, False), fill=INK)
        frames.append(img)
    return encode_gif(frames, out)


def gif_split(ctx, out):
    a, b = ctx["nominal"], ctx["dim"]
    ta, tb = clip_frames(a, 40, 420, 300), clip_frames(b, 40, 420, 300)
    frames = []
    for i in range(52):
        img = frame()
        brand(img, "deterministic replays")
        headline(img, "Same robot. One change.", 56, 38)
        k = min(i, 39)
        framed(img, ta[k], (24, 128))
        framed(img, tb[k], (W - 24 - 420, 128))
        d = ImageDraw.Draw(img)
        d.text((24, 440), "Trained conditions", font=font(22), fill=INK)
        d.text((W - 444, 440), cond_text(b), font=font(19), fill=AMBER)
        if i >= 36:
            badge(img, (36, 140), verdict(a)[0], GREEN, 26)
            badge(img, (W - 432, 140), "FAILURE", RED, 26)
            d = ImageDraw.Draw(img)
            d.text((24, 480), "200 / 200 in trained conditions", font=font(20, False), fill=GREEN)
            d.text((W - 444, 480), "Dim the light: it reaches for empty air", font=font(19, False), fill=RED)
        d.text((24, 548), "Stress Twin finds these breaking points automatically, before deployment.",
               font=font(20), fill=INK)
        frames.append(img)
    return encode_gif(frames, out)


def gif_grid(ctx, out):
    reps = ctx["grid"]
    tw, th = 420, 180
    tiles = [clip_frames(r, 40, tw, th) for r in reps]
    pos = [(24, 108), (W - 24 - tw, 108), (24, 108 + th + 46), (W - 24 - tw, 108 + th + 46)]
    frames = []
    for i in range(52):
        img = frame()
        brand(img, "found automatically · replayed from seed")
        headline(img, "4 ways it breaks", 50, 36)
        d = ImageDraw.Draw(img)
        for (x, y), t, r in zip(pos, tiles, reps):
            k = min(i, len(t) - 1)
            framed(img, t[k], (x, y), border=RED if i >= 36 else (40, 56, 96))
            d = ImageDraw.Draw(img)
            title = CAUSE_TITLE.get(r.cause, r.cause)
            d.text((x, y + th + 8), title, font=font(18), fill=RED)
            st = stressors(r.scenario, 1)
            compact = {"bump during approach": "bump", "object hue shift": "hue shift", "object mass": "mass",
                       "object size": "size", "scene light": "light"}
            detail = f"{compact.get(st[0]['label'], st[0]['label'])} {st[0]['text']}" if st else ""
            room = tw - d.textlength(title + "   ", font=font(18))
            while detail and d.textlength(detail, font=font(15, False)) > room:
                detail = detail[:-2]
            d.text((x + d.textlength(title + "   ", font=font(18)), y + th + 10), detail, font=font(15, False), fill=MUTED)
        d.text((24, 562), "Every cause is diagnosed from telemetry, not guessed.", font=font(19), fill=INK)
        frames.append(img)
    return encode_gif(frames, out)


def perception_sweep(scenario, lights):
    perc = Perception.shared()
    out = []
    for lv in lights:
        s = dict(scenario, light=float(lv))
        m = mujoco.MjModel.from_xml_string(scene.build_xml(s))
        dd = mujoco.MjData(m)
        dd.qpos[[m.joint(n).qposadr[0] for n in ("gx", "gy", "gz")]] = scene.HAND_HOME
        mujoco.mj_forward(m, dd)
        rnd = mujoco.Renderer(m, IMG, IMG)
        img = add_noise(render_overhead(m, dd, rnd), s["image_noise"], s["seed"])
        rnd.close()
        est = perc.predict(img)
        out.append((lv, img, est))
    return out


def gif_robot_view(ctx, out):
    s = ctx["sweep_scenario"]
    lights = np.concatenate([np.full(6, 1.0), np.linspace(1.0, 0.1, 44), np.full(10, 0.1)])
    sweep = perception_sweep(s, lights)
    true_xy = np.array([s["obj_x"], s["obj_y"]])
    cam = scene.camera_shift(s)
    frames = []
    for lv, img, est in sweep:
        canvas = frame()
        brand(canvas, "the policy's own camera, real CNN output")
        headline(canvas, "What the robot sees as the lights dim", 52, 32)
        view = Image.fromarray(img).resize((400, 400), Image.NEAREST)
        dv = ImageDraw.Draw(view)
        z = s["obj_size"] * 2
        tx, ty = world_to_overhead_px(true_xy, cam, z, 400)
        ex, ey = world_to_overhead_px(est, cam, z, 400)
        err = float(np.linalg.norm(est - true_xy))
        bad = err > PERCEPTION_TOL
        dv.ellipse([tx - 20, ty - 20, tx + 20, ty + 20], outline=GREEN, width=4)
        col = RED if bad else AMBER
        dv.line([ex - 16, ey - 16, ex + 16, ey + 16], fill=col, width=6)
        dv.line([ex - 16, ey + 16, ex + 16, ey - 16], fill=col, width=6)
        framed(canvas, view, (24, 120), border=RED if bad else (40, 56, 96))
        d = ImageDraw.Draw(canvas)
        x0 = 460
        d.text((x0, 124), "scene light", font=font(18, False), fill=MUTED)
        d.text((x0, 148), f"{lv:.2f}", font=font(52), fill=INK)
        d.rounded_rectangle([x0, 214, W - 24, 230], 8, fill=CARD)
        lo, hi = params.BY_NAME["light"].nominal
        fx = lambda v: x0 + (v - 0.08) / 0.92 * (W - 24 - x0)
        d.rounded_rectangle([fx(lo), 214, fx(hi), 230], 8, fill=(22, 101, 52))
        d.ellipse([fx(lv) - 11, 211, fx(lv) + 11, 233], fill=INK)
        d.text((fx(lo), 236), "trained range", font=font(14, False), fill=GREEN)
        d.text((x0, 284), "perception error", font=font(18, False), fill=MUTED)
        d.text((x0, 308), f"{err * 1000:.0f} mm", font=font(52), fill=RED if bad else INK)
        d.text((x0, 374), f"grasp tolerance {PERCEPTION_TOL * 1000:.0f} mm", font=font(16, False), fill=MUTED)
        if bad:
            badge(canvas, (x0, 412), "GRASP WILL MISS", RED, 24)
        else:
            badge(canvas, (x0, 412), "ON TARGET", GREEN, 24, glow=False)
        d = ImageDraw.Draw(canvas)
        d.text((x0, 480), "o  true cube", font=font(18), fill=GREEN)
        d.text((x0, 506), "x  where the CNN thinks it is", font=font(18), fill=col)
        d.text((24, 548), "Trained only on bright scenes, the vision model drifts as light falls.", font=font(18), fill=INK)
        frames.append(canvas)
    return encode_gif(frames, out, fps=12)


def gif_map(ctx, out):
    history, env = ctx["history"], ctx["env"]
    rows = [n for n in STRESSABLE if n != "distractor_hue_gap"]
    bounds = factor_boundaries(history)
    x0, x1, y0, rh = 200, W - 150, 112, 34
    order = list(range(len(history)))
    steps = 44
    frames = []
    for i in range(steps + 10):
        n = int(len(order) * min(1.0, i / steps))
        img = frame()
        brand(img, f"{n} of {len(history)} simulated episodes")
        headline(img, "Mapping where it breaks", 52, 34)
        d = ImageDraw.Draw(img)
        fails = 0
        for r_i, name in enumerate(rows):
            p = params.BY_NAME[name]
            y = y0 + r_i * rh
            X = lambda v: x0 + (v - p.low) / (p.high - p.low) * (x1 - x0)
            d.text((x0 - 12 - d.textlength(p.label, font=font(15, False)), y - 9), p.label, font=font(15, False), fill=INK)
            d.rounded_rectangle([x0, y - 7, x1, y + 7], 7, fill=CARD)
            d.rounded_rectangle([X(p.nominal[0]), y - 7, max(X(p.nominal[1]), X(p.nominal[0]) + 3), y + 7], 4, fill=(22, 101, 52))
        for h in history[:n]:
            off = params.off_nominal(h["scenario"])
            if not off:
                continue
            fails += not h["success"]
            name = h["mode"].split("|")[1] if not h["success"] else off[0]
            if name not in rows:
                continue
            p = params.BY_NAME[name]
            y = y0 + rows.index(name) * rh + ((h["scenario"]["seed"] % 100) / 100 - 0.5) * 8
            x = x0 + (h["scenario"][name] - p.low) / (p.high - p.low) * (x1 - x0)
            col = GREEN if h["success"] else RED
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
        if i >= steps:
            for r_i, name in enumerate(rows):
                b = bounds[name]
                txt = " · ".join(f"{'≤' if side == 'low' else '≥'} {fmt_value(name, b[side]['mildest_failure'])}"
                                 for side in ("low", "high") if b[side]["mildest_failure"] is not None)
                d.text((x1 + 12, y0 + r_i * rh - 9), txt or "robust", font=font(15), fill=RED if txt else GREEN)
        d.text((24, 552), f"failures found: {sum(not h['success'] for h in history[:n])}", font=font(22), fill=RED)
        d.rounded_rectangle([330, 560, 360, 572], 6, fill=(22, 101, 52))
        d.text((368, 556), "trained range", font=font(16, False), fill=MUTED)
        d.ellipse([500, 560, 512, 572], fill=GREEN)
        d.text((518, 556), "passed", font=font(16, False), fill=MUTED)
        d.ellipse([594, 560, 606, 572], fill=RED)
        d.text((612, 556), "failed", font=font(16, False), fill=MUTED)
        frames.append(img)
    return encode_gif(frames, out)


def gif_safety(ctx, out):
    env, val = ctx["env"], ctx["val"]
    limits = [l["text"] for l in env["limits"]]
    frames = []
    for i in range(56):
        img = frame()
        brand(img, f"validated on {val['n_inside']} held-out scenarios")
        headline(img, "A safety case you can check", 52, 34)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([24, 112, 470, 540], 14, fill=CARD)
        d.text((44, 128), "Safe operating envelope", font=font(22), fill=INK)
        shown = min(len(limits), i // 3)
        for j, l in enumerate(limits[:shown]):
            cy = 186 + j * 44
            d.line([(46, cy), (53, cy + 7), (66, cy - 8)], fill=GREEN, width=4)
            d.text((76, 174 + j * 44), l, font=font(20, False), fill=INK)
        p = max(0.0, min(1.0, (i - 12) / 26))
        ease = 1 - (1 - p) ** 3
        d.rounded_rectangle([494, 112, W - 24, 320], 14, fill=CARD)
        d.text((514, 128), "pass rate INSIDE the envelope", font=font(18, False), fill=MUTED)
        d.text((514, 152), f"{val['pass_rate_inside'] * 100 * ease:.0f}%", font=font(110), fill=GREEN)
        lo, hi = val["ci95_inside"]
        d.text((514, 284), f"95% CI {lo:.0%}–{hi:.0%}  ·  n = {val['n_inside']}", font=font(15, False), fill=MUTED)
        d.rounded_rectangle([494, 336, W - 24, 540], 14, fill=CARD)
        d.text((514, 352), "pass rate OUTSIDE it", font=font(18, False), fill=MUTED)
        d.text((514, 376), f"{val['pass_rate_outside'] * 100 * ease:.0f}%", font=font(110), fill=RED)
        d.text((514, 508), f"n = {val['n_outside']}  ·  none of these were in the search", font=font(15, False), fill=MUTED)
        d.text((24, 556), "Found by PRIM box peeling, then tested on fresh scenarios.", font=font(18), fill=INK)
        frames.append(img)
    return encode_gif(frames, out)


def main(run="runs/demo", outdir="../submission/gallery-gif"):
    run_p, out = Path(run), Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    history = load_history(run_p)
    cl = {c["mode"]: c for c in clusters(history)}
    camp = json.loads((run_p / "campaign.json").read_text())
    ctx = {
        "history": history, "env": prim_envelope(history), "val": json.loads((run_p / "validation.json").read_text()),
        "summary": camp["summary"],
        "nominal": replay(params.sample_nominal(np.random.default_rng(3))),
        "dim": replay(cl["perception_miss|light"]["representative"]["scenario"]),
        "hero": replay(cl["perception_miss|hue_shift"]["representative"]["scenario"]),
        "sweep_scenario": params.sample_nominal(np.random.default_rng(3)),
    }
    grid_modes = ["slip|obj_mass", "disturbance|disturbance", "perception_miss|hue_shift", "collision|obj_size"]
    ctx["grid"] = [replay(cl[m]["representative"]["scenario"]) for m in grid_modes if m in cl]
    jobs = [("01-cover.gif", gif_cover), ("02-same-robot-one-change.gif", gif_split), ("03-four-ways-it-breaks.gif", gif_grid),
            ("04-what-the-robot-sees.gif", gif_robot_view), ("05-mapping-where-it-breaks.gif", gif_map),
            ("06-safety-case.gif", gif_safety)]
    for name, fn in jobs:
        size = fn(ctx, out / name)
        print(f"{name}: {size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
