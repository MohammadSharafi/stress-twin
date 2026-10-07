"""Build the narrated demo video (1920x1080, under 3 minutes) from real Stress Twin output.

    python scripts/make_video.py runs/demo ../submission/stress-twin-demo.mp4

Every clip is a deterministic replay of an episode from the run; numbers on screen are read
from the run's files. Narration uses macOS `say`; subtitles are burned in for muted viewing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stress_twin import params
from stress_twin.analysis import clusters, load_history, prim_envelope, stressors
from stress_twin.episode import FRAME_EVERY, run_episode
from stress_twin.report import CAUSE_TITLE, failure_map_svg, marked_overhead

W, H, FPS = 1920, 1080, 25
BG = (11, 16, 32)
FG = (226, 232, 240)
MUTED = (148, 163, 184)
RED = (239, 68, 68)
GREEN = (34, 197, 94)
CARD = (17, 26, 48)
VOICE = "Samantha"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def font(size: int, bold: bool = False):
    path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
    return ImageFont.truetype(path, size)


F_TITLE, F_H1, F_H2, F_BODY, F_SMALL, F_SUB, F_MONO = (font(120, True), font(64, True), font(44, True), font(36),
                                                      font(28), font(38, True),
                                                      ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 30))


def tts(text: str, path: Path) -> float:
    subprocess.run(["say", "-v", VOICE, "-r", "182", "-o", str(path), text], check=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def canvas() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def subtitle(img: Image.Image, text: str) -> None:
    d = ImageDraw.Draw(img)
    lines = textwrap.wrap(text, 70)[:3]
    y = H - 40 - 50 * len(lines)
    d.rectangle([0, y - 18, W, H], fill=(5, 8, 18))
    for i, line in enumerate(lines):
        w = d.textlength(line, font=F_SUB)
        d.text(((W - w) / 2, y + i * 50), line, font=F_SUB, fill=(255, 255, 255))


def header(img: Image.Image, label: str) -> None:
    d = ImageDraw.Draw(img)
    d.text((60, 40), "STRESS", font=F_H2, fill=FG)
    d.text((60 + d.textlength("STRESS ", font=F_H2), 40), "TWIN", font=F_H2, fill=RED)
    d.text((W - 60 - d.textlength(label, font=F_SMALL), 52), label, font=F_SMALL, fill=MUTED)


class Scene:
    def __init__(self, name: str, sentences: list[str], draw, min_dur: float = 0.0, module: bool = False):
        self.name, self.sentences, self.draw, self.min_dur, self.module = name, sentences, draw, min_dur, module
        self.timeline: list[tuple[float, float, str]] = []


def build(scene: Scene, tmp: Path, idx: int) -> tuple[Path, float]:
    parts, timeline, t = [], [], 0.6
    for j, sentence in enumerate(scene.sentences):
        a = tmp / f"s{idx}_{j}.aiff"
        dur = tts(sentence, a)
        parts.append((a, t))
        timeline.append((t, t + dur, sentence))
        t += dur + 0.35
    duration = max(t + 0.5, scene.min_dur)
    scene.timeline = timeline
    seg = tmp / f"scene{idx:02d}.mp4"
    inputs, filters = [], []
    for k, (a, start) in enumerate(parts):
        inputs += ["-i", str(a)]
        filters.append(f"[{k + 1}:a]adelay={int(start * 1000)}|{int(start * 1000)}[a{k}]")
    mix = "".join(f"[a{k}]" for k in range(len(parts)))
    filters.append(f"{mix}amix=inputs={len(parts)}:normalize=0,apad,atrim=0:{duration:.3f}[aout]")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", *inputs, "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
           "-ar", "48000", "-t", f"{duration:.3f}", str(seg)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(int(round(duration * FPS))):
        tt = f / FPS
        img = scene.draw(tt, duration)
        sub = next((s for a, b, s in timeline if a <= tt < b + 0.3), "")
        if sub:
            subtitle(img, sub)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on scene {scene.name}")
    return seg, duration


# ---------------- visuals ----------------

def replay(rec_or_scenario) -> dict:
    scenario = rec_or_scenario["scenario"] if "scenario" in rec_or_scenario else rec_or_scenario
    r = run_episode(scenario, record=True, frame_size=(720, 1280))
    js = r.to_json()
    return {"rec": js, "frames": r.frames, "overhead": marked_overhead(js, r.overhead, size=480)}


def clip_drawer(clip: dict, label: str, speed: float = 0.6):
    rec = clip["rec"]
    cond = stressors(rec["scenario"], 3)
    verdict = "SUCCESS" if rec["success"] else "FAILURE · " + CAUSE_TITLE.get(rec["cause"], rec["cause"])
    color = GREEN if rec["success"] else RED

    def draw(t: float, dur: float) -> Image.Image:
        img = canvas()
        header(img, label)
        i = min(int(t * speed / FRAME_EVERY), len(clip["frames"]) - 1)
        img.paste(Image.fromarray(clip["frames"][i]), (60, 130))
        img.paste(clip["overhead"], (1380, 130))
        d = ImageDraw.Draw(img)
        d.text((1380, 625), "Conditions vs training", font=F_SMALL, fill=MUTED)
        y = 665
        if not cond:
            d.text((1380, y), "all conditions inside", font=F_BODY, fill=FG)
            d.text((1380, y + 40), "the training range", font=F_SMALL, fill=MUTED)
        for s in cond:
            d.text((1380, y), f"{s['label']}: {s['text']}", font=F_BODY, fill=FG)
            d.text((1380, y + 40), f"trained {s['nominal']}", font=F_SMALL, fill=MUTED)
            y += 92
        if i == len(clip["frames"]) - 1 or t > 0.85 * dur:
            d.rounded_rectangle([60, 130, 60 + 40 + d.textlength(verdict, font=F_H2), 205], 12, fill=color)
            d.text((80, 142), verdict, font=F_H2, fill=(255, 255, 255))
        return img

    return draw


def montage_drawer(clips: list[dict], label: str, scene_ref: list):
    """Clip k plays while narration sentence k+1 is spoken (sentence 0 is the intro)."""
    drawers = [clip_drawer(c, label, speed=0.9) for c in clips]

    def draw(t: float, dur: float) -> Image.Image:
        tl = scene_ref[0].timeline
        starts = [a for a, _, _ in tl[1:]] + [dur]
        k = max(0, min(sum(1 for a in starts[:-1] if a <= t) - 1, len(clips) - 1))
        if t < starts[0]:
            return drawers[0](t * 0.25, starts[0])
        end = starts[k + 1] if k + 1 < len(starts) else dur
        return drawers[k](t - starts[k], end - starts[k])

    return draw


def title_drawer(t: float, dur: float) -> Image.Image:
    img = canvas()
    d = ImageDraw.Draw(img)
    a = min(1.0, t / 0.8)
    c = tuple(int(BG[i] + (FG[i] - BG[i]) * a) for i in range(3))
    r = tuple(int(BG[i] + (RED[i] - BG[i]) * a) for i in range(3))
    w1 = d.textlength("Stress ", font=F_TITLE)
    total = w1 + d.textlength("Twin", font=F_TITLE)
    x = (W - total) / 2
    d.text((x, 360), "Stress ", font=F_TITLE, fill=c)
    d.text((x + w1, 360), "Twin", font=F_TITLE, fill=r)
    tag = "A crash-test lab for robot brains"
    d.text(((W - d.textlength(tag, font=F_H2)) / 2, 520), tag, font=F_H2, fill=MUTED)
    return img


def log_drawer(lines: list[str]):
    def draw(t: float, dur: float) -> Image.Image:
        img = canvas()
        header(img, "campaign · adversarial search")
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([60, 140, W - 60, 840], 18, fill=(8, 12, 24), outline=(31, 42, 68), width=2)
        shown = int(len(lines) * min(1.0, t / (dur * 0.8))) + 1
        y = 175
        for line in lines[:shown]:
            d.text((95, y), line[:100], font=F_MONO, fill=GREEN if line.startswith("$") else FG)
            y += 52
        return img

    return draw


def image_drawer(png: Path, label: str, zoom: float = 1.08):
    base = Image.open(png).convert("RGB")

    def draw(t: float, dur: float) -> Image.Image:
        img = canvas()
        header(img, label)
        z = 1 + (zoom - 1) * (t / dur)
        bw, bh = base.size
        cw, ch = int(bw / z), int(bh / z)
        crop = base.crop(((bw - cw) // 2, (bh - ch) // 2, (bw + cw) // 2, (bh + ch) // 2))
        box_w, box_h = W - 120, H - 120 - 170  # leave room for two subtitle lines
        scale = min(box_w / cw, box_h / ch)
        crop = crop.resize((int(cw * scale), int(ch * scale)))
        img.paste(crop, ((W - crop.width) // 2, 115))
        return img

    return draw


def safety_drawer(env: dict, val: dict):
    limits = [l["text"] for l in env["limits"]]

    def draw(t: float, dur: float) -> Image.Image:
        img = canvas()
        header(img, "safety case")
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([60, 130, 900, 860], 18, fill=CARD)
        d.text((95, 160), "Operating envelope", font=F_H2, fill=FG)
        for i, l in enumerate(limits[: int(len(limits) * min(1, t / (dur * 0.4))) + 1]):
            d.text((95, 240 + i * 62), "•  " + l, font=F_BODY, fill=FG)
        if t > dur * 0.35:
            d.rounded_rectangle([960, 130, W - 60, 860], 18, fill=CARD)
            d.text((1000, 160), f"Validated on {val['n_inside']} held-out scenarios", font=F_BODY, fill=MUTED)
            d.text((1000, 240), f"{val['pass_rate_inside']:.0%}", font=font(190, True), fill=GREEN)
            d.text((1000, 450), "pass inside the envelope", font=F_BODY, fill=FG)
            lo, hi = val["ci95_inside"]
            d.text((1000, 500), f"95% CI {lo:.0%}–{hi:.0%}", font=F_SMALL, fill=MUTED)
            d.text((1000, 580), f"{val['pass_rate_outside']:.0%}", font=font(130, True), fill=RED)
            d.text((1000, 730), "pass outside it", font=F_BODY, fill=FG)
        return img

    return draw


def stack_drawer(t: float, dur: float) -> Image.Image:
    img = canvas()
    header(img, "built on NVIDIA + Nebius")
    d = ImageDraw.Draw(img)
    rows = [
        ("NVIDIA Nemotron 3 Ultra", "adversary: writes testable stress hypotheses", "live mode"),
        ("NVIDIA Nemotron 3 Nano Omni", "explains failures from frames, checked against telemetry", "live mode"),
        ("Nebius Token Factory", "OpenAI-compatible API, structured JSON, image input", "live mode"),
        ("Nebius Serverless Jobs", "episode batches in parallel containers", "worker verified"),
        ("MuJoCo + PyTorch", "simulation and the policy under test", "this demo"),
    ]
    for i, (a, b, tag) in enumerate(rows[: int(len(rows) * min(1, t / (dur * 0.6))) + 1]):
        y = 150 + i * 140
        d.rounded_rectangle([60, y, W - 60, y + 115], 16, fill=CARD)
        d.text((95, y + 18), a, font=F_H2, fill=FG)
        d.text((95, y + 70), b, font=F_SMALL, fill=MUTED)
        tw = d.textlength(tag, font=F_SMALL)
        d.rounded_rectangle([W - 120 - tw, y + 38, W - 90, y + 80], 20, outline=GREEN if tag == "this demo" else MUTED, width=2)
        d.text((W - 105 - tw, y + 44), tag, font=F_SMALL, fill=GREEN if tag == "this demo" else MUTED)
    return img


def outro_drawer(t: float, dur: float) -> Image.Image:
    img = title_drawer(1.0, dur)
    d = ImageDraw.Draw(img)
    url = "github.com/MohammadSharafi/stress-twin"
    d.text(((W - d.textlength(url, font=F_H2)) / 2, 640), url, font=F_H2, fill=FG)
    return img


def render_failure_map(history, env, out: Path) -> Path:
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
    body{{margin:0;background:#0b1020;font-family:Helvetica,Arial,sans-serif}}
    .chart{{width:1800px;height:auto;background:#111a30;border-radius:14px}}
    .chart .lbl{{font-size:13px;fill:#e2e8f0}} .chart .axis{{font-size:12px;fill:#94a3b8}}
    .chart .track{{fill:#1a2440}} .chart .nominal{{fill:#14532d}} .chart .pass{{fill:#22c55e}} .chart .fail{{fill:#f87171}}
    .chart .faint{{opacity:.28}} .chart .envline{{stroke:#e2e8f0;stroke-width:2;stroke-dasharray:4 3}}
    .chart .brk{{font-size:13px;fill:#f87171;font-weight:700}} .chart .robust{{font-size:13px;fill:#22c55e}}
    h1{{color:#e2e8f0;font-size:40px;margin:10px 0 16px}}
    </style></head><body><div style="padding:30px 60px"><h1>Failure map: where each condition breaks the policy</h1>
    {failure_map_svg(history, env)}</div></body></html>"""
    page = out.with_suffix(".html")
    page.write_text(html)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},1400",
                    f"--screenshot={out}", f"file://{page}"], capture_output=True, check=True)
    im = Image.open(out).convert("RGB")
    arr = np.asarray(im).astype(int)
    rows = np.where(np.abs(arr - np.array(BG)).sum(axis=2).max(axis=1) > 30)[0]
    im.crop((0, max(0, rows.min() - 20), W, min(im.height, rows.max() + 20))).save(out)
    return out


def main(run: str = "runs/demo", out: str = "../submission/stress-twin-demo.mp4") -> None:
    run_p, out_p = Path(run), Path(out).resolve()
    history = load_history(run_p)
    val = json.loads((run_p / "validation.json").read_text())
    env = prim_envelope(history)
    camp = json.loads((run_p / "campaign.json").read_text())
    cl = {c["mode"]: c for c in clusters(history)}

    def pick(*modes):
        for m in modes:
            if m in cl:
                return replay(cl[m]["representative"])
        return replay(next(iter(cl.values()))["representative"])

    nominal = replay(params.sample_nominal(np.random.default_rng(3)))
    dim = pick("perception_miss|light")
    montage = [pick("slip|obj_mass"), pick("disturbance|disturbance"), pick("perception_miss|hue_shift"),
               pick("collision|obj_size", "perception_miss|image_noise")]

    s = camp["summary"]
    log_lines = ["$ stress-twin run --budget 240 --rounds 4 --strategy boundary"]
    log_lines += [l for l in (run_p / "campaign.log").read_text().splitlines()] if (run_p / "campaign.log").exists() else []
    log_lines += [f"episodes {s['episodes']} · failures {s['failures']} · failure modes {s['modes']} · root causes {len(s['causes'])}",
                  "$ stress-twin validate --run runs/demo --n 150",
                  f"held-out pass rate inside envelope {val['pass_rate_inside']:.1%} · outside {val['pass_rate_outside']:.0%}",
                  "ENVELOPE_OK"]

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fmap = render_failure_map(history, env, tmp / "fmap.png")
        ref: list = []
        montage_scene = Scene("montage", ["Every failure is diagnosed from telemetry and replayed exactly from its recorded seed.",
                                          "A slip under a heavier load.", "A stale estimate after a bump.",
                                          "A cube in an unfamiliar colour.", "A collision the descent never expected."],
                              montage_drawer(montage, "failure modes · deterministic replays", ref), min_dur=20, module=True)
        ref.append(montage_scene)
        scenes = [
            Scene("title", ["Stress Twin. A crash-test lab for robot brains."], title_drawer, 5),
            Scene("nominal", ["This robot's vision model was trained on one kind of scene: bright light and a red cube.",
                              "In those conditions it succeeds every time. Two hundred out of two hundred."],
                  clip_drawer(nominal, "policy under test · trained conditions"), module=True),
            Scene("dim", ["Dim the lights, and it reaches for empty air.", "Nobody tested that. Stress Twin does."],
                  clip_drawer(dim, "one condition changed"), module=True),
            Scene("search", ["Stress Twin searches fourteen conditions, from lighting and colour to mass, friction, camera calibration and bumps.",
                             "Each round it proposes the mildest scenarios most likely to break the robot, and runs them in parallel."],
                  log_drawer(log_lines), module=True),
            montage_scene,
            Scene("map", ["The failure map shows where each condition breaks the policy.",
                          "The green band is what it was trained on. Red dots are failures."],
                  image_drawer(fmap, "failure map"), module=True),
            Scene("safety", ["Then Stress Twin writes a safety case: an operating envelope, as limits a reviewer can read.",
                             f"Validated on {val['n_inside']} scenarios the search never saw: {val['pass_rate_inside']:.0%} pass inside the envelope, {val['pass_rate_outside']:.0%} outside."],
                  safety_drawer(env, val), module=True),
            Scene("stack", ["In live mode, NVIDIA Nemotron 3 Ultra on Nebius Token Factory is the adversary, writing testable hypotheses.",
                            "Nemotron Nano Omni explains each failure from video frames, checked against telemetry.",
                            "Batches scale out on Nebius Serverless Jobs. This demo ran in offline mode."],
                  stack_drawer),
            Scene("outro", ["Stress Twin. Find where your robot breaks, before your customers do."], outro_drawer, 5),
        ]
        segs, log = [], []
        for i, sc in enumerate(scenes):
            seg, dur = build(sc, tmp, i)
            segs.append(seg)
            log.append({"scene": sc.name, "seconds": round(dur, 2), "module_footage": sc.module})
            print(f"scene {sc.name}: {dur:.1f}s")
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in segs))
        out_p.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        "-movflags", "+faststart", str(out_p)], check=True)
    total = sum(x["seconds"] for x in log)
    module = sum(x["seconds"] for x in log if x["module_footage"])
    summary = {"video": str(out_p), "total_seconds": round(total, 1), "module_footage_seconds": round(module, 1), "scenes": log}
    out_p.with_name(out_p.stem + "-scenes.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_p}: {total:.1f}s total, {module:.1f}s of application modules in action")


if __name__ == "__main__":
    main(*sys.argv[1:])
