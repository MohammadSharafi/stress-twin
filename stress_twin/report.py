"""Self-contained HTML safety report for a campaign run."""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import params
from .analysis import (clusters, factor_boundaries, fmt_nominal, fmt_value, load_history, prim_envelope,
                       stressors)
from .analyst import llm_caption, offline_caption
from .campaign import summarize
from .diagnose import CAUSES
from .episode import FRAME_EVERY, run_episode
from .perception import world_to_overhead_px
from .scene import camera_shift
from .strategies import STRESSABLE

CAUSE_TITLE = {"perception_miss": "Perception miss", "disturbance": "Stale estimate after bump",
               "collision": "Descent collision", "grasp_miss": "Grasp miss", "slip": "Slip during lift"}
CAUSE_COLOR = {"perception_miss": "#c2410c", "disturbance": "#7c3aed", "collision": "#b91c1c",
               "grasp_miss": "#0e7490", "slip": "#a16207"}
MAX_CLIPS = 8
e = html.escape


def _font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def marked_overhead(rec: dict, overhead: np.ndarray, size: int = 240) -> Image.Image:
    img = Image.fromarray(overhead).resize((size, size), Image.NEAREST)
    d = ImageDraw.Draw(img)
    cam = camera_shift(rec["scenario"])
    obj_z = rec["scenario"]["obj_size"] * 2
    tx, ty = world_to_overhead_px(rec["true_xy"], cam, z=obj_z, size=size)
    ex, ey = world_to_overhead_px(rec["est_xy"], cam, z=obj_z, size=size)
    d.ellipse([tx - 13, ty - 13, tx + 13, ty + 13], outline=(34, 197, 94), width=3)
    d.line([ex - 10, ey - 10, ex + 10, ey + 10], fill=(239, 68, 68), width=4)
    d.line([ex - 10, ey + 10, ex + 10, ey - 10], fill=(239, 68, 68), width=4)
    d.rectangle([0, 0, size, 22], fill=(15, 23, 42))
    d.text((8, 4), "ROBOT VIEW  o true  x perceived", fill=(226, 232, 240), font=_font(12))
    return img


def make_gif(rec: dict, frames: list, overhead: np.ndarray) -> bytes:
    left = marked_overhead(rec, overhead)
    caption = f"{CAUSE_TITLE.get(rec['cause'], rec['cause']).upper()}  ·  " + ", ".join(
        f"{s['label']} {s['text']}" for s in stressors(rec["scenario"], 2))
    out = []
    font = _font(15)
    for i, f in enumerate(frames):
        canvas = Image.new("RGB", (560, 276), (15, 23, 42))
        canvas.paste(left, (0, 0))
        canvas.paste(Image.fromarray(f), (240, 0))
        d = ImageDraw.Draw(canvas)
        d.text((10, 248), caption[:70], fill=(248, 113, 113), font=font)
        d.text((488, 248), f"t={i * FRAME_EVERY:.1f}s", fill=(148, 163, 184), font=_font(13))
        out.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=96))
    buf = io.BytesIO()
    out[0].save(buf, format="GIF", save_all=True, append_images=out[1:], duration=int(FRAME_EVERY * 1000),
                loop=0, optimize=True)
    return buf.getvalue()


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------- SVG charts ----------

def failure_map_svg(history: list[dict], env: dict) -> str:
    bounds = factor_boundaries(history)
    rows = [n for n in STRESSABLE if bounds[n]["tested"] > 0]
    W, label_w, right_w, row_h, top = 980, 170, 200, 40, 30
    plot_w = W - label_w - right_w
    H = top + row_h * len(rows) + 30
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Per-factor failure map" class="chart">']
    out.append(f'<text x="{label_w}" y="18" class="axis">full tested range →</text>')
    out.append(f'<text x="{W - right_w + 14}" y="18" class="axis">policy breaks at</text>')
    for i, name in enumerate(rows):
        p = params.BY_NAME[name]
        y = top + i * row_h + row_h / 2

        def X(v):
            return label_w + (v - p.low) / (p.high - p.low) * plot_w

        out.append(f'<text x="{label_w - 12}" y="{y + 4}" text-anchor="end" class="lbl"><title>{e(p.description)}</title>{e(p.label)}</text>')
        out.append(f'<rect x="{label_w}" y="{y - 9}" width="{plot_w}" height="18" rx="9" class="track"/>')
        n0, n1 = X(p.nominal[0]), X(p.nominal[1])
        out.append(f'<rect x="{n0 - 1}" y="{y - 9}" width="{max(n1 - n0, 2) + 2}" height="18" rx="4" class="nominal"/>')
        for h in history:
            off = params.off_nominal(h["scenario"])
            if name not in off[:3]:
                continue
            primary = (h["mode"] or "").endswith("|" + name) if not h["success"] else off[0] == name
            cls = ("pass" if h["success"] else "fail") + ("" if primary else " faint")
            jitter = ((h["scenario"]["seed"] % 1000) / 1000 - 0.5) * 10
            out.append(f'<circle cx="{X(h["scenario"][name]):.1f}" cy="{y + jitter:.1f}" r="{4 if primary else 3}" class="{cls}"/>')
        lo, hi = env["box"].get(name, [p.low, p.high])
        for b in (lo, hi):
            if p.low + 1e-9 < b < p.high - 1e-9:
                out.append(f'<line x1="{X(b):.1f}" x2="{X(b):.1f}" y1="{y - 15}" y2="{y + 15}" class="envline"/>')
        brk = []
        for side, sym in (("low", "≤"), ("high", "≥")):
            mf = bounds[name][side]["mildest_failure"]
            if mf is not None:
                brk.append(f"{sym} {fmt_value(name, mf)}")
        txt = " · ".join(brk) if brk else f"no failure in {bounds[name]['tested']} tests"
        out.append(f'<text x="{W - right_w + 14}" y="{y + 4}" class="{"brk" if brk else "robust"}">{e(txt)}</text>')
    out.append("</svg>")
    return "".join(out)


def progress_svg(history: list[dict]) -> str:
    W, H, L, B = 980, 220, 50, 30
    seen, modes, fails = set(), [], []
    nf = 0
    for h in history:
        if not h["success"]:
            nf += 1
            seen.add(h["mode"])
        modes.append(len(seen))
        fails.append(nf)
    n = len(history)
    ymax = max(max(fails) if fails else 1, 1)

    def pts(series, scale):
        return " ".join(f"{L + i / max(n - 1, 1) * (W - L - 20):.1f},{H - B - v / scale * (H - B - 20):.1f}"
                        for i, v in enumerate(series))

    mmax = max(max(modes) if modes else 1, 1)
    rounds = sorted({h["round"] for h in history})
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Discovery progress">']
    for r in rounds[1:]:
        i = next(k for k, h in enumerate(history) if h["round"] == r)
        x = L + i / max(n - 1, 1) * (W - L - 20)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="14" y2="{H - B}" class="grid"/>'
                   f'<text x="{x + 4:.1f}" y="24" class="axis">round {r + 1}</text>')
    out.append(f'<line x1="{L}" x2="{W - 20}" y1="{H - B}" y2="{H - B}" class="axisline"/>')
    out.append(f'<polyline points="{pts(fails, ymax)}" class="line-fail"/>')
    out.append(f'<polyline points="{pts(modes, mmax)}" class="line-mode"/>')
    out.append(f'<text x="{L}" y="{H - 8}" class="axis">episodes → {n}</text>')
    out.append(f'<text x="{W - 20}" y="{H - 8}" text-anchor="end" class="axis">'
               f'<tspan class="k-fail">■</tspan> failures ({fails[-1] if fails else 0}) '
               f'<tspan class="k-mode">■</tspan> distinct failure modes ({modes[-1] if modes else 0})</text>')
    out.append("</svg>")
    return "".join(out)


# ---------- report ----------

def recommendations(cl: list[dict], bounds: dict) -> list[str]:
    causes = {c["cause"] for c in cl}
    recs = []
    perc_factors = sorted({c["factor"] for c in cl if c["cause"] == "perception_miss"})
    if perc_factors:
        parts = []
        for f in perc_factors:
            b = bounds.get(f, {})
            for side in ("low", "high"):
                mf = (b.get(side) or {}).get("mildest_failure")
                if mf is not None:
                    parts.append(f"{params.BY_NAME[f].label} {'down to' if side == 'low' else 'up to'} {fmt_value(f, mf)}")
        recs.append("Retrain perception with data covering " + ", ".join(parts) + " and beyond. The exported failure set is a starting point.")
    if "disturbance" in causes:
        recs.append("Re-perceive the object before descending (close the loop). A single perception pass cannot survive a bump.")
    if "slip" in causes:
        recs.append("Raise grip force or cap payload: slips appear when mass is high relative to friction. Add a slip check during lift.")
    if "collision" in causes:
        recs.append("Estimate object height and orientation before descending; the fixed grasp height collides with tall or rotated objects.")
    if "grasp_miss" in causes:
        recs.append("Add grasp verification (finger contact check) and a retry.")
    return recs


def build_report(run: Path, analyst_mode: str = "auto", log=print) -> Path:
    run = Path(run)
    history = load_history(run)
    meta = json.loads((run / "campaign.json").read_text()) if (run / "campaign.json").exists() else {}
    validation = json.loads((run / "validation.json").read_text()) if (run / "validation.json").exists() else None
    summary = summarize(history)
    cl = clusters(history)
    bounds = factor_boundaries(history)
    env = prim_envelope(history)

    client = None
    if analyst_mode in ("auto", "llm"):
        try:
            from .llm import TokenFactory

            client = TokenFactory.from_env(log_path=run / "llm_calls.jsonl")
            if not client.has("vision"):
                raise RuntimeError("no vision model on Token Factory")
        except Exception as err:  # noqa: BLE001
            if analyst_mode == "llm":
                raise
            log(f"analyst: offline ({err})")
            client = None

    cards = []
    captions = []
    for c in cl[:MAX_CLIPS]:
        rec = c["representative"]
        replay = run_episode(rec["scenario"], record=True)
        same = replay.cause == rec["cause"]
        cap = offline_caption(rec)
        if client is not None:
            try:
                cap = llm_caption(client, rec, marked_overhead(rec, replay.overhead), replay.frames)
            except Exception as err:  # noqa: BLE001
                log(f"analyst call failed for {c['mode']}: {err}; using offline caption")
        cap["replay_matches"] = same
        captions.append({"mode": c["mode"], **cap})
        gif = base64.b64encode(make_gif(rec, replay.frames, replay.overhead)).decode()
        cond = stressors(rec["scenario"])
        grounded = ('<span class="badge ok">matches telemetry</span>' if cap["grounded"]
                    else f'<span class="badge warn">analyst said {e(cap["cause"])}; telemetry says {e(rec["cause"])}</span>')
        cards.append(f"""
      <article class="card cluster">
        <img src="data:image/gif;base64,{gif}" alt="Replay of {e(c['mode'])}" width="560" height="276" loading="lazy">
        <div class="cbody">
          <div class="chips"><span class="cause" style="--c:{CAUSE_COLOR.get(c['cause'], '#475569')}">{e(CAUSE_TITLE.get(c['cause'], c['cause']))}</span>
            <span class="muted">{c['count']} failure{'s' if c['count'] != 1 else ''} · main factor <b>{e(params.BY_NAME[c['factor']].label) if c['factor'] in params.BY_NAME else e(c['factor'])}</b></span></div>
          <p class="mildest">Mildest breaking case (stress {c['min_stress']:.2f}): {'; '.join(f"{e(s['label'])} <b>{e(s['text'])}</b> <span class='muted'>(trained {e(s['nominal'])})</span>" for s in cond) or 'nominal'}</p>
          <p>{e(cap['explanation'])}</p>
          <p class="muted small">Evidence: {e(cap['visual_evidence'])} · analyst: {e(cap['analyst'])} {grounded}{'' if same else ' <span class="badge warn">replay differed</span>'}</p>
        </div>
      </article>""")
    (run / "captions.json").write_text(json.dumps(captions, indent=2))

    with open(run / "failures.jsonl", "w") as f:
        for h in history:
            if not h["success"]:
                f.write(json.dumps({"scenario": h["scenario"], "cause": h["cause"], "mode": h["mode"]}) + "\n")

    if validation:
        lo, hi = validation["ci95_inside"]
        verdict = (f"Safe operating envelope validated: <b>{validation['pass_rate_inside']:.0%}</b> pass rate on "
                   f"{validation['n_inside']} fresh scenarios inside it (95% CI {lo:.0%}–{hi:.0%}), versus "
                   f"{validation['pass_rate_outside']:.0%} outside.")
        residual = validation["failures_inside"]
    else:
        verdict = ("Safe operating envelope estimated from campaign data "
                   f"({env['train_pass_rate']:.0%} pass rate on {env['train_support']} campaign episodes inside it). "
                   "Not yet validated on held-out scenarios: run <code>stress-twin validate</code>.")
        residual = []

    llm_calls = [json.loads(l) for l in (run / "llm_calls.jsonl").read_text().splitlines()] if (run / "llm_calls.jsonl").exists() else []
    planner_notes = json.loads((run / "planner_notes.json").read_text()) if (run / "planner_notes.json").exists() else []
    hyp = [h for h in history if h["source"] == "llm" and h.get("expected_cause")]
    mode_chip = ("Token Factory · " + ", ".join(sorted({c['model'] for c in llm_calls if c.get('ok')}))
                 if any(c.get("ok") for c in llm_calls) else "Offline mode (no Token Factory calls)")

    llm_html = ""
    if llm_calls or hyp:
        by = {}
        for c in llm_calls:
            k = (c["role"], c["model"])
            b = by.setdefault(k, {"n": 0, "ok": 0, "pt": 0, "ct": 0, "lat": []})
            b["n"] += 1
            b["ok"] += int(bool(c.get("ok")))
            b["pt"] += c.get("prompt_tokens") or 0
            b["ct"] += c.get("completion_tokens") or 0
            b["lat"].append(c.get("latency_s") or 0)
        rows = "".join(f"<tr><td>{e(r)}</td><td><code>{e(m)}</code></td><td>{b['ok']}/{b['n']}</td><td>{b['pt']:,}</td>"
                       f"<td>{b['ct']:,}</td><td>{np.mean(b['lat']):.1f}s</td></tr>" for (r, m), b in by.items())
        conf = sum(1 for h in hyp if h.get("confirmed"))
        notes = "".join(f"<li><b>Round {n['round'] + 1}</b>: {n['accepted']}/{n['requested']} proposals accepted. {e(n['analysis'][:500])}</li>"
                        for n in planner_notes)
        examples = "".join(f"<li>{e(h['hypothesis'])} → <b>{'confirmed' if h.get('confirmed') else ('failed: ' + h['cause'] if not h['success'] else 'refuted')}</b></li>"
                           for h in hyp[:12])
        llm_html = f"""
    <section id="adversary"><h2>Nemotron adversary</h2>
      <p>{len(hyp)} hypothesis-driven scenarios proposed; {conf} confirmed ({(conf / len(hyp) if hyp else 0):.0%}).</p>
      <ul class="tight">{notes}</ul>
      <details><summary>Hypotheses (first 12)</summary><ul class="tight">{examples}</ul></details>
      <table><thead><tr><th>Role</th><th>Model</th><th>OK</th><th>Prompt tokens</th><th>Completion tokens</th><th>Mean latency</th></tr></thead><tbody>{rows}</tbody></table>
    </section>"""

    limits = "".join(f"<li>{e(l['text'])}</li>" for l in env["limits"]) or "<li>No limits needed: no failures found.</li>"
    robust = [n for n in STRESSABLE if bounds[n]["tested"] >= 5 and bounds[n]["failures"] == 0]
    residual_html = ("".join(f"<li>{e(r['cause'])}: {e(', '.join(r['stressors']))}</li>" for r in residual[:10])
                     if residual else "<li>None observed in validation.</li>" if validation else "<li>Run validation to measure.</li>")
    recs = "".join(f"<li>{e(r)}</li>" for r in recommendations(cl, bounds))
    run_date = dt.datetime.fromtimestamp((run / "results.jsonl").stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stress Twin Report</title>
<style>
:root {{ --bg:#f8fafc; --fg:#0f172a; --muted:#64748b; --card:#ffffff; --line:#e2e8f0; --track:#eef2f7;
  --nominal:#bbf7d0; --pass:#16a34a; --fail:#dc2626; --accent:#ef4444; --mode:#2563eb; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0b1020; --fg:#e2e8f0; --muted:#94a3b8;
  --card:#111a30; --line:#1f2a44; --track:#1a2440; --nominal:#14532d; --pass:#22c55e; --fail:#f87171; --mode:#60a5fa; }} }}
:root[data-theme="dark"] {{ --bg:#0b1020; --fg:#e2e8f0; --muted:#94a3b8; --card:#111a30; --line:#1f2a44; --track:#1a2440;
  --nominal:#14532d; --pass:#22c55e; --fail:#f87171; --mode:#60a5fa; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif }}
main {{ max-width:1060px; margin:0 auto; padding:32px 16px 64px }}
h1 {{ font-size:40px; letter-spacing:-1px; margin:0 }} h1 span {{ color:var(--accent) }}
h2 {{ font-size:22px; margin:44px 0 10px }}
.sub {{ color:var(--muted); margin:4px 0 18px }}
.verdict {{ background:var(--card); border:1px solid var(--line); border-left:5px solid var(--accent); padding:14px 18px; border-radius:10px }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:18px 0 }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px }}
.stat b {{ display:block; font-size:28px }} .stat span {{ color:var(--muted); font-size:13px }}
.chip {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 10px; font-size:13px; color:var(--muted) }}
.chart {{ width:100%; height:auto; background:var(--card); border:1px solid var(--line); border-radius:10px }}
.chart .lbl {{ font-size:13px; fill:var(--fg) }} .chart .axis {{ font-size:12px; fill:var(--muted) }}
.chart .track {{ fill:var(--track) }} .chart .nominal {{ fill:var(--nominal) }}
.chart .pass {{ fill:var(--pass) }} .chart .fail {{ fill:var(--fail) }} .chart .faint {{ opacity:.28 }}
.chart .envline {{ stroke:var(--fg); stroke-width:2; stroke-dasharray:4 3 }}
.chart .brk {{ font-size:13px; fill:var(--fail); font-weight:600 }} .chart .robust {{ font-size:13px; fill:var(--pass) }}
.chart .grid {{ stroke:var(--line) }} .chart .axisline {{ stroke:var(--muted) }}
.chart .line-fail {{ fill:none; stroke:var(--fail); stroke-width:2.5 }} .chart .line-mode {{ fill:none; stroke:var(--mode); stroke-width:2.5 }}
.chart .k-fail {{ fill:var(--fail) }} .chart .k-mode {{ fill:var(--mode) }}
.legend {{ color:var(--muted); font-size:13px; margin:6px 2px }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin:0 4px 0 12px; vertical-align:-1px }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden }}
.cluster {{ display:grid; grid-template-columns:560px 1fr; margin:14px 0 }}
.cluster img {{ width:100%; height:auto; display:block; background:#0f172a }}
.cbody {{ padding:14px 18px }} .cbody p {{ margin:8px 0 }}
.cause {{ background:var(--c); color:#fff; border-radius:6px; padding:2px 9px; font-weight:700; font-size:13px; margin-right:8px }}
.muted {{ color:var(--muted) }} .small {{ font-size:13px }}
.badge {{ border-radius:999px; padding:1px 8px; font-size:12px; margin-left:6px }}
.badge.ok {{ background:#dcfce7; color:#166534 }} .badge.warn {{ background:#fef3c7; color:#92400e }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px }}
.box {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 18px }}
.box h3 {{ margin:2px 0 8px; font-size:16px }} ul.tight li {{ margin:4px 0 }}
table {{ width:100%; border-collapse:collapse; margin-top:12px; font-size:14px }} th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line) }}
code {{ font-size:13px }} footer {{ color:var(--muted); font-size:13px; margin-top:48px }}
@media (max-width: 860px) {{ .cluster {{ grid-template-columns:1fr }} .grid2 {{ grid-template-columns:1fr }} h1 {{ font-size:30px }} }}
</style></head>
<body><main>
  <h1>Stress <span>Twin</span> safety report</h1>
  <p class="sub">Policy under test: gantry pick-and-place, CNN perception trained on nominal scenes + scripted grasp ·
    strategy <b>{e(str(meta.get('strategy', '?')))}</b> · run {e(run.name)} · {run_date} · <span class="chip">{e(mode_chip)}</span></p>
  <div class="verdict">{verdict}</div>
  <div class="stats">
    <div class="stat"><b>{summary['episodes']}</b><span>simulated episodes</span></div>
    <div class="stat"><b>{summary['failures']}</b><span>failures</span></div>
    <div class="stat"><b>{summary['modes']}</b><span>distinct failure modes</span></div>
    <div class="stat" title="Stress measures how far a scenario is from the training conditions: 0 means fully inside them."><b>{summary['subtle_failures']}</b><span>subtle failures (stress ≤ 0.30)</span></div>
    <div class="stat"><b>{len(summary['causes'])}</b><span>root causes</span></div>
  </div>

  <p class="small muted">Stress = how far a scenario sits outside the training conditions, summed over all conditions on a 0–1
    scale per condition. A subtle failure breaks the policy with only a small departure from what it was trained on.</p>

  <section id="failure-map"><h2>Failure map</h2>
    <p class="muted">Each row is one condition, drawn across its full tested range. Green band: what the policy was trained on.
      Dots: episodes where that condition was pushed (solid = main stressor). Dashed lines: safe envelope limits.</p>
    {failure_map_svg(history, env)}
    <div class="legend"><i style="background:var(--pass)"></i>passed <i style="background:var(--fail)"></i>failed
      <i style="background:var(--nominal);border-radius:2px"></i>training range</div>
  </section>

  <section id="progress"><h2>Discovery progress</h2>{progress_svg(history)}</section>

  <section id="clusters"><h2>Failure modes</h2>
    <p class="muted">Each mode is a root cause paired with the condition that triggers it, shown with its mildest breaking case
      replayed deterministically from the recorded seed.</p>
    {''.join(cards) or '<p>No failures found.</p>'}
  </section>

  <section id="safety-case"><h2>Safety case</h2>
    <div class="grid2">
      <div class="box"><h3>Operating envelope</h3><p class="small muted">Found by PRIM box peeling over {env['train_total']} episodes.</p>
        <ul class="tight">{limits}</ul>
        <p class="small muted">Other conditions held at their trained range. Robust across the full tested range:
          {e(', '.join(params.BY_NAME[n].label for n in robust)) or 'none established'}.</p></div>
      <div class="box"><h3>Residual risk inside the envelope</h3><ul class="tight">{residual_html}</ul>
        <h3>Recommended fixes</h3><ul class="tight">{recs}</ul></div>
    </div>
    <p class="small muted">Retraining set: <code>failures.jsonl</code> ({summary['failures']} scenarios, replayable from their seeds).</p>
  </section>
  {llm_html}
  <footer>Stress Twin · MuJoCo {_mj_version()} simulation · failure causes are diagnosed deterministically from telemetry and every
    model-written explanation is checked against that diagnosis · clips are re-rendered from the recorded seed.</footer>
</main></body></html>"""
    path = run / "report.html"
    path.write_text(page)
    return path


def _mj_version() -> str:
    import mujoco

    return mujoco.__version__
