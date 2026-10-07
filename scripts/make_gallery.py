"""Render subtitle-free 3:2 gallery stills (1920x1280) from the same visuals as the demo video."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import make_video as mv  # noqa: E402
from stress_twin import params  # noqa: E402
from stress_twin.analysis import clusters, load_history, prim_envelope  # noqa: E402


def pad32(img: Image.Image) -> Image.Image:
    out = Image.new("RGB", (1920, 1280), mv.BG)
    out.paste(img, (0, 100))
    return out


def main(run="runs/demo", outdir="../submission/gallery"):
    run_p, out = Path(run), Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("0[2-6]-*.png"):
        old.unlink()
    history = load_history(run_p)
    val = json.loads((run_p / "validation.json").read_text())
    env = prim_envelope(history)
    cl = {c["mode"]: c for c in clusters(history)}
    shots = []
    nominal = mv.replay(params.sample_nominal(np.random.default_rng(3)))
    shots.append(("02-trained-conditions-success", mv.clip_drawer(nominal, "policy under test · trained conditions")(9.9, 10.0)))
    dim = mv.replay(cl["perception_miss|light"]["representative"])
    shots.append(("03-dim-light-perception-miss", mv.clip_drawer(dim, "one condition changed")(6.5, 6.6)))
    hue = mv.replay(cl["perception_miss|hue_shift"]["representative"])
    shots.append(("04-unfamiliar-colour-replay", mv.clip_drawer(hue, "failure modes · deterministic replays", 0.9)(4.9, 5.0)))
    with tempfile.TemporaryDirectory() as td:
        fmap = mv.render_failure_map(history, env, Path(td) / "fmap.png")
        shots.append(("05-failure-map", mv.image_drawer(fmap, "failure map", zoom=1.0)(0, 1)))
    shots.append(("06-validated-safety-case", mv.safety_drawer(env, val)(10, 10)))
    for name, img in shots:
        pad32(img).save(out / f"{name}.png")
        print("wrote", out / f"{name}.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
