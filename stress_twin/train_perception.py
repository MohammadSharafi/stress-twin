"""Train the perception CNN of the policy under test on nominal scenes only.

    python -m stress_twin.train_perception --n 6000 --epochs 40
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import mujoco
import numpy as np
import torch
from torch import nn

from . import params, scene
from .perception import IMG, WEIGHTS, PerceptionNet, add_noise, render_overhead, to_tensor


def _render_one(seed: int) -> tuple[np.ndarray, np.ndarray]:
    s = params.sample_nominal(np.random.default_rng(seed))
    model = mujoco.MjModel.from_xml_string(scene.build_xml(s))
    data = mujoco.MjData(model)
    data.qpos[[model.joint(n).qposadr[0] for n in ("gx", "gy", "gz")]] = scene.HAND_HOME
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, IMG, IMG)
    img = add_noise(render_overhead(model, data, r), s["image_noise"], s["seed"])
    r.close()
    return img, np.array([s["obj_x"], s["obj_y"]], dtype=np.float32)


def make_dataset(n: int, base_seed: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    with mp.get_context("spawn").Pool(workers) as pool:
        out = pool.map(_render_one, range(base_seed, base_seed + n), chunksize=32)
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()

    t0 = time.time()
    X, Y = make_dataset(args.n, 10_000, args.workers)
    Xv, Yv = make_dataset(600, 900_000, args.workers)
    print(f"rendered {len(X)} train / {len(Xv)} val images in {time.time() - t0:.1f}s")

    torch.manual_seed(0)
    net = PerceptionNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.epochs * (len(X) // 128 + 1))
    xt, yt = to_tensor(X), torch.from_numpy(Y)
    xv, yv = to_tensor(Xv), torch.from_numpy(Yv)
    loss_fn = nn.MSELoss()
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
        if epoch % 5 == 4 or epoch == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                err = (net(xv) - yv).norm(dim=1)
            print(f"epoch {epoch + 1}: val mean err {err.mean() * 1000:.1f} mm, p95 {err.quantile(0.95) * 1000:.1f} mm")
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), WEIGHTS)
    print(f"saved {WEIGHTS} ({WEIGHTS.stat().st_size // 1024} KB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
