"""Perception for the policy under test: a small CNN that regresses the target object's
table position from the overhead camera image.

It is trained only on nominal scenes (see train_perception.py). That is deliberate: it
mirrors how real learned policies are trained on a narrow data distribution and then
deployed into a wider world.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from . import scene

IMG = 64
POS_SCALE = 0.12
WEIGHTS = Path(__file__).parent / "assets" / "perception.pt"


class PerceptionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 2), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)) * POS_SCALE


def render_overhead(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer) -> np.ndarray:
    renderer.update_scene(data, camera="overhead")
    return renderer.render().copy()


def add_noise(img: np.ndarray, std: float, seed: int) -> np.ndarray:
    if std <= 0:
        return img
    rng = np.random.default_rng(seed + 104729)
    noisy = img.astype(np.float32) / 255.0 + rng.normal(0.0, std, img.shape)
    return (np.clip(noisy, 0, 1) * 255).astype(np.uint8)


def to_tensor(imgs: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(imgs, dtype=np.float32) / 255.0)
    if x.ndim == 3:
        x = x[None]
    return x.permute(0, 3, 1, 2).contiguous()


class Perception:
    """Loads trained weights once per process."""

    _shared: "Perception | None" = None

    def __init__(self, weights: Path = WEIGHTS) -> None:
        torch.set_num_threads(1)
        self.net = PerceptionNet()
        self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.net.eval()

    @classmethod
    def shared(cls) -> "Perception":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> np.ndarray:
        return self.net(to_tensor(img))[0].numpy().astype(float)


def world_to_overhead_px(xy, cam_xy, z: float = 0.0, size: int = IMG) -> tuple[float, float]:
    """Project a table point into the overhead image (for drawing markers on clips)."""
    half = (scene.CAM_HEIGHT - z) * np.tan(np.deg2rad(scene.CAM_FOVY / 2))
    col = size / 2 + (xy[0] - cam_xy[0]) / half * size / 2
    row = size / 2 - (xy[1] - cam_xy[1]) / half * size / 2
    return col, row
