"""Pinhole cameras in the OpenCV convention (x right, y down, z forward; world -> camera).

Pixel (i, j) covers [i, i+1) x [j, j+1); its centre is at (i + 0.5, j + 0.5), as in COLMAP. The
rasterizer, the compressor's metadata and the WebGPU viewer all share this convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Camera:
    R: np.ndarray  # (3, 3) world -> camera rotation
    t: np.ndarray  # (3,) world -> camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    name: str = ""
    image_path: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def center(self) -> np.ndarray:
        """Camera position in world space."""
        return -self.R.T @ self.t

    @property
    def viewmat(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m

    @property
    def fov_y(self) -> float:
        return 2 * math.atan(self.height / (2 * self.fy))

    def scaled(self, s: float) -> Camera:
        """The same camera for an image resized by factor ``s`` (0.5 = half resolution)."""
        w, h = max(1, round(self.width * s)), max(1, round(self.height * s))
        sx, sy = w / self.width, h / self.height
        return Camera(self.R, self.t, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, w, h,
                      self.name, self.image_path, dict(self.meta))

    def to_dict(self) -> dict:
        return {"R": self.R.tolist(), "t": self.t.tolist(), "fx": self.fx, "fy": self.fy, "cx": self.cx,
                "cy": self.cy, "width": self.width, "height": self.height, "name": self.name}

    @staticmethod
    def from_dict(d: dict) -> Camera:
        return Camera(np.asarray(d["R"], float), np.asarray(d["t"], float), d["fx"], d["fy"], d["cx"], d["cy"],
                      int(d["width"]), int(d["height"]), d.get("name", ""))

    @staticmethod
    def look_at(eye, target, up, fov_y: float, width: int, height: int) -> Camera:
        """A camera at ``eye`` looking at ``target``; ``up`` is the world up direction."""
        eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
        f = target - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, up)
        if np.linalg.norm(r) < 1e-8:
            r = np.cross(f, np.array([1.0, 0, 0]) if abs(f[0]) < 0.9 else np.array([0, 1.0, 0]))
        r /= np.linalg.norm(r)
        d = np.cross(f, r)  # camera y points down (OpenCV)
        R = np.stack([r, d, f])
        fy = height / (2 * math.tan(fov_y / 2))
        return Camera(R, -R @ eye, fy, fy, width / 2, height / 2, width, height)


def qvec_to_rotmat(q) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) -> rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def camera_tensors(cam: Camera, device) -> dict[str, torch.Tensor]:
    return {"R": torch.as_tensor(cam.R, dtype=torch.float32, device=device),
            "t": torch.as_tensor(cam.t, dtype=torch.float32, device=device),
            "campos": torch.as_tensor(cam.center, dtype=torch.float32, device=device)}


def scene_frame(cams: list[Camera]) -> dict:
    """Centre, radius ("extent") and a robust up vector of a set of training cameras."""
    centers = np.stack([c.center for c in cams])
    mid = centers.mean(0)
    extent = float(np.linalg.norm(centers - mid, axis=1).max()) * 1.1
    up = -np.mean([c.R[1] for c in cams], axis=0)  # camera -y is "up" in the image
    up = up / (np.linalg.norm(up) + 1e-12)
    # Look-at target: the point closest to all optical axes (least squares).
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c in cams:
        d = c.R[2]
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c.center
    try:
        target = np.linalg.solve(A + 1e-6 * np.eye(3), b)
    except np.linalg.LinAlgError:
        target = mid
    if not np.all(np.isfinite(target)) or np.linalg.norm(target - mid) > 3 * extent:
        target = mid
    return {"center": mid, "extent": max(extent, 1e-3), "up": up, "target": target}
