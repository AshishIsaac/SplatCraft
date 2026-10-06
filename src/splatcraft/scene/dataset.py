"""Scene loading: COLMAP folders (real captures) and NeRF-synthetic ``transforms_*.json`` folders.

Images are resized once to the training resolution and cached next to the scene (``_cache_<w>``),
so later runs start immediately.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .cameras import Camera, scene_frame
from .colmap_io import read_model


@dataclass
class SceneData:
    name: str
    kind: str  # "colmap" | "blender"
    train: list[Camera]
    test: list[Camera]
    points: np.ndarray
    colors: np.ndarray
    white_background: bool
    frame: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def detect_kind(path: Path) -> str:
    if (path / "transforms_train.json").is_file():
        return "blender"
    return "colmap"


def load_scene(path: str | Path, max_width: int = 1600, test_every: int = 8, max_images: int = 0,
               random_init: int = 100_000) -> SceneData:
    path = Path(path)
    kind = detect_kind(path)
    if kind == "blender":
        train, test, pts, cols = _load_blender(path)
        white = True
    else:
        img_dir = path / "images"
        cams, pts, cols, warnings = read_model(path, img_dir)
        cams = [c for c in cams if Path(c.image_path).is_file()]
        if not cams:
            raise FileNotFoundError(f"none of the registered images exist in {img_dir}")
        if max_images and len(cams) > max_images:
            idx = np.linspace(0, len(cams) - 1, max_images).round().astype(int)
            cams = [cams[i] for i in sorted(set(idx))]
        if test_every and len(cams) >= 2 * test_every:
            test = [c for i, c in enumerate(cams) if i % test_every == 0]
            train = [c for i, c in enumerate(cams) if i % test_every != 0]
        else:
            train, test = cams, cams[:: max(1, len(cams) // 4)]
        white = False
    sd = SceneData(path.name, kind, train, test, pts, cols, white)
    if kind == "colmap":
        sd.warnings = warnings
    # Resize every camera so the image is at most ``max_width`` wide.
    s = min(1.0, max_width / max(c.width for c in sd.train)) if max_width else 1.0
    if s < 1.0:
        sd.train = [c.scaled(s) for c in sd.train]
        sd.test = [c.scaled(s) for c in sd.test]
    sd.frame = scene_frame(sd.train)
    if len(sd.points) == 0:  # no SfM points (synthetic scenes): random init inside the camera hull
        rng = np.random.default_rng(0)
        r = 1.3 if kind == "blender" else sd.frame["extent"] / 1.1 * 0.8  # 1.3: the 3DGS synthetic init cube
        sd.points = rng.uniform(-r, r, (random_init, 3)) + (sd.frame["target"] if kind != "blender" else 0)
        sd.colors = rng.uniform(0, 1, (random_init, 3)).astype(np.float32)
    return sd


def _load_blender(path: Path):
    out = []
    for split in ("train", "test"):
        meta = json.loads((path / f"transforms_{split}.json").read_text())
        cams = []
        for fr in meta["frames"]:
            c2w = np.asarray(fr["transform_matrix"], float)
            c2w[:3, 1:3] *= -1  # OpenGL (y up, z back) -> OpenCV (y down, z forward)
            w2c = np.linalg.inv(c2w)
            fp = fr["file_path"]
            img = path / (fp if fp.endswith(".png") else fp + ".png")
            w = h = 800
            if img.is_file():
                im = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
                if im is not None:
                    h, w = im.shape[:2]
            f = 0.5 * w / math.tan(0.5 * meta["camera_angle_x"])
            cams.append(Camera(w2c[:3, :3], w2c[:3, 3], f, f, w / 2, h / 2, w, h, img.stem, str(img)))
        out.append(cams)
    return out[0], out[1], np.zeros((0, 3)), np.zeros((0, 3), np.float32)


def load_image(cam: Camera, white_background: bool, cache: bool = True) -> np.ndarray:
    """-> (H, W, 3) uint8 RGB at the camera's resolution (alpha composited onto the background)."""
    src = Path(cam.image_path)
    cpath = src.parent / f"_cache_{cam.width}x{cam.height}" / (src.stem + ".png")
    if cache and cpath.is_file():
        im = cv2.imread(str(cpath), cv2.IMREAD_UNCHANGED)
        if im is not None:
            return im[..., ::-1].copy() if im.ndim == 3 else np.repeat(im[..., None], 3, 2)
    im = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(src)
    if im.dtype == np.uint16:
        im = (im / 257).astype(np.uint8)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[1] != cam.width or im.shape[0] != cam.height:
        im = cv2.resize(im, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
    if im.shape[2] == 4:
        a = im[..., 3:4].astype(np.float32) / 255
        bg = 255.0 if white_background else 0.0
        im = (im[..., :3].astype(np.float32) * a + bg * (1 - a)).round().astype(np.uint8)
    rgb = im[..., ::-1].copy()
    if cache:
        try:
            cpath.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(cpath), im)
        except OSError:
            pass
    return rgb


class ImageBank:
    """All training images resident on the device as uint8 (4x smaller than float)."""

    def __init__(self, cams: list[Camera], white_background: bool, device) -> None:
        self.cams = cams
        self.images = [torch.from_numpy(load_image(c, white_background)).to(device) for c in cams]

    def __len__(self) -> int:
        return len(self.cams)

    def get(self, i: int) -> torch.Tensor:
        return self.images[i].float() / 255.0
