"""Held-out view evaluation: PSNR, SSIM, LPIPS (the three numbers every 3DGS paper reports)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..gs.gaussians import GaussianModel
from ..gs.losses import psnr, ssim
from ..gs.rasterizer import render
from ..paths import models_dir
from ..scene.cameras import Camera
from ..scene.dataset import load_image

_LPIPS: dict = {}


def lpips_model(net: str, device):
    """LPIPS with its backbone cached under the workspace (``models/torch``), not the user profile."""
    key = (net, str(device))
    if key not in _LPIPS:
        os.environ.setdefault("TORCH_HOME", str(models_dir() / "torch"))
        import lpips

        _LPIPS[key] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    return _LPIPS[key]


def load_run_cameras(run_dir: Path) -> tuple[list[Camera], list[Camera], dict]:
    meta = json.loads((Path(run_dir) / "cameras.json").read_text())

    def cams(key):
        out = []
        for d in meta[key]:
            c = Camera.from_dict(d)
            c.image_path = d.get("image_path", "")
            out.append(c)
        return out

    return cams("train"), cams("test"), meta


@torch.no_grad()
def evaluate(g: GaussianModel, cams: list[Camera], white_background: bool, lpips_net: str = "vgg",
             max_views: int = 0, save_dir: Path | None = None, log=print) -> dict:
    dev = g.device
    bg = torch.ones(3, device=dev) if white_background else torch.zeros(3, device=dev)
    if max_views and len(cams) > max_views:
        idx = np.linspace(0, len(cams) - 1, max_views).round().astype(int)
        cams = [cams[i] for i in idx]
    lp = lpips_model(lpips_net, dev) if lpips_net else None
    rows, times = [], []
    for i, cam in enumerate(cams):
        gt = torch.from_numpy(load_image(cam, white_background)).to(dev).float() / 255
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = render(g, cam, bg)["image"].clamp(0, 1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        row = {"view": cam.name, "psnr": psnr(img, gt), "ssim": float(ssim(img, gt))}
        if lp is not None:
            row["lpips"] = float(lp(img.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1))
        rows.append(row)
        if save_dir is not None and i < 6:
            import cv2

            save_dir.mkdir(parents=True, exist_ok=True)
            both = torch.cat([gt, img], 1).cpu().numpy()
            cv2.imwrite(str(save_dir / f"{i:03d}_{cam.name}.jpg"), (both[..., ::-1] * 255).round().astype(np.uint8))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips") if rows and k in rows[0]}
    res = {**{k: round(v, 4) for k, v in mean.items()}, "views": len(rows), "gaussians": g.num,
           "resolution": [cams[0].width, cams[0].height] if cams else None,
           "train_renderer_ms": round(1000 * float(np.median(times[1:] or times)), 2) if times else None,
           "per_view": rows}
    log(f"eval: PSNR {res.get('psnr', 0):.2f}  SSIM {res.get('ssim', 0):.4f}  LPIPS {res.get('lpips', float('nan')):.4f}"
        f"  ({len(rows)} views, {g.num:,} Gaussians)")
    return res
