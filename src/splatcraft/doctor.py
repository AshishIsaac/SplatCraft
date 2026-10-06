"""``splatcraft doctor``: is everything installed, and does a tiny scene actually train?"""

from __future__ import annotations

import importlib
import platform
import time

import numpy as np


def _ok(label: str, good: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if good else '!!'}] {label}{(': ' + detail) if detail else ''}", flush=True)
    return good


def smoke_train(device: str | None = None, iterations: int = 400) -> dict:
    """Fit a random Gaussian scene to 8 rendered views of another one; PSNR must climb."""
    import torch

    from .config import load_config
    from .gs.gaussians import GaussianModel
    from .gs.losses import psnr
    from .gs.rasterizer import render
    from .gs.trainer import pick_device
    from .scene.cameras import Camera

    dev = pick_device(device)
    rng = np.random.default_rng(0)
    gt = GaussianModel.from_points(rng.normal(0, 0.4, (400, 3)), rng.uniform(0, 1, (400, 3)), 1, dev, 0.8)
    cams = [Camera.look_at([3 * np.cos(a), 0.6, 3 * np.sin(a)], [0, 0, 0], [0, -1, 0], 0.8, 64, 64)
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    bg = torch.zeros(3, device=dev)
    with torch.no_grad():
        targets = [render(gt, c, bg)["image"].clamp(0, 1) for c in cams]
    cfg = load_config("smoke")
    g = GaussianModel.from_points(rng.normal(0, 0.6, (300, 3)), np.full((300, 3), 0.5), 1, dev, 0.1)
    g.setup_optimizer(cfg["train"]["lr"], 3.0)
    with torch.no_grad():
        first = float(np.mean([psnr(render(g, c, bg)["image"], t) for c, t in zip(cams, targets)]))
    for it in range(iterations):
        i = it % len(cams)
        loss = (render(g, cams[i], bg)["image"] - targets[i]).abs().mean()
        loss.backward()
        g.optimizer.step()
        g.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        last = float(np.mean([psnr(render(g, c, bg)["image"], t) for c, t in zip(cams, targets)]))
    return {"psnr_start": first, "psnr_end": last, "device": str(dev)}


def doctor(full: bool = False) -> int:
    print(f"SplatCraft doctor (Python {platform.python_version()}, {platform.system()})")
    good = True
    try:
        import torch
    except ImportError as e:
        _ok("PyTorch", False, str(e))
        return 1
    cuda = torch.cuda.is_available()
    _ok("PyTorch", True, f"{torch.__version__}, " + (f"CUDA on {torch.cuda.get_device_name(0)}" if cuda else "CPU only"))
    from .gs.rasterizer import triton_available

    tri = triton_available()
    _ok("Triton kernels", tri or not cuda,
        "fused GPU rasterizer" if tri else "not used: training runs on the tensor-op rasterizer (slower)")
    for mod, why in (("pycolmap", "camera poses from videos"), ("lpips", "LPIPS metric"), ("fastapi", "web app"),
                     ("cv2", "video frames")):
        try:
            m = importlib.import_module(mod)
            good &= _ok(mod, True, f"{getattr(m, '__version__', '')} ({why})")
        except ImportError as e:
            good &= _ok(mod, False, f"{e} ({why})")
    from .paths import WEB_DIR, home

    good &= _ok("viewer files", (WEB_DIR / "index.html").is_file(), str(WEB_DIR))
    _ok("workspace", True, str(home()))
    t0 = time.time()
    try:
        r = smoke_train()
        good &= _ok("smoke training", r["psnr_end"] > r["psnr_start"] + 3,
                    f"PSNR {r['psnr_start']:.1f} -> {r['psnr_end']:.1f} dB in {time.time() - t0:.1f} s on {r['device']}")
    except Exception as e:  # noqa: BLE001
        good &= _ok("smoke training", False, repr(e))
    from .compress.codec import decode_scene, encode_scene
    from .gs.gaussians import GaussianModel

    rng = np.random.default_rng(1)
    g = GaussianModel.from_points(rng.normal(0, 1, (2000, 3)), rng.uniform(0, 1, (2000, 3)), 3)
    cc = {"entropy": "rans", "min_opacity": 0, "position_bits": 16, "scale_bits": 8, "rotation_bits": 8,
          "opacity_bits": 8, "color_bits": 8, "sh_codebook": 64, "kmeans_iters": 3, "chunks": 2}
    data, _ = encode_scene(g, cc, log=lambda *_: None)
    good &= _ok("codec round trip", decode_scene(data).num == 2000, f"{len(data)} bytes")
    if full:
        from .paths import data_dir

        for s in ("lego", "bonsai"):
            _ok(f"dataset {s}", (data_dir() / s / ".complete").is_file())
    print("all good" if good else "some checks failed")
    return 0 if good else 1
