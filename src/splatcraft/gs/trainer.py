"""The 3DGS optimisation loop: render a random training view, L1 + D-SSIM, Adam, densify / prune."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from pathlib import Path

import torch
import yaml

from ..scene.dataset import ImageBank, SceneData
from .gaussians import GaussianModel
from .losses import photometric_loss, psnr
from .rasterizer import render

Progress = Callable[[dict], None]


def pick_device(pref: str | None = None) -> torch.device:
    if pref:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exp_lr(step: int, total: int, lr0: float, lr1: float) -> float:
    t = min(max(step / max(total, 1), 0.0), 1.0)
    return math.exp(math.log(lr0) * (1 - t) + math.log(lr1) * t)


@torch.no_grad()
def importance(g: GaussianModel, cams, bg, cfg_train: dict) -> torch.Tensor:
    """Global significance: summed blend weight (alpha x transmittance) over all pixels of all views."""
    total = torch.zeros(g.num, device=g.device)
    for cam in cams:
        out = render(g, cam, bg, K=cfg_train["slab"], budget=cfg_train["slab_budget"], want_weights=True)
        total += out["weights"]
    return total


def train(scene: SceneData, cfg: dict, out_dir: str | Path, device: str | None = None,
          progress: Progress | None = None, should_stop: Callable[[], bool] | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)
    tc, dc, cc = cfg["train"], cfg["densify"], cfg["compaction"]
    seed = int(tc.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    _save_cameras(scene, out_dir / "cameras.json")

    say = progress or (lambda d: None)
    say({"stage": "load", "message": f"loading {len(scene.train)} training images"})
    bank = ImageBank(scene.train, scene.white_background, dev)
    extent = scene.frame["extent"]
    g = GaussianModel.from_points(scene.points, scene.colors, tc["sh_degree"], dev, tc["init_opacity"])
    g.setup_optimizer(tc["lr"], extent)
    bg_const = torch.ones(3, device=dev) if scene.white_background else torch.zeros(3, device=dev)

    iters = int(tc["iterations"])
    order: list[int] = []
    log = open(out_dir / "train_log.jsonl", "w", encoding="utf-8")  # noqa: SIM115 - closed after the loop
    t0 = time.time()
    ema_loss, ema_psnr = None, None
    compact_at = set(int(x) for x in (cc.get("at") or []))
    stopped = False
    for it in range(1, iters + 1):
        for group in g.optimizer.param_groups:
            if group["name"] == "means":
                group["lr"] = exp_lr(it, iters, tc["lr"]["means"] * extent, tc["lr"]["means_final"] * extent)
        if it % tc["sh_increase_every"] == 0 and g.active_sh_degree < g.max_sh_degree:
            g.active_sh_degree += 1
        if not order:
            order = list(range(len(bank)))
            random.shuffle(order)
        idx = order.pop()
        cam, gt = bank.cams[idx], bank.get(idx)
        bg = torch.rand(3, device=dev) if tc.get("random_background") else bg_const

        out = render(g, cam, bg, K=tc["slab"], budget=tc["slab_budget"])
        p = out["proj"]
        p.mean2d.retain_grad()
        loss, parts = photometric_loss(out["image"], gt, tc["lambda_dssim"])
        loss.backward()

        with torch.no_grad():
            if it < dc["stop"] and p.mean2d.grad is not None:
                g.add_densify_stats(p.mean2d.grad, p.extent, cam.width, cam.height)
            stats = None
            if dc["start"] < it < dc["stop"] and it % dc["every"] == 0:
                max_screen = dc["max_screen"] if it > dc["screen_size_after"] else None
                stats = g.densify_and_prune(dc["grad_threshold"], extent, dc["percent_dense"], dc["min_opacity"],
                                            max_screen, tc["max_gaussians"])
            elif dc["opacity_reset_every"] and it % dc["opacity_reset_every"] == 0 and it < dc["stop"]:
                g.reset_opacity()
            else:
                g.optimizer.step()
            g.optimizer.zero_grad(set_to_none=True)
            if it in compact_at and g.num > 1000:
                imp = importance(g, bank.cams, bg_const, tc)
                k = max(1, int(g.num * cc["keep"]))
                keep = torch.zeros(g.num, dtype=torch.bool, device=dev)
                keep[imp.topk(k).indices] = True
                n0 = g.num
                g.prune(keep)
                stats = {"compacted": n0 - g.num, "total": g.num}

        l, ps = float(loss.detach()), psnr(out["image"].detach(), gt)
        ema_loss = l if ema_loss is None else 0.9 * ema_loss + 0.1 * l
        ema_psnr = ps if ema_psnr is None else 0.9 * ema_psnr + 0.1 * ps
        if it % 10 == 0 or it == 1 or stats or it == iters:
            rec = {"it": it, "loss": round(ema_loss, 5), "psnr": round(ema_psnr, 3), "n": g.num,
                   "sh": g.active_sh_degree, "elapsed": round(time.time() - t0, 1), **parts}
            if stats:
                rec["densify"] = stats
            log.write(json.dumps(rec) + "\n")
            if it % 50 == 0 or it == iters or it == 1:
                log.flush()
                eta = (time.time() - t0) / it * (iters - it)
                say({"stage": "train", "iteration": it, "iterations": iters, "loss": rec["loss"],
                     "psnr": rec["psnr"], "gaussians": g.num, "eta_s": round(eta),
                     "message": f"iteration {it}/{iters}  PSNR {ema_psnr:.2f}  {g.num:,} Gaussians"})
        if should_stop and should_stop():
            stopped = True
            break
    log.close()
    g.save_ply(out_dir / "point_cloud.ply")
    summary = {"iterations": it, "stopped_early": stopped, "gaussians": g.num, "train_seconds": round(time.time() - t0, 1),
               "train_psnr_ema": round(ema_psnr or 0, 3), "device": str(dev),
               "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
               "peak_mem_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 2) if dev.type == "cuda" else None,
               "train_resolution": [scene.train[0].width, scene.train[0].height], "scene": scene.name}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    say({"stage": "train", "iteration": it, "iterations": iters, "message": "training finished", "done": True})
    return summary


def _save_cameras(scene: SceneData, path: Path) -> None:
    f = scene.frame
    path.write_text(json.dumps({
        "scene": scene.name, "kind": scene.kind, "white_background": scene.white_background,
        "frame": {"center": list(map(float, f["center"])), "extent": float(f["extent"]),
                  "up": list(map(float, f["up"])), "target": list(map(float, f["target"]))},
        "train": [c.to_dict() | {"image_path": c.image_path} for c in scene.train],
        "test": [c.to_dict() | {"image_path": c.image_path} for c in scene.test],
    }))
