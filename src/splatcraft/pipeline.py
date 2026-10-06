"""High-level steps on a run directory: train -> compress -> evaluate -> publish.

A run directory holds::

    config.yaml  cameras.json  train_log.jsonl  train_summary.json
    point_cloud.ply          the trained scene (standard 3DGS layout)
    scene.splatc             the compressed, streamable scene
    compress_report.json     sizes per attribute, ratio vs .ply
    metrics.json             PSNR / SSIM / LPIPS of the .ply and of the decoded .splatc
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .compress.codec import decode_scene, encode_scene
from .eval.metrics import evaluate, load_run_cameras
from .gs.gaussians import GaussianModel
from .gs.trainer import importance, pick_device, train
from .paths import home
from .scene.dataset import load_scene


def run_config(run_dir: Path) -> dict:
    return yaml.safe_load((Path(run_dir) / "config.yaml").read_text())


def train_run(data: Path, cfg: dict, run_dir: Path, device: str | None = None, progress=None, should_stop=None) -> dict:
    d = cfg["data"]
    scene = load_scene(data, d["max_width"], d["test_every"], d.get("max_images", 0), d.get("random_init", 100_000))
    for w in scene.warnings:
        print("warning:", w)
    return train(scene, cfg, run_dir, device, progress, should_stop)


def view_meta(meta: dict, n_views: int = 12) -> dict:
    """Default viewer camera + a few capture poses, stored in the .splatc header."""
    f = meta["frame"]
    train = meta["train"]
    c0 = train[0]
    R, t = np.asarray(c0["R"]), np.asarray(c0["t"])
    eye = (-R.T @ t).tolist()
    idx = np.linspace(0, len(train) - 1, min(n_views, len(train))).round().astype(int)
    keep = ("R", "t", "fx", "fy", "cx", "cy", "width", "height", "name")
    return {"target": f["target"], "up": f["up"], "eye": eye, "extent": f["extent"],
            "fov_y": 2 * math.atan(c0["height"] / (2 * c0["fy"])),
            "background": [1, 1, 1] if meta["white_background"] else [0, 0, 0],
            "cameras": [{k: train[i][k] for k in keep} for i in sorted(set(idx.tolist()))]}


def compress_run(run_dir: Path, cfg: dict | None = None, out: Path | None = None, device: str | None = None,
                 log=print) -> dict:
    run_dir = Path(run_dir)
    cfg = cfg or run_config(run_dir)
    dev = pick_device(device)
    g = GaussianModel.load_ply(run_dir / "point_cloud.ply", dev)
    train_cams, _, meta = load_run_cameras(run_dir)
    bg = torch.ones(3, device=dev) if meta["white_background"] else torch.zeros(3, device=dev)
    t0 = time.time()
    log(f"compress: importance over {len(train_cams)} training views")
    imp = importance(g, train_cams, bg, cfg["train"])
    data, report = encode_scene(g, cfg["compress"], imp, view_meta(meta), log=log)
    out = Path(out or run_dir / "scene.splatc")
    out.write_bytes(data)
    report["seconds"] = round(time.time() - t0, 1)
    report["ply_file_bytes"] = (run_dir / "point_cloud.ply").stat().st_size
    report["ratio_vs_ply_file"] = round(report["ply_file_bytes"] / len(data), 2)
    (run_dir / "compress_report.json").write_text(json.dumps(report, indent=2))
    log(f"compress: {report['ply_file_bytes'] / 1e6:.1f} MB .ply -> {len(data) / 1e6:.2f} MB .splatc "
        f"({report['ratio_vs_ply_file']}x, {report['bytes_per_gaussian']} B/Gaussian)")
    return report


def evaluate_run(run_dir: Path, cfg: dict | None = None, device: str | None = None, which=("ply", "splatc"),
                 log=print) -> dict:
    run_dir = Path(run_dir)
    cfg = cfg or run_config(run_dir)
    dev = pick_device(device)
    _, test, meta = load_run_cameras(run_dir)
    ec = cfg["eval"]
    res = {}
    if "ply" in which and (run_dir / "point_cloud.ply").is_file():
        g = GaussianModel.load_ply(run_dir / "point_cloud.ply", dev)
        res["ply"] = evaluate(g, test, meta["white_background"], ec["lpips_net"], ec["max_test_views"],
                              run_dir / "renders" / "ply", log=log)
        res["ply"]["bytes"] = (run_dir / "point_cloud.ply").stat().st_size
    if "splatc" in which and (run_dir / "scene.splatc").is_file():
        g = decode_scene((run_dir / "scene.splatc").read_bytes(), dev)
        res["splatc"] = evaluate(g, test, meta["white_background"], ec["lpips_net"], ec["max_test_views"],
                                 run_dir / "renders" / "splatc", log=log)
        res["splatc"]["bytes"] = (run_dir / "scene.splatc").stat().st_size
    res["lpips_net"] = ec["lpips_net"]
    old = {}
    if (run_dir / "metrics.json").is_file():
        old = json.loads((run_dir / "metrics.json").read_text())
    (run_dir / "metrics.json").write_text(json.dumps({**old, **res}, indent=2))
    return res


def publish(run_dir: Path, name: str, dest: Path | None = None) -> Path:
    """Copy a run's .splatc where the server (and the static viewer) serve scenes from."""
    src = Path(run_dir) / "scene.splatc"
    if not src.is_file():
        raise FileNotFoundError(f"{src} does not exist: run `splatcraft compress {run_dir}` first")
    dest = Path(dest or home() / "scenes")
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{name}.splatc"
    shutil.copyfile(src, out)
    return out
