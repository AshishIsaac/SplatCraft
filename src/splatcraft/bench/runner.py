"""``splatcraft bench``: the PSNR / SSIM / LPIPS vs file size vs WebGPU FPS table.

For every scene: train (or reuse) a run, evaluate the uncompressed .ply, encode several .splatc
variants (a rate-distortion sweep plus an entropy-coder ablation), evaluate each decoded file on the
held-out views, then time every variant in the browser on every GPU available. Writes
``benchmarks/results/benchmark.json`` and ``benchmarks/results/BENCHMARK.md``.
"""

from __future__ import annotations

import copy
import json
import platform
import shutil
import time
from pathlib import Path

import torch

from ..compress.codec import decode_scene, encode_scene
from ..config import load_config, set_dotted
from ..eval.metrics import evaluate, load_run_cameras
from ..gs.gaussians import GaussianModel
from ..gs.trainer import importance, pick_device
from ..paths import REFERENCE_DIR, data_dir, home, runs_dir
from ..pipeline import run_config, train_run, view_meta

# name -> (compress overrides, evaluate quality?)  "size-only" variants decode to the same model.
VARIANTS = {
    "splatc": ({}, True),
    "splatc-small": ({"prune_keep": 0.6, "sh_codebook": 2048}, True),
    "splatc-tiny": ({"prune_keep": 0.35, "sh_codebook": 1024}, True),
    "splatc-deflate": ({"entropy": "deflate"}, False),
    "splatc-no-entropy": ({"entropy": "none"}, False),
}
SIZES = ((1280, 720), (1920, 1080))
# Per-scene settings of the committed benchmark runs (a 6 GB GPU; lego converges with ~300k Gaussians).
SCENE_SET = {"lego": ["train.max_gaussians=300000"]}


def _row(scene, variant, m: dict | None, nbytes: int, n: int) -> dict:
    r = {"scene": scene, "variant": variant, "bytes": nbytes, "gaussians": n, "fps": {}}
    if m:
        r.update({k: round(m[k], 4) for k in ("psnr", "ssim", "lpips") if k in m})
    return r


def bench_scene(scene: str, preset: str, train: bool, log=print) -> tuple[list[dict], dict]:
    cfg = set_dotted(load_config(preset), SCENE_SET.get(scene, []))
    run = runs_dir() / f"{scene}_{preset}"
    if not (run / "point_cloud.ply").is_file():
        if not train:
            raise SystemExit(f"{run} has no trained model; run without --no-train")
        log(f"== training {scene} ({preset})")
        train_run(data_dir() / scene, cfg, run, progress=lambda d: d.get("iteration", 0) % 2000 == 0
                  and d.get("message") and log("   " + d["message"]))
    cfg = run_config(run)
    dev = pick_device()
    summary = json.loads((run / "train_summary.json").read_text())
    train_cams, test, meta = load_run_cameras(run)
    ec = cfg["eval"]
    g = GaussianModel.load_ply(run / "point_cloud.ply", dev)
    log(f"== {scene}: evaluating .ply ({g.num:,} Gaussians)")
    m = evaluate(g, test, meta["white_background"], ec["lpips_net"], ec["max_test_views"], log=log)
    rows = [_row(scene, "3dgs-ply", m, (run / "point_cloud.ply").stat().st_size, g.num)]
    bg = torch.ones(3, device=dev) if meta["white_background"] else torch.zeros(3, device=dev)
    imp = importance(g, train_cams, bg, cfg["train"])
    vdir = run / "variants"
    vdir.mkdir(exist_ok=True)
    published = []
    for name, (over, quality) in VARIANTS.items():
        cc = {**copy.deepcopy(cfg["compress"]), **over}
        t0 = time.time()
        data, rep = encode_scene(g, cc, imp, view_meta(meta), log=lambda *_: None)
        (vdir / f"{name}.splatc").write_bytes(data)
        log(f"   {name}: {len(data) / 1e6:.2f} MB ({rep['ratio_vs_ply']}x) in {time.time() - t0:.0f} s")
        mm = None
        if quality:
            gd = decode_scene(data, dev)
            mm = evaluate(gd, test, meta["white_background"], ec["lpips_net"], ec["max_test_views"], log=log)
        row = _row(scene, name, mm, len(data), rep["gaussians"])
        row["ratio_vs_ply"] = round(rows[0]["bytes"] / len(data), 2)
        rows.append(row)
        if quality:
            pub = home() / "scenes" / f"bench-{scene}-{name}.splatc"
            pub.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(vdir / f"{name}.splatc", pub)
            published.append((pub.stem, row))
    info = {"scene": scene, "preset": preset, "iterations": summary["iterations"],
            "train_minutes": round(summary["train_seconds"] / 60, 1), "train_gpu": summary.get("gpu"),
            "train_resolution": summary.get("train_resolution"), "test_views": m["views"],
            "eval_resolution": m["resolution"], "lpips_net": ec["lpips_net"], "published": published}
    return rows, info


def write_tables(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(result, indent=1))
    devs = result["devices"]
    lines = ["# SplatCraft benchmark", "",
             f"Generated {result['date']} by `splatcraft bench{' --quick' if result['preset'] == 'quick' else ''}`. "
             "Quality is measured on held-out test views after decoding each file. FPS = 120 frames of a "
             "360-degree orbit rendered back to back by the WebGPU viewer (at most 2 frames in flight, not "
             "capped by the display) / wall-clock time. On a fast discrete GPU this is bounded by the "
             "browser's per-frame submission cost; the GPU-time table below shows the actual GPU work.", ""]
    for s in result["scenes"]:
        lines.append(f"- **{s['scene']}**: {s['iterations']:,} iterations ({s['train_minutes']} min on "
                     f"{s['train_gpu']}), trained at {s['train_resolution'][0]}x{s['train_resolution'][1]}, "
                     f"{s['test_views']} test views, LPIPS-{s['lpips_net']}")
    if result.get("gpu_names"):
        lines.append("- GPUs: " + "; ".join(f"`{k}` = {v}" for k, v in result["gpu_names"].items()))
    lines.append("")
    hdr = ["Scene", "File", "Size", "vs .ply", "Gaussians", "PSNR ↑", "SSIM ↑", "LPIPS ↓"] + [f"FPS {d}" for d in devs]
    lines += ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * 2 + ["---:"] * (len(hdr) - 2)) + "|"]
    for r in result["rows"]:
        def f(k, fmt, r=r):
            return fmt.format(r[k]) if r.get(k) is not None else "–"

        cells = [r["scene"], f"`{r['variant']}`", f"{r['bytes'] / 1e6:.2f} MB",
                 f"{r.get('ratio_vs_ply', 1.0):.1f}x", f"{r['gaussians']:,}", f("psnr", "{:.2f}"),
                 f("ssim", "{:.3f}"), f("lpips", "{:.3f}")] + [str(r["fps"].get(d, "–")) for d in devs]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "FPS columns: `<GPU> @ <width>x<height>`. `splatc-deflate` / `splatc-no-entropy` decode to "
              "exactly the same scene as `splatc` (entropy-coder ablation: size only).", ""]
    if result.get("browser_runs"):
        lines += ["## Per-stage GPU time (ms, WebGPU timestamp queries)", "",
                  "| GPU | Scene | Resolution | preprocess | scan | emit | radix sort | ranges | raster | total |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for b in result["browser_runs"]:
            st = b.get("stage_ms") or {}
            if not st:
                continue
            tot = sum(st.values())
            lines.append(f"| {b['gpu']} | {b['scene']} | {b['width']}x{b['height']} | "
                         + " | ".join(f"{st.get(k, 0):.2f}" for k in ("preprocess", "scan", "emit", "sort", "ranges", "raster"))
                         + f" | {tot:.2f} |")
    (out_dir / "BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmarks(quick: bool = False, scenes=("lego", "bonsai"), browser: bool = True, train: bool = True,
                   out_dir: Path | None = None, log=print) -> int:
    preset = "quick" if quick else "full"
    rows, infos = [], []
    for s in scenes:
        r, i = bench_scene(s, preset, train, log)
        rows += r
        infos.append(i)
    runs = []
    devices = []
    if browser:
        from .browser import measure_fps

        names = [n for i in infos for n, _ in i["published"]]
        log(f"== WebGPU frame rate: {len(names)} files x {len(SIZES)} resolutions")
        runs = measure_fps(names, SIZES, log=log)
        by_name = {n: row for i in infos for n, row in i["published"]}
        for b in runs:
            dev = f"{b['gpu']} @ {b['width']}x{b['height']}"
            if dev not in devices:
                devices.append(dev)
            by_name[b["scene"]]["fps"][dev] = b["fps"]
            b["scene"] = b["scene"].replace("bench-", "")
    for i in infos:
        i.pop("published")
    result = {"date": time.strftime("%Y-%m-%d"), "preset": preset, "platform": platform.platform(),
              "torch": torch.__version__, "devices": devices, "scenes": infos, "rows": rows, "browser_runs": runs}
    out = Path(out_dir or REFERENCE_DIR)
    write_tables(result, out)
    log(f"wrote {out / 'BENCHMARK.md'}")
    return 0
