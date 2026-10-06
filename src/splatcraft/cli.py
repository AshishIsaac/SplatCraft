"""``splatcraft`` command line.

    splatcraft data download [lego|bonsai|all]
    splatcraft capture VIDEO|FOLDER|ZIP --name NAME        frames + COLMAP poses
    splatcraft train DATA [-c quick|full|smoke] [--name RUN] [--set train.iterations=3000]
    splatcraft compress RUN [--set compress.sh_codebook=2048]
    splatcraft eval RUN
    splatcraft publish RUN NAME                           make it viewable at /v/NAME
    splatcraft process VIDEO --name NAME                  capture -> train -> compress -> publish
    splatcraft bench [--quick]                            PSNR/SSIM/LPIPS x size x FPS table
    splatcraft serve [--port 8770]                        upload page, viewer, share links
    splatcraft doctor                                     check the installation (10 s smoke test)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import __version__
from .config import load_config, set_dotted
from .paths import data_dir, home, runs_dir


def _resolve_data(arg: str) -> Path:
    p = Path(arg)
    for cand in (p, data_dir() / arg, home() / "captures" / arg / "scene"):
        if cand.is_dir():
            return cand
    raise SystemExit(f"no scene found for {arg!r} (tried {p}, {data_dir() / arg}, captures/{arg}/scene). "
                     "Download one with `splatcraft data download lego` or capture with `splatcraft capture`.")


def _resolve_run(arg: str) -> Path:
    p = Path(arg)
    if (p / "cameras.json").is_file():
        return p
    if (runs_dir() / arg / "cameras.json").is_file():
        return runs_dir() / arg
    raise SystemExit(f"no run directory {arg!r} (looked in . and {runs_dir()})")


def _progress(d: dict) -> None:
    msg = d.get("message")
    if msg and (d.get("stage") != "train" or d.get("iteration", 0) % 500 == 0 or d.get("done")):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_data(a) -> int:
    from .scene.download import SCENES, download_scene

    names = list(SCENES) if a.name == "all" else [a.name]
    for n in names:
        download_scene(n)
    return 0


def cmd_models(a) -> int:
    import torch

    from .eval.metrics import lpips_model

    for net in ("vgg", "alex"):
        lpips_model(net, torch.device("cpu"))
        print(f"LPIPS ({net}) ready")
    return 0


def cmd_capture(a) -> int:
    from .scene.sfm import prepare_capture

    out = home() / "captures" / a.name
    scene = prepare_capture(Path(a.source), out, frames=a.frames)
    print(f"scene ready: {scene}\nnext: splatcraft train {a.name}")
    return 0


def cmd_train(a) -> int:
    from .pipeline import train_run

    cfg = set_dotted(load_config(a.config), a.set)
    data = _resolve_data(a.data)
    name = a.name or f"{data.parent.name if data.name == 'scene' else data.name}_{cfg.get('preset', 'custom')}"
    out = runs_dir() / name
    summary = train_run(data, cfg, out, a.device, progress=_progress)
    print(json.dumps(summary, indent=1))
    if not a.no_compress:
        return cmd_compress(argparse.Namespace(run=str(out), set=a.set, config=None, device=a.device,
                                               publish=a.publish, eval=a.eval))
    return 0


def cmd_compress(a) -> int:
    from .pipeline import compress_run, evaluate_run, publish, run_config

    run = _resolve_run(a.run)
    cfg = set_dotted(run_config(run), a.set)
    compress_run(run, cfg, device=a.device)
    if a.eval:
        evaluate_run(run, cfg, a.device)
    if a.publish:
        print(f"published: {publish(run, a.publish)}  ->  open /v/{a.publish} in `splatcraft serve`")
    return 0


def cmd_eval(a) -> int:
    from .pipeline import evaluate_run, run_config

    run = _resolve_run(a.run)
    res = evaluate_run(run, set_dotted(run_config(run), a.set), a.device)
    print(json.dumps({k: {m: v[m] for m in ("psnr", "ssim", "lpips", "gaussians", "bytes") if m in v}
                      for k, v in res.items() if isinstance(v, dict)}, indent=1))
    return 0


def cmd_publish(a) -> int:
    from .pipeline import publish

    print(publish(_resolve_run(a.run), a.name))
    return 0


def cmd_process(a) -> int:
    from .pipeline import compress_run, publish, train_run
    from .scene.sfm import prepare_capture

    cfg = set_dotted(load_config(a.config), a.set)
    scene = prepare_capture(Path(a.source), home() / "captures" / a.name, frames=a.frames)
    run = runs_dir() / a.name
    train_run(scene, cfg, run, a.device, progress=_progress)
    compress_run(run, cfg, device=a.device)
    print(f"published: {publish(run, a.name)}")
    return 0


def cmd_serve(a) -> int:
    from .server.app import serve

    serve(a.host, a.port, open_browser=not a.no_browser)
    return 0


def cmd_bench(a) -> int:
    from .bench.runner import run_benchmarks

    return run_benchmarks(quick=a.quick, scenes=a.scenes, browser=not a.no_browser, train=not a.no_train)


def cmd_doctor(a) -> int:
    from .doctor import doctor

    return doctor(full=a.full)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="splatcraft", description="Train, compress and stream 3D Gaussian Splatting scenes.")
    ap.add_argument("--version", action="version", version=f"splatcraft {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, run=False):
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. train.iterations=3000")
        p.add_argument("--device", default=None, help="cuda | cpu (default: cuda when available)")
        if not run:
            p.add_argument("-c", "--config", default="quick", help="preset: smoke | quick | full | path.yaml")

    p = sub.add_parser("data", help="download benchmark scenes")
    p.add_argument("action", choices=["download"])
    p.add_argument("name", nargs="?", default="all", choices=["lego", "bonsai", "all"])
    p.set_defaults(fn=cmd_data)

    p = sub.add_parser("models", help="download the LPIPS weights")
    p.add_argument("action", choices=["download"])
    p.set_defaults(fn=cmd_models)

    p = sub.add_parser("capture", help="video / photos -> COLMAP scene")
    p.add_argument("source")
    p.add_argument("--name", required=True)
    p.add_argument("--frames", type=int, default=120)
    p.set_defaults(fn=cmd_capture)

    p = sub.add_parser("train", help="train a scene (then compress it)")
    p.add_argument("data", help="scene folder, downloaded scene name, or capture name")
    p.add_argument("--name", help="run name (default: <scene>_<preset>)")
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--eval", action="store_true", help="also compute PSNR / SSIM / LPIPS on held-out views")
    p.add_argument("--publish", metavar="NAME", help="publish the compressed scene as NAME")
    common(p)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("compress", help="(re)compress a trained run")
    p.add_argument("run")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--publish", metavar="NAME")
    common(p, run=True)
    p.set_defaults(fn=cmd_compress)

    p = sub.add_parser("eval", help="PSNR / SSIM / LPIPS of a run (.ply and .splatc)")
    p.add_argument("run")
    common(p, run=True)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("publish", help="serve a run's .splatc under a name")
    p.add_argument("run")
    p.add_argument("name")
    p.set_defaults(fn=cmd_publish)

    p = sub.add_parser("process", help="video -> trained, compressed, published scene")
    p.add_argument("source")
    p.add_argument("--name", required=True)
    p.add_argument("--frames", type=int, default=120)
    common(p)
    p.set_defaults(fn=cmd_process)

    p = sub.add_parser("serve", help="web app: upload, progress, viewer, share links")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("SPLATCRAFT_PORT", 8770)))
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("bench", help="build the benchmark table")
    p.add_argument("--quick", action="store_true", help="7k-iteration models instead of the 30k paper schedule")
    p.add_argument("--scenes", nargs="*", default=["lego", "bonsai"])
    p.add_argument("--no-browser", action="store_true", help="skip the WebGPU FPS measurements")
    p.add_argument("--no-train", action="store_true", help="reuse existing runs")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("doctor", help="check the installation")
    p.add_argument("--full", action="store_true", help="also check the datasets and a browser")
    p.set_defaults(fn=cmd_doctor)

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
