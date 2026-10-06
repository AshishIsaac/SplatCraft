"""End to end on a tiny synthetic NeRF-format scene (no downloads): load -> train -> compress -> eval."""

import json
import math

import cv2
import numpy as np
import torch

from splatcraft.config import load_config, set_dotted
from splatcraft.gs.gaussians import GaussianModel
from splatcraft.gs.rasterizer import render
from splatcraft.scene.cameras import Camera


def _make_scene(root, n_train=8, n_test=2, size=48):
    """Render a random Gaussian 'object' from a ring of cameras into a transforms_*.json scene."""
    rng = np.random.default_rng(0)
    gt = GaussianModel.from_points(rng.normal(0, 0.3, (300, 3)), rng.uniform(0, 1, (300, 3)), 0, "cpu", 0.9)
    fov = 0.7
    for split, n, off in (("train", n_train, 0.0), ("test", n_test, 0.4)):
        (root / split).mkdir(parents=True)
        frames = []
        for i in range(n):
            a = 2 * math.pi * i / n + off
            eye = np.array([3 * math.cos(a), 3 * math.sin(a), 1.0])
            cam = Camera.look_at(eye, [0, 0, 0], [0, 0, 1], fov, size, size)  # z up, as in Blender
            with torch.no_grad():
                img = render(gt, cam, torch.ones(3))["image"].clamp(0, 1).numpy()
            cv2.imwrite(str(root / split / f"r_{i}.png"), (img[..., ::-1] * 255).round().astype(np.uint8))
            c2w = np.eye(4)
            c2w[:3, :3] = cam.R.T
            c2w[:3, 3] = eye
            c2w[:3, 1:3] *= -1  # OpenCV -> OpenGL, as stored by Blender
            frames.append({"file_path": f"./{split}/r_{i}", "transform_matrix": c2w.tolist()})
        (root / f"transforms_{split}.json").write_text(json.dumps({"camera_angle_x": 2 * math.atan(math.tan(fov / 2)),
                                                                 "frames": frames}))
    return root


def test_train_compress_eval(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATCRAFT_HOME", str(tmp_path / "home"))
    from splatcraft.pipeline import compress_run, evaluate_run, publish, train_run

    data = _make_scene(tmp_path / "toy")
    cfg = set_dotted(load_config("smoke"), ["data.max_width=48", "data.random_init=1500", "train.iterations=120",
                                            "densify.start=20", "densify.stop=100", "densify.every=20",
                                            "compaction.at=[90]", "eval.lpips_net=null", "compress.sh_codebook=32"])
    run = tmp_path / "run"
    summary = train_run(data, cfg, run, device="cpu")
    assert summary["iterations"] == 120 and (run / "point_cloud.ply").is_file()
    rep = compress_run(run, cfg, device="cpu", log=lambda *_: None)
    assert rep["ratio_vs_ply_file"] > 3
    res = evaluate_run(run, cfg, device="cpu", log=lambda *_: None)
    assert res["ply"]["psnr"] > 18, res["ply"]["psnr"]  # learned something (blank image scores ~12)
    assert res["splatc"]["psnr"] > res["ply"]["psnr"] - 1.5  # compression costs little
    assert publish(run, "toy").is_file()


def test_blender_loader_roundtrips_poses(tmp_path):
    from splatcraft.scene.dataset import load_scene

    _make_scene(tmp_path / "toy", 4, 1)
    sc = load_scene(tmp_path / "toy", 48)
    assert sc.kind == "blender" and sc.white_background and len(sc.train) == 4 and len(sc.test) == 1
    # every camera looks at the origin: it projects to the image centre
    for cam in sc.train:
        pc = cam.R @ np.zeros(3) + cam.t
        assert pc[2] > 0
        assert abs(cam.fx * pc[0] / pc[2] + cam.cx - 24) < 0.5 and abs(cam.fy * pc[1] / pc[2] + cam.cy - 24) < 0.5
