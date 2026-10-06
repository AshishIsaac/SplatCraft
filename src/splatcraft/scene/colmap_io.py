"""Reader for COLMAP sparse models (binary or text): cameras, registered images, 3D points.

Self-contained (numpy only) so that training never needs pycolmap; pycolmap is only used to *run*
structure-from-motion on new videos (``scene/sfm.py``).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .cameras import Camera, qvec_to_rotmat

# model_id -> (name, number of params)
CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
                 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
                 8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12)}


def _intrinsics(model: str, params) -> tuple[float, float, float, float, bool]:
    """-> fx, fy, cx, cy, has_distortion."""
    p = list(params)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"):
        return p[0], p[0], p[1], p[2], model not in ("SIMPLE_PINHOLE",) and any(abs(v) > 1e-9 for v in p[3:])
    return p[0], p[1], p[2], p[3], model != "PINHOLE" and any(abs(v) > 1e-9 for v in p[4:])


def _read(f, fmt: str):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras_binary(path) -> dict:
    cams = {}
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            cid, model_id, w, h = _read(f, "iiQQ")
            name, n = CAMERA_MODELS[model_id]
            cams[cid] = {"model": name, "width": w, "height": h, "params": _read(f, "d" * n)}
    return cams


def read_images_binary(path) -> dict:
    imgs = {}
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _read(f, "idddddddi")
            name = b""
            while (ch := f.read(1)) != b"\x00":
                name += ch
            n2d = _read(f, "Q")[0]
            f.seek(24 * n2d, 1)  # skip the 2D observations (x, y, point3D_id)
            imgs[iid] = {"q": (qw, qx, qy, qz), "t": (tx, ty, tz), "camera_id": cid, "name": name.decode()}
    return imgs


def read_points_binary(path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        n = _read(f, "Q")[0]
        xyz = np.empty((n, 3))
        rgb = np.empty((n, 3), np.uint8)
        for i in range(n):
            vals = _read(f, "QdddBBBd")
            xyz[i] = vals[1:4]
            rgb[i] = vals[4:7]
            f.seek(8 * _read(f, "Q")[0], 1)  # track
    return xyz, rgb


def read_cameras_text(path) -> dict:
    cams = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            p = line.split()
            cams[int(p[0])] = {"model": p[1], "width": int(p[2]), "height": int(p[3]),
                               "params": tuple(map(float, p[4:]))}
    return cams


def read_images_text(path) -> dict:
    imgs = {}
    lines = [ln for ln in Path(path).read_text().splitlines() if not ln.startswith("#")]
    for i in range(0, len(lines) - 1, 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        imgs[int(p[0])] = {"q": tuple(map(float, p[1:5])), "t": tuple(map(float, p[5:8])), "camera_id": int(p[8]),
                           "name": " ".join(p[9:])}
    return imgs


def read_points_text(path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            p = line.split()
            xyz.append(list(map(float, p[1:4])))
            rgb.append(list(map(int, p[4:7])))
    return np.asarray(xyz, float).reshape(-1, 3), np.asarray(rgb, np.uint8).reshape(-1, 3)


def find_model(root: Path) -> Path:
    for cand in (root / "sparse" / "0", root / "sparse", root):
        if (cand / "cameras.bin").is_file() or (cand / "cameras.txt").is_file():
            return cand
    raise FileNotFoundError(f"no COLMAP model (sparse/0/cameras.bin) under {root}")


def read_model(root: Path, images_dir: Path) -> tuple[list[Camera], np.ndarray, np.ndarray, list[str]]:
    """-> cameras (sorted by image name), points xyz, points rgb (0..1), warnings."""
    m = find_model(root)
    if (m / "cameras.bin").is_file():
        cams, imgs = read_cameras_binary(m / "cameras.bin"), read_images_binary(m / "images.bin")
        xyz, rgb = read_points_binary(m / "points3D.bin") if (m / "points3D.bin").is_file() else (None, None)
    else:
        cams, imgs = read_cameras_text(m / "cameras.txt"), read_images_text(m / "images.txt")
        xyz, rgb = read_points_text(m / "points3D.txt") if (m / "points3D.txt").is_file() else (None, None)
    warnings = []
    out = []
    for im in sorted(imgs.values(), key=lambda d: d["name"]):
        c = cams[im["camera_id"]]
        fx, fy, cx, cy, dist = _intrinsics(c["model"], c["params"])
        if dist:
            warnings.append(f"{im['name']}: {c['model']} distortion ignored (undistort the images for best quality)")
        out.append(Camera(qvec_to_rotmat(im["q"]), np.asarray(im["t"], float), fx, fy, cx, cy, c["width"],
                          c["height"], im["name"], str(images_dir / im["name"])))
    if xyz is None or len(xyz) == 0:
        xyz, rgb = np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    return out, xyz, rgb.astype(np.float32) / 255.0, sorted(set(warnings))[:3]
