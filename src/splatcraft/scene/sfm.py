"""Capture -> posed images: frame extraction from a video, then COLMAP structure-from-motion.

Pose seeding is deliberately off-the-shelf (pycolmap: SIFT, sequential + exhaustive matching,
incremental mapping, undistortion to PINHOLE). The output is a standard COLMAP scene folder::

    <scene>/images/*.jpg      undistorted frames
    <scene>/sparse/0/*.bin    PINHOLE cameras, poses, sparse points (the 3DGS initialisation)
"""

from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp"}


def sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (min(640, g.shape[1]), int(g.shape[0] * min(640, g.shape[1]) / g.shape[1])))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def extract_frames(video: Path, out_dir: Path, target: int = 120, max_side: int = 1600, log=print) -> int:
    """Pick ``target`` frames spread evenly over the video, the sharpest one in each time window."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out_dir.mkdir(parents=True, exist_ok=True)
    if total <= 0:  # some containers do not report a frame count: read everything once
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        total = len(frames)
        reader = lambda i: frames[i]  # noqa: E731
    else:
        frames = None

        def reader(i):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            return f if ok else None
    target = max(8, min(target, total))
    edges = np.linspace(0, total, target + 1).astype(int)
    log(f"video: {total} frames at {fps:.0f} fps ({total / fps:.0f} s) -> {target} sharpest frames")
    n = 0
    for k in range(target):
        lo, hi = edges[k], max(edges[k] + 1, edges[k + 1])
        cand = np.unique(np.linspace(lo, hi - 1, min(5, hi - lo)).astype(int))  # up to 5 candidates per window
        best, best_s = None, -1.0
        for i in cand:
            f = reader(int(i))
            if f is None:
                continue
            s = sharpness(f)
            if s > best_s:
                best, best_s = f, s
        if best is None:
            continue
        h, w = best.shape[:2]
        sc = min(1.0, max_side / max(h, w))
        if sc < 1:
            best = cv2.resize(best, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"frame_{n:04d}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    return n


def gather_images(src: Path, out_dir: Path, max_images: int = 200, max_side: int = 1600) -> int:
    """A folder or .zip of photos -> resized copies in ``out_dir`` (evenly subsampled to ``max_images``)."""
    tmp = None
    if src.suffix.lower() == ".zip":
        tmp = out_dir.parent / "_unzip"
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        src = tmp
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT and not p.name.startswith("._"))
    if not files:
        raise ValueError(f"no images found in {src}")
    if len(files) > max_images:
        files = [files[i] for i in np.linspace(0, len(files) - 1, max_images).round().astype(int)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        im = cv2.imread(str(f))
        if im is None:
            continue
        h, w = im.shape[:2]
        sc = min(1.0, max_side / max(h, w))
        if sc < 1:
            im = cv2.resize(im, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"img_{i:04d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return len(list(out_dir.glob("*.jpg")))


def run_sfm(image_dir: Path, work: Path, sequential: bool = True, log=print) -> Path:
    """pycolmap SfM on ``image_dir``; returns the scene folder (undistorted images + sparse/0)."""
    import pycolmap

    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    n_img = len(list(image_dir.glob("*")))
    t0 = time.time()
    log(f"sfm: SIFT features on {n_img} images")
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = 8192
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext)
    if sequential and n_img > 60:
        log("sfm: sequential matching (video order, 20 neighbours + quadratic overlap)")
        pair = pycolmap.SequentialPairingOptions()
        pair.overlap = 20
        pair.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=pair)
    else:
        log("sfm: exhaustive matching")
        pycolmap.match_exhaustive(db)
    log("sfm: incremental mapping")
    sparse = work / "sparse_raw"
    sparse.mkdir(exist_ok=True)
    recs = pycolmap.incremental_mapping(db, image_dir, sparse)
    if not recs:
        raise RuntimeError("structure-from-motion failed: no images could be registered. Use a slower, "
                           "steadier walk-around with more texture and overlap.")
    best_id, best = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    log(f"sfm: registered {best.num_reg_images()}/{n_img} images, {best.num_points3D()} points "
        f"({len(recs)} model(s), {time.time() - t0:.0f} s)")
    if best.num_reg_images() < max(6, 0.3 * n_img):
        raise RuntimeError(f"structure-from-motion registered only {best.num_reg_images()} of {n_img} images")
    scene = work / "scene"
    if scene.exists():
        shutil.rmtree(scene)
    log("sfm: undistorting to PINHOLE")
    pycolmap.undistort_images(scene, sparse / str(best_id), image_dir)
    (scene / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for f in (scene / "sparse").glob("*.bin"):
        f.replace(scene / "sparse" / "0" / f.name)
    return scene


def prepare_capture(src: Path, out: Path, frames: int = 120, log=print) -> Path:
    """Video / photo folder / zip -> COLMAP scene folder ready for training."""
    src, out = Path(src), Path(out)
    raw = out / "frames"
    if raw.exists():
        shutil.rmtree(raw)
    if src.is_file() and src.suffix.lower() in VIDEO_EXT:
        extract_frames(src, raw, frames, log=log)
        seq = True
    else:
        gather_images(src, raw, max_images=max(frames, 8))
        seq = False
    return run_sfm(raw, out, sequential=seq, log=log)
