"""Benchmark scenes from public Hugging Face dataset mirrors (no account or token needed).

* ``lego``   NeRF-synthetic (Mildenhall et al. 2020): 100 train / 200 test views, known poses, ~130 MB.
* ``bonsai`` Mip-NeRF 360 (Barron et al. 2022): a real indoor capture with COLMAP poses and SfM points,
             292 photos, ~370 MB.

Downloads resume (files already present with the right size are skipped) and run in parallel.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..paths import data_dir

SCENES = {
    "lego": {"repo": "pablovela5620/nerf-synthetic-mirror", "prefix": "lego",
             "include": ["transforms_train.json", "transforms_test.json", "train/", "test/"],
             "exclude": ["_depth_", "_normal_"], "about": "NeRF-synthetic lego (synthetic, known poses)"},
    "bonsai": {"repo": "nvs-bench/mipnerf360", "prefix": "bonsai", "include": ["images/", "sparse/"],
               "exclude": [], "about": "Mip-NeRF 360 bonsai (real capture, COLMAP)"},
}
API = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}?recursive=true"
FILE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "splatcraft"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_files(name: str) -> list[dict]:
    s = SCENES[name]
    items = json.loads(_get(API.format(repo=s["repo"], path=s["prefix"])))
    out = []
    for it in items:
        if it.get("type") != "file":
            continue
        rel = it["path"][len(s["prefix"]) + 1:]
        if any(rel.startswith(p) or rel == p for p in s["include"]) and not any(x in rel for x in s["exclude"]):
            out.append({"path": it["path"], "rel": rel, "size": it.get("size", 0)})
    return out


def download_scene(name: str, dest: Path | None = None, log=print, workers: int = 8) -> Path:
    if name not in SCENES:
        raise SystemExit(f"unknown scene {name!r}; choose from {', '.join(SCENES)}")
    dest = Path(dest or data_dir() / name)
    marker = dest / ".complete"
    if marker.is_file():
        log(f"{name}: already downloaded ({dest})")
        return dest
    files = list_files(name)
    total = sum(f["size"] for f in files)
    log(f"{name}: {len(files)} files, {total / 1e6:.0f} MB -> {dest}")
    repo = SCENES[name]["repo"]
    done = {"bytes": 0, "n": 0}

    def fetch(f):
        target = dest / f["rel"]
        if target.is_file() and target.stat().st_size == f["size"]:
            return f["size"]
        target.parent.mkdir(parents=True, exist_ok=True)
        url = FILE.format(repo=repo, path=urllib.parse.quote(f["path"]))
        for attempt in range(4):
            try:
                data = _get(url, timeout=120)
                tmp = target.with_suffix(target.suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(target)
                return len(data)
            except OSError:
                if attempt == 3:
                    raise
        return 0

    with ThreadPoolExecutor(workers) as ex:
        for size in ex.map(fetch, files):
            done["bytes"] += size
            done["n"] += 1
            if done["n"] % 25 == 0 or done["n"] == len(files):
                log(f"{name}: {done['n']}/{len(files)} files ({done['bytes'] / 1e6:.0f}/{total / 1e6:.0f} MB)")
    marker.write_text("ok")
    return dest
