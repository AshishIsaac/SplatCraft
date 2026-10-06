"""Background job queue: uploaded video -> frames -> COLMAP -> 3DGS training -> compression -> share link.

One worker thread (one GPU). Every job lives in ``<home>/jobs/<id>/`` with a ``status.json`` the
web page polls, so jobs survive a server restart (unfinished ones are marked interrupted).
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

from ..config import load_config
from ..paths import home, jobs_dir, runs_dir

STAGES = ["queued", "frames", "poses", "train", "compress", "done"]


def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", s.strip().lower()).strip("-")
    return s[:40] or "scene"


class JobManager:
    def __init__(self, runner=None) -> None:
        self.q: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.cancel: set[str] = set()
        self.runner = runner or run_job
        jobs_dir().mkdir(parents=True, exist_ok=True)
        for st in self.list():  # a restart interrupts whatever was running
            if st["state"] in ("queued", "running"):
                self.update(st["id"], state="failed", message="interrupted by a server restart - upload again")
        self.worker = threading.Thread(target=self._loop, daemon=True, name="splatcraft-jobs")
        self.worker.start()

    # ---- status files -------------------------------------------------------------------------
    def path(self, jid: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{12}", jid):
            raise KeyError(jid)
        return jobs_dir() / jid

    def get(self, jid: str) -> dict:
        p = self.path(jid) / "status.json"
        if not p.is_file():
            raise KeyError(jid)
        return json.loads(p.read_text(encoding="utf-8"))

    def update(self, jid: str, **kw) -> dict:
        with self.lock:
            p = self.path(jid) / "status.json"
            st = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
            st.update(kw, updated=time.time())
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(st), encoding="utf-8")
            tmp.replace(p)
            return st

    def list(self) -> list[dict]:
        out = []
        for d in jobs_dir().glob("*/status.json"):
            try:
                out.append(json.loads(d.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda s: s.get("created", 0), reverse=True)

    # ---- lifecycle ------------------------------------------------------------------------------
    def submit(self, filename: str, data_path: Path | None, name: str, preset: str = "quick") -> dict:
        jid = uuid.uuid4().hex[:12]
        d = jobs_dir() / jid
        d.mkdir(parents=True)
        src = d / ("input" + Path(filename).suffix.lower())
        if data_path is not None:
            Path(data_path).replace(src)
        base = slugify(name or Path(filename).stem)
        scene = base
        n = 2
        while (home() / "scenes" / f"{scene}.splatc").exists():
            scene = f"{base}-{n}"
            n += 1
        self.update(jid, id=jid, name=scene, file=filename, input=str(src), preset=preset, state="queued",
                    stage="queued", progress=0.0, message="waiting for the GPU", created=time.time())
        self.q.put(jid)
        return self.get(jid)

    def request_cancel(self, jid: str) -> dict:
        st = self.get(jid)
        if st["state"] in ("queued", "running"):
            self.cancel.add(jid)
            if st["state"] == "queued":
                return self.update(jid, state="cancelled", message="cancelled")
        return self.get(jid)

    def _loop(self) -> None:
        while True:
            jid = self.q.get()
            if jid in self.cancel or self.get(jid)["state"] != "queued":
                continue
            self.update(jid, state="running", started=time.time())
            try:
                self.runner(self, jid)
                if jid in self.cancel:
                    self.update(jid, state="cancelled", message="cancelled")
                else:
                    self.update(jid, state="done", stage="done", progress=1.0, finished=time.time())
            except Exception as e:  # noqa: BLE001 - report every failure on the page
                (self.path(jid) / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                self.update(jid, state="failed", message=str(e)[:400])


def run_job(m: JobManager, jid: str) -> None:
    """The real pipeline (replaced by a fake in the API tests)."""
    from ..pipeline import compress_run, publish, train_run
    from ..scene.sfm import VIDEO_EXT, prepare_capture

    st = m.get(jid)
    d = m.path(jid)
    cfg = load_config(st.get("preset", "quick"))
    stop = lambda: jid in m.cancel  # noqa: E731
    src = Path(st["input"])
    weights = {"frames": (0.0, 0.05), "poses": (0.05, 0.3), "train": (0.3, 0.95), "compress": (0.95, 1.0)}

    def at(stage, frac, msg):
        lo, hi = weights[stage]
        m.update(jid, stage=stage, progress=round(lo + (hi - lo) * frac, 4), message=msg)

    def log(msg):
        stage = "poses" if msg.startswith("sfm") else "frames"
        frac = {"sfm: SIFT": 0.1, "sfm: sequential": 0.3, "sfm: exhaustive": 0.3, "sfm: incremental": 0.55,
                "sfm: registered": 0.9, "sfm: undistorting": 0.95}
        f = next((v for k, v in frac.items() if msg.startswith(k)), 0.5)
        at(stage, f, msg)

    at("frames", 0.0, "extracting the sharpest frames" if src.suffix in VIDEO_EXT else "reading photos")
    scene = prepare_capture(src, d / "capture", frames=int(cfg.get("capture", {}).get("frames", 120)), log=log)
    if stop():
        return

    def prog(p):
        if p.get("stage") == "train" and p.get("iterations"):
            at("train", p["iteration"] / p["iterations"], p["message"])
            m.update(jid, eta_s=p.get("eta_s"), gaussians=p.get("gaussians"), psnr=p.get("psnr"))
        elif p.get("message"):
            at("train", 0.0, p["message"])

    run = runs_dir() / f"job-{jid}"
    train_run(scene, cfg, run, progress=prog, should_stop=stop)
    if stop():
        return
    at("compress", 0.1, "compressing (pruning, codebooks, entropy coding)")
    rep = compress_run(run, cfg, log=lambda s: at("compress", 0.5, s))
    publish(run, st["name"])
    m.update(jid, scene=st["name"], url=f"/v/{st['name']}", bytes=rep["bytes"], ratio=rep["ratio_vs_ply_file"],
             message=f"ready: {rep['bytes'] / 1e6:.1f} MB, {rep['gaussians']:,} Gaussians")
