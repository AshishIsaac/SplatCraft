"""FastAPI web app.

    /                       upload a video, watch jobs, browse scenes, see the benchmark table
    /v/<scene>              the WebGPU viewer for a scene (the share link)
    /viewer/...             viewer assets (web/)
    /scenes/<scene>.splatc  scene files (streamed; Content-Length lets the viewer show progress)
    /api/...                jobs, scenes, benchmark submissions, health
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..compress.codec import read_header
from ..paths import REFERENCE_DIR, WEB_DIR, home, scenes_dirs
from .jobs import JobManager, slugify

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = int(os.environ.get("SPLATCRAFT_MAX_UPLOAD_MB", 2048)) * 1024 * 1024
ALLOWED = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp", ".zip"}


def find_scene(name: str) -> Path | None:
    if not name or slugify(name) != name.lower() or "/" in name or "\\" in name:
        return None  # names are slugs: no path tricks
    for d in scenes_dirs():
        p = d / f"{name}.splatc"
        if p.is_file():
            return p
    return None


def scene_info(p: Path) -> dict:
    try:
        with open(p, "rb") as f:
            head = f.read(1 << 20)
        h, _ = read_header(head)
        count, deg = h["count"], h["sh_degree"]
    except (OSError, ValueError, KeyError):
        count, deg = None, None
    return {"name": p.stem, "bytes": p.stat().st_size, "gaussians": count, "sh_degree": deg,
            "url": f"/v/{p.stem}", "file": f"/scenes/{p.name}", "modified": p.stat().st_mtime,
            "demo": p.parent.resolve() == (WEB_DIR / "scenes").resolve()}


def create_app(jobs: JobManager | None = None) -> FastAPI:
    app = FastAPI(title="SplatCraft", version=__version__)
    app.state.jobs = jobs or JobManager()
    (home() / "scenes").mkdir(parents=True, exist_ok=True)
    bench_file = home() / "bench_submissions.jsonl"

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/v/{name}", response_class=HTMLResponse)
    def viewer(name: str):
        if not find_scene(name):
            raise HTTPException(404, f"no scene named {name!r}")
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        inject = (f'<base href="/viewer/">\n<script>window.SPLATCRAFT_DEFAULT_SCENE = {json.dumps(name)};'
                  f' window.SPLATCRAFT_SCENE_BASE = "/scenes/";</script>')
        return html.replace("<head>", "<head>\n" + inject, 1)

    @app.get("/viewer/", include_in_schema=False)
    def viewer_root():
        return RedirectResponse("/")

    @app.get("/scenes/{fname}")
    def scene_file(fname: str):
        if not fname.endswith(".splatc"):
            raise HTTPException(404)
        p = find_scene(fname[: -len(".splatc")])
        if not p:
            raise HTTPException(404, "scene not found")
        return FileResponse(p, media_type="application/octet-stream",
                            headers={"Cache-Control": "public, max-age=3600", "Access-Control-Allow-Origin": "*"})

    @app.get("/api/scenes")
    def scenes():
        seen, out = set(), []
        for d in scenes_dirs():
            for p in sorted(d.glob("*.splatc")):
                if p.stem not in seen:
                    seen.add(p.stem)
                    out.append(scene_info(p))
        return sorted(out, key=lambda s: (s["demo"], -s["modified"]))

    @app.post("/api/jobs")
    async def create_job(file: UploadFile = File(...),  # noqa: B008 - FastAPI idiom
                         name: str = Form(""), preset: str = Form("quick")):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED:
            raise HTTPException(400, f"unsupported file type {ext or '?'}: upload a video ({', '.join(sorted(ALLOWED - {'.zip'}))}) or a .zip of photos")
        if preset not in ("smoke", "quick", "full"):
            raise HTTPException(400, "preset must be smoke, quick or full")
        fd, tmp_name = tempfile.mkstemp(dir=home(), suffix=ext)
        os.close(fd)  # Windows: an open handle would block the later move into the job folder
        tmp = Path(tmp_name)
        size = 0
        with open(tmp, "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    f.close()
                    tmp.unlink(missing_ok=True)
                    raise HTTPException(413, f"file larger than {MAX_UPLOAD >> 20} MB")
                f.write(chunk)
        return app.state.jobs.submit(file.filename, tmp, name, preset)

    @app.get("/api/jobs")
    def list_jobs():
        return app.state.jobs.list()

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        try:
            return app.state.jobs.get(jid)
        except KeyError:
            raise HTTPException(404, "no such job") from None

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        try:
            return app.state.jobs.request_cancel(jid)
        except KeyError:
            raise HTTPException(404, "no such job") from None

    @app.post("/api/bench")
    @app.post("/viewer/api/bench")
    async def submit_bench(req: Request):
        try:
            res = await req.json()
        except ValueError:
            raise HTTPException(400, "expected JSON") from None
        keep = {k: res.get(k) for k in ("scene", "gaussians", "file_bytes", "width", "height", "frames", "fps",
                                        "frame_ms", "gpu_ms", "latency_ms", "instances", "gpu", "user_agent",
                                        "first_pixel_ms", "load_ms", "date", "stage_ms")}
        if not isinstance(keep.get("fps"), int | float):
            raise HTTPException(400, "missing fps")
        keep["received"] = time.time()
        with open(bench_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(keep) + "\n")
        return {"ok": True}

    @app.get("/api/bench")
    def get_bench():
        ref = {}
        p = REFERENCE_DIR / "benchmark.json"
        if p.is_file():
            ref = json.loads(p.read_text(encoding="utf-8"))
        subs = []
        if bench_file.is_file():
            for line in bench_file.read_text(encoding="utf-8").splitlines()[-200:]:
                with contextlib.suppress(ValueError):
                    subs.append(json.loads(line))
        return {"reference": ref, "submissions": subs[::-1]}

    app.mount("/viewer", StaticFiles(directory=WEB_DIR, html=True), name="viewer")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def serve(host: str = "127.0.0.1", port: int = 8770, open_browser: bool = True) -> None:
    import uvicorn

    app = create_app()
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
    ready = os.environ.get("SPLATCRAFT_READY_FILE")

    @app.on_event("startup")
    def _ready():
        print(f"SplatCraft is running: {url}", flush=True)
        if ready:
            Path(ready).write_text("ok")
        if open_browser:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")
