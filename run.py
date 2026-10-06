#!/usr/bin/env python3
"""SplatCraft one-command launcher.

    python run.py                  first run: set everything up, then open the web app
                                   later runs: open the web app straight away
    python run.py --cpu            force the CPU build of PyTorch (no NVIDIA GPU needed; training is slow)
    python run.py --home DIR       keep the workspace (datasets, runs, scenes: ~2 GB+) in DIR; remembered
    python run.py --venv DIR       put the private Python environment in DIR; remembered
    python run.py --reinstall      rebuild the environment from scratch
    python run.py --port 8770 --no-browser
    python run.py --host 0.0.0.0   serve on the network (open share links on phones on the same Wi-Fi)

The first run performs the one-time setup in a private virtual environment (.venv): PyTorch
(CUDA build with an NVIDIA GPU, otherwise CPU), the Triton kernel compiler (fast training
rasterizer), SplatCraft and its libraries (pycolmap for camera poses, LPIPS, FastAPI), the two
benchmark scenes (NeRF-synthetic lego and Mip-NeRF 360 bonsai, about 500 MB) and the LPIPS weights
(about 760 MB), while a setup window shows the progress. A failed or interrupted setup resumes
where it stopped. Later runs start immediately.

This file deliberately uses only the Python standard library: nothing else is installed yet
when it first runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".setup"
SETTINGS = STATE_DIR / "launcher.json"  # remembered --home / --venv
INSTALL_MARK = STATE_DIR / "install.json"
LOG_FILE = STATE_DIR / "install.log"
PROGRESS_FILE = STATE_DIR / "install_progress.json"  # completed steps, so a failed install resumes
MIN_PY = (3, 10)
MAX_PY = (3, 12)  # newest version the pinned wheels (requirements/*.txt) are verified on
INSTALL_VERSION = 1  # bump to force existing environments to re-run the installer
NAME = "SplatCraft"


# --------------------------------------------------------------------------- settings / environment


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(s: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(s, indent=1), encoding="utf-8")


SET = load_settings()
VENV = Path(SET["venv"]) if SET.get("venv") else ROOT / ".venv"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def has_nvidia_gpu() -> bool:
    candidates = [shutil.which("nvidia-smi")]
    if os.name == "nt":
        candidates.append(r"C:\Windows\System32\nvidia-smi.exe")
    for exe in filter(None, candidates):
        try:
            out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=15)
            if out.returncode == 0 and "GPU" in out.stdout:
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def removable_drive(p: Path) -> bool:
    """True for a USB stick / SD card (Windows only; elsewhere we can't tell cheaply)."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        return ctypes.windll.kernel32.GetDriveTypeW(f"{p.drive}\\") == 2  # DRIVE_REMOVABLE
    except (AttributeError, OSError):
        return False


def project_fingerprint() -> str:
    h = hashlib.sha256()
    for f in ["pyproject.toml", *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "requirements").glob("*.txt"))]:
        p = ROOT / f
        if p.is_file():
            h.update(p.read_bytes())
    h.update(str(INSTALL_VERSION).encode())
    h.update(str(VENV).encode())
    return h.hexdigest()[:16]


def install_needed(profile: str, force: bool) -> bool:
    if force or not venv_python().is_file() or not INSTALL_MARK.is_file():
        return True
    try:
        mark = json.loads(INSTALL_MARK.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    if mark.get("fingerprint") != project_fingerprint() or mark.get("home") != os.environ.get("SPLATCRAFT_HOME", ""):
        return True
    # The CUDA build of PyTorch also runs on the CPU, so `--cpu` doesn't need its own install.
    return mark.get("profile") != profile and not (mark.get("profile") == "gpu" and profile == "cpu")


def venv_works() -> bool:
    try:
        return subprocess.run([str(venv_python()), "-c", "import sys"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def plan(profile: str) -> list[dict]:
    """Installation steps: (key, title, command, required)."""
    py = str(venv_python())
    # Generous retries/timeouts: multi-GB CUDA wheels over consumer internet do get interrupted.
    pip = [py, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--retries", "10", "--timeout", "60"]
    torch_req = "torch-gpu.txt" if profile == "gpu" else "torch-cpu.txt"
    sc = [py, "-m", "splatcraft"]
    steps = [
        {"key": "venv", "title": "Create private Python environment", "cmd": [sys.executable, "-m", "venv", str(VENV)],
         "required": True},
        {"key": "pip", "title": "Update package installer", "cmd": [*pip, "-U", "pip", "setuptools", "wheel"],
         "required": True},
        {"key": "torch", "title": f"Install PyTorch ({'CUDA, about 2.5 GB' if profile == 'gpu' else 'CPU'})",
         "cmd": [*pip, "-r", str(ROOT / "requirements" / torch_req)], "required": True},
        {"key": "app", "title": "Install SplatCraft and its libraries (pycolmap, LPIPS, FastAPI)",
         "cmd": [*pip, "-e", f"{ROOT}[browser]"], "required": True},
    ]
    if profile == "gpu" and os.name == "nt":
        # Linux wheels of PyTorch already ship Triton; on Windows it is a separate (optional) package.
        steps.append({"key": "triton", "title": "Install Triton (fused GPU training kernels, optional)",
                      "cmd": [*pip, "-r", str(ROOT / "requirements" / "triton-windows.txt")], "required": False})
    steps += [
        {"key": "data", "title": "Download benchmark scenes: lego + bonsai (about 500 MB)",
         "cmd": [*sc, "data", "download", "all"], "required": False},
        {"key": "models", "title": "Download LPIPS weights (about 760 MB)",
         "cmd": [*sc, "models", "download"], "required": False},
        {"key": "verify", "title": "Verify installation (trains a tiny scene)", "cmd": [*sc, "doctor"], "required": True},
    ]
    return steps


# --------------------------------------------------------------------------- installer


class Installer(threading.Thread):
    """Runs the plan, streaming progress events to the UI through a queue."""

    def __init__(self, steps: list[dict], profile: str, events: queue.Queue, fresh: bool = False) -> None:
        super().__init__(daemon=True)
        self.steps = steps
        self.profile = profile
        self.events = events
        self.ok = False
        self.fresh = fresh
        self.done = set() if fresh else self._load_progress()

    def _exec(self, cmd: list[str], log) -> int:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                                    env={**os.environ, "PIP_PROGRESS_BAR": "off", "PYTHONUNBUFFERED": "1"})
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                line = line.strip()
                if line:
                    self.events.put(("log", _short(line)))
            return proc.wait()
        except OSError as e:
            log.write(f"{e}\n")
            return -1

    def _load_progress(self) -> set:
        try:
            d = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            if d.get("fingerprint") == project_fingerprint() and d.get("profile") == self.profile and venv_python().is_file():
                return set(d.get("done", []))
        except (OSError, ValueError):
            pass
        return set()

    def _save_progress(self) -> None:
        PROGRESS_FILE.write_text(json.dumps({"fingerprint": project_fingerprint(), "profile": self.profile,
                                             "done": sorted(self.done)}), encoding="utf-8")

    def run(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as log:
            for i, st in enumerate(self.steps):
                if st["key"] in self.done and st["key"] != "verify":
                    self.events.put(("step", i, "done", "already done"))
                    continue
                self.events.put(("step", i, "active", ""))
                if st["key"] == "venv" and VENV.exists():
                    if not self.fresh and venv_works():
                        # An update or profile change: keep the installed packages and let pip add only
                        # what changed, instead of downloading everything again. --reinstall starts clean.
                        self.done.add("venv")
                        self._save_progress()
                        self.events.put(("step", i, "done", "reusing existing environment"))
                        continue
                    shutil.rmtree(VENV, ignore_errors=True)
                t0 = time.time()
                rc = -1
                for attempt in (1, 2):  # one automatic retry: pip resumes partial downloads
                    if attempt == 2:
                        self.events.put(("log", "retrying after a failure (network hiccup?) ..."))
                    rc = self._exec(st["cmd"], log)
                    if rc == 0 or st["key"] in ("venv", "verify"):
                        break
                dt = time.time() - t0
                if rc == 0:
                    self.done.add(st["key"])
                    self._save_progress()
                    self.events.put(("step", i, "done", f"{dt:.0f}s"))
                elif not st["required"]:
                    self.events.put(("step", i, "skipped", "optional - continuing without it"))
                else:
                    self.events.put(("step", i, "failed", f"exit code {rc}"))
                    self.events.put(("fail", f"'{st['title']}' failed. Details: {LOG_FILE}"))
                    return
        INSTALL_MARK.write_text(json.dumps({
            "fingerprint": project_fingerprint(), "profile": self.profile, "python": platform.python_version(),
            "home": os.environ.get("SPLATCRAFT_HOME", ""), "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2), encoding="utf-8")
        self.ok = True
        self.events.put(("installed",))


def _short(line: str, n: int = 96) -> str:
    line = line.replace(str(ROOT), ".")
    return line if len(line) <= n else line[: n - 1] + "..."


# --------------------------------------------------------------------------- UI


class SetupWindow:
    """Animated tkinter window: spinner, step checklist, progress bar, live log line."""

    BG, PANEL, FG, MUTED, ACCENT, OK, WARN, ERR = (
        "#16181d", "#1f2229", "#e8e8e8", "#8b8f98", "#3987e5", "#2fb67c", "#e0a000", "#e5534b")

    def __init__(self, titles: list[str], subtitle: str) -> None:
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title(f"{NAME} - setup")
        self.root.configure(bg=self.BG)
        self.root.geometry("820x560")
        self.root.minsize(700, 480)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.closed = False
        self.t0 = time.time()

        head = tk.Frame(self.root, bg=self.BG)
        head.pack(fill="x", padx=28, pady=(24, 8))
        self.canvas = tk.Canvas(head, width=64, height=64, bg=self.BG, highlightthickness=0)
        self.canvas.pack(side="left")
        txt = tk.Frame(head, bg=self.BG)
        txt.pack(side="left", padx=16)
        tk.Label(txt, text=NAME, font=("Segoe UI", 22, "bold"), fg=self.FG, bg=self.BG).pack(anchor="w")
        self.subtitle = tk.Label(txt, text=subtitle, font=("Segoe UI", 11), fg=self.MUTED, bg=self.BG)
        self.subtitle.pack(anchor="w")

        body = tk.Frame(self.root, bg=self.PANEL)
        body.pack(fill="both", expand=True, padx=28, pady=8)
        self.rows = []
        for t in titles:
            r = tk.Frame(body, bg=self.PANEL)
            r.pack(fill="x", padx=18, pady=5)
            icon = tk.Canvas(r, width=20, height=20, bg=self.PANEL, highlightthickness=0)
            icon.pack(side="left")
            lbl = tk.Label(r, text=t, font=("Segoe UI", 11), fg=self.MUTED, bg=self.PANEL)
            lbl.pack(side="left", padx=10)
            det = tk.Label(r, text="", font=("Segoe UI", 9), fg=self.MUTED, bg=self.PANEL)
            det.pack(side="right")
            self.rows.append({"icon": icon, "label": lbl, "detail": det, "status": "pending"})

        foot = tk.Frame(self.root, bg=self.BG)
        foot.pack(fill="x", padx=28, pady=(6, 20))
        self.bar = tk.Canvas(foot, height=6, bg=self.PANEL, highlightthickness=0)
        self.bar.pack(fill="x")
        self.status = tk.Label(foot, text="", font=("Consolas", 9), fg=self.MUTED, bg=self.BG, anchor="w")
        self.status.pack(fill="x", pady=(8, 0))
        self.elapsed = tk.Label(foot, text="", font=("Segoe UI", 9), fg=self.MUTED, bg=self.BG, anchor="e")
        self.elapsed.pack(fill="x")
        self._draw_icons()

    def _on_close(self) -> None:
        self.closed = True
        self.root.destroy()

    # ---- state ---------------------------------------------------------------
    def set_step(self, i: int, status: str, detail: str = "") -> None:
        row = self.rows[i]
        row["status"] = status
        row["detail"].configure(text=detail)
        color = {"active": self.FG, "done": self.FG, "failed": self.ERR, "skipped": self.WARN}.get(status, self.MUTED)
        row["label"].configure(fg=color)
        self._draw_icons()

    def set_status(self, text: str, color: str | None = None) -> None:
        self.status.configure(text=text, fg=color or self.MUTED)

    # ---- drawing -------------------------------------------------------------
    def _draw_icons(self) -> None:
        for row in self.rows:
            c = row["icon"]
            c.delete("all")
            s = row["status"]
            if s == "done":
                c.create_oval(2, 2, 18, 18, fill=self.OK, outline="")
                c.create_line(6, 10, 9, 13, 14, 7, fill="white", width=2)
            elif s == "failed":
                c.create_oval(2, 2, 18, 18, fill=self.ERR, outline="")
                c.create_line(7, 7, 13, 13, fill="white", width=2)
                c.create_line(13, 7, 7, 13, fill="white", width=2)
            elif s == "skipped":
                c.create_oval(2, 2, 18, 18, fill=self.WARN, outline="")
                c.create_line(6, 10, 14, 10, fill="white", width=2)
            elif s == "pending":
                c.create_oval(3, 3, 17, 17, outline=self.MUTED, width=2)

    def animate(self) -> None:
        if self.closed:
            return
        t = time.time() - self.t0
        # Big spinner: two arcs rotating in opposite directions.
        c = self.canvas
        c.delete("all")
        a = (t * 240) % 360
        c.create_arc(6, 6, 58, 58, start=a, extent=110, style="arc", outline=self.ACCENT, width=5)
        c.create_arc(16, 16, 48, 48, start=-a * 1.4, extent=80, style="arc", outline=self.OK, width=4)
        pulse = 3 + 2 * math.sin(t * 4)
        c.create_oval(32 - pulse, 32 - pulse, 32 + pulse, 32 + pulse, fill=self.FG, outline="")
        # Small spinner on the active row.
        for row in self.rows:
            if row["status"] == "active":
                ic = row["icon"]
                ic.delete("all")
                ic.create_arc(2, 2, 18, 18, start=(t * 360) % 360, extent=270, style="arc", outline=self.ACCENT, width=3)
        # Progress bar: filled part = completed steps, moving shimmer = work in progress.
        w = max(self.bar.winfo_width(), 1)
        done = sum(r["status"] in ("done", "skipped") for r in self.rows)
        frac = done / max(len(self.rows), 1)
        b = self.bar
        b.delete("all")
        b.create_rectangle(0, 0, w * frac, 6, fill=self.ACCENT, outline="")
        sx = (t * 0.35 % 1.0) * w
        b.create_rectangle(sx, 0, min(sx + w * 0.15, w), 6, fill="#86b6ef", outline="")
        m, s = divmod(int(t), 60)
        self.elapsed.configure(text=f"{m:d}:{s:02d} elapsed - this happens only once")
        self.root.after(33, self.animate)



def run_install_ui(profile: str, app_cmd: list[str], env: dict, fresh: bool = False) -> int:
    """Install with an animated window, then keep animating until the web app is up."""
    steps = plan(profile)
    titles = [s["title"] for s in steps] + [f"Start the {NAME} web app"]
    try:
        win = SetupWindow(titles, f"First-time setup ({'NVIDIA GPU' if profile == 'gpu' else 'CPU'} profile)")
    except Exception:  # no display / no tkinter (headless servers, minimal Linux)
        return run_install_console(profile, app_cmd, env, fresh)

    events: queue.Queue = queue.Queue()
    inst = Installer(steps, profile, events, fresh=fresh)
    inst.start()
    state = {"proc": None, "rc": None, "failed": False}
    ready = STATE_DIR / "app_ready"

    def poll() -> None:
        if win.closed:
            return
        try:
            while True:
                ev = events.get_nowait()
                if ev[0] == "step":
                    win.set_step(ev[1], ev[2], ev[3])
                elif ev[0] == "log":
                    win.set_status(ev[1])
                elif ev[0] == "fail":
                    state["failed"] = True
                    win.set_status(ev[1], SetupWindow.ERR)
                    win.subtitle.configure(text="Setup failed - close this window, fix the issue and run again (it resumes)")
                elif ev[0] == "installed":
                    win.set_step(len(steps), "active")
                    win.set_status(f"Starting {NAME} ... (opening the web app in your browser)")
                    ready.unlink(missing_ok=True)
                    state["proc"] = subprocess.Popen(app_cmd, cwd=ROOT, env={**env, "SPLATCRAFT_READY_FILE": str(ready)})
        except queue.Empty:
            pass
        proc = state["proc"]
        if proc is not None:
            if ready.exists():  # the web app is up: hand over seamlessly
                win.set_step(len(steps), "done")
                win.subtitle.configure(text=f"Setup complete - next time {NAME} starts straight away")
                win.root.after(900, win._on_close)
                return
            rc = proc.poll()
            if rc is not None:
                state["rc"] = rc
                win.set_step(len(steps), "failed", f"exit code {rc}")
                win.set_status(f"{NAME} exited early - see the terminal for details", SetupWindow.ERR)
                return
        win.root.after(100, poll)

    win.root.after(100, poll)
    win.root.after(33, win.animate)
    win.root.mainloop()
    if state["failed"]:
        return 1
    proc = state["proc"]
    if proc is None:
        if inst.is_alive():
            print("Setup window closed: the installation stops now and resumes on the next run.")
        return 1
    try:
        return proc.wait()
    except KeyboardInterrupt:
        return 0


def run_install_console(profile: str, app_cmd: list[str], env: dict, fresh: bool = False) -> int:
    steps = plan(profile)
    events: queue.Queue = queue.Queue()
    inst = Installer(steps, profile, events, fresh=fresh)
    inst.start()
    spin = "|/-\\"
    k = 0
    print(f"{NAME} first-time setup ({profile} profile). Details: {LOG_FILE}")
    while inst.is_alive() or not events.empty():
        try:
            ev = events.get(timeout=0.1)
            if ev[0] == "step" and ev[2] != "active":
                print(f"\r[{ev[2]:>7}] {steps[ev[1]]['title']} {ev[3]}".ljust(100))
            elif ev[0] == "fail":
                print(ev[1])
        except queue.Empty:
            k += 1
            print(f"\r{spin[k % 4]} installing...", end="", flush=True)
    if not inst.ok:
        return 1
    return launch(app_cmd, env)


def launch(app_cmd: list[str], env: dict) -> int:
    try:
        return subprocess.call(app_cmd, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 0


# --------------------------------------------------------------------------- main


def _take(argv: list[str], flag: str) -> tuple[str | None, list[str]]:
    if flag in argv:
        i = argv.index(flag)
        if i + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value")
        return argv[i + 1], argv[:i] + argv[i + 2:]
    return None, argv


def main(argv: list[str]) -> int:
    global VENV
    if sys.version_info < MIN_PY:
        print(f"{NAME} needs Python >= {MIN_PY[0]}.{MIN_PY[1]} (this is {platform.python_version()})")
        return 1
    home, argv = _take(argv, "--home")
    venv, argv = _take(argv, "--venv")
    port, argv = _take(argv, "--port")
    host, argv = _take(argv, "--host")
    settings = load_settings()
    if home:
        settings["home"] = str(Path(home).expanduser().resolve())
    if venv:
        settings["venv"] = str(Path(venv).expanduser().resolve())
        VENV = Path(settings["venv"])
    if home or venv:
        save_settings(settings)
    force_cpu = "--cpu" in argv
    reinstall = "--reinstall" in argv
    no_browser = "--no-browser" in argv
    unknown = [a for a in argv if a not in ("--cpu", "--reinstall", "--no-browser")]
    if unknown:
        print(__doc__)
        print(f"unknown option(s): {' '.join(unknown)}")
        return 2
    if settings.get("home"):
        os.environ["SPLATCRAFT_HOME"] = settings["home"]
    profile = "cpu" if force_cpu or not has_nvidia_gpu() else "gpu"
    work = Path(settings.get("home") or ROOT / "workspace")
    if removable_drive(VENV) or removable_drive(work):
        print(f"Note: {VENV if removable_drive(VENV) else work} is on a removable drive (USB stick / SD card).\n"
              "Installing PyTorch there takes a long time and training reads the images constantly.\n"
              "Consider an internal drive:  python run.py --home C:\\SplatCraftData --venv C:\\SplatCraftData\\venv\n")
    needed = install_needed(profile, reinstall)
    if sys.version_info[:2] > MAX_PY and needed:
        # Checked before downloading anything: the pinned wheels may not exist for newer Pythons.
        print(f"{NAME}'s first-time setup needs Python {MIN_PY[0]}.{MIN_PY[1]} to {MAX_PY[0]}.{MAX_PY[1]} "
              f"(this is {platform.python_version()}).\n"
              "Install Python 3.12 (it can sit next to your current Python):\n"
              "  https://www.python.org/downloads/release/python-31210/\n"
              "then run:  py -3.12 run.py      (Windows)\n"
              "           python3.12 run.py    (Linux / macOS)")
        return 1
    app_cmd = [str(venv_python()), "-m", "splatcraft", "serve"]
    if host:
        app_cmd += ["--host", host]
    if port:
        app_cmd += ["--port", port]
    if no_browser:
        app_cmd.append("--no-browser")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}
    env["SPLATCRAFT_PROFILE"] = profile
    if needed:
        if no_browser:
            return run_install_console(profile, app_cmd, env, fresh=reinstall)
        return run_install_ui(profile, app_cmd, env, fresh=reinstall)
    return launch(app_cmd, env)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(130)
