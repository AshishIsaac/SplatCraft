"""Where things live. Everything downloaded or produced goes under SPLATCRAFT_HOME (default: ./workspace)."""

from __future__ import annotations

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]
CONFIG_DIR = Path(os.environ.get("SPLATCRAFT_CONFIGS", REPO_ROOT / "configs"))
WEB_DIR = Path(os.environ.get("SPLATCRAFT_WEB", REPO_ROOT / "web"))
REFERENCE_DIR = REPO_ROOT / "benchmarks" / "results"


def home() -> Path:
    return Path(os.environ.get("SPLATCRAFT_HOME", REPO_ROOT / "workspace")).expanduser().resolve()


def data_dir() -> Path:
    return home() / "data"


def runs_dir() -> Path:
    return home() / "runs"


def jobs_dir() -> Path:
    return home() / "jobs"


def models_dir() -> Path:
    return home() / "models"


def scenes_dirs() -> list[Path]:
    """Folders the server publishes .splatc scenes from: the workspace and the demo scenes shipped in web/."""
    return [home() / "scenes", WEB_DIR / "scenes"]
