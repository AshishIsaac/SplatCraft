"""run.py: stdlib only, sane install plan, argument handling."""

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _launcher():
    spec = importlib.util.spec_from_file_location("sc_run", REPO / "run.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["sc_run"] = m
    sys.dont_write_bytecode = True  # keep the repository root clean
    spec.loader.exec_module(m)
    return m


def test_launcher_plan_and_options():
    r = _launcher()
    keys = [s["key"] for s in r.plan("cpu")]
    assert keys == ["venv", "pip", "torch", "app", "data", "models", "verify"]
    gpu = next(s for s in r.plan("gpu") if s["key"] == "torch")["cmd"]
    cpu = next(s for s in r.plan("cpu") if s["key"] == "torch")["cmd"]
    assert gpu[-1].endswith("torch-gpu.txt") and cpu[-1].endswith("torch-cpu.txt")
    for s in r.plan("gpu"):
        if s["key"] in ("data", "models", "triton"):
            assert not s["required"]  # optional downloads must not block the app
    assert r.project_fingerprint() == r.project_fingerprint()
    assert r._take(["--home", "D:/x", "--cpu"], "--home") == ("D:/x", ["--cpu"])
    assert r.main(["--bogus"]) == 2


def test_launcher_uses_only_stdlib():
    tree = ast.parse((REPO / "run.py").read_text(encoding="utf-8"))
    mods = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods


def test_requirement_files_exist():
    for f in ("torch-cpu.txt", "torch-gpu.txt", "triton-windows.txt"):
        assert (REPO / "requirements" / f).is_file()
