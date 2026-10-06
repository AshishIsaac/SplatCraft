"""Copy benchmark-run scenes into web/scenes/ (the live demo) and write web/scenes/index.json.

    python scripts/make_demo_scenes.py lego=lego_full:splatc bonsai=bonsai_full:splatc-small

Each argument is NAME=RUN[:VARIANT]; VARIANT is a file in RUN/variants/ (default: RUN/scene.splatc).
"""

from __future__ import annotations

import json
import shutil
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from splatcraft.paths import runs_dir  # noqa: E402


def write_index(scenes_dir: Path) -> list[dict]:
    out = []
    for p in sorted(scenes_dir.glob("*.splatc")):
        b = p.read_bytes()
        h = json.loads(b[12:12 + struct.unpack_from("<I", b, 8)[0]])
        out.append({"name": p.stem, "bytes": p.stat().st_size, "gaussians": h["count"]})
    (scenes_dir / "index.json").write_text(json.dumps(out))
    return out


def main(args: list[str]) -> int:
    dest = REPO / "web" / "scenes"
    dest.mkdir(parents=True, exist_ok=True)
    for a in args:
        name, _, spec = a.partition("=")
        run, _, variant = spec.partition(":")
        src = runs_dir() / run / ("variants" if variant else "") / (f"{variant}.splatc" if variant else "scene.splatc")
        shutil.copyfile(src, dest / f"{name}.splatc")
        print(f"{src} -> web/scenes/{name}.splatc ({src.stat().st_size / 1e6:.2f} MB)")
    print(json.dumps(write_index(dest), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
