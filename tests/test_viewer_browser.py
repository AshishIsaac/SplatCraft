"""The WebGPU viewer renders the same pixels as the PyTorch rasterizer (real Chrome, via Playwright).

Runs on any WebGPU adapter: a real GPU locally, Chrome's SwiftShader (CPU) adapter on CI runners.
Skipped when Playwright or a Chrome/Edge install is missing.
"""

import asyncio
import os
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.browser
REPO = Path(__file__).resolve().parents[1]
SCENE = REPO / "web" / "scenes" / "lego.splatc"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    pytest.importorskip("playwright")
    if not SCENE.is_file():
        pytest.skip("demo scene missing")
    os.environ["SPLATCRAFT_HOME"] = str(tmp_path_factory.mktemp("home"))
    import uvicorn

    from splatcraft.server.app import create_app
    from splatcraft.server.jobs import JobManager

    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(JobManager(runner=lambda m, j: None)), host="127.0.0.1",
                                        port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    yield port
    srv.should_exit = True


async def _render(port: int, cams: list[dict]) -> tuple[list[np.ndarray], str]:
    from playwright.async_api import async_playwright

    args = ["--enable-unsafe-webgpu"]
    if os.environ.get("SPLATCRAFT_SWIFTSHADER"):
        args += ["--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"]
    async with async_playwright() as p:
        channel = os.environ.get("SPLATCRAFT_BROWSER_CHANNEL", "chrome")
        try:
            browser = await p.chromium.launch(channel=channel or None, headless=True, args=args)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"no browser: {e}")
        page = await browser.new_page(viewport={"width": 640, "height": 480})
        await page.goto(f"http://127.0.0.1:{port}/v/lego")
        try:
            await page.wait_for_function("window.splatcraft && window.splatcraft.state.loaded > 0", timeout=60000)
            await page.evaluate("window.splatcraft.ready")
        except Exception:  # noqa: BLE001
            if await page.evaluate("!document.querySelector('#nogpu').hidden"):
                pytest.skip("browser has no WebGPU adapter")
            raise
        gpu = await page.evaluate("document.querySelector('#stat-gpu').textContent")
        out = []
        for c in cams:
            px = await page.evaluate("c => splatcraft.renderCamera(c)", c)
            out.append(np.array(px, np.uint8).reshape(c["height"], c["width"], 4)[..., :3])
        await browser.close()
        return out, gpu


def test_viewer_matches_pytorch(server):
    from splatcraft.compress.codec import decode_scene, read_header
    from splatcraft.gs.losses import psnr
    from splatcraft.gs.rasterizer import render
    from splatcraft.scene.cameras import Camera

    data = SCENE.read_bytes()
    h, _ = read_header(data)
    # half-resolution capture cameras keep the CPU reference render fast
    cams = [dict(c, fx=c["fx"] / 2, fy=c["fy"] / 2, cx=c["cx"] / 2, cy=c["cy"] / 2, width=c["width"] // 2,
                 height=c["height"] // 2) for c in h["view"]["cameras"][:2]]
    web, gpu = asyncio.run(_render(server, cams))
    g = decode_scene(data)
    bg = torch.tensor(h["view"]["background"], dtype=torch.float32)
    for c, w in zip(cams, web):
        with torch.no_grad():
            ref = render(g, Camera.from_dict(c), bg, backend="torch")["image"].clamp(0, 1)
        ref8 = torch.round(ref * 255) / 255
        p = psnr(torch.from_numpy(w).float() / 255, ref8)
        assert p > 45, f"WebGPU ({gpu}) vs PyTorch: {p:.1f} dB"
