"""WebGPU frame-rate measurements in a real browser (Chrome / Edge through Playwright).

Each GPU the browser can reach is measured separately: on Windows/Linux laptops with two GPUs Chrome
picks the integrated one by default and the discrete one with ``--force_high_performance_gpu``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

GPU_MODES = {"default": [], "high-performance": ["--force_high_performance_gpu"]}


def _start_server(port: int):
    import uvicorn

    from ..server.app import create_app

    cfg = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.1)
    return srv


async def _measure(scenes: list[str], port: int, sizes, frames: int, flags: list[str], channel: str, log) -> list[dict]:
    from playwright.async_api import async_playwright

    out = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel=channel, headless=True,
                                          args=["--enable-unsafe-webgpu", "--enable-dawn-features=allow_unsafe_apis",
                                                *flags])
        for scene in scenes:
            for w, h in sizes:
                page = await browser.new_page(viewport={"width": w, "height": h})
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))  # noqa: B023
                await page.goto(f"http://127.0.0.1:{port}/v/{scene}?ui=0")
                try:
                    await page.wait_for_function("window.splatcraft && window.splatcraft.state.loaded > 0", timeout=60000)
                    await page.evaluate("window.splatcraft.ready")
                    res = await page.evaluate(f"splatcraft.bench({{frames: {frames}, width: {w}, height: {h}}})")
                    res["scene"] = scene
                    out.append(res)
                    log(f"  {scene} {w}x{h} on {res['gpu']}: {res['fps']} fps ({res['frame_ms']} ms/frame)")
                except Exception as e:  # noqa: BLE001
                    log(f"  {scene} {w}x{h}: failed ({e}; {errors[:2]})")
                await page.close()
        await browser.close()
    return out


def measure_fps(scenes: list[str], sizes=((1280, 720), (1920, 1080)), frames: int = 120, port: int = 8799,
                channel: str = "chrome", log=print) -> list[dict]:
    """-> one result dict per (GPU mode, scene, size); scenes must be published (home()/scenes)."""
    srv = _start_server(port)
    results, seen = [], set()
    try:
        for mode, flags in GPU_MODES.items():
            try:
                res = asyncio.run(_measure(scenes, port, sizes, frames, flags, channel, log))
            except Exception as e:  # noqa: BLE001 - no browser / no WebGPU
                log(f"browser benchmark ({mode}) unavailable: {e}")
                continue
            gpus = {r["gpu"] for r in res}
            if gpus and gpus <= seen:  # same adapter as the previous mode: skip duplicates
                continue
            seen |= gpus
            for r in res:
                r["mode"] = mode
            results += res
    finally:
        srv.should_exit = True
    return results


if __name__ == "__main__":  # pragma: no cover
    import sys

    print(json.dumps(measure_fps(sys.argv[1:] or ["lego"]), indent=1))
