"""README screenshots: the static viewer (as on GitHub Pages) and the progressive-loading sequence.

    python scripts/screenshots.py            (needs Chrome + playwright; writes docs/img/*.png)
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "img"
PORT = 8813


async def main() -> None:
    from playwright.async_api import async_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True,
                                    args=["--enable-unsafe-webgpu", "--force_high_performance_gpu"])
        for scene, view in (("lego", None), ("bonsai", 2)):
            pg = await b.new_page(viewport={"width": 1280, "height": 720})
            pg.on("pageerror", lambda e: print("pageerror", e))
            await pg.goto(f"http://127.0.0.1:{PORT}/?scene={scene}")
            await pg.wait_for_function("window.splatcraft && window.splatcraft.state.loaded > 0", timeout=60000)
            await pg.evaluate("window.splatcraft.ready")
            if view is not None:
                for _ in range(view):
                    await pg.keyboard.press("v")
            await pg.wait_for_timeout(1500)
            await pg.screenshot(path=str(OUT / f"{scene}.png"))
            await pg.close()
        if "--static-only" in sys.argv:
            await b.close()
            return
        # progressive loading on a throttled (~4 Mbit/s) connection
        pg = await b.new_page(viewport={"width": 960, "height": 540})
        cdp = await pg.context.new_cdp_session(pg)
        await cdp.send("Network.enable")
        await cdp.send("Network.emulateNetworkConditions", {"offline": False, "latency": 40,
                                                            "downloadThroughput": 500_000, "uploadThroughput": 100_000})
        await pg.goto(f"http://127.0.0.1:{PORT}/?scene=bonsai")
        shots, seen = 0, set()
        t0 = time.time()
        while shots < 6 and time.time() - t0 < 90:  # every chunk; the README shows chunks 1, 3, 5, 6
            n = await pg.evaluate("window.splatcraft?.state?.loaded || 0")
            if n and n not in seen:
                seen.add(n)
                await pg.wait_for_timeout(250)
                await pg.screenshot(path=str(OUT / f"progressive_{shots}.png"))
                print("progressive", shots, n)
                shots += 1
            await pg.wait_for_timeout(100)
        await b.close()


if __name__ == "__main__":
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "-d", str(REPO / "web")],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    try:
        asyncio.run(main())
    finally:
        srv.terminate()
