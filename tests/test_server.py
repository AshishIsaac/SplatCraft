"""Web API: upload -> job -> published scene -> viewer link, with a fake (instant) pipeline."""

import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DEMO = Path(__file__).resolve().parents[1] / "web" / "scenes"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATCRAFT_HOME", str(tmp_path))
    from splatcraft.server.app import create_app
    from splatcraft.server.jobs import JobManager

    def fake_runner(m, jid):
        st = m.get(jid)
        assert Path(st["input"]).read_bytes() == b"not really a video"
        m.update(jid, stage="train", progress=0.5, message="iteration 5/10")
        src = next(DEMO.glob("*.splatc"))
        (tmp_path / "scenes").mkdir(exist_ok=True)
        shutil.copyfile(src, tmp_path / "scenes" / f"{st['name']}.splatc")
        m.update(jid, scene=st["name"], url=f"/v/{st['name']}")

    return TestClient(create_app(JobManager(runner=fake_runner)))


def _wait(c, jid, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = c.get(f"/api/jobs/{jid}").json()
        if st["state"] in ("done", "failed", "cancelled"):
            return st
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_upload_to_share_link(client):
    r = client.post("/api/jobs", files={"file": ("My Desk.mp4", b"not really a video", "video/mp4")},
                    data={"name": "My Desk!", "preset": "quick"})
    assert r.status_code == 200, r.text
    st = _wait(client, r.json()["id"])
    assert st["state"] == "done" and st["url"] == "/v/my-desk"
    page = client.get("/v/my-desk")
    assert page.status_code == 200 and 'SPLATCRAFT_DEFAULT_SCENE = "my-desk"' in page.text
    f = client.get("/scenes/my-desk.splatc")
    assert f.status_code == 200 and f.content[:6] == b"SPLATC"
    assert "my-desk" in [s["name"] for s in client.get("/api/scenes").json()]
    # a second upload with the same name gets a fresh slug instead of overwriting
    r2 = client.post("/api/jobs", files={"file": ("x.mov", b"not really a video", "video/quicktime")},
                     data={"name": "my desk"})
    assert _wait(client, r2.json()["id"])["url"] == "/v/my-desk-2"


def test_rejects_bad_uploads(client):
    assert client.post("/api/jobs", files={"file": ("a.exe", b"x", "application/octet-stream")}).status_code == 400
    r = client.post("/api/jobs", files={"file": ("a.mp4", b"x", "video/mp4")}, data={"preset": "huge"})
    assert r.status_code == 400


def test_scene_paths_are_safe(client):
    assert client.get("/scenes/..%2F..%2Fsecret.splatc").status_code == 404
    assert client.get("/v/..%2Fetc").status_code == 404
    assert client.get("/api/jobs/not-a-job").status_code == 404


def test_viewer_assets_and_bench(client):
    assert client.get("/viewer/js/renderer.js").status_code == 200
    assert "SplatCraft" in client.get("/").text
    r = client.post("/viewer/api/bench", json={"scene": "lego", "fps": 120.5, "gpu": "test gpu", "width": 640,
                                                 "height": 480})
    assert r.status_code == 200
    assert client.get("/api/bench").json()["submissions"][0]["gpu"] == "test gpu"
    assert client.post("/api/bench", json={"no": "fps"}).status_code == 400
