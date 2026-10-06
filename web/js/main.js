// SplatCraft viewer: streams a .splatc scene and renders it with the hand-written WebGPU pipeline.
//
// URL parameters:
//   scene=<name | url>   scenes/<name>.splatc, or any URL to a .splatc file
//   scale=0.5..2         render resolution relative to the canvas' CSS pixels x devicePixelRatio
//   bench=1              run the benchmark once loaded (fixed 360 degree orbit, uncapped frame rate)
//   ui=0                 hide the interface (clean screenshots)

import { OrbitCamera, attachControls } from "./camera.js";
import { streamSplatc } from "./loader.js";
import { SplatRenderer } from "./renderer.js";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const fmtMB = (b) => (b / 1e6).toFixed(b < 1e7 ? 2 : 1) + " MB";
const fmtN = (n) => n.toLocaleString("en-US");

function sceneUrl() {
  const s = params.get("scene") || window.SPLATCRAFT_DEFAULT_SCENE || "lego";
  if (/[/.]/.test(s)) return { url: s, name: s.split("/").pop().replace(/\.splatc$/, "") };
  return { url: `${window.SPLATCRAFT_SCENE_BASE || "scenes/"}${encodeURIComponent(s)}.splatc`, name: s };
}

const state = {
  renderer: null, camera: null, header: null, dirty: true, loaded: 0, bytes: 0, totalBytes: 0,
  frameMs: [], scale: Number(params.get("scale")) || 1, viewIdx: -1, t0: performance.now(),
  autoScale: !params.get("scale"), // adapt the render resolution to keep interaction smooth
  firstPixelMs: null, doneMs: null,
};
const ready = {};
ready.promise = new Promise((res, rej) => { ready.resolve = res; ready.reject = rej; });

function fail(msg) {
  $("#error").hidden = false;
  $("#error-msg").textContent = msg;
  $("#loading").hidden = true;
  ready.reject(new Error(msg));
}

function canvasSize() {
  const c = $("#view");
  const dpr = Math.min(window.devicePixelRatio || 1, 2) * state.scale;
  const w = Math.max(16, Math.round(c.clientWidth * dpr));
  const h = Math.max(16, Math.round(c.clientHeight * dpr));
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; state.dirty = true; }
  return [w, h];
}

function updateHud() {
  const h = state.header;
  if (!h) return;
  const fm = state.frameMs.slice(-30).sort((a, b) => a - b);
  const med = fm.length ? fm[fm.length >> 1] : 0;
  $("#stat-fps").textContent = med ? `${(1000 / med).toFixed(0)} fps` : "–";
  $("#stat-ms").textContent = med ? `${med.toFixed(1)} ms` : "";
  const st = state.renderer.stageMs;
  $("#stat-gpu-ms").textContent = st ? `${Object.values(st).reduce((a, b) => a + b, 0).toFixed(2)} ms` : "n/a";
  $("#stat-n").textContent = `${fmtN(state.loaded)} / ${fmtN(h.count)}`;
  $("#stat-mb").textContent = state.totalBytes ? `${fmtMB(state.bytes)} / ${fmtMB(state.totalBytes)}` : fmtMB(state.bytes);
  const r = state.renderer;
  $("#stat-inst").textContent = r.lastTotal ? fmtN(r.lastTotal) : "–";
  $("#stat-res").textContent = `${r.size[0]}×${r.size[1]}`;
  const frac = state.totalBytes ? state.bytes / state.totalBytes : state.loaded / h.count;
  $("#progress").style.width = `${(100 * Math.min(1, frac)).toFixed(1)}%`;
  $("#progress-wrap").classList.toggle("done", state.loaded >= h.count);
}

// Slow GPU (older phones, integrated graphics)? Render fewer pixels while keeping ~25+ fps.
let lastAdapt = 0;
function adaptScale() {
  if (!state.autoScale || state.frameMs.length < 12 || performance.now() - lastAdapt < 700) return;
  const recent = state.frameMs.slice(-12).sort((a, b) => a - b)[6];
  let s = state.scale;
  if (recent > 40 && s > 0.5) s = Math.max(0.5, s * 0.8);
  else if (recent < 14 && s < 1) s = Math.min(1, s * 1.15);
  if (s !== state.scale) {
    state.scale = s;
    state.frameMs.length = 0;
    lastAdapt = performance.now();
    state.dirty = true;
    $("#res-scale").title = `Render resolution (auto: ${Math.round(s * 100)} %)`;
  }
}

let lastT = performance.now();
let controls = null;
function loop(t) {
  const dt = (t - lastT) / 1000;
  lastT = t;
  controls?.tick(dt);
  if (state.autorotate) { state.camera.orbit(dt * 0.35, 0); state.dirty = true; }
  const [w, h] = canvasSize();
  if (state.dirty && state.renderer && state.camera && !state.benchmarking) {
    const t0 = performance.now();
    if (state.renderer.render(state.camera.frame(w, h))) {
      state.dirty = false;
      if (state.firstPixelMs === null) state.firstPixelMs = performance.now() - state.t0;
      // only time frames of the fully loaded scene (uploads during streaming would count otherwise)
      const warm = state.loaded < (state.header?.count || 0) || (state.framesDone = (state.framesDone || 0) + 1) <= 1;
      state.renderer.device.queue.onSubmittedWorkDone().then(() => {
        if (warm) return;
        state.frameMs.push(performance.now() - t0);
        if (state.frameMs.length > 120) state.frameMs.shift();
        adaptScale();
      });
    }
  }
  updateHud();
  requestAnimationFrame(loop);
}

// ---- benchmark: a fixed 360 degree orbit rendered back to back (not capped by the display refresh)
async function bench({ frames = 120, width, height } = {}) {
  const r = state.renderer;
  r.enableProfiling();
  const [cw, ch] = canvasSize();
  width = width || cw;
  height = height || ch;
  state.benchmarking = true;
  const cam = new OrbitCamera(state.header.view);
  const times = [];
  const settle = async () => { // let the instance buffer resize for this view before timing
    for (let i = 0; i < 6; i++) {
      r.render(cam.frame(width, height), { toCanvas: false });
      await r.device.queue.onSubmittedWorkDone();
      await new Promise((res) => setTimeout(res, 20));
    }
  };
  let wall = 0;
  try {
    await settle();
    // Frames are submitted back to back with at most 2 in flight, like a real render loop, so the
    // result is throughput (not CPU->GPU round-trip latency). GPU time per stage comes from timestamps.
    const gpu = [];
    const t0 = performance.now();
    let prev = Promise.resolve();
    for (let i = 0; i < frames; i++) {
      cam.orbit((2 * Math.PI) / frames, 0);
      const ts = performance.now();
      r.render(cam.frame(width, height), { toCanvas: false });
      const done = r.device.queue.onSubmittedWorkDone().then(() => times.push(performance.now() - ts));
      await prev;
      prev = done;
      if (r.stageMs) gpu.push(Object.values(r.stageMs).reduce((a, b) => a + b, 0));
    }
    await prev;
    wall = performance.now() - t0;
    state.lastGpuMs = gpu.length ? gpu.sort((a, b) => a - b)[gpu.length >> 1] : null;
  } finally {
    state.benchmarking = false;
    state.dirty = true;
  }
  const s = [...times].sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  const info = r.info || {};
  const res = {
    scene: sceneUrl().name, gaussians: state.loaded, file_bytes: state.totalBytes || state.bytes,
    width, height, frames, frame_ms: +(wall / frames).toFixed(3), p95_latency_ms: +q(0.95).toFixed(3),
    fps: +((1000 * frames) / wall).toFixed(1), latency_ms: +q(0.5).toFixed(2),
    gpu_ms: state.lastGpuMs && +state.lastGpuMs.toFixed(3), stage_ms: r.stageMs || null, instances: r.lastTotal,
    gpu: [info.vendor, info.architecture, info.description, info.device].filter(Boolean).join(" ") || "unknown",
    user_agent: navigator.userAgent, first_pixel_ms: state.firstPixelMs && Math.round(state.firstPixelMs),
    load_ms: state.doneMs && Math.round(state.doneMs), date: new Date().toISOString(),
  };
  return res;
}

async function runBenchUI() {
  const btn = $("#btn-bench");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const res = await bench();
    $("#bench-out").hidden = false;
    $("#bench-fps").textContent = `${res.fps} fps`;
    $("#bench-detail").textContent =
      `${res.frame_ms} ms/frame${res.gpu_ms ? ` (GPU ${res.gpu_ms} ms)` : ""} · ${res.width}×${res.height} · ${fmtN(res.gaussians)} Gaussians · ${res.gpu}`;
    $("#bench-json").value = JSON.stringify(res, null, 1);
    fetch("api/bench", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(res) })
      .then((r) => { if (r.ok) $("#bench-saved").hidden = false; }).catch(() => {});
    window.splatcraft.lastBench = res;
  } finally {
    btn.disabled = false;
    btn.textContent = "Benchmark";
  }
}

function cycleView(step) {
  const cams = state.header?.view?.cameras || [];
  if (!cams.length) return;
  state.viewIdx = (state.viewIdx + step + cams.length) % cams.length;
  state.camera.setExact(cams[state.viewIdx]);
  $("#btn-view").textContent = `View ${state.viewIdx + 1}/${cams.length}`;
  state.dirty = true;
}

async function main() {
  const { url, name } = sceneUrl();
  document.title = `${name} · SplatCraft`;
  $("#scene-name").textContent = name;
  fillScenePicker(name);
  if (params.get("ui") === "0") document.body.classList.add("no-ui");
  let r;
  try {
    r = await SplatRenderer.create($("#view"));
  } catch (e) {
    $("#nogpu").hidden = false;
    $("#loading").hidden = true;
    ready.reject(e);
    return;
  }
  state.renderer = r;
  r.enableProfiling(); // GPU time per stage when the browser exposes timestamp queries
  r.onCapacity = () => { state.dirty = true; };
  r.lost.then(() => fail(`GPU device lost: ${r.lostInfo?.message || "unknown reason"}`));
  const info = r.info || {};
  $("#stat-gpu").textContent = info.description || [info.vendor, info.architecture].filter(Boolean).join(" ") || "WebGPU";
  requestAnimationFrame(loop);
  try {
    await streamSplatc(url, {
      onHeader(h, total) {
        state.header = h;
        state.totalBytes = total;
        state.camera = new OrbitCamera(h.view);
        const el = $("#view");
        controls = attachControls(el, state.camera, () => { state.dirty = true; state.autorotate = false; });
        $("#loading-msg").textContent = "Decoding…";
      },
      onCodebook(cb) { r.setScene(state.header, cb); },
      onChunk(rec, first, n) {
        r.uploadChunk(rec, first, n);
        state.loaded = first + n;
        state.dirty = true;
        $("#loading").hidden = true;
      },
      onProgress(b, total) { state.bytes = b; state.totalBytes = total || state.totalBytes; },
    });
    state.doneMs = performance.now() - state.t0;
  } catch (e) {
    fail(`Could not load ${url}: ${e.message}`);
    return;
  }
  ready.resolve();
  if (params.get("bench") === "1") runBenchUI();
}

// Scene picker: the server's scene list, or scenes/index.json on static hosting (GitHub Pages).
async function fillScenePicker(current) {
  let list = null;
  for (const url of ["/api/scenes", "scenes/index.json"]) {
    try {
      const r = await fetch(url);
      if (r.ok) { list = await r.json(); break; }
    } catch { /* not available here */ }
  }
  if (!list || list.length < 2) return;
  const sel = $("#scene-pick");
  sel.innerHTML = list.map((s) => `<option value="${s.name}">${s.name} (${fmtMB(s.bytes)})</option>`).join("");
  sel.value = current;
  sel.hidden = false;
  sel.onchange = () => {
    const served = location.pathname.includes("/v/");
    location.href = served ? `/v/${encodeURIComponent(sel.value)}` : `?scene=${encodeURIComponent(sel.value)}`;
  };
}

// UI wiring
$("#btn-reset").onclick = () => { state.camera?.reset(); state.viewIdx = -1; $("#btn-view").textContent = "Views"; state.dirty = true; };
$("#btn-view").onclick = () => cycleView(1);
$("#btn-spin").onclick = () => { state.autorotate = !state.autorotate; };
$("#btn-bench").onclick = runBenchUI;
$("#btn-share").onclick = async () => {
  const link = location.href.split("#")[0];
  try { await navigator.clipboard.writeText(link); $("#btn-share").textContent = "Link copied"; } catch { prompt("Share this link", link); }
  setTimeout(() => { $("#btn-share").textContent = "Share"; }, 1600);
};
$("#bench-close").onclick = () => { $("#bench-out").hidden = true; };
$("#res-scale").value = String(state.scale);
$("#res-scale").onchange = (e) => { state.scale = Number(e.target.value); state.autoScale = false; state.dirty = true; };
window.addEventListener("resize", () => { state.dirty = true; });
window.addEventListener("keydown", (e) => {
  if (e.key === "v") cycleView(1);
  if (e.key === "r") $("#btn-reset").click();
});

// Programmatic hooks (used by the browser tests and the benchmark runner).
window.splatcraft = {
  ready: ready.promise, state, bench,
  async renderCamera(c) { // exact capture camera -> RGBA Uint8Array (as a plain array for transport)
    const r = state.renderer;
    const cam = new OrbitCamera(state.header.view);
    cam.setExact(c);
    state.benchmarking = true;
    try {
      for (let i = 0; i < 4; i++) { // re-render until the instance buffer is large enough for this view
        r.render(cam.frame(c.width, c.height), { toCanvas: false });
        await r.device.queue.onSubmittedWorkDone();
        await new Promise((res) => setTimeout(res, 60));
        if (!r.readPending && r.lastTotal <= r.capacity) break;
      }
      return Array.from(await r.readPixels());
    } finally {
      state.benchmarking = false;
      state.dirty = true;
    }
  },
};

main();
