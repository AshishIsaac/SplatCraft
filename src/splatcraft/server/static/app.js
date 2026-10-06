// SplatCraft dashboard: upload, job progress, scene gallery, benchmark table.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const mb = (b) => (b / 1e6).toFixed(b < 1e7 ? 2 : 1) + " MB";
const num = (n) => (n == null ? "–" : Number(n).toLocaleString("en-US"));
const dur = (s) => (s == null ? "" : s < 90 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`);
let file = null;

function pickFile(f) {
  file = f;
  $("#file-name").textContent = f ? `${f.name} (${mb(f.size)})` : "";
  $("#go").disabled = !f;
  if (f && !$("#name").value) $("#name").value = f.name.replace(/\.[^.]+$/, "").slice(0, 40);
}
$("#file").onchange = (e) => pickFile(e.target.files[0]);
const drop = $("#drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));

$("#upload").onsubmit = (e) => {
  e.preventDefault();
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  fd.append("name", $("#name").value);
  fd.append("preset", $("#preset").value);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  $("#upload-bar").hidden = false;
  $("#go").disabled = true;
  xhr.upload.onprogress = (ev) => {
    if (ev.lengthComputable) $("#upload-bar > div").style.width = `${(100 * ev.loaded) / ev.total}%`;
    $("#upload-msg").textContent = `Uploading… ${mb(ev.loaded)}`;
  };
  xhr.onload = () => {
    $("#upload-bar").hidden = true;
    if (xhr.status >= 200 && xhr.status < 300) {
      $("#upload-msg").textContent = "Uploaded. Your scene is being built below; you can close this page and come back.";
      pickFile(null);
      $("#file").value = "";
      $("#name").value = "";
      refreshJobs();
    } else {
      let msg = xhr.statusText;
      try { msg = JSON.parse(xhr.responseText).detail || msg; } catch {}
      $("#upload-msg").textContent = `Upload failed: ${msg}`;
      $("#go").disabled = false;
    }
  };
  xhr.onerror = () => { $("#upload-msg").textContent = "Upload failed (network)"; $("#go").disabled = false; };
  xhr.send(fd);
};

async function copy(text, btn) {
  try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; } catch { prompt("Link", text); }
  setTimeout(() => { btn.textContent = "Copy link"; }, 1500);
}
window.copyLink = (path, btn) => copy(location.origin + path, btn);
window.cancelJob = async (id) => { await fetch(`/api/jobs/${id}/cancel`, { method: "POST" }); refreshJobs(); };

async function refreshJobs() {
  const jobs = await (await fetch("/api/jobs")).json();
  $("#jobs-section").hidden = jobs.length === 0;
  $("#jobs").innerHTML = jobs.slice(0, 20).map((j) => {
    const running = j.state === "running" || j.state === "queued";
    const eta = j.state === "running" && j.stage === "train" && j.eta_s ? ` · ~${dur(j.eta_s)} left` : "";
    return `<div class="card job">
      <div><span class="title">${esc(j.name)}</span><span class="chip ${esc(j.state)}">${esc(j.state === "running" ? j.stage : j.state)}</span>
        <span class="muted" style="font-size:13px"> ${esc(j.file)}${eta}</span></div>
      <div class="actions">
        ${j.url ? `<a class="btn primary" href="${esc(j.url)}">Open</a><button onclick="copyLink('${esc(j.url)}', this)">Copy link</button>` : ""}
        ${running ? `<button onclick="cancelJob('${esc(j.id)}')">Cancel</button>` : ""}
      </div>
      ${running ? `<div class="bar"><div style="width:${(100 * (j.progress || 0)).toFixed(1)}%"></div></div>` : ""}
      <div class="msg">${esc(j.message)}</div>
    </div>`;
  }).join("");
  if (jobs.some((j) => j.state === "running" || j.state === "queued")) setTimeout(refreshJobs, 2000);
  else refreshScenes();
}

async function refreshScenes() {
  const scenes = await (await fetch("/api/scenes")).json();
  $("#scenes").innerHTML = scenes.length ? scenes.map((s) => `<div class="card scene">
      <div class="name">${esc(s.name)}${s.demo ? '<span class="chip">demo</span>' : ""}</div>
      <div class="meta">${mb(s.bytes)} · ${num(s.gaussians)} Gaussians · SH degree ${esc(s.sh_degree)}</div>
      <div class="actions"><a class="btn primary" href="${esc(s.url)}">Open</a>
        <button onclick="copyLink('${esc(s.url)}', this)">Copy link</button>
        <a class="btn" href="${esc(s.file)}" download>.splatc</a></div>
    </div>`).join("") : '<div class="empty">No scenes yet: upload a video above.</div>';
}

function table(el, cols, rows) {
  el.innerHTML = `<thead><tr>${cols.map((c) => `<th class="${c.l ? "l" : ""}">${esc(c.h)}</th>`).join("")}</tr></thead>
    <tbody>${rows.length ? rows.map((r) => `<tr>${cols.map((c) => `<td class="${c.l ? "l" : ""}">${c.f(r)}</td>`).join("")}</tr>`).join("")
      : `<tr><td class="l muted" colspan="${cols.length}">No measurements yet</td></tr>`}</tbody>`;
}

async function refreshBench() {
  const b = await (await fetch("/api/bench")).json();
  const rows = b.reference?.rows || [];
  table($("#bench-ref"), [
    { h: "Scene", l: 1, f: (r) => esc(r.scene) }, { h: "Model", l: 1, f: (r) => esc(r.variant) },
    { h: "PSNR ↑", f: (r) => r.psnr?.toFixed(2) ?? "–" }, { h: "SSIM ↑", f: (r) => r.ssim?.toFixed(3) ?? "–" },
    { h: "LPIPS ↓", f: (r) => r.lpips?.toFixed(3) ?? "–" }, { h: "Size", f: (r) => (r.bytes ? mb(r.bytes) : "–") },
    { h: "Gaussians", f: (r) => num(r.gaussians) },
    ...(b.reference?.devices || []).map((d) => ({ h: `FPS ${d}`, f: (r) => r.fps?.[d] ?? "–" })),
  ], rows);
  table($("#bench-subs"), [
    { h: "Date", l: 1, f: (r) => esc((r.date || "").slice(0, 10)) }, { h: "GPU", l: 1, f: (r) => esc(r.gpu) },
    { h: "Scene", l: 1, f: (r) => esc(r.scene) }, { h: "Resolution", f: (r) => `${r.width}×${r.height}` },
    { h: "FPS", f: (r) => r.fps }, { h: "ms / frame", f: (r) => r.frame_ms ?? "–" },
    { h: "Gaussians", f: (r) => num(r.gaussians) },
  ], b.submissions || []);
}

fetch("/api/health").then((r) => r.json()).then((h) => { $("#version").textContent = `v${h.version}`; });
refreshJobs();
refreshScenes();
refreshBench();
