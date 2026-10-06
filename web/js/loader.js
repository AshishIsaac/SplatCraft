// Streaming .splatc loader: parses the header as soon as it arrives, then decodes the SH codebook and
// every chunk the moment its last byte is in, so the first (most important) Gaussians render while the
// rest of the file is still downloading. Mirrors src/splatcraft/compress/codec.py::decode_scene.

import { inflateRaw, ransDecode } from "./rans.js";

const MAGIC = [83, 80, 76, 65, 84, 67, 0, 1]; // "SPLATC\0\1"
export const RECORD = 16; // floats per decoded Gaussian (see renderer.js)

async function unpack(bytes, method, raw) {
  let out;
  if (method === "rans") out = ransDecode(bytes);
  else if (method === "deflate") out = await inflateRaw(bytes);
  else out = bytes;
  if (out.length !== raw) throw new Error(`stream decoded to ${out.length} bytes, expected ${raw}`);
  return out;
}

const deq = (q, lo, hi, levels) => lo + (q / levels) * (hi - lo);

export async function decodeCodebook(h, payload) {
  const sh = h.sh;
  if (!sh) return null;
  const q = await unpack(payload.subarray(sh.offset, sh.offset + sh.length), h.entropy, sh.raw);
  const K = sh.codebook_size, D = sh.dims;
  const cb = new Float32Array(K * D);
  for (let d = 0; d < D; d++) {
    const lo = sh.min[d], hi = sh.max[d];
    for (let k = 0; k < K; k++) cb[k * D + d] = deq(q[d * K + k], lo, hi, 255);
  }
  return cb;
}

export async function decodeChunk(h, payload, ch) {
  const n = ch.count;
  const st = {};
  for (const s of ch.streams) {
    st[s.name] = await unpack(payload.subarray(s.offset, s.offset + s.length), h.entropy, s.raw);
  }
  const rec = new Float32Array(n * RECORD);
  const recU = new Uint32Array(rec.buffer);
  const P = h.position, S = h.scale, C = h.color;
  const pl = (1 << P.bits) - 1, sl = (1 << S.bits) - 1, rl = (1 << h.rotation.bits) - 1;
  const ol = (1 << h.opacity.bits) - 1, cl = (1 << C.bits) - 1;
  ["x", "y", "z"].forEach((ax, a) => {
    const b0 = st[`p${ax}0`], b1 = st[`p${ax}1`], b2 = st[`p${ax}2`];
    const mask = b2 ? 0xffffff : 0xffff; // deltas wrap modulo 2^(8 * bytes)
    let acc = 0;
    for (let i = 0; i < n; i++) {
      let z = b0[i] | (b1[i] << 8);
      if (b2) z |= b2[i] << 16;
      acc = (acc + ((z >>> 1) ^ -(z & 1))) & mask; // zig-zag delta along the Morton curve
      rec[i * RECORD + a] = deq(acc, P.min[a], P.max[a], pl);
    }
  });
  const sc = st.scale, rot = st.rot, op = st.opacity, col = st.color, s0 = st.sh0, s1 = st.sh1;
  for (let i = 0; i < n; i++) {
    const o = i * RECORD;
    rec[o + 3] = op[i] / ol;
    for (let a = 0; a < 3; a++) rec[o + 4 + a] = Math.exp(deq(sc[a * n + i], S.min[a], S.max[a], sl));
    recU[o + 7] = s0 ? s0[i] | (s1[i] << 8) : 0;
    let w = rot[i] / rl;
    let x = (rot[n + i] / rl) * 2 - 1, y = (rot[2 * n + i] / rl) * 2 - 1, z = (rot[3 * n + i] / rl) * 2 - 1;
    const len = Math.hypot(w, x, y, z) || 1;
    rec[o + 8] = w / len; rec[o + 9] = x / len; rec[o + 10] = y / len; rec[o + 11] = z / len;
    for (let a = 0; a < 3; a++) rec[o + 12 + a] = deq(col[a * n + i], C.min[a], C.max[a], cl);
  }
  return rec;
}

function parseHeader(bytes) {
  for (let i = 0; i < 8; i++) if (bytes[i] !== MAGIC[i]) throw new Error("not a .splatc file");
  const hl = new DataView(bytes.buffer, bytes.byteOffset).getUint32(8, true);
  if (bytes.length < 12 + hl) return null;
  return { header: JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + hl))), base: 12 + hl };
}

// Fetches `url` and calls: onHeader(header, totalBytes), onCodebook(Float32Array|null),
// onChunk(records, firstIndex, count), onProgress(receivedBytes, totalBytes). Resolves when done.
export async function streamSplatc(url, cb) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  let buf = new Uint8Array(total || 1 << 20);
  let len = 0;
  let parsed = null, payload = null, nextChunk = 0, first = 0, codebookDone = false;
  const reader = res.body.getReader();

  const pump = async (final) => {
    if (!parsed) {
      if (len < 12) return;
      parsed = parseHeader(buf.subarray(0, len));
      if (!parsed) return;
      cb.onHeader?.(parsed.header, total);
    }
    const h = parsed.header;
    payload = buf.subarray(parsed.base, len);
    if (!codebookDone) {
      if (h.sh && payload.length < h.sh.offset + h.sh.length) return;
      cb.onCodebook?.(await decodeCodebook(h, payload));
      codebookDone = true;
    }
    while (nextChunk < h.chunks.length) {
      const ch = h.chunks[nextChunk];
      if (payload.length < ch.offset + ch.length) break;
      const rec = await decodeChunk(h, payload, ch);
      cb.onChunk?.(rec, first, ch.count, nextChunk);
      first += ch.count;
      nextChunk++;
      await new Promise((r) => setTimeout(r, 0)); // let a frame render between chunks
    }
    if (final && nextChunk < h.chunks.length) throw new Error("file ended before all chunks arrived");
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (len + value.length > buf.length) {
      const bigger = new Uint8Array(Math.max(buf.length * 2, len + value.length));
      bigger.set(buf.subarray(0, len));
      buf = bigger;
    }
    buf.set(value, len);
    len += value.length;
    cb.onProgress?.(len, total);
    await pump(false);
  }
  await pump(true);
  return parsed.header;
}
