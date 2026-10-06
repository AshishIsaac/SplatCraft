// Static order-0 rANS decoder with interleaved lanes: a transcription of
// src/splatcraft/compress/rans.py::decode (see that file for the stream layout).

export function ransDecode(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const n = dv.getUint32(0, true);
  const L = dv.getUint16(4, true);
  const scaleBits = dv.getUint8(6);
  let pos = 7;
  const freq = new Uint32Array(256);
  const present = buf.subarray(pos, pos + 32);
  pos += 32;
  for (let s = 0; s < 256; s++) {
    if (present[s >> 3] & (1 << (s & 7))) {
      freq[s] = dv.getUint16(pos, true);
      pos += 2;
    }
  }
  const counts = new Uint32Array(L);
  let totalWords = 0;
  for (let l = 0; l < L; l++) {
    counts[l] = dv.getUint32(pos, true);
    totalWords += counts[l];
    pos += 4;
  }
  // Copy the words to an aligned buffer (stream offsets inside the file can be odd).
  const words = new Uint16Array(buf.slice(pos, pos + 2 * totalWords).buffer);
  const out = new Uint8Array(n);
  if (n === 0) return out;
  const M = 1 << scaleBits;
  const mask = M - 1;
  const cum = new Uint32Array(256);
  const slot2sym = new Uint8Array(M);
  let c = 0;
  for (let s = 0; s < 256; s++) {
    cum[s] = c;
    slot2sym.fill(s, c, c + freq[s]);
    c += freq[s];
  }
  let base = 0;
  for (let l = 0; l < L; l++) {
    let p = base;
    let x = words[p] * 65536 + words[p + 1];
    p += 2;
    for (let i = l; i < n; i += L) {
      const slot = x & mask;
      const s = slot2sym[slot];
      out[i] = s;
      x = freq[s] * (x >>> scaleBits) + slot - cum[s];
      if (x < 65536) x = x * 65536 + words[p++];
    }
    base += counts[l];
  }
  return out;
}

export async function inflateRaw(bytes) {
  const ds = new DecompressionStream("deflate-raw");
  const stream = new Blob([bytes]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
