"""The ``.splatc`` scene format: pruned, quantised, vector-quantised, entropy-coded, progressive.

Layout::

    "SPLATC\\0\\1"  u32 header_length  header (UTF-8 JSON)  payload

The payload holds the SH codebook first, then *chunks*: disjoint sets of Gaussians ordered by
importance (summed blend weight over the training views), so the first few hundred KB already
render a recognisable scene while the rest streams in. Inside a chunk, Gaussians are in Morton
(Z-curve) order so neighbouring positions are close and their deltas small. Every attribute plane
is a separate stream coded with ``rans`` (or ``deflate``); see docs/format.md for the field table.

Per Gaussian (before entropy coding): position 3x16 bit, log-scale 3x8, rotation 4x8, opacity 8,
base colour 3x8, SH codebook index 16 bit  ->  15 bytes vs 248 bytes of float32 in a 3DGS .ply.
"""

from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import torch

from ..gs.gaussians import GaussianModel, inverse_sigmoid
from . import rans
from .kmeans import kmeans

MAGIC = b"SPLATC\x00\x01"
VERSION = 1


# ----------------------------------------------------------------------------- helpers

def _q(v: np.ndarray, lo, hi, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    span = np.maximum(np.asarray(hi, np.float64) - lo, 1e-12)
    t = np.clip((v - lo) / span, 0, 1)
    return np.round(t * levels).astype(np.uint32)


def _dq(q: np.ndarray, lo, hi, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    return (np.asarray(lo, np.float64) + q.astype(np.float64) / levels * (np.asarray(hi, np.float64) - lo)).astype(
        np.float32)


def _pack(data: np.ndarray, method: str) -> bytes:
    raw = np.ascontiguousarray(data, np.uint8).tobytes()
    if method == "rans":
        return rans.encode(raw)
    if method == "deflate":
        c = zlib.compressobj(9, zlib.DEFLATED, -15)
        return c.compress(raw) + c.flush()
    return raw


def _unpack(buf: bytes, method: str, n: int) -> np.ndarray:
    if method == "rans":
        out = rans.decode(buf)
    elif method == "deflate":
        out = np.frombuffer(zlib.decompress(buf, -15), np.uint8)
    else:
        out = np.frombuffer(buf, np.uint8)
    if len(out) != n:
        raise ValueError(f"stream decoded to {len(out)} bytes, expected {n}")
    return out


def morton3(q: np.ndarray) -> np.ndarray:
    """Interleave three 16-bit coordinates into a 48-bit Z-order key."""
    def spread(v):
        v = v.astype(np.uint64) & np.uint64(0xFFFF)
        v = (v | (v << np.uint64(32))) & np.uint64(0x0000FFFF0000FFFF)  # noqa: E501 - classic bit tricks
        v = (v | (v << np.uint64(16))) & np.uint64(0x00FF0000FF0000FF)
        v = (v | (v << np.uint64(8))) & np.uint64(0xF00F00F00F00F00F)
        v = (v | (v << np.uint64(4))) & np.uint64(0x30C30C30C30C30C3)
        v = (v | (v << np.uint64(2))) & np.uint64(0x9249249249249249)
        return v
    return spread(q[:, 0]) | (spread(q[:, 1]) << np.uint64(1)) | (spread(q[:, 2]) << np.uint64(2))


def _zigzag(d: np.ndarray) -> np.ndarray:
    d = d.astype(np.int64)
    return ((d << 1) ^ (d >> 63)).astype(np.uint32)


def _unzigzag(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.int64)
    return (z >> 1) ^ -(z & 1)


def chunk_sizes(n: int, levels: int) -> list[int]:
    """Doubling chunk sizes (1, 2, 4, ...) so the first chunk arrives fast."""
    levels = max(1, min(levels, max(1, n // 256)))
    w = np.array([2.0 ** i for i in range(levels)])
    sizes = np.floor(w / w.sum() * n).astype(int)
    sizes[-1] += n - sizes.sum()
    return [int(s) for s in sizes if s > 0]


# ----------------------------------------------------------------------------- encode

def encode_scene(g: GaussianModel, cc: dict, importance: torch.Tensor | None = None, view: dict | None = None,
                 log=print) -> tuple[bytes, dict]:
    method = cc.get("entropy", "rans")
    with torch.no_grad():
        means = g.means.detach().float().cpu().numpy().astype(np.float64)
        log_s = g.log_scales.detach().float().cpu().numpy().astype(np.float64)
        quats = torch.nn.functional.normalize(g.quats.detach().float(), dim=-1).cpu().numpy()
        opac = torch.sigmoid(g.opacity_logits.detach().float()).cpu().numpy()[:, 0]
        dc = g.sh_dc.detach().float().cpu().numpy()[:, 0, :]
        rest = g.sh_rest.detach().float()
        n_rest = rest.shape[1]
        imp = (importance.detach().float().cpu().numpy() if importance is not None
               else opac * np.exp(log_s.sum(1)) ** (1 / 3))

    n0 = len(means)
    keep = (opac >= cc.get("min_opacity", 0.0)) & np.isfinite(means).all(1) & np.isfinite(log_s).all(1)
    if importance is not None:
        keep &= imp > 0
    pk = float(cc.get("prune_keep", 1.0))
    if pk < 1.0:
        k = max(1, int(keep.sum() * pk))
        thr = np.sort(imp[keep])[::-1][k - 1]
        keep &= imp >= thr
    order = np.argsort(-imp[keep], kind="stable")
    sel = np.nonzero(keep)[0][order]
    means, log_s, quats, opac, dc, imp = means[sel], log_s[sel], quats[sel], opac[sel], dc[sel], imp[sel]
    n = len(sel)
    log(f"compress: kept {n:,} of {n0:,} Gaussians")

    pos_bits, sb, rb, ob, cb = (int(cc[k]) for k in ("position_bits", "scale_bits", "rotation_bits",
                                                         "opacity_bits", "color_bits"))
    if not 8 < pos_bits <= 24:
        raise ValueError("position_bits must be 9..24")
    pos_bytes = 2 if pos_bits <= 16 else 3
    bmin, bmax = means.min(0), means.max(0)
    s_lo, s_hi = np.percentile(log_s, 0.05, axis=0), log_s.max(0)
    c_lo, c_hi = np.percentile(dc, 0.02, axis=0), np.percentile(dc, 99.98, axis=0)

    # Spherical harmonics beyond the DC term: one shared VQ codebook (45 floats -> one 16-bit index).
    sh_info, labels = None, None
    payload = bytearray()
    if n_rest:
        x = rest[torch.as_tensor(sel, device=rest.device)].reshape(n, -1)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        x = x.to(dev)
        K = min(int(cc["sh_codebook"]), n)
        log(f"compress: k-means SH codebook ({K} x {x.shape[1]})")
        cent, lab = kmeans(x, K, int(cc["kmeans_iters"]), torch.as_tensor(imp, device=dev, dtype=torch.float32))
        cent = cent.cpu().numpy().astype(np.float64)
        labels = lab.cpu().numpy().astype(np.uint32)
        d_lo, d_hi = cent.min(0), cent.max(0)
        q = _q(cent, d_lo, d_hi, 8).astype(np.uint8)
        blob = _pack(q.T.copy(), method)  # dimension-major: each coefficient's values sit together
        sh_info = {"codebook_size": int(K), "dims": int(x.shape[1]), "bits": 8, "min": d_lo.tolist(),
                   "max": d_hi.tolist(), "offset": 0, "length": len(blob), "raw": int(q.size)}
        payload += blob

    sizes = chunk_sizes(n, int(cc.get("chunks", 1)))
    chunks = []
    attr_bytes: dict[str, int] = {}
    start = 0
    for size in sizes:
        idx = np.arange(start, start + size)
        start += size
        qp = _q(means[idx], bmin, bmax, pos_bits)
        mo = np.argsort(morton3(qp), kind="stable")
        idx, qp = idx[mo], qp[mo]
        streams = {}
        # Positions: per-axis delta along the Morton curve, zig-zag, split into byte planes.
        # Deltas wrap modulo 2^(8 * bytes) so the zig-zag value fits the stored bytes exactly.
        mod = 1 << (8 * pos_bytes)
        d = np.diff(qp.astype(np.int64), axis=0, prepend=0)
        d = (d + mod // 2) % mod - mod // 2
        z = _zigzag(d)
        for a, ax in enumerate("xyz"):
            streams[f"p{ax}0"] = (z[:, a] & 0xFF).astype(np.uint8)
            streams[f"p{ax}1"] = ((z[:, a] >> 8) & 0xFF).astype(np.uint8)
            if pos_bytes > 2:
                streams[f"p{ax}2"] = ((z[:, a] >> 16) & 0xFF).astype(np.uint8)
        streams["scale"] = _q(log_s[idx], s_lo, s_hi, sb).astype(np.uint8).T.ravel()
        qt = quats[idx].copy()
        qt[qt[:, 0] < 0] *= -1  # q and -q are the same rotation: keep w >= 0
        streams["rot"] = np.concatenate([_q(qt[:, :1], 0.0, 1.0, rb), _q(qt[:, 1:], -1.0, 1.0, rb)], 1
                                        ).astype(np.uint8).T.ravel()
        streams["opacity"] = _q(opac[idx], 0.0, 1.0, ob).astype(np.uint8)
        streams["color"] = _q(dc[idx], c_lo, c_hi, cb).astype(np.uint8).T.ravel()
        if labels is not None:
            streams["sh0"] = (labels[idx] & 0xFF).astype(np.uint8)
            streams["sh1"] = (labels[idx] >> 8).astype(np.uint8)
        chunk = {"count": int(size), "offset": len(payload), "streams": []}
        for name, arr in streams.items():
            blob = _pack(arr, method)
            chunk["streams"].append({"name": name, "offset": len(payload), "length": len(blob), "raw": int(arr.size)})
            payload += blob
            key = "position" if name[0] == "p" and len(name) == 3 else "sh_index" if name.startswith("sh") else name
            attr_bytes[key] = attr_bytes.get(key, 0) + len(blob)
        chunk["length"] = len(payload) - chunk["offset"]
        chunks.append(chunk)

    sh_degree = int(round(np.sqrt(n_rest + 1))) - 1
    header = {
        "format": "splatc", "version": VERSION, "count": int(n), "sh_degree": sh_degree, "entropy": method,
        "position": {"bits": pos_bits, "min": bmin.tolist(), "max": bmax.tolist()},
        "scale": {"bits": sb, "min": s_lo.tolist(), "max": s_hi.tolist()},
        "rotation": {"bits": rb}, "opacity": {"bits": ob},
        "color": {"bits": cb, "min": c_lo.tolist(), "max": c_hi.tolist()},
        "sh": sh_info, "chunks": chunks, "view": view or {},
    }
    hb = json.dumps(header, separators=(",", ":")).encode()
    data = MAGIC + struct.pack("<I", len(hb)) + hb + bytes(payload)
    if sh_info:
        attr_bytes["sh_codebook"] = sh_info["length"]
    attr_bytes["header"] = len(hb) + 12
    report = {"input_gaussians": n0, "gaussians": n, "bytes": len(data), "ply_bytes": n0 * (17 + 3 * n_rest) * 4, "attribute_bytes": attr_bytes,
              "bytes_per_gaussian": round(len(data) / max(n, 1), 2), "chunks": [c["count"] for c in chunks],
              "entropy": method}
    report["ratio_vs_ply"] = round(report["ply_bytes"] / len(data), 2)
    return data, report


# ----------------------------------------------------------------------------- decode

def read_header(data: bytes) -> tuple[dict, int]:
    if data[:8] != MAGIC:
        raise ValueError("not a .splatc file")
    (hl,) = struct.unpack_from("<I", data, 8)
    return json.loads(data[12:12 + hl].decode()), 12 + hl


def decode_scene(data: bytes, device="cpu") -> GaussianModel:
    h, base = read_header(data)
    method = h["entropy"]
    payload = memoryview(data)[base:]

    def stream(s):
        return _unpack(bytes(payload[s["offset"]:s["offset"] + s["length"]]), method, s["raw"])

    sh = h.get("sh")
    cent = None
    if sh:
        q = stream(sh).reshape(sh["dims"], sh["codebook_size"]).T
        cent = _dq(q, np.asarray(sh["min"]), np.asarray(sh["max"]), 8)
    P, S, C = h["position"], h["scale"], h["color"]
    parts = {k: [] for k in ("means", "log_s", "quat", "opac", "dc", "rest")}
    for ch in h["chunks"]:
        n = ch["count"]
        st = {s["name"]: stream(s) for s in ch["streams"]}
        qp = np.zeros((n, 3), np.int64)
        for a, ax in enumerate("xyz"):
            z = st[f"p{ax}0"].astype(np.int64) | (st[f"p{ax}1"].astype(np.int64) << 8)
            if f"p{ax}2" in st:
                z |= st[f"p{ax}2"].astype(np.int64) << 16
            mod = 1 << (8 * (3 if f"p{ax}2" in st else 2))
            qp[:, a] = np.cumsum(_unzigzag(z)) % mod
        parts["means"].append(_dq(qp, np.asarray(P["min"]), np.asarray(P["max"]), P["bits"]))
        parts["log_s"].append(_dq(st["scale"].reshape(3, n).T, np.asarray(S["min"]), np.asarray(S["max"]), S["bits"]))
        r = st["rot"].reshape(4, n).T
        rb = h["rotation"]["bits"]
        parts["quat"].append(np.concatenate([_dq(r[:, :1], 0.0, 1.0, rb), _dq(r[:, 1:], -1.0, 1.0, rb)], 1))
        parts["opac"].append(_dq(st["opacity"], 0.0, 1.0, h["opacity"]["bits"]))
        parts["dc"].append(_dq(st["color"].reshape(3, n).T, np.asarray(C["min"]), np.asarray(C["max"]), C["bits"]))
        if cent is not None:
            lab = st["sh0"].astype(np.int64) | (st["sh1"].astype(np.int64) << 8)
            parts["rest"].append(cent[lab])
    cat = {k: np.concatenate(v) for k, v in parts.items() if v}
    deg = h["sh_degree"]
    g = GaussianModel(deg, device)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=g.device)  # noqa: E731
    g.means = t(cat["means"])
    g.log_scales = t(cat["log_s"])
    g.quats = t(cat["quat"])
    g.opacity_logits = inverse_sigmoid(t(cat["opac"]).clamp(1e-4, 1 - 1e-4))[:, None]
    g.sh_dc = t(cat["dc"])[:, None, :]
    nr = (deg + 1) ** 2 - 1
    g.sh_rest = t(cat["rest"]).reshape(-1, nr, 3) if "rest" in cat else torch.zeros(len(g.means), nr, 3,
                                                                                  device=g.device)
    g.active_sh_degree = deg
    g._reset_stats()
    return g
