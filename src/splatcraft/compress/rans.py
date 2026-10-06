"""Static order-0 rANS entropy coder over bytes, with interleaved lanes.

Format of one coded stream (little endian)::

    u32  n            number of symbols
    u16  lanes        independent rANS states (symbol i belongs to lane i % lanes)
    u8   scale_bits   frequencies sum to 1 << scale_bits
    u8[32]            bitmap of the symbols that occur
    u16[k]            their normalised frequencies (k = popcount of the bitmap)
    u32[lanes]        number of 16-bit words in each lane
    u16[...]          lane 0 words, lane 1 words, ...  (each lane: final state hi, lo, then renorm words)

State x lives in [2^16, 2^32); the encoder emits at most one 16-bit word per symbol, so every lane
is a sequence of (decode symbol, maybe read one word) steps. Lanes are what make this fast in numpy
(the encoder and decoder below are vectorised across lanes) and in JavaScript (``web/js/rans.js``
is a straight transcription of ``decode``).
"""

from __future__ import annotations

import struct

import numpy as np

RANS_L = 1 << 16
SCALE_BITS = 14


def _lanes_for(n: int) -> int:
    return int(np.clip(n // 2048, 1, 256))


def normalize_freqs(counts: np.ndarray, scale_bits: int = SCALE_BITS) -> np.ndarray:
    """Scale counts to sum to 2^scale_bits, keeping every occurring symbol >= 1."""
    M = 1 << scale_bits
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total == 0:
        return np.zeros(256, np.int64)
    f = np.where(counts > 0, np.maximum(1, np.floor(counts * M / total)), 0).astype(np.int64)
    diff = M - f.sum()
    order = np.argsort(-counts)
    i = 0
    while diff != 0:  # hand out / take back the rounding remainder, most frequent symbols first
        s = order[i % 256]
        if counts[s] > 0 and (diff > 0 or f[s] > 1):
            step = 1 if diff > 0 else -1
            f[s] += step
            diff -= step
        i += 1
    return f


def encode(data: np.ndarray | bytes, scale_bits: int = SCALE_BITS, lanes: int | None = None) -> bytes:
    sym = np.frombuffer(bytes(data), np.uint8) if isinstance(data, bytes | bytearray) else np.ascontiguousarray(
        data, np.uint8).ravel()
    n = len(sym)
    L = lanes or _lanes_for(n)
    freqs = normalize_freqs(np.bincount(sym, minlength=256), scale_bits)
    cum = np.concatenate([[0], np.cumsum(freqs)[:-1]]).astype(np.uint64)
    steps = -(-n // L)
    pad = np.zeros(steps * L, np.uint8)
    pad[:n] = sym
    grid = pad.reshape(steps, L)  # grid[j, l] = symbol j*L + l
    present = (np.arange(steps * L).reshape(steps, L) < n)
    x = np.full(L, RANS_L, np.uint64)
    words = np.zeros((steps, L), np.uint16)
    emitted = np.zeros((steps, L), bool)
    f_tab = freqs.astype(np.uint64)
    shift = np.uint64(32 - scale_bits)
    for j in range(steps - 1, -1, -1):
        s = grid[j]
        act = present[j]
        f = f_tab[s]
        f_safe = np.where(act, f, np.uint64(1))
        x_max = f_safe << shift  # ((RANS_L >> scale_bits) << 16) * f
        ren = act & (x >= x_max)
        words[j, ren] = (x[ren] & np.uint64(0xFFFF)).astype(np.uint16)
        emitted[j] = ren
        x = np.where(ren, x >> np.uint64(16), x)
        nx = ((x // f_safe) << np.uint64(scale_bits)) + (x % f_safe) + cum[s]
        x = np.where(act, nx, x)
    lane_words = []
    for l in range(L):
        w = words[emitted[:, l], l]
        head = np.array([int(x[l]) >> 16, int(x[l]) & 0xFFFF], np.uint16)
        lane_words.append(np.concatenate([head, w]))
    bitmap = np.packbits((freqs > 0).astype(np.uint8), bitorder="little")
    out = [struct.pack("<IHB", n, L, scale_bits), bitmap.tobytes(), freqs[freqs > 0].astype("<u2").tobytes(),
           np.array([len(w) for w in lane_words], "<u4").tobytes()]
    out += [w.astype("<u2").tobytes() for w in lane_words]
    return b"".join(out)


def decode(buf: bytes | memoryview) -> np.ndarray:
    buf = memoryview(buf)
    n, L, scale_bits = struct.unpack_from("<IHB", buf, 0)
    pos = 7
    present = np.unpackbits(np.frombuffer(buf[pos:pos + 32], np.uint8), bitorder="little").astype(bool)
    pos += 32
    k = int(present.sum())
    freqs = np.zeros(256, np.int64)
    freqs[present] = np.frombuffer(buf[pos:pos + 2 * k], "<u2")
    pos += 2 * k
    counts = np.frombuffer(buf[pos:pos + 4 * L], "<u4").astype(np.int64)
    pos += 4 * L
    allw = np.frombuffer(buf[pos:pos + 2 * int(counts.sum())], "<u2").astype(np.uint64)
    if n == 0:
        return np.zeros(0, np.uint8)
    M = 1 << scale_bits
    slot2sym = np.repeat(np.arange(256), freqs).astype(np.uint8)
    cum = np.concatenate([[0], np.cumsum(freqs)[:-1]]).astype(np.uint64)
    f_tab = freqs.astype(np.uint64)
    base = np.concatenate([[0], np.cumsum(counts)[:-1]])
    x = (allw[base] << np.uint64(16)) | allw[base + 1]
    ptr = base + 2
    steps = -(-n // L)
    out = np.zeros(steps * L, np.uint8)
    mask = np.uint64(M - 1)
    sb = np.uint64(scale_bits)
    lane_ids = np.arange(L)
    for j in range(steps):
        act = j * L + lane_ids < n
        slot = x & mask
        s = slot2sym[slot.astype(np.int64)]
        out[j * L:(j + 1) * L] = s
        nx = f_tab[s] * (x >> sb) + slot - cum[s]
        x = np.where(act, nx, x)
        ren = act & (x < RANS_L)
        if ren.any():
            idx = ptr[ren]
            x[ren] = (x[ren] << np.uint64(16)) | allw[idx]
            ptr[ren] += 1
    return out[:n]
