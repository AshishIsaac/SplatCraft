# The `.splatc` format (version 1)

A compressed, streamable 3D Gaussian Splatting scene. Writer: `src/splatcraft/compress/codec.py`
(`encode_scene`). Readers: the same file (`decode_scene`, Python) and `web/js/loader.js` (browser,
streaming). Both are tested against each other: the WebGPU render of a decoded file matches the
PyTorch render of the same file to > 45 dB PSNR (`tests/test_viewer_browser.py`).

## Layout

```
offset 0   8 bytes   magic "SPLATC\0\1"
offset 8   u32 LE    header length H
offset 12  H bytes   header, UTF-8 JSON
offset 12+H          payload: SH codebook stream, then chunk 0 streams, chunk 1 streams, ...
```

All `offset` values in the header are relative to the start of the payload. The payload is ordered
so that a reader can act on every section as soon as its last byte arrives: header -> codebook ->
chunk 0 (the most important Gaussians) -> chunk 1 -> ...

## Header

| Field | Meaning |
|---|---|
| `count` | number of Gaussians (sum of the chunk counts) |
| `sh_degree` | spherical-harmonic degree (0-3) |
| `entropy` | stream coding: `rans` (default), `deflate` (raw DEFLATE, RFC 1951) or `none` |
| `position` | `{bits, min[3], max[3]}`: uniform quantisation box for the means |
| `scale` | `{bits, min[3], max[3]}`: per-axis range of the *log* scales |
| `rotation`, `opacity` | `{bits}` |
| `color` | `{bits, min[3], max[3]}`: range of the SH DC coefficients |
| `sh` | `{codebook_size K, dims D, bits, min[D], max[D], offset, length, raw}` or `null` for degree 0 |
| `chunks` | `[{count, offset, length, streams: [{name, offset, length, raw}]}]` |
| `view` | viewer defaults: `target`, `up`, `eye`, `fov_y`, `extent`, `background`, and up to 12 capture `cameras` (`R`, `t`, `fx`, `fy`, `cx`, `cy`, `width`, `height`; OpenCV convention, world -> camera) |

## Streams

Every stream is a byte array (`raw` bytes before coding). Quantisation is
`q = round((v - min) / (max - min) * (2^bits - 1))`, decoding is the inverse.

| Stream | Bytes per Gaussian | Content |
|---|---|---|
| `px0 px1 py0 py1 pz0 pz1` | 6 | positions: 16-bit per axis. Within a chunk, Gaussians are in Morton (Z-curve) order of their quantised position; each axis stores the zig-zag coded delta to the previous Gaussian, **modulo 2^16**, split into a low-byte and a high-byte plane |
| `scale` | 3 | log-scale, 8 bit per axis, planar (all x, then all y, then all z) |
| `rot` | 4 | unit quaternion with w >= 0: w in [0, 1], x, y, z in [-1, 1], 8 bit each, planar |
| `opacity` | 1 | sigmoid(opacity), 8 bit |
| `color` | 3 | SH DC coefficient per channel, 8 bit, planar |
| `sh0 sh1` | 2 | index into the SH codebook (low byte, high byte) |

The SH codebook stream holds `K x D` bytes in *dimension-major* order (all K values of
coefficient 0, then coefficient 1, ...), dequantised per dimension with `sh.min[d]`, `sh.max[d]`.
A codeword's D = 3 * ((degree + 1)^2 - 1) floats are ordered `[coefficient][channel]` (45 floats at
degree 3).

Byte planes and planar layouts are what make the entropy coder effective: each plane has a narrow,
stable distribution (high bytes of small deltas are almost always 0, 1 or 255).

## rANS streams

`entropy = "rans"`: static order-0 rANS over bytes with interleaved lanes
(`src/splatcraft/compress/rans.py`, `web/js/rans.js`):

```
u32 n, u16 lanes, u8 scale_bits, u8[32] symbol bitmap, u16[k] frequencies (k = bitmap popcount),
u32[lanes] words per lane, then each lane's u16 words (state hi, state lo, renormalisation words)
```

Symbol `i` belongs to lane `i % lanes`. State lives in [2^16, 2^32), renormalisation emits/reads
one 16-bit word at a time, so decoding a lane is: `slot = x & (M-1); s = slot2sym[slot];
x = f[s] * (x >> scale_bits) + slot - cum[s]; if x < 2^16: x = (x << 16) | next_word`.
On the lego demo scene the 73 coded streams total 1.4 % above their order-0 entropy, frequency tables
and lane headers included (0.4 % on a single 1M-symbol stream).

## Chunks and progressive loading

Gaussians are sorted by importance (summed alpha-blending weight over all training views) and cut
into chunks of doubling size (1, 2, 4, ... parts of the total). Chunk 0 is therefore small and
holds the Gaussians that matter most: the viewer shows a recognisable scene after a few percent of
the download and refines it as the remaining chunks stream in.
