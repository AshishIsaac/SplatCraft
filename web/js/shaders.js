// WGSL for the tile-based splat renderer. Per frame:
//   preprocess  3D Gaussian -> 2D splat (EWA projection, SH colour, opacity-aware tile rectangle)
//   scan        exclusive prefix sum of per-Gaussian tile counts -> instance offsets
//   emit        one (tile | depth) key per (Gaussian, overlapped tile) instance
//   sort        4-pass LSD radix sort of the 32-bit keys (histogram -> scan -> stable scatter)
//   ranges      start/end of every tile's run in the sorted instance list
//   raster      one workgroup per 16x16 tile: front-to-back alpha blending with early termination
//   blit        output texture -> canvas
// Conventions match the PyTorch trainer exactly (src/splatcraft/gs/rasterizer.py).

const UNIFORMS = /* wgsl */ `
struct Uniforms {
  view: mat4x4<f32>,     // world -> camera (OpenCV: x right, y down, z forward)
  campos: vec4<f32>,
  focal: vec4<f32>,      // fx, fy, cx, cy (pixels)
  bg: vec4<f32>,
  size: vec4<u32>,       // width, height, tilesX, tilesY
  info: vec4<u32>,       // gaussian count, sh degree, depth bits, instance capacity
  extra: vec4<f32>,      // near plane, 2D blur, -, -
};
// Exact test: does the ellipse q(d) = a dx^2 + 2 b dx dy + c dy^2 <= k2 (alpha >= 1/255) touch tile (tx, ty)?
// q is convex, so outside the tile its minimum over the tile lies on an edge, where it is a 1D
// quadratic with a closed-form clamped minimiser. Conservative: skipping a tile never drops a pixel.
fn tileHit(mean: vec2<f32>, co: vec3<f32>, k2: f32, tx: u32, ty: u32) -> bool {
  let lo = vec2<f32>(f32(tx), f32(ty)) * 16.0 - mean;
  let hi = lo + vec2<f32>(16.0);
  if (lo.x <= 0.0 && hi.x >= 0.0 && lo.y <= 0.0 && hi.y >= 0.0) { return true; }
  let a = co.x; let b = co.y; let c = co.z;
  var m = 3.0e38;
  for (var e = 0u; e < 2u; e++) {
    let dx = select(lo.x, hi.x, e == 1u);          // vertical edges x = const
    let dy = clamp(-b * dx / c, lo.y, hi.y);
    m = min(m, a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
    let dy2 = select(lo.y, hi.y, e == 1u);         // horizontal edges y = const
    let dx2 = clamp(-b * dy2 / a, lo.x, hi.x);
    m = min(m, a * dx2 * dx2 + 2.0 * b * dx2 * dy2 + c * dy2 * dy2);
  }
  return m <= k2;
}

struct Splat {
  mean: vec2<f32>,
  conic: vec4<f32>,      // inverse 2D covariance a, b, c + opacity
  color: vec4<f32>,      // rgb + camera depth
  rect: vec2<u32>,       // tile rectangle: x0 | y0 << 16, x1 | y1 << 16 (exclusive)
};
`;

export const PREPROCESS = UNIFORMS + /* wgsl */ `
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> gauss: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> shcb: array<f32>;
@group(0) @binding(3) var<storage, read_write> splats: array<Splat>;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;

const C0 = 0.28209479177387814;
const C1 = 0.4886025119029199;

fn cb3(o: u32) -> vec3<f32> { return vec3<f32>(shcb[o], shcb[o + 1u], shcb[o + 2u]); }

// Fully unrolled (no dynamically indexed private array: that spills to scratch memory on some GPUs).
fn shColor(d: vec3<f32>, dc: vec3<f32>, idx: u32, deg: u32) -> vec3<f32> {
  var c = C0 * dc;
  if (deg == 0u) { return c; }
  let b = idx * 45u;
  let x = d.x; let y = d.y; let z = d.z;
  c += -C1 * y * cb3(b) + C1 * z * cb3(b + 3u) - C1 * x * cb3(b + 6u);
  if (deg == 1u) { return c; }
  let xx = x * x; let yy = y * y; let zz = z * z;
  c += 1.0925484305920792 * x * y * cb3(b + 9u) - 1.0925484305920792 * y * z * cb3(b + 12u)
     + 0.31539156525252005 * (2.0 * zz - xx - yy) * cb3(b + 15u) - 1.0925484305920792 * x * z * cb3(b + 18u)
     + 0.5462742152960396 * (xx - yy) * cb3(b + 21u);
  if (deg == 2u) { return c; }
  c += -0.5900435899266435 * y * (3.0 * xx - yy) * cb3(b + 24u) + 2.890611442640554 * x * y * z * cb3(b + 27u)
     - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * cb3(b + 30u)
     + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * cb3(b + 33u)
     - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * cb3(b + 36u) + 1.445305721320277 * z * (xx - yy) * cb3(b + 39u)
     - 0.5900435899266435 * x * (xx - 3.0 * yy) * cb3(b + 42u);
  return c;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.info.x) { return; }
  counts[i] = 0u;
  let g0 = gauss[i * 4u];
  let g1 = gauss[i * 4u + 1u];
  let q = normalize(gauss[i * 4u + 2u]);
  let g3 = gauss[i * 4u + 3u];
  let pos = g0.xyz;
  let opacity = g0.w;
  if (opacity < 1.0 / 255.0) { return; }
  let pc = (U.view * vec4<f32>(pos, 1.0)).xyz;
  if (pc.z < U.extra.x) { return; }
  let W = f32(U.size.x); let H = f32(U.size.y);
  let fx = U.focal.x; let fy = U.focal.y; let cx = U.focal.z; let cy = U.focal.w;
  let z = pc.z;
  let tx = clamp(pc.x / z, -(cx + 0.15 * W) / fx, (W - cx + 0.15 * W) / fx) * z;
  let ty = clamp(pc.y / z, -(cy + 0.15 * H) / fy, (H - cy + 0.15 * H) / fy) * z;
  // 3D covariance M M^T with M = R(q) diag(s)
  let w = q.x; let x = q.y; let y = q.z; let qz = q.w;
  let R = mat3x3<f32>(  // columns
    vec3<f32>(1.0 - 2.0 * (y * y + qz * qz), 2.0 * (x * y + w * qz), 2.0 * (x * qz - w * y)),
    vec3<f32>(2.0 * (x * y - w * qz), 1.0 - 2.0 * (x * x + qz * qz), 2.0 * (y * qz + w * x)),
    vec3<f32>(2.0 * (x * qz + w * y), 2.0 * (y * qz - w * x), 1.0 - 2.0 * (x * x + y * y)));
  let M = mat3x3<f32>(R[0] * g1.x, R[1] * g1.y, R[2] * g1.z);
  let S = M * transpose(M);
  let V = mat3x3<f32>(U.view[0].xyz, U.view[1].xyz, U.view[2].xyz);
  // rows of T = J V
  let vr0 = vec3<f32>(V[0].x, V[1].x, V[2].x);
  let vr1 = vec3<f32>(V[0].y, V[1].y, V[2].y);
  let vr2 = vec3<f32>(V[0].z, V[1].z, V[2].z);
  let t0 = (fx / z) * vr0 - (fx * tx / (z * z)) * vr2;
  let t1 = (fy / z) * vr1 - (fy * ty / (z * z)) * vr2;
  let St0 = S * t0;
  let St1 = S * t1;
  let a = dot(t0, St0) + U.extra.y;
  let b = dot(t0, St1);
  let c = dot(t1, St1) + U.extra.y;
  let det = a * c - b * b;
  if (det <= 0.0) { return; }
  let mean = vec2<f32>(fx * pc.x / z + cx, fy * pc.y / z + cy);
  let k2 = 2.0 * log(255.0 * opacity);   // same expression as in emit (counts must match exactly)
  let k = sqrt(k2);
  let ext = vec2<f32>(k * sqrt(a), k * sqrt(c));
  if (mean.x + ext.x <= 0.0 || mean.x - ext.x >= W || mean.y + ext.y <= 0.0 || mean.y - ext.y >= H) { return; }
  let tw = i32(U.size.z); let th = i32(U.size.w);
  let x0 = clamp(i32(floor((mean.x - ext.x) / 16.0)), 0, tw);
  let x1 = clamp(i32(floor((mean.x + ext.x) / 16.0)) + 1, 0, tw);
  let y0 = clamp(i32(floor((mean.y - ext.y) / 16.0)), 0, th);
  let y1 = clamp(i32(floor((mean.y + ext.y) / 16.0)) + 1, 0, th);
  let box = (x1 - x0) * (y1 - y0);
  if (box <= 0) { return; }
  let co = vec3<f32>(c / det, -b / det, a / det);
  var cnt = box;
  if (box > 1 && box <= 1024) { // huge splats (close to the camera) keep their box: the test would cost more
    cnt = 0;
    for (var ty = y0; ty < y1; ty++) {
      for (var tx = x0; tx < x1; tx++) {
        if (tileHit(mean, co, k2, u32(tx), u32(ty))) { cnt++; }
      }
    }
    if (cnt == 0) { return; }
  }
  let dir = normalize(pos - U.campos.xyz);
  let col = max(shColor(dir, g3.xyz, bitcast<u32>(g1.w), U.info.y) + 0.5, vec3<f32>(0.0));
  var s: Splat;
  s.mean = mean;
  s.conic = vec4<f32>(co, opacity);
  s.color = vec4<f32>(col, z);
  s.rect = vec2<u32>(u32(x0) | (u32(y0) << 16u), u32(x1) | (u32(y1) << 16u));
  splats[i] = s;
  counts[i] = u32(cnt);
}
`;

export const SCAN = /* wgsl */ `
@group(0) @binding(0) var<uniform> S: vec4<u32>;   // x = length
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> sums: array<u32>;
var<workgroup> tmp: array<u32, 256>;

@compute @workgroup_size(256)
fn scanBlock(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let n = S.x;
  let i0 = wg.x * 512u + li * 2u;
  var a = 0u; var b = 0u;
  if (i0 < n) { a = data[i0]; }
  if (i0 + 1u < n) { b = data[i0 + 1u]; }
  let s = a + b;
  tmp[li] = s;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off << 1u) {
    var t = 0u;
    if (li >= off) { t = tmp[li - off]; }
    workgroupBarrier();
    tmp[li] = tmp[li] + t;
    workgroupBarrier();
  }
  let excl = tmp[li] - s;
  if (i0 < n) { data[i0] = excl; }
  if (i0 + 1u < n) { data[i0 + 1u] = excl + a; }
  if (li == 255u) { sums[wg.x] = tmp[255]; }
}

@compute @workgroup_size(256)
fn addBlock(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let n = S.x;
  let v = sums[wg.x];
  let i0 = wg.x * 512u + li * 2u;
  if (i0 < n) { data[i0] = data[i0] + v; }
  if (i0 + 1u < n) { data[i0 + 1u] = data[i0 + 1u] + v; }
}
`;

export const TOTAL = UNIFORMS + /* wgsl */ `
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let n = U.info.x;
  var total = 0u;
  if (n > 0u) { total = offsets[n - 1u] + counts[n - 1u]; }
  info[0] = min(total, U.info.w);
  info[1] = total;
}
`;

export const EMIT = UNIFORMS + /* wgsl */ `
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> vals: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.info.x || counts[i] == 0u) { return; }
  let s = splats[i];
  let cap = U.info.w;
  let depthBits = U.info.z;
  // Positive floats sort like their bit patterns; drop the sign bit, keep the top depthBits bits.
  let dq = (bitcast<u32>(s.color.w) << 1u) >> (32u - depthBits);
  let x0 = s.rect.x & 0xffffu; let y0 = s.rect.x >> 16u;
  let x1 = s.rect.y & 0xffffu; let y1 = s.rect.y >> 16u;
  var o = offsets[i];
  let end = min(o + counts[i], cap);
  let box = (x1 - x0) * (y1 - y0);
  let exact = box > 1u && box <= 1024u;   // must match preprocess
  let k2 = 2.0 * log(255.0 * s.conic.w);
  for (var y = y0; y < y1; y++) {
    for (var x = x0; x < x1; x++) {
      if (exact && !tileHit(s.mean, s.conic.xyz, k2, x, y)) { continue; }
      if (o >= end) { return; }
      keys[o] = ((y * U.size.z + x) << depthBits) | dq;
      vals[o] = i;
      o++;
    }
  }
}
`;

const SORT_COMMON = /* wgsl */ `
struct SortParams { shift: u32, numBlocks: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> P: SortParams;
@group(0) @binding(1) var<storage, read> info: array<u32>;
var<workgroup> nW: u32;
`;

export const SORT_HIST = SORT_COMMON + /* wgsl */ `
@group(0) @binding(2) var<storage, read> keysIn: array<u32>;
@group(0) @binding(3) var<storage, read_write> table: array<u32>;
var<workgroup> hist: array<atomic<u32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  atomicStore(&hist[li], 0u);
  if (li == 0u) { nW = info[0]; }
  let n = workgroupUniformLoad(&nW);
  let base = wg.x * 1024u;
  for (var k = 0u; k < 4u; k++) {
    let idx = base + k * 256u + li;
    if (idx < n) { atomicAdd(&hist[(keysIn[idx] >> P.shift) & 255u], 1u); }
  }
  workgroupBarrier();
  table[li * P.numBlocks + wg.x] = atomicLoad(&hist[li]);
}
`;

// Stable scatter: the block is sorted locally by its 8-bit digit with eight 1-bit split passes
// (each a workgroup prefix sum), then every key goes to scanned_table[digit][block] + local rank.
export const SORT_SCATTER = SORT_COMMON + /* wgsl */ `
@group(0) @binding(2) var<storage, read> keysIn: array<u32>;
@group(0) @binding(3) var<storage, read> valsIn: array<u32>;
@group(0) @binding(4) var<storage, read> table: array<u32>;
@group(0) @binding(5) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(6) var<storage, read_write> valsOut: array<u32>;
var<workgroup> sk: array<u32, 1024>;
var<workgroup> sv: array<u32, 1024>;
var<workgroup> sums: array<u32, 256>;
var<workgroup> dstart: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  if (li == 0u) { nW = info[0]; }
  let n = workgroupUniformLoad(&nW);
  let base = wg.x * 1024u;
  if (base >= n) { return; }
  for (var k = 0u; k < 4u; k++) {
    let p = k * 256u + li;
    let idx = base + p;
    if (idx < n) { sk[p] = keysIn[idx]; sv[p] = valsIn[idx]; } else { sk[p] = 0xffffffffu; sv[p] = 0u; }
  }
  workgroupBarrier();
  for (var bit = 0u; bit < 8u; bit++) {
    var k4: array<u32, 4>;
    var v4: array<u32, 4>;
    var f4: array<u32, 4>;
    var cnt = 0u;
    for (var j = 0u; j < 4u; j++) {
      let p = li * 4u + j;
      k4[j] = sk[p];
      v4[j] = sv[p];
      f4[j] = 1u - ((k4[j] >> (P.shift + bit)) & 1u);
      cnt += f4[j];
    }
    sums[li] = cnt;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off << 1u) {
      var t = 0u;
      if (li >= off) { t = sums[li - off]; }
      workgroupBarrier();
      sums[li] = sums[li] + t;
      workgroupBarrier();
    }
    var zeros = sums[li] - cnt;
    let totalZeros = sums[255];
    workgroupBarrier();
    for (var j = 0u; j < 4u; j++) {
      let p = li * 4u + j;
      var dst: u32;
      if (f4[j] == 1u) { dst = zeros; zeros += 1u; } else { dst = totalZeros + (p - zeros); }
      sk[dst] = k4[j];
      sv[dst] = v4[j];
    }
    workgroupBarrier();
  }
  for (var k = 0u; k < 4u; k++) {
    let p = k * 256u + li;
    let d = (sk[p] >> P.shift) & 255u;
    if (p == 0u || ((sk[p - 1u] >> P.shift) & 255u) != d) { dstart[d] = p; }
  }
  workgroupBarrier();
  for (var k = 0u; k < 4u; k++) {
    let p = k * 256u + li;
    if (base + p < n) {
      let d = (sk[p] >> P.shift) & 255u;
      let dst = table[d * P.numBlocks + wg.x] + (p - dstart[d]);
      keysOut[dst] = sk[p];
      valsOut[dst] = sv[p];
    }
  }
}
`;

export const RANGES = UNIFORMS + /* wgsl */ `
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> info: array<u32>;
@group(0) @binding(2) var<storage, read> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> ranges: array<vec2<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = info[0];
  let db = U.info.z;
  for (var k = 0u; k < 4u; k++) {
    let i = gid.x * 4u + k;
    if (i >= n) { return; }
    let t = keys[i] >> db;
    if (i == 0u || (keys[i - 1u] >> db) != t) { ranges[t].x = i; }
    if (i == n - 1u || (keys[i + 1u] >> db) != t) { ranges[t].y = i + 1u; }
  }
}
`;

// One 64-thread workgroup per 16x16 tile; each thread shades a 2x2 pixel quad, so every splat read
// from workgroup memory is reused for 4 pixels (4x less shared-memory traffic than 1 pixel / thread,
// which is what limits integrated GPUs).
export const RASTER = UNIFORMS + /* wgsl */ `
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> vals: array<u32>;
@group(0) @binding(3) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(4) var outTex: texture_storage_2d<rgba8unorm, write>;
const BATCH = 64u;
var<workgroup> smean: array<vec2<f32>, 64>;
var<workgroup> sconic: array<vec4<f32>, 64>;
var<workgroup> scolor: array<vec4<f32>, 64>;
var<workgroup> rangeW: vec2<u32>;
var<workgroup> doneW: atomic<u32>;
var<workgroup> doneSnap: u32;
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let tile = wg.y * U.size.z + wg.x;
  if (li == 0u) { rangeW = ranges[tile]; atomicStore(&doneW, 0u); }
  let r = workgroupUniformLoad(&rangeW);
  let px0 = wg.x * 16u + lid.x * 2u;
  let py0 = wg.y * 16u + lid.y * 2u;
  let base = vec2<f32>(f32(px0) + 0.5, f32(py0) + 0.5);
  var T = vec4<f32>(1.0);              // quad pixels: (0,0) (1,0) (0,1) (1,1)
  var C0 = vec3<f32>(0.0); var C1 = vec3<f32>(0.0); var C2 = vec3<f32>(0.0); var C3 = vec3<f32>(0.0);
  var live = vec4<bool>(px0 < U.size.x && py0 < U.size.y, px0 + 1u < U.size.x && py0 < U.size.y,
                        px0 < U.size.x && py0 + 1u < U.size.y, px0 + 1u < U.size.x && py0 + 1u < U.size.y);
  var done = !any(live);
  if (done) { atomicAdd(&doneW, 1u); }
  let n = select(0u, r.y - r.x, r.y > r.x);
  let rounds = (n + BATCH - 1u) / BATCH;
  for (var b = 0u; b < rounds; b++) {
    workgroupBarrier();
    if (li == 0u) { doneSnap = atomicLoad(&doneW); }
    if (workgroupUniformLoad(&doneSnap) >= 64u) { break; }
    let idx = r.x + b * BATCH + li;
    if (idx < r.y) {
      let s = splats[vals[idx]];
      smean[li] = s.mean;
      sconic[li] = s.conic;
      scolor[li] = s.color;
    }
    workgroupBarrier();
    if (!done) {
      let m = min(BATCH, r.y - (r.x + b * BATCH));
      for (var j = 0u; j < m; j++) {
        let co = sconic[j];
        let d = smean[j] - base;         // offset to pixel (0,0); neighbours are d - (1,0), d - (0,1), ...
        let dx = vec4<f32>(d.x, d.x - 1.0, d.x, d.x - 1.0);
        let dy = vec4<f32>(d.y, d.y, d.y - 1.0, d.y - 1.0);
        let power = -0.5 * (co.x * dx * dx + co.z * dy * dy) - co.y * dx * dy;
        var alpha = min(vec4<f32>(0.99), co.w * exp(min(power, vec4<f32>(0.0))));
        alpha = select(vec4<f32>(0.0), alpha, (power <= vec4<f32>(0.0)) & (alpha >= vec4<f32>(1.0 / 255.0)) & live);
        if (!any(alpha > vec4<f32>(0.0))) { continue; }
        let nt = T * (1.0 - alpha);
        // a pixel stops (and skips this splat) once its transmittance would drop below 1e-4
        let stop = live & (nt < vec4<f32>(0.0001));
        live = live & !stop;
        let w = select(vec4<f32>(0.0), alpha * T, live);
        let c = scolor[j].xyz;
        C0 += c * w.x; C1 += c * w.y; C2 += c * w.z; C3 += c * w.w;
        T = select(T, nt, live);
        if (!any(live)) { done = true; atomicAdd(&doneW, 1u); break; }
      }
    }
  }
  let bg = U.bg.xyz;
  if (px0 < U.size.x && py0 < U.size.y) { textureStore(outTex, vec2<i32>(i32(px0), i32(py0)), vec4<f32>(C0 + T.x * bg, 1.0)); }
  if (px0 + 1u < U.size.x && py0 < U.size.y) { textureStore(outTex, vec2<i32>(i32(px0 + 1u), i32(py0)), vec4<f32>(C1 + T.y * bg, 1.0)); }
  if (px0 < U.size.x && py0 + 1u < U.size.y) { textureStore(outTex, vec2<i32>(i32(px0), i32(py0 + 1u)), vec4<f32>(C2 + T.z * bg, 1.0)); }
  if (px0 + 1u < U.size.x && py0 + 1u < U.size.y) { textureStore(outTex, vec2<i32>(i32(px0 + 1u), i32(py0 + 1u)), vec4<f32>(C3 + T.w * bg, 1.0)); }
}
`;

export const BLIT = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  return textureLoad(src, vec2<i32>(pos.xy), 0);
}
`;
