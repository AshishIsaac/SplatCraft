// WebGPU tile-based Gaussian splat renderer (compute only, no Three.js, no rasterisation pipeline
// for the splats themselves). See shaders.js for the per-frame stages.
//
// Decoded Gaussian record (16 floats, written by loader.js):
//   0-2 position, 3 opacity, 4-6 scale (linear), 7 SH codebook index (u32 bits),
//   8-11 rotation quaternion (w, x, y, z), 12-14 SH DC colour, 15 unused

import * as S from "./shaders.js";

const UNIFORM_BYTES = 160;
const BLOCK = 1024; // radix sort keys per workgroup

class Scanner {
  // Exclusive prefix sum over `n` u32s in `buffer`, as a stack of 512-element block scans.
  constructor(r, buffer, n) {
    this.levels = [];
    let buf = buffer, len = n;
    for (;;) {
      const groups = Math.max(1, Math.ceil(len / 512));
      const sums = r.buffer(groups * 4, GPUBufferUsage.STORAGE);
      const uni = r.buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      r.device.queue.writeBuffer(uni, 0, new Uint32Array([len, 0, 0, 0]));
      const entries = [uni, buf, sums];
      this.levels.push({ groups, scanBG: r.bind(r.pipes.scanBlock, entries), addBG: r.bind(r.pipes.addBlock, entries),
                         owned: [sums, uni] });
      if (groups <= 1) break;
      buf = sums;
      len = groups;
    }
  }
  encode(pass, pipes) {
    for (const l of this.levels) {
      pass.setPipeline(pipes.scanBlock);
      pass.setBindGroup(0, l.scanBG);
      pass.dispatchWorkgroups(l.groups);
    }
    for (let k = this.levels.length - 2; k >= 0; k--) {
      const l = this.levels[k];
      pass.setPipeline(pipes.addBlock);
      pass.setBindGroup(0, l.addBG);
      pass.dispatchWorkgroups(l.groups);
    }
  }
  destroy() {
    for (const l of this.levels) for (const b of l.owned) b.destroy();
  }
}

export class SplatRenderer {
  static async create(canvas) {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser");
    const adapter = (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }))
      || (await navigator.gpu.requestAdapter());
    if (!adapter) throw new Error("no WebGPU adapter found");
    const L = adapter.limits;
    const device = await adapter.requestDevice({
      requiredFeatures: adapter.features.has("timestamp-query") ? ["timestamp-query"] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
        maxBufferSize: L.maxBufferSize,
        maxComputeWorkgroupsPerDimension: L.maxComputeWorkgroupsPerDimension,
      },
    });
    return new SplatRenderer(adapter, device, canvas);
  }

  constructor(adapter, device, canvas) {
    this.adapter = adapter;
    this.device = device;
    this.canvas = canvas;
    this.info = adapter.info || {};
    this.lost = device.lost.then((i) => { this.lostInfo = i; });
    this.ctx = canvas ? canvas.getContext("webgpu") : null;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    if (this.ctx) this.ctx.configure({ device, format: this.format, alphaMode: "opaque" });
    const mod = (code) => device.createShaderModule({ code });
    const cp = (code, entryPoint = "main") =>
      device.createComputePipeline({ layout: "auto", compute: { module: mod(code), entryPoint } });
    const scan = mod(S.SCAN);
    this.pipes = {
      preprocess: cp(S.PREPROCESS), total: cp(S.TOTAL), emit: cp(S.EMIT), hist: cp(S.SORT_HIST),
      scatter: cp(S.SORT_SCATTER), ranges: cp(S.RANGES), raster: cp(S.RASTER),
      scanBlock: device.createComputePipeline({ layout: "auto", compute: { module: scan, entryPoint: "scanBlock" } }),
      addBlock: device.createComputePipeline({ layout: "auto", compute: { module: scan, entryPoint: "addBlock" } }),
    };
    const blit = mod(S.BLIT);
    this.blitPipe = device.createRenderPipeline({
      layout: "auto", vertex: { module: blit, entryPoint: "vs" },
      fragment: { module: blit, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
    this.uniformData = new ArrayBuffer(UNIFORM_BYTES);
    this.uniforms = this.buffer(UNIFORM_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.infoBuf = this.buffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.infoRead = this.buffer(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    this.readPending = false;
    this.count = 0;
    this.capacity = 0;
    this.lastTotal = 0;
    this.overflow = false;
    this.size = [0, 0];
  }

  buffer(size, usage) {
    return this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
  }

  bind(pipe, resources) {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: resources.map((r, i) => ({ binding: i, resource: r instanceof GPUBuffer ? { buffer: r } : r })),
    });
  }

  // ---- scene ---------------------------------------------------------------------------------
  setScene(header, codebook) {
    this.destroyScene();
    this.header = header;
    this.total = header.count;
    this.shDegree = header.sh_degree;
    const N = Math.max(1, header.count);
    const st = GPUBufferUsage.STORAGE;
    this.gauss = this.buffer(N * 64, st | GPUBufferUsage.COPY_DST);
    const cb = codebook && codebook.length ? codebook : new Float32Array(4);
    this.shcb = this.buffer(cb.byteLength, st | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(this.shcb, 0, cb);
    this.splats = this.buffer(N * 64, st);
    this.counts = this.buffer(N * 4, st | GPUBufferUsage.COPY_SRC);
    this.offsets = this.buffer(N * 4, st | GPUBufferUsage.COPY_DST);
    this.count = 0;
    this.sceneBGs = null;
    this.smallFrames = 0;
    this.setCapacity(Math.min(Math.max(N * 4, 1 << 18), this.maxCapacity()));
  }

  maxCapacity() {
    const lim = Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize);
    return Math.min(Math.floor(lim / 4), (this.device.limits.maxComputeWorkgroupsPerDimension || 65535) * BLOCK);
  }

  uploadChunk(records, first, n) {
    this.device.queue.writeBuffer(this.gauss, first * 64, records, 0, n * 16);
    this.count = Math.max(this.count, first + n);
    this.countScan?.destroy();
    this.countScan = null; // rebuilt for the new count on the next frame
    this.sceneBGs = null;
  }

  setCapacity(cap) {
    cap = Math.ceil(cap / BLOCK) * BLOCK;
    for (const b of [this.keysA, this.valsA, this.keysB, this.valsB, this.table, ...(this.sortParams || [])]) b?.destroy();
    this.tableScan?.destroy();
    this.capacity = cap;
    const st = GPUBufferUsage.STORAGE;
    this.keysA = this.buffer(cap * 4, st);
    this.valsA = this.buffer(cap * 4, st);
    this.keysB = this.buffer(cap * 4, st);
    this.valsB = this.buffer(cap * 4, st);
    this.numBlocks = cap / BLOCK;
    this.table = this.buffer(256 * this.numBlocks * 4, st);
    this.tableScan = new Scanner(this, this.table, 256 * this.numBlocks);
    this.sortParams = [0, 8, 16, 24].map((shift) => {
      const b = this.buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(b, 0, new Uint32Array([shift, this.numBlocks, 0, 0]));
      return b;
    });
    this.sortBGs = this.sortParams.map((p, i) => {
      const [kin, vin, kout, vout] = i % 2 === 0
        ? [this.keysA, this.valsA, this.keysB, this.valsB] : [this.keysB, this.valsB, this.keysA, this.valsA];
      return { hist: this.bind(this.pipes.hist, [p, this.infoBuf, kin, this.table]),
               scatter: this.bind(this.pipes.scatter, [p, this.infoBuf, kin, vin, this.table, kout, vout]) };
    });
    this.sceneBGs = null;
    this.frameBGs = null;
  }

  destroyScene() {
    for (const b of [this.gauss, this.shcb, this.splats, this.counts, this.offsets]) b?.destroy();
    this.countScan?.destroy();
    this.countScan = null;
  }

  // ---- frame -----------------------------------------------------------------------------------
  ensureTarget(w, h) {
    if (this.size[0] === w && this.size[1] === h && this.outTex) return;
    this.outTex?.destroy();
    this.ranges?.destroy();
    this.size = [w, h];
    this.tilesX = Math.ceil(w / 16);
    this.tilesY = Math.ceil(h / 16);
    const tiles = this.tilesX * this.tilesY;
    this.tileBits = Math.max(1, Math.ceil(Math.log2(tiles)));
    this.outTex = this.device.createTexture({
      size: [w, h], format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.ranges = this.buffer(tiles * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.frameBGs = null;
    this.blitBG = this.bind(this.blitPipe, [this.outTex.createView()]);
  }

  writeUniforms(cam) {
    const f = new Float32Array(this.uniformData);
    const u = new Uint32Array(this.uniformData);
    f.set(cam.view, 0);
    f.set([...cam.campos, 0], 16);
    f.set([cam.fx, cam.fy, cam.cx, cam.cy], 20);
    const bg = cam.background || this.header?.view?.background || [0, 0, 0];
    f.set([...bg, 1], 24);
    u.set([cam.width, cam.height, this.tilesX, this.tilesY], 28);
    u.set([this.count, this.shDegree || 0, 32 - this.tileBits, this.capacity], 32);
    f.set([cam.near ?? 0.01, 0.3, 0, 0], 36);
    this.device.queue.writeBuffer(this.uniforms, 0, this.uniformData);
  }

  buildBindGroups() {
    const p = this.pipes;
    if (!this.countScan) this.countScan = new Scanner(this, this.offsets, Math.max(1, this.count));
    if (!this.sceneBGs) {
      this.sceneBGs = {
        preprocess: this.bind(p.preprocess, [this.uniforms, this.gauss, this.shcb, this.splats, this.counts]),
        total: this.bind(p.total, [this.uniforms, this.counts, this.offsets, this.infoBuf]),
      };
      this.frameBGs = null;
    }
    if (!this.frameBGs) {
      this.frameBGs = {
        emit: this.bind(p.emit, [this.uniforms, this.splats, this.counts, this.offsets, this.keysA, this.valsA]),
        ranges: this.bind(p.ranges, [this.uniforms, this.infoBuf, this.keysA, this.ranges]),
        raster: this.bind(p.raster, [this.uniforms, this.splats, this.valsA, this.ranges, this.outTex.createView()]),
      };
    }
  }

  encodeFrame(enc) {
    const p = this.pipes;
    const N = this.count;
    const prof = this.profiler;
    let q = 0;
    // One compute pass per stage; with timestamp queries each pass records its GPU start/end time.
    const stage = (name, fn) => {
      const desc = {};
      if (prof && q + 2 <= prof.slots) {
        desc.timestampWrites = { querySet: prof.qs, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 };
        prof.names[q / 2] = name;
        q += 2;
      }
      const pass = enc.beginComputePass(desc);
      fn(pass);
      pass.end();
    };
    const run = (pass, pipe, bg, x, y = 1) => {
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(x, y);
    };
    enc.clearBuffer(this.ranges);
    stage("preprocess", (pass) => run(pass, p.preprocess, this.sceneBGs.preprocess, Math.ceil(N / 256)));
    enc.copyBufferToBuffer(this.counts, 0, this.offsets, 0, N * 4);
    stage("scan", (pass) => {
      this.countScan.encode(pass, p);
      run(pass, p.total, this.sceneBGs.total, 1);
    });
    stage("emit", (pass) => run(pass, p.emit, this.frameBGs.emit, Math.ceil(N / 256)));
    stage("sort", (pass) => {
      for (const bg of this.sortBGs) {
        run(pass, p.hist, bg.hist, this.numBlocks);
        this.tableScan.encode(pass, p);
        run(pass, p.scatter, bg.scatter, this.numBlocks);
      }
    });
    stage("ranges", (pass) => run(pass, p.ranges, this.frameBGs.ranges, Math.ceil(this.capacity / 1024)));
    stage("raster", (pass) => run(pass, p.raster, this.frameBGs.raster, this.tilesX, this.tilesY));
    if (prof && q) {
      enc.resolveQuerySet(prof.qs, 0, q, prof.resolve, 0);
      if (!prof.pending) enc.copyBufferToBuffer(prof.resolve, 0, prof.read, 0, q * 8);
      prof.used = q;
    }
  }

  // GPU time per stage (ms), when the adapter supports timestamp queries. Results land in this.stageMs.
  enableProfiling() {
    if (!this.device.features.has("timestamp-query") || this.profiler) return !!this.profiler;
    const slots = 16;
    this.profiler = {
      qs: this.device.createQuerySet({ type: "timestamp", count: slots }), slots, names: [], pending: false,
      resolve: this.buffer(slots * 8, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC),
      read: this.buffer(slots * 8, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
    };
    return true;
  }

  readProfile() {
    const prof = this.profiler;
    if (!prof || prof.pending || !prof.used) return;
    prof.pending = true;
    const used = prof.used;
    prof.read.mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigInt64Array(prof.read.getMappedRange().slice(0, used * 8));
      prof.read.unmap();
      const out = {};
      for (let i = 0; i < used / 2; i++) out[prof.names[i]] = Number(t[2 * i + 1] - t[2 * i]) / 1e6;
      this.stageMs = out;
      prof.pending = false;
    }).catch(() => { prof.pending = false; });
  }

  // Render one frame for `cam` ({view, campos, fx, fy, cx, cy, width, height}). With `toCanvas`, the
  // result is blitted to the canvas; it is always left in this.outTex.
  render(cam, { toCanvas = true } = {}) {
    if (!this.header || this.count === 0) return false;
    this.ensureTarget(cam.width, cam.height);
    this.writeUniforms(cam);
    this.buildBindGroups();
    const enc = this.device.createCommandEncoder();
    this.encodeFrame(enc);
    if (toCanvas && this.ctx) {
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store",
                             clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      });
      rp.setPipeline(this.blitPipe);
      rp.setBindGroup(0, this.blitBG);
      rp.draw(3);
      rp.end();
    }
    const readNow = !this.readPending;
    if (readNow) enc.copyBufferToBuffer(this.infoBuf, 0, this.infoRead, 0, 16);
    this.device.queue.submit([enc.finish()]);
    if (readNow) this.checkCapacity();
    this.readProfile();
    return true;
  }

  // Instance-buffer sizing, read back one frame late without stalling: grow at once on overflow,
  // shrink after a while when far too big (the sort's cost scales with the capacity, not the count).
  checkCapacity() {
    this.readPending = true;
    this.infoRead.mapAsync(GPUMapMode.READ).then(() => {
      const v = new Uint32Array(this.infoRead.getMappedRange().slice(0));
      this.infoRead.unmap();
      this.readPending = false;
      const total = v[1];
      this.lastTotal = total;
      const want = Math.min(this.maxCapacity(), Math.max(1 << 16, Math.ceil(total * 1.5)));
      this.overflow = total > this.maxCapacity();
      if (total > this.capacity) {
        this.smallFrames = 0;
        this.setCapacity(want);
        this.onCapacity?.(want);
      } else if (total * 3 < this.capacity && ++this.smallFrames > 20) {
        this.smallFrames = 0;
        this.setCapacity(want);
      } else if (total * 3 >= this.capacity) {
        this.smallFrames = 0;
      }
    }).catch(() => { this.readPending = false; });
  }

  async readPixels() {
    const [w, h] = this.size;
    const bpr = Math.ceil((w * 4) / 256) * 256;
    const buf = this.device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.outTex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) out.set(src.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
    buf.unmap();
    buf.destroy();
    return out;
  }
}
