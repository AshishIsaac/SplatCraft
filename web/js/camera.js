// Orbit camera producing OpenCV-convention view matrices (x right, y down, z forward), identical to
// splatcraft.scene.cameras.Camera.look_at in Python.

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return mul(a, 1 / l); };

function rotate(v, axis, ang) { // Rodrigues
  const c = Math.cos(ang), s = Math.sin(ang);
  return add(add(mul(v, c), mul(cross(axis, v), s)), mul(axis, dot(axis, v) * (1 - c)));
}

export function viewFromRt(R, t) { // row-major R (3x3 nested), t -> column-major mat4
  const m = new Float32Array(16);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) m[c * 4 + r] = R[r][c];
    m[12 + r] = t[r];
  }
  m[15] = 1;
  return m;
}

export class OrbitCamera {
  constructor(view = {}) {
    this.home = view;
    this.reset();
  }

  reset() {
    const v = this.home;
    this.target = v.target ? [...v.target] : [0, 0, 0];
    this.up = norm(v.up || [0, -1, 0]);
    this.eye = v.eye ? [...v.eye] : add(this.target, [0, 0, -4]);
    this.fovY = v.fov_y || 0.9;
    this.exact = null;
    this.changed = true;
  }

  get distance() { return Math.hypot(...sub(this.eye, this.target)); }

  // Snap to a capture camera ({R, t, fx, fy, cx, cy, width, height}); kept exact until the user moves.
  setExact(c) {
    this.exact = c;
    const center = mul([0, 1, 2].map((j) => c.R[0][j] * c.t[0] + c.R[1][j] * c.t[1] + c.R[2][j] * c.t[2]), -1);
    const fwd = c.R[2];
    const d = Math.max(this.distance, 1e-3);
    this.eye = center;
    this.target = add(center, mul(fwd, d));
    this.fovY = 2 * Math.atan(c.height / (2 * c.fy));
    this.changed = true;
  }

  leaveExact() {
    this.exact = null;
  }

  basis() {
    const f = norm(sub(this.target, this.eye));
    let r = cross(f, this.up);
    if (Math.hypot(...r) < 1e-6) r = cross(f, Math.abs(f[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]);
    r = norm(r);
    return { f, r, d: cross(f, r) };
  }

  orbit(dx, dy) { // radians
    this.leaveExact();
    let off = sub(this.eye, this.target);
    off = rotate(off, this.up, -dx);
    const { r } = this.basis();
    const rotated = rotate(off, r, dy);
    const cosUp = dot(norm(rotated), this.up);
    if (Math.abs(cosUp) < 0.995) off = rotated;
    this.eye = add(this.target, off);
    this.changed = true;
  }

  pan(dx, dy) { // fractions of the view height
    this.leaveExact();
    const { r, d } = this.basis();
    const s = this.distance * 2 * Math.tan(this.fovY / 2);
    const mv = add(mul(r, -dx * s), mul(d, -dy * s));
    this.eye = add(this.eye, mv);
    this.target = add(this.target, mv);
    this.changed = true;
  }

  zoom(factor) {
    this.leaveExact();
    const off = sub(this.eye, this.target);
    const d = Math.hypot(...off);
    const nd = Math.min(Math.max(d * factor, 1e-3), 1e4);
    this.eye = add(this.target, mul(off, nd / d));
    this.changed = true;
  }

  fly(forward, right) { // fractions of the orbit distance
    this.leaveExact();
    const { f, r } = this.basis();
    const mv = add(mul(f, forward * this.distance), mul(r, right * this.distance));
    this.eye = add(this.eye, mv);
    this.target = add(this.target, mv);
    this.changed = true;
  }

  // -> renderer camera for a width x height target
  frame(width, height, extra = {}) {
    if (this.exact) {
      const c = this.exact;
      const sx = width / c.width, sy = height / c.height;
      const center = mul([0, 1, 2].map((j) => c.R[0][j] * c.t[0] + c.R[1][j] * c.t[1] + c.R[2][j] * c.t[2]), -1);
      return { view: viewFromRt(c.R, c.t), campos: center, fx: c.fx * sx, fy: c.fy * sy, cx: c.cx * sx, cy: c.cy * sy,
               width, height, ...extra };
    }
    const { f, r, d } = this.basis();
    const R = [r, d, f];
    const t = R.map((row) => -dot(row, this.eye));
    const fy = height / (2 * Math.tan(this.fovY / 2));
    return { view: viewFromRt(R, t), campos: [...this.eye], fx: fy, fy, cx: width / 2, cy: height / 2, width, height,
             ...extra };
  }
}

// Mouse, touch and keyboard controls. Calls onChange() after every interaction.
export function attachControls(el, cam, onChange) {
  const ptrs = new Map();
  let last = null;
  el.addEventListener("contextmenu", (e) => e.preventDefault());
  el.addEventListener("pointerdown", (e) => {
    el.setPointerCapture(e.pointerId);
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey });
    last = null;
  });
  const end = (e) => { ptrs.delete(e.pointerId); last = null; };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
  el.addEventListener("pointermove", (e) => {
    const p = ptrs.get(e.pointerId);
    if (!p) return;
    const h = el.clientHeight || 1;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;
    if (ptrs.size === 1) {
      if (p.button === 2 || p.button === 1 || p.shift || e.shiftKey) cam.pan(dx / h, dy / h);
      else cam.orbit((dx / h) * Math.PI, (dy / h) * Math.PI);
    } else if (ptrs.size === 2) {
      const [a, b] = [...ptrs.values()];
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      if (last) {
        if (dist > 0 && last.dist > 0) cam.zoom(last.dist / dist);
        cam.pan((mid.x - last.mid.x) / h, (mid.y - last.mid.y) / h);
      }
      last = { mid, dist };
    }
    onChange();
  });
  el.addEventListener("wheel", (e) => {
    e.preventDefault();
    cam.zoom(Math.exp(e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)));
    onChange();
  }, { passive: false });
  const keys = new Set();
  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement) return;
    keys.add(e.key.toLowerCase());
  });
  window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
  window.addEventListener("blur", () => keys.clear());
  return {
    tick(dt) { // WASD / arrow keys fly, Q / E orbit
      const s = Math.min(dt, 0.1) * 0.8;
      let moved = false;
      const k = (a, b) => keys.has(a) || keys.has(b);
      if (k("w", "arrowup")) { cam.fly(s, 0); moved = true; }
      if (k("s", "arrowdown")) { cam.fly(-s, 0); moved = true; }
      if (k("a", "arrowleft")) { cam.fly(0, -s); moved = true; }
      if (k("d", "arrowright")) { cam.fly(0, s); moved = true; }
      if (keys.has("q")) { cam.orbit(-s * 2, 0); moved = true; }
      if (keys.has("e")) { cam.orbit(s * 2, 0); moved = true; }
      if (moved) onChange();
    },
  };
}
