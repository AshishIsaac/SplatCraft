"""Fused Triton kernels for the tile blend (forward + backward), ~20x faster than the tensor-op path.

One program per 16x16 tile; the 256 pixels are a vector, Gaussians are consumed front to back in
blocks of ``B``. Within a block, transmittance comes from a cumulative sum of log(1 - alpha), so the
whole block is composited with no sequential loop. A tile stops once every pixel's transmittance is
below 1e-4 (forward and backward take exactly the same decision, so gradients stay consistent).

Backward walks front to back as well: the colour "behind" Gaussian i, which the chain rule needs, is
``C_final - prefix_i`` (C_final includes the background), so no back-to-front division trick and no
per-pixel storage beyond the final colour and transmittance.

Per-Gaussian gradients are reduced over the tile's pixels and ``atomic_add``-ed into global buffers.
The math is the same as ``rasterizer._slab`` (tested against it and against the dense reference).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NP = 9  # packed per-Gaussian record: mx, my, conic a, b, c, opacity, r, g, b


@triton.jit
def _fwd(pk, gid, starts, counts, bg, out_c, out_t, wts, tiles_x, W, H,
         B: tl.constexpr, WANT_W: tl.constexpr):
    t = tl.program_id(0)
    start = tl.load(starts + t)
    count = tl.load(counts + t)
    pix = tl.arange(0, 256)
    px = ((t % tiles_x) * 16 + pix % 16).to(tl.float32) + 0.5
    py = ((t // tiles_x) * 16 + pix // 16).to(tl.float32) + 0.5
    inside = (px < W) & (py < H)
    T = tl.full((256,), 1.0, tl.float32)
    cr = tl.zeros((256,), tl.float32)
    cg = tl.zeros((256,), tl.float32)
    cb = tl.zeros((256,), tl.float32)
    j0 = 0
    tmax = 1.0
    while (j0 < count) & (tmax >= 1e-4):
        offs = j0 + tl.arange(0, B)
        m = offs < count
        g = tl.load(gid + start + offs, mask=m, other=0)
        base = pk + g * 9
        mx = tl.load(base + 0, mask=m, other=0.0)
        my = tl.load(base + 1, mask=m, other=0.0)
        ca = tl.load(base + 2, mask=m, other=0.0)
        cbb = tl.load(base + 3, mask=m, other=0.0)
        cc = tl.load(base + 4, mask=m, other=0.0)
        op = tl.load(base + 5, mask=m, other=0.0)
        r = tl.load(base + 6, mask=m, other=0.0)
        gg = tl.load(base + 7, mask=m, other=0.0)
        bb = tl.load(base + 8, mask=m, other=0.0)
        dx = px[None, :] - mx[:, None]
        dy = py[None, :] - my[:, None]
        power = -0.5 * (ca[:, None] * dx * dx + cc[:, None] * dy * dy) - cbb[:, None] * dx * dy
        alpha = tl.minimum(op[:, None] * tl.exp(tl.minimum(power, 0.0)), 0.99)
        alpha = tl.where(m[:, None] & (power <= 0.0) & (alpha >= 1.0 / 255.0), alpha, 0.0)
        lg = tl.log(1.0 - alpha)
        cum = tl.cumsum(lg, axis=0)
        w = alpha * T[None, :] * tl.exp(cum - lg)
        cr += tl.sum(w * r[:, None], axis=0)
        cg += tl.sum(w * gg[:, None], axis=0)
        cb += tl.sum(w * bb[:, None], axis=0)
        T = T * tl.exp(tl.sum(lg, axis=0))
        if WANT_W:
            tl.atomic_add(wts + g, tl.sum(tl.where(inside[None, :], w, 0.0), axis=1), mask=m)
        j0 += B
        tmax = tl.max(T, axis=0)
    o = t * 256 + pix
    tl.store(out_c + o * 3 + 0, cr + T * tl.load(bg + 0))
    tl.store(out_c + o * 3 + 1, cg + T * tl.load(bg + 1))
    tl.store(out_c + o * 3 + 2, cb + T * tl.load(bg + 2))
    tl.store(out_t + o, T)


@triton.jit
def _bwd(pk, gid, starts, counts, c_fin, t_fin, g_c, g_a, grads, tiles_x, B: tl.constexpr):
    t = tl.program_id(0)
    start = tl.load(starts + t)
    count = tl.load(counts + t)
    pix = tl.arange(0, 256)
    px = ((t % tiles_x) * 16 + pix % 16).to(tl.float32) + 0.5
    py = ((t // tiles_x) * 16 + pix // 16).to(tl.float32) + 0.5
    o = t * 256 + pix
    Cr = tl.load(c_fin + o * 3 + 0)
    Cg = tl.load(c_fin + o * 3 + 1)
    Cb = tl.load(c_fin + o * 3 + 2)
    Tf = tl.load(t_fin + o)
    gr = tl.load(g_c + o * 3 + 0)
    gg_ = tl.load(g_c + o * 3 + 1)
    gb = tl.load(g_c + o * 3 + 2)
    ga = tl.load(g_a + o)
    T = tl.full((256,), 1.0, tl.float32)
    pr = tl.zeros((256,), tl.float32)
    pg = tl.zeros((256,), tl.float32)
    pb = tl.zeros((256,), tl.float32)
    j0 = 0
    tmax = 1.0
    while (j0 < count) & (tmax >= 1e-4):
        offs = j0 + tl.arange(0, B)
        m = offs < count
        g = tl.load(gid + start + offs, mask=m, other=0)
        base = pk + g * 9
        mx = tl.load(base + 0, mask=m, other=0.0)
        my = tl.load(base + 1, mask=m, other=0.0)
        ca = tl.load(base + 2, mask=m, other=0.0)
        cbb = tl.load(base + 3, mask=m, other=0.0)
        cc = tl.load(base + 4, mask=m, other=0.0)
        op = tl.load(base + 5, mask=m, other=0.0)
        r = tl.load(base + 6, mask=m, other=0.0)
        gg = tl.load(base + 7, mask=m, other=0.0)
        bb = tl.load(base + 8, mask=m, other=0.0)
        dx = px[None, :] - mx[:, None]
        dy = py[None, :] - my[:, None]
        power = -0.5 * (ca[:, None] * dx * dx + cc[:, None] * dy * dy) - cbb[:, None] * dx * dy
        G = tl.exp(tl.minimum(power, 0.0))
        raw = op[:, None] * G
        alpha = tl.minimum(raw, 0.99)
        live = m[:, None] & (power <= 0.0) & (alpha >= 1.0 / 255.0)
        alpha = tl.where(live, alpha, 0.0)
        lg = tl.log(1.0 - alpha)
        cum = tl.cumsum(lg, axis=0)
        Tx = T[None, :] * tl.exp(cum - lg)
        w = alpha * Tx
        wr = w * r[:, None]
        wg = w * gg[:, None]
        wb = w * bb[:, None]
        # colour behind Gaussian i (later Gaussians + background) = C_final - prefix through i
        sr = Cr[None, :] - (pr[None, :] + tl.cumsum(wr, axis=0))
        sg = Cg[None, :] - (pg[None, :] + tl.cumsum(wg, axis=0))
        sb = Cb[None, :] - (pb[None, :] + tl.cumsum(wb, axis=0))
        inv = 1.0 / (1.0 - alpha)
        dA = (gr[None, :] * (r[:, None] * Tx - sr * inv) + gg_[None, :] * (gg[:, None] * Tx - sg * inv)
              + gb[None, :] * (bb[:, None] * Tx - sb * inv) + ga[None, :] * Tf[None, :] * inv)
        dA = tl.where(live & (raw < 0.99), dA, 0.0)
        dP = dA * alpha
        tl.atomic_add(grads + g * 9 + 0, tl.sum((ca[:, None] * dx + cbb[:, None] * dy) * dP, axis=1), mask=m)
        tl.atomic_add(grads + g * 9 + 1, tl.sum((cc[:, None] * dy + cbb[:, None] * dx) * dP, axis=1), mask=m)
        tl.atomic_add(grads + g * 9 + 2, tl.sum(-0.5 * dx * dx * dP, axis=1), mask=m)
        tl.atomic_add(grads + g * 9 + 3, tl.sum(-dx * dy * dP, axis=1), mask=m)
        tl.atomic_add(grads + g * 9 + 4, tl.sum(-0.5 * dy * dy * dP, axis=1), mask=m)
        tl.atomic_add(grads + g * 9 + 5, tl.sum(dA * G, axis=1), mask=m)
        tl.atomic_add(grads + g * 9 + 6, tl.sum(w * gr[None, :], axis=1), mask=m)
        tl.atomic_add(grads + g * 9 + 7, tl.sum(w * gg_[None, :], axis=1), mask=m)
        tl.atomic_add(grads + g * 9 + 8, tl.sum(w * gb[None, :], axis=1), mask=m)
        pr += tl.sum(wr, axis=0)
        pg += tl.sum(wg, axis=0)
        pb += tl.sum(wb, axis=0)
        T = T * tl.exp(tl.sum(lg, axis=0))
        j0 += B
        tmax = tl.max(T, axis=0)


BLOCK = 16


class TritonRasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, bg, b, want_weights: bool):
        from .rasterizer import _untile

        packed = packed.contiguous().float()
        T = b.tiles_x * b.tiles_y
        dev = packed.device
        c = torch.empty(T * 256, 3, device=dev)
        tr = torch.empty(T * 256, device=dev)
        wts = torch.zeros(len(packed) if want_weights else 1, device=dev)
        gid = b.gid.to(torch.int32).contiguous()
        starts = b.starts.to(torch.int32).contiguous()
        counts = b.counts.to(torch.int32).contiguous()
        if T:
            _fwd[(T,)](packed, gid, starts, counts, bg.float().contiguous(), c, tr, wts, b.tiles_x, b.width,
                       b.height, B=BLOCK, WANT_W=want_weights, num_warps=4)
        ctx.save_for_backward(packed, gid, starts, counts, c, tr)
        ctx.b = b
        img = _untile(c.view(T, 256, 3), b)
        alpha = _untile((1 - tr).view(T, 256, 1), b)[..., 0]
        ctx.mark_non_differentiable(wts)
        return img, alpha, wts

    @staticmethod
    def backward(ctx, g_img, g_alpha, _):
        from .rasterizer import _tile

        packed, gid, starts, counts, c, tr = ctx.saved_tensors
        b = ctx.b
        T = b.tiles_x * b.tiles_y
        gc = _tile(g_img.float(), b).reshape(T * 256, 3).contiguous()
        if g_alpha is None:
            ga = torch.zeros(T * 256, device=packed.device)
        else:
            ga = _tile(g_alpha.float()[..., None], b).reshape(T * 256).contiguous()
        grads = torch.zeros_like(packed)
        if T:
            _bwd[(T,)](packed, gid, starts, counts, c, tr, gc, ga, grads, b.tiles_x, B=BLOCK, num_warps=4)
        return grads, None, None, None
