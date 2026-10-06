"""Differentiable tile-based Gaussian splat rasterizer, written in plain PyTorch.

This is the same algorithm as the CUDA rasterizer of Kerbl et al. (2023), expressed with tensor ops so
it runs on any PyTorch device without compiling anything:

1. **Project** (``project``): every 3D Gaussian becomes a 2D Gaussian: mean, inverse covariance
   ("conic"), opacity, colour from spherical harmonics. This part is ordinary autograd.
2. **Bin** (``bin_gaussians``): each Gaussian is duplicated once per 16x16 screen tile its 3-sigma
   ellipse (tightened by its opacity) overlaps; the (tile, Gaussian) pairs are sorted by tile, then
   by depth. Integer bookkeeping only, no gradients.
3. **Blend** (``_Rasterize``): front-to-back alpha compositing per tile, in *depth slabs* of K
   Gaussians: every slab is one batched tensor op over all still-active tiles x K Gaussians x 256
   pixels. Tiles whose pixels are all saturated (T < 1e-4) or out of Gaussians drop out.

The backward pass is a custom ``autograd.Function``: it stores only the per-pixel transmittance at the
start of every slab, then walks the slabs back to front, re-computes each one under autograd and
chains the transmittance gradient into the slab in front of it. Memory is therefore O(one slab)
instead of O(all Gaussian x pixel interactions), which is what makes training fit on a 6 GB GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .sh import eval_sh

TILE = 16
PIX = TILE * TILE
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
T_EPS = 1e-4


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) quaternions (w, x, y, z), not necessarily normalised -> (N, 3, 3)."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)


@dataclass
class Projected:
    mean2d: torch.Tensor  # (N, 2) pixels
    conic: torch.Tensor  # (N, 3) inverse 2D covariance (a, b, c) -> [[a, b], [b, c]]
    opacity: torch.Tensor  # (N,)
    color: torch.Tensor  # (N, 3)
    depth: torch.Tensor  # (N,) camera z
    extent: torch.Tensor  # (N, 2) half-size of the screen bounding box in pixels (0 = culled)


def project(means, log_scales, quats, opacity_logits, sh, sh_degree: int, cam, *, near: float = 0.01,
            blur: float = 0.3, scale_modifier: float = 1.0) -> Projected:
    """EWA splatting: 3D Gaussians -> screen-space 2D Gaussians (differentiable)."""
    dev = means.device
    R = torch.as_tensor(cam.R, dtype=means.dtype, device=dev)
    t = torch.as_tensor(cam.t, dtype=means.dtype, device=dev)
    pc = means @ R.T + t
    z = pc[:, 2]
    zc = z.clamp_min(near)
    # Clamp the Jacobian's linearisation point to a little outside the view frustum (as in 3DGS):
    # huge off-screen Gaussians otherwise get absurd 2D covariances.
    W, H = cam.width, cam.height
    lo_x, hi_x = -(cam.cx + 0.15 * W) / cam.fx, (W - cam.cx + 0.15 * W) / cam.fx
    lo_y, hi_y = -(cam.cy + 0.15 * H) / cam.fy, (H - cam.cy + 0.15 * H) / cam.fy
    tx = (pc[:, 0] / zc).clamp(lo_x, hi_x) * zc
    ty = (pc[:, 1] / zc).clamp(lo_y, hi_y) * zc
    zero = torch.zeros_like(zc)
    J = torch.stack([cam.fx / zc, zero, -cam.fx * tx / (zc * zc),
                     zero, cam.fy / zc, -cam.fy * ty / (zc * zc)], -1).reshape(-1, 2, 3)
    M = quat_to_rotmat(quats) * (torch.exp(log_scales) * scale_modifier)[:, None, :]
    # Tiny per-Gaussian matrix products as broadcast multiply-sums: batched GEMMs of 2x3 / 3x3
    # matrices are ~20x slower on the GPU (and dominate the backward pass).
    cov3d = (M[:, :, None, :] * M[:, None, :, :]).sum(-1)  # M M^T
    T = (J[:, :, :, None] * R[None, None, :, :]).sum(2)  # J R
    TC = (T[:, :, :, None] * cov3d[:, None, :, :]).sum(2)  # T cov3d
    cov2d = (TC[:, :, None, :] * T[:, None, :, :]).sum(-1)  # T cov3d T^T
    a = cov2d[:, 0, 0] + blur
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + blur
    det = (a * c - b * b).clamp_min(1e-12)
    conic = torch.stack([c / det, -b / det, a / det], -1)
    mean2d = torch.stack([cam.fx * pc[:, 0] / zc + cam.cx, cam.fy * pc[:, 1] / zc + cam.cy], -1)
    opacity = torch.sigmoid(opacity_logits.reshape(-1))

    campos = -R.T @ t
    dirs = F.normalize(means - campos, dim=-1)
    color = (eval_sh(sh_degree, sh, dirs) + 0.5).clamp_min(0.0)

    with torch.no_grad():
        # Opacity-aware extent: alpha = o * exp(-d^2 / 2) drops below 1/255 at d = sqrt(2 ln(255 o)).
        k = torch.sqrt(2 * torch.log((opacity * 255).clamp_min(1.0)))
        ext = torch.stack([k * a.sqrt(), k * c.sqrt()], -1)
        on = (z > near) & (k > 0) & torch.isfinite(ext).all(-1) & (a * c - b * b > 0)
        on &= (mean2d[:, 0] + ext[:, 0] > 0) & (mean2d[:, 0] - ext[:, 0] < W)
        on &= (mean2d[:, 1] + ext[:, 1] > 0) & (mean2d[:, 1] - ext[:, 1] < H)
        ext = torch.where(on[:, None], ext, torch.zeros_like(ext))
    return Projected(mean2d, conic, opacity, color, z, ext)


@dataclass
class Binning:
    gid: torch.Tensor  # (P,) Gaussian index of every (tile, Gaussian) pair, sorted by tile then depth
    starts: torch.Tensor  # (T,) first pair of each tile
    counts: torch.Tensor  # (T,) number of pairs of each tile
    tiles_x: int
    tiles_y: int
    width: int
    height: int


@torch.no_grad()
def bin_gaussians(p: Projected, width: int, height: int) -> Binning:
    dev = p.mean2d.device
    tw, th = math.ceil(width / TILE), math.ceil(height / TILE)
    ids = torch.nonzero(p.extent[:, 0] > 0).squeeze(1)
    ids = ids[torch.argsort(p.depth[ids])]  # front to back
    m, e = p.mean2d[ids], p.extent[ids]
    x0 = ((m[:, 0] - e[:, 0]) / TILE).floor().clamp(0, tw).long()
    x1 = ((m[:, 0] + e[:, 0]) / TILE).floor().add(1).clamp(0, tw).long()
    y0 = ((m[:, 1] - e[:, 1]) / TILE).floor().clamp(0, th).long()
    y1 = ((m[:, 1] + e[:, 1]) / TILE).floor().add(1).clamp(0, th).long()
    nx, ny = x1 - x0, y1 - y0
    cnt = nx * ny
    total = int(cnt.sum())
    T = tw * th
    if total == 0:
        z = torch.zeros(T, dtype=torch.long, device=dev)
        return Binning(torch.zeros(0, dtype=torch.long, device=dev), z, z.clone(), tw, th, width, height)
    rep = torch.repeat_interleave(torch.arange(len(ids), device=dev), cnt)
    first = torch.cumsum(cnt, 0) - cnt
    local = torch.arange(total, device=dev) - first[rep]
    tile = (y0[rep] + local // nx[rep]) * tw + (x0[rep] + local % nx[rep])
    tile, order = torch.sort(tile, stable=True)  # stable: depth order survives within a tile
    gid = ids[rep[order]]
    counts = torch.bincount(tile, minlength=T)
    starts = torch.cumsum(counts, 0) - counts
    return Binning(gid, starts, counts, tw, th, width, height)


def _tile_pixels(tiles: torch.Tensor, tw: int):
    """Pixel-centre coordinates of the 256 pixels of each tile: two (A, 1, 256) tensors."""
    off = torch.arange(PIX, device=tiles.device)
    px = (tiles % tw)[:, None] * TILE + (off % TILE)[None] + 0.5
    py = (tiles // tw)[:, None] * TILE + (off // TILE)[None] + 0.5
    return px[:, None, :], py[:, None, :]


def _gather(b: Binning, tiles: torch.Tensor, s: int, K: int):
    k = s * K + torch.arange(K, device=tiles.device)
    valid = k[None, :] < b.counts[tiles, None]
    idx = (b.starts[tiles, None] + k[None, :]).clamp_max(max(len(b.gid) - 1, 0))
    return b.gid[idx], valid


def _slab(m, cn, o, c, valid, px, py, Ts):
    """Composite one slab of K Gaussians over A tiles. Returns colour (A, 256, 3), T_end (A, 256), weights."""
    dx = px - m[..., 0:1]
    dy = py - m[..., 1:2]
    power = -0.5 * (cn[..., 0:1] * dx * dx + cn[..., 2:3] * dy * dy) - cn[..., 1:2] * dx * dy
    alpha = (o[..., None] * torch.exp(power.clamp_max(0.0))).clamp_max(ALPHA_MAX)
    keep = valid[..., None] & (power <= 0) & (alpha >= ALPHA_MIN)
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    trans = torch.cumprod(1 - alpha, dim=1)
    t_excl = torch.cat([torch.ones_like(trans[:, :1]), trans[:, :-1]], 1) * Ts[:, None, :]
    w = alpha * t_excl
    return torch.einsum("akp,akc->apc", w, c), trans[:, -1] * Ts, w


def _plan(b: Binning, K: int, budget: int):
    """Tile groups per slab such that a group's (tiles x K x 256) stays under ``budget`` elements."""
    per = max(1, budget // (K * PIX))
    return per


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean2d, conic, opacity, color, bg, b: Binning, K: int, budget: int, want_weights: bool):
        dev = mean2d.device
        T = b.tiles_x * b.tiles_y
        dt = mean2d.dtype
        acc = torch.zeros(T, PIX, 3, device=dev, dtype=dt)
        trans = torch.ones(T, PIX, device=dev, dtype=dt)
        weights = torch.zeros(len(mean2d), device=dev, dtype=dt) if want_weights else None
        slabs = []
        per = _plan(b, K, budget)
        active = torch.nonzero(b.counts > 0).squeeze(1)
        s = 0
        while active.numel():
            done = []
            for g0 in range(0, active.numel(), per):
                tiles = active[g0:g0 + per]
                gid, valid = _gather(b, tiles, s, K)
                px, py = (v.to(dt) for v in _tile_pixels(tiles, b.tiles_x))
                Ts = trans[tiles]
                col, Te, w = _slab(mean2d[gid], conic[gid], opacity[gid], color[gid], valid, px, py, Ts)
                slabs.append((s, tiles, Ts))
                acc[tiles] += col
                trans[tiles] = Te
                if want_weights:  # only pixels inside the image (edge tiles overhang it)
                    inside = ((px < b.width) & (py < b.height)).to(dt)
                    weights.index_add_(0, gid.reshape(-1), (w * inside).sum(-1).reshape(-1))
                done.append(tiles[(b.counts[tiles] <= (s + 1) * K) | (Te.max(1).values < T_EPS)])
            fin = torch.cat(done) if done else active[:0]
            active = active[~torch.isin(active, fin)]
            s += 1
        img = acc + trans[..., None] * bg
        ctx.save_for_backward(mean2d, conic, opacity, color, bg)
        ctx.b, ctx.K, ctx.slabs = b, K, slabs
        out_img = _untile(img, b)
        out_alpha = _untile((1 - trans)[..., None], b)[..., 0]
        if weights is not None:
            ctx.mark_non_differentiable(weights)
            return out_img, out_alpha, weights
        return out_img, out_alpha, torch.zeros(0, device=dev)

    @staticmethod
    def backward(ctx, g_img, g_alpha, _gw):
        mean2d, conic, opacity, color, bg = ctx.saved_tensors
        b, K = ctx.b, ctx.K
        gC = _tile(g_img, b)  # (T, 256, 3)
        gT = (gC * bg).sum(-1)  # dL/dT_final
        if g_alpha is not None:
            gT = gT - _tile(g_alpha[..., None], b)[..., 0]
        g_m = torch.zeros_like(mean2d)
        g_cn = torch.zeros_like(conic)
        g_o = torch.zeros_like(opacity)
        g_c = torch.zeros_like(color)
        for s, tiles, Ts in reversed(ctx.slabs):
            gid, valid = _gather(b, tiles, s, K)
            px, py = (v.to(mean2d.dtype) for v in _tile_pixels(tiles, b.tiles_x))
            with torch.enable_grad():
                m = mean2d[gid].detach().requires_grad_(True)
                cn = conic[gid].detach().requires_grad_(True)
                o = opacity[gid].detach().requires_grad_(True)
                c = color[gid].detach().requires_grad_(True)
                ts = Ts.detach().requires_grad_(True)
                col, Te, _ = _slab(m, cn, o, c, valid, px, py, ts)
                dm, dcn, do, dc, dts = torch.autograd.grad((col, Te), (m, cn, o, c, ts), (gC[tiles], gT[tiles]))
            flat = gid.reshape(-1)
            g_m.index_add_(0, flat, dm.reshape(-1, 2))
            g_cn.index_add_(0, flat, dcn.reshape(-1, 3))
            g_o.index_add_(0, flat, do.reshape(-1))
            g_c.index_add_(0, flat, dc.reshape(-1, 3))
            gT[tiles] = dts
        return g_m, g_cn, g_o, g_c, None, None, None, None, None


def _untile(x: torch.Tensor, b: Binning) -> torch.Tensor:
    C = x.shape[-1]
    x = x.reshape(b.tiles_y, b.tiles_x, TILE, TILE, C).permute(0, 2, 1, 3, 4)
    return x.reshape(b.tiles_y * TILE, b.tiles_x * TILE, C)[: b.height, : b.width]


def _tile(x: torch.Tensor, b: Binning) -> torch.Tensor:
    H, W = b.height, b.width
    x = F.pad(x, (0, 0, 0, b.tiles_x * TILE - W, 0, b.tiles_y * TILE - H))
    C = x.shape[-1]
    x = x.reshape(b.tiles_y, TILE, b.tiles_x, TILE, C).permute(0, 2, 1, 3, 4)
    return x.reshape(b.tiles_y * b.tiles_x, PIX, C)


_TRITON: bool | None = None


def triton_available() -> bool:
    global _TRITON
    if _TRITON is None:
        try:
            from . import triton_raster  # noqa: F401

            _TRITON = torch.cuda.is_available()
        except Exception:  # no triton wheel for this platform / no GPU
            _TRITON = False
    return _TRITON


def rasterize(p: Projected, width: int, height: int, bg: torch.Tensor, *, K: int = 32,
              budget: int = 6_000_000, want_weights: bool = False, backend: str = "auto"):
    """-> image (H, W, 3), alpha (H, W), binning, per-Gaussian blend weights (if ``want_weights``).

    ``backend``: "triton" (fused GPU kernels), "torch" (tensor ops, any device) or "auto".
    """
    b = bin_gaussians(p, width, height)
    use_triton = backend == "triton" or (backend == "auto" and p.mean2d.is_cuda
                                         and p.mean2d.dtype == torch.float32 and triton_available())
    if use_triton:
        from .triton_raster import TritonRasterize

        packed = torch.cat([p.mean2d, p.conic, p.opacity[:, None], p.color], 1)
        img, alpha, w = TritonRasterize.apply(packed, bg.to(p.mean2d), b, want_weights)
    else:
        img, alpha, w = _Rasterize.apply(p.mean2d, p.conic, p.opacity, p.color, bg.to(p.mean2d), b, K, budget,
                                         want_weights)
    return img, alpha, b, (w if want_weights else None)


def render(g, cam, bg: torch.Tensor, sh_degree: int | None = None, **kw) -> dict:
    """Render a ``GaussianModel`` (or anything with the same attributes) from ``cam``."""
    deg = g.active_sh_degree if sh_degree is None else sh_degree
    p = project(g.means, g.log_scales, g.quats, g.opacity_logits, g.sh, deg, cam)
    img, alpha, b, w = rasterize(p, cam.width, cam.height, bg, **kw)
    return {"image": img, "alpha": alpha, "proj": p, "binning": b, "weights": w}


def reference_rasterize(p: Projected, width: int, height: int, bg: torch.Tensor) -> torch.Tensor:
    """Dense O(N x H x W) rasterizer with plain autograd: the ground truth for gradient tests."""
    on = torch.nonzero(p.extent[:, 0] > 0).squeeze(1)
    on = on[torch.argsort(p.depth[on])]
    ys, xs = torch.meshgrid(torch.arange(height, device=bg.device), torch.arange(width, device=bg.device),
                            indexing="ij")
    px, py = xs.reshape(1, -1).to(p.mean2d) + 0.5, ys.reshape(1, -1).to(p.mean2d) + 0.5
    m, cn, o, c = p.mean2d[on], p.conic[on], p.opacity[on], p.color[on]
    dx, dy = px - m[:, 0:1], py - m[:, 1:2]
    power = -0.5 * (cn[:, 0:1] * dx * dx + cn[:, 2:3] * dy * dy) - cn[:, 1:2] * dx * dy
    alpha = (o[:, None] * torch.exp(power.clamp_max(0.0))).clamp_max(ALPHA_MAX)
    alpha = torch.where((power <= 0) & (alpha >= ALPHA_MIN), alpha, torch.zeros_like(alpha))
    trans = torch.cumprod(1 - alpha, 0)
    t_excl = torch.cat([torch.ones_like(trans[:1]), trans[:-1]], 0)
    img = (alpha * t_excl).T @ c + trans[-1][:, None] * bg
    return img.reshape(height, width, 3)
