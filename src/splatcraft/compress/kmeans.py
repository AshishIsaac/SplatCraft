"""Weighted k-means on the GPU (vector quantisation codebooks)."""

from __future__ import annotations

import torch


@torch.no_grad()
def assign(x: torch.Tensor, centroids: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    out = torch.empty(len(x), dtype=torch.long, device=x.device)
    c2 = (centroids * centroids).sum(1)
    for i in range(0, len(x), chunk):
        xi = x[i:i + chunk]
        d = c2[None, :] - 2 * xi @ centroids.T  # |x|^2 is constant per row
        out[i:i + chunk] = d.argmin(1)
    return out


@torch.no_grad()
def kmeans(x: torch.Tensor, k: int, iters: int = 15, weights: torch.Tensor | None = None,
           seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (centroids (k, D), labels (N,)). Importance ``weights`` bias centroids toward visible Gaussians."""
    n = len(x)
    if n <= k:
        return x.clone(), torch.arange(n, device=x.device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.ones(n, device=x.device) if weights is None else weights.clamp_min(1e-12)
    # Initialise by importance-weighted sampling without replacement (cheap k-means++ stand-in).
    init = torch.multinomial(w.cpu().double(), k, replacement=False, generator=gen).to(x.device)
    c = x[init].clone()
    labels = assign(x, c)
    for _ in range(iters):
        sums = torch.zeros_like(c).index_add_(0, labels, x * w[:, None])
        mass = torch.zeros(k, device=x.device).index_add_(0, labels, w)
        empty = mass <= 0
        c = torch.where(empty[:, None], c, sums / mass.clamp_min(1e-12)[:, None])
        if empty.any():  # re-seed dead codewords on the points the codebook currently fits worst
            err = ((x - c[labels]) ** 2).sum(1) * w
            c[empty] = x[err.topk(int(empty.sum())).indices]
        new = assign(x, c)
        if torch.equal(new, labels):
            break
        labels = new
    return c, labels
