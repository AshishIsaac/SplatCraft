"""The .splatc codec: every attribute survives encode -> decode within its quantisation step."""

import numpy as np
import pytest
import torch

from splatcraft.compress.codec import chunk_sizes, decode_scene, encode_scene, morton3, read_header
from splatcraft.gs.gaussians import GaussianModel

CC = {"entropy": "rans", "min_opacity": 0.0, "prune_keep": 1.0, "position_bits": 16, "scale_bits": 8,
      "rotation_bits": 8, "opacity_bits": 8, "color_bits": 8, "sh_codebook": 64, "kmeans_iters": 4, "chunks": 3}


def _model(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    g = GaussianModel.from_points(rng.normal(0, 3, (n, 3)), rng.uniform(0, 1, (n, 3)), 3)
    g.log_scales = torch.as_tensor(rng.normal(-4, 1, (n, 3)), dtype=torch.float32)
    g.quats = torch.as_tensor(rng.normal(0, 1, (n, 4)), dtype=torch.float32)
    g.opacity_logits = torch.as_tensor(rng.normal(0, 2, (n, 1)), dtype=torch.float32)
    g.sh_rest = torch.as_tensor(rng.normal(0, 0.05, (n, 15, 3)), dtype=torch.float32)
    return g


def _match(a: GaussianModel, b: GaussianModel):
    """Index into `a` for each Gaussian of `b` (codec reorders by importance + Morton code)."""
    from scipy.spatial import cKDTree

    return cKDTree(a.means.numpy()).query(b.means.numpy())[1]


@pytest.mark.parametrize("entropy", ["rans", "deflate", "none"])
def test_roundtrip_attributes(entropy):
    g = _model()
    data, rep = encode_scene(g, {**CC, "entropy": entropy}, log=lambda *_: None)
    d = decode_scene(data)
    assert d.num == g.num == rep["gaussians"]
    idx = _match(g, d)
    span = (g.means.max(0).values - g.means.min(0).values)
    assert ((d.means - g.means[idx]).abs() <= span / 65535 * 0.51 + 1e-5).all()
    assert len(set(idx.tolist())) == g.num  # a bijection: nothing lost or duplicated
    assert (d.opacity - g.opacity[idx]).abs().max() < 1 / 255 * 0.51 + 1e-4
    q1 = torch.nn.functional.normalize(d.quats, dim=-1)
    q2 = torch.nn.functional.normalize(g.quats[idx], dim=-1)
    assert (q1 * q2).sum(-1).abs().min() > 0.99
    assert (d.sh_dc - g.sh_dc[idx]).abs().median() < 0.02
    # SH rest goes through a 64-entry codebook: only the error statistics are bounded
    assert (d.sh_rest - g.sh_rest[idx]).pow(2).mean() < g.sh_rest.pow(2).mean()


def test_far_apart_positions_wrap_correctly():
    """Deltas that exceed +-32767 quantisation steps (regression: positions came back scrambled)."""
    g = _model(2000)
    g.means[::2] += 1000.0  # two far-apart clusters -> huge Morton-order jumps
    data, _ = encode_scene(g, CC, log=lambda *_: None)
    d = decode_scene(data)
    idx = _match(g, d)
    span = (g.means.max(0).values - g.means.min(0).values)
    assert ((d.means - g.means[idx]).abs() <= span / 65535 * 0.51 + 1e-4).all()


def test_header_chunks_and_pruning():
    g = _model(5000)
    imp = torch.rand(5000)
    data, rep = encode_scene(g, {**CC, "prune_keep": 0.5, "chunks": 4}, importance=imp, log=lambda *_: None)
    h, _ = read_header(data)
    assert h["count"] == rep["gaussians"] == 2500
    assert [c["count"] for c in h["chunks"]] == chunk_sizes(2500, 4)
    assert h["chunks"][0]["count"] < h["chunks"][-1]["count"]  # progressive: small first chunk


def test_compression_ratio():
    g = _model(20000)
    data, rep = encode_scene(g, {**CC, "sh_codebook": 1024}, log=lambda *_: None)
    assert rep["ratio_vs_ply"] > 10


def test_morton_orders_locality():
    q = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [65535, 65535, 65535]], np.uint32)
    m = morton3(q)
    assert m[0] < m[1] < m[2] < m[3]
