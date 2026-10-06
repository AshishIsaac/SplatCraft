"""The tile rasterizer (custom backward with slab re-computation) against dense plain-autograd ground truth."""

import math

import numpy as np
import pytest
import torch

from splatcraft.gs.rasterizer import project, rasterize, reference_rasterize
from splatcraft.scene.cameras import Camera


def _scene(n=60, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    means = (torch.rand(n, 3, generator=g, dtype=dtype) - 0.5) * 2
    log_scales = torch.log(torch.rand(n, 3, generator=g, dtype=dtype) * 0.25 + 0.03)
    quats = torch.randn(n, 4, generator=g, dtype=dtype)
    opac = torch.randn(n, 1, generator=g, dtype=dtype)
    sh = torch.randn(n, 16, 3, generator=g, dtype=dtype) * 0.3
    return [t.requires_grad_(True) for t in (means, log_scales, quats, opac, sh)]


def _cam(w=45, h=37):
    return Camera.look_at([0.3, -0.4, -3.0], [0, 0, 0], [0, -1, 0], math.radians(50), w, h)


@pytest.mark.parametrize("K", [4, 32])
def test_forward_matches_reference(K):
    params = _scene()
    cam = _cam()
    bg = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64)
    p = project(*params, 3, cam)
    img, alpha, *_ = rasterize(p, cam.width, cam.height, bg, K=K)
    ref = reference_rasterize(p, cam.width, cam.height, bg)
    assert img.shape == (cam.height, cam.width, 3)
    assert torch.allclose(img, ref, atol=1e-3), (img - ref).abs().max()
    assert float(alpha.max()) > 0.5


@pytest.mark.parametrize("K", [3, 32])
def test_gradients_match_reference(K):
    cam = _cam()
    bg = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64)
    w = torch.rand(cam.height, cam.width, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)

    def grads(fn):
        params = _scene()
        p = project(*params, 3, cam)
        out = fn(p)
        (out * w).sum().backward()
        return [t.grad.clone() for t in params]

    g_tile = grads(lambda p: rasterize(p, cam.width, cam.height, bg, K=K)[0])
    g_ref = grads(lambda p: reference_rasterize(p, cam.width, cam.height, bg))
    for a, b, name in zip(g_tile, g_ref, ["means", "log_scales", "quats", "opacity", "sh"]):
        rel = (a - b).norm() / (b.norm() + 1e-12)
        # Tiny mismatch allowed: the tile path stops a tile once every pixel's transmittance < 1e-4.
        assert rel < 1e-3, f"{name}: relative gradient error {float(rel):.2e}"


def test_empty_view_is_background():
    params = _scene(10)
    cam = Camera.look_at([0, 0, 5], [0, 0, 10], [0, -1, 0], 1.0, 20, 20)  # looking away
    bg = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    img, alpha, *_ = rasterize(project(*params, 3, cam), 20, 20, bg)
    assert torch.allclose(img, bg.expand(20, 20, 3))
    assert float(alpha.abs().max()) == 0


def test_weights_count_visible_contribution():
    params = _scene()
    cam = _cam()
    bg = torch.zeros(3, dtype=torch.float64)
    p = project(*params, 3, cam)
    _, alpha, _, wts = rasterize(p, cam.width, cam.height, bg, want_weights=True)
    assert wts.shape == (60,)
    # Sum of all blend weights = total coverage (sum of alpha over pixels).
    assert np.isclose(float(wts.sum()), float(alpha.sum()), rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_matches_torch_path():
    from splatcraft.gs.rasterizer import triton_available

    if not triton_available():
        pytest.skip("triton not installed")
    cam = _cam(70, 53)
    bg = torch.tensor([0.2, 0.5, 0.9], device="cuda")
    w = torch.rand(cam.height, cam.width, 3, generator=torch.Generator().manual_seed(1)).cuda()
    wa = torch.rand(cam.height, cam.width, generator=torch.Generator().manual_seed(2)).cuda()

    def run(backend):
        params = [t.detach().float().cuda().requires_grad_(True) for t in _scene(300)]
        p = project(*params, 3, cam)
        img, alpha, _, wts = rasterize(p, cam.width, cam.height, bg, backend=backend, want_weights=True)
        ((img * w).sum() + (alpha * wa).sum()).backward()
        return img.detach(), wts, [t.grad for t in params]

    img_t, w_t, g_t = run("triton")
    img_r, w_r, g_r = run("torch")
    assert (img_t - img_r).abs().max() < 1e-4
    assert torch.allclose(w_t, w_r, rtol=1e-3, atol=1e-4)
    for a, b, name in zip(g_t, g_r, ["means", "log_scales", "quats", "opacity", "sh"]):
        rel = float((a - b).norm() / (b.norm() + 1e-12))
        assert rel < 2e-3, f"{name}: {rel:.2e}"
