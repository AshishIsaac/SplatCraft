"""Real spherical harmonics up to degree 3 (the basis and sign convention of the 3DGS reference code).

The WebGPU viewer's ``preprocess.wgsl`` evaluates exactly the same polynomial.
"""

from __future__ import annotations

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435)


def num_coeffs(degree: int) -> int:
    return (degree + 1) ** 2


def sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor:
    """(N, 3) unit directions -> (N, (degree+1)^2) basis values."""
    x, y, z = dirs.unbind(-1)
    cols = [torch.full_like(x, C0)]
    if degree >= 1:
        cols += [-C1 * y, C1 * z, -C1 * x]
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        cols += [C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy)]
    if degree >= 3:
        cols += [C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy),
                 C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy), C3[5] * z * (xx - yy),
                 C3[6] * x * (xx - 3 * yy)]
    return torch.stack(cols, -1)


def eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """sh: (N, K, 3) with K >= (degree+1)^2, dirs: (N, 3) unit vectors -> (N, 3) colour (before the +0.5).

    One basis matrix and one contraction: slicing ``sh[:, k]`` per coefficient instead would make the
    backward pass allocate a full-size zero gradient per coefficient.
    """
    k = num_coeffs(degree)
    basis = sh_basis(degree, dirs)
    return torch.einsum("nk,nkc->nc", basis, sh[:, :k])


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    return sh * C0 + 0.5
