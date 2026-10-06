"""The trainable scene: anisotropic 3D Gaussians with spherical-harmonic colour.

Parameters live in their unconstrained form (log-scales, unnormalised quaternions, opacity logits),
exactly as in 3DGS, so the ``.ply`` this writes opens in any standard Gaussian-splat viewer.
Densification (clone / split) and pruning edit the tensors *and* the Adam moments in place.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .sh import num_coeffs, rgb_to_sh

PARAMS = ("means", "log_scales", "quats", "opacity_logits", "sh_dc", "sh_rest")


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


@torch.no_grad()
def knn_mean_dist(xyz: torch.Tensor, k: int = 3, chunk: int = 4096) -> torch.Tensor:
    """Mean distance to the ``k`` nearest other points (chunked brute force on the GPU)."""
    out = torch.empty(len(xyz), device=xyz.device)
    for i in range(0, len(xyz), chunk):
        d = torch.cdist(xyz[i:i + chunk], xyz)
        d = d.topk(min(k + 1, len(xyz)), largest=False).values[:, 1:]
        out[i:i + chunk] = d.mean(1) if d.numel() else 1.0
    return out


class GaussianModel:
    def __init__(self, max_sh_degree: int = 3, device="cpu") -> None:
        self.max_sh_degree = max_sh_degree
        self.active_sh_degree = 0
        self.device = torch.device(device)
        z = torch.zeros(0, device=self.device)
        self.means = z.reshape(0, 3)
        self.log_scales = z.reshape(0, 3)
        self.quats = z.reshape(0, 4)
        self.opacity_logits = z.reshape(0, 1)
        self.sh_dc = z.reshape(0, 1, 3)
        self.sh_rest = z.reshape(0, num_coeffs(max_sh_degree) - 1, 3)
        self.optimizer: torch.optim.Optimizer | None = None
        self._reset_stats()

    # ------------------------------------------------------------------ construction / io
    @classmethod
    def from_points(cls, xyz: np.ndarray, rgb: np.ndarray, max_sh_degree: int = 3, device="cpu",
                    init_opacity: float = 0.1) -> GaussianModel:
        g = cls(max_sh_degree, device)
        pts = torch.as_tensor(xyz, dtype=torch.float32, device=g.device)
        col = torch.as_tensor(rgb, dtype=torch.float32, device=g.device)
        n = len(pts)
        dist = knn_mean_dist(pts).clamp_min(1e-7)
        g.means = pts
        g.log_scales = torch.log(dist)[:, None].repeat(1, 3)
        g.quats = torch.zeros(n, 4, device=g.device)
        g.quats[:, 0] = 1
        g.opacity_logits = inverse_sigmoid(torch.full((n, 1), init_opacity, device=g.device))
        g.sh_dc = rgb_to_sh(col)[:, None, :]
        g.sh_rest = torch.zeros(n, num_coeffs(max_sh_degree) - 1, 3, device=g.device)
        g._reset_stats()
        return g

    @property
    def num(self) -> int:
        return len(self.means)

    @property
    def sh(self) -> torch.Tensor:
        return torch.cat([self.sh_dc, self.sh_rest], 1)

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits).squeeze(1)

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    def params(self) -> dict[str, torch.Tensor]:
        return {k: getattr(self, k) for k in PARAMS}

    def to(self, device) -> GaussianModel:
        self.device = torch.device(device)
        for k in PARAMS:
            setattr(self, k, getattr(self, k).to(self.device))
        self._reset_stats()
        return self

    def save_ply(self, path: str | Path) -> None:
        """Standard 3DGS ``point_cloud.ply`` layout (readable by every splat viewer)."""
        n = self.num
        rest = self.sh_rest.detach().transpose(1, 2).reshape(n, -1)  # channel-major, as 3DGS writes it
        cols = [("x", self.means[:, 0]), ("y", self.means[:, 1]), ("z", self.means[:, 2]),
                ("nx", torch.zeros(n)), ("ny", torch.zeros(n)), ("nz", torch.zeros(n))]
        cols += [(f"f_dc_{i}", self.sh_dc[:, 0, i]) for i in range(3)]
        cols += [(f"f_rest_{i}", rest[:, i]) for i in range(rest.shape[1])]
        cols += [("opacity", self.opacity_logits[:, 0])]
        cols += [(f"scale_{i}", self.log_scales[:, i]) for i in range(3)]
        cols += [(f"rot_{i}", self.quats[:, i]) for i in range(4)]
        data = np.stack([c.detach().float().cpu().numpy() for _, c in cols], 1).astype("<f4")
        header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
        header += "".join(f"property float {name}\n" for name, _ in cols) + "end_header\n"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(header.encode())
            f.write(data.tobytes())

    @classmethod
    def load_ply(cls, path: str | Path, device="cpu") -> GaussianModel:
        names, n, raw = _read_ply(path)
        col = {name: raw[:, i] for i, name in enumerate(names)}
        n_rest = sum(1 for k in names if k.startswith("f_rest_"))
        deg = int(round(math.sqrt(n_rest / 3 + 1))) - 1
        g = cls(deg, device)
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=g.device)  # noqa: E731
        g.means = t(np.stack([col["x"], col["y"], col["z"]], 1))
        g.log_scales = t(np.stack([col[f"scale_{i}"] for i in range(3)], 1))
        g.quats = t(np.stack([col[f"rot_{i}"] for i in range(4)], 1))
        g.opacity_logits = t(col["opacity"][:, None])
        g.sh_dc = t(np.stack([col[f"f_dc_{i}"] for i in range(3)], 1)[:, None, :])
        rest = np.stack([col[f"f_rest_{i}"] for i in range(n_rest)], 1) if n_rest else np.zeros((n, 0))
        g.sh_rest = t(rest.reshape(n, 3, -1).transpose(0, 2, 1))
        g.active_sh_degree = deg
        g._reset_stats()
        return g

    # ------------------------------------------------------------------ optimisation
    def setup_optimizer(self, lr: dict, extent: float) -> None:
        groups = []
        for k in PARAMS:
            t = getattr(self, k).detach().requires_grad_(True)
            setattr(self, k, t)
            base = lr[k] * (extent if k == "means" else 1.0)
            groups.append({"params": [t], "lr": base, "name": k})
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def _replace(self, new: dict[str, torch.Tensor], keep_state: torch.Tensor | None = None,
                 n_added: int = 0, reset: dict[str, torch.Tensor] | None = None) -> None:
        """Swap every parameter for ``new[k]`` and remap the Adam moments the same way.

        ``keep_state`` selects the surviving rows' moments; ``n_added`` new rows get zero moments.
        """
        opt = self.optimizer
        for group in opt.param_groups if opt else []:
            k = group["name"]
            old = group["params"][0]
            st = opt.state.pop(old, None)
            t = new[k].detach().requires_grad_(True)
            if st is not None:
                for m in ("exp_avg", "exp_avg_sq"):
                    v = st[m] if keep_state is None else st[m][keep_state]
                    if n_added:
                        v = torch.cat([v, torch.zeros((n_added, *v.shape[1:]), device=v.device)])
                    if reset is not None and k in reset:
                        v = torch.zeros_like(v)
                    st[m] = v
                opt.state[t] = st
            group["params"][0] = t
            setattr(self, k, t)
        if opt is None:
            for k, v in new.items():
                setattr(self, k, v)

    def prune(self, keep: torch.Tensor) -> None:
        self._replace({k: getattr(self, k).detach()[keep] for k in PARAMS}, keep_state=keep)
        self.grad_accum, self.denom, self.max_radii = self.grad_accum[keep], self.denom[keep], self.max_radii[keep]

    def append(self, new: dict[str, torch.Tensor]) -> None:
        n_new = len(new["means"])
        self._replace({k: torch.cat([getattr(self, k).detach(), new[k]]) for k in PARAMS}, n_added=n_new)
        z = torch.zeros(n_new, device=self.device)
        self.grad_accum = torch.cat([self.grad_accum, z])
        self.denom = torch.cat([self.denom, z])
        self.max_radii = torch.cat([self.max_radii, z])

    def _reset_stats(self) -> None:
        n = len(self.means)
        self.grad_accum = torch.zeros(n, device=self.device)
        self.denom = torch.zeros(n, device=self.device)
        self.max_radii = torch.zeros(n, device=self.device)

    @torch.no_grad()
    def add_densify_stats(self, mean2d_grad: torch.Tensor, extent2d: torch.Tensor, width: int, height: int) -> None:
        vis = extent2d[:, 0] > 0
        # 3DGS measures the positional gradient in NDC units: d/dndc = d/dpixel * size / 2.
        g = mean2d_grad * torch.tensor([width * 0.5, height * 0.5], device=mean2d_grad.device)
        self.grad_accum[vis] += g[vis].norm(dim=-1)
        self.denom[vis] += 1
        self.max_radii[vis] = torch.maximum(self.max_radii[vis], extent2d[vis].max(-1).values)

    @torch.no_grad()
    def densify_and_prune(self, grad_thresh: float, extent: float, percent_dense: float, min_opacity: float,
                          max_screen: float | None, max_gaussians: int) -> dict:
        grads = self.grad_accum / self.denom.clamp_min(1)
        big = self.scales.max(1).values > percent_dense * extent
        hot = grads >= grad_thresh
        budget = max_gaussians - self.num
        clone, split = hot & ~big, hot & big
        n_req = int(clone.sum() + split.sum())
        if n_req > budget:  # over the cap: keep only the hottest candidates
            if budget <= 0:
                clone[:] = False
                split[:] = False
            else:
                thr = grads[hot].topk(budget).values[-1]
                clone &= grads >= thr
                split &= grads >= thr
        n0 = self.num
        # Clone: an identical copy (the optimizer then pulls the two apart).
        new = {k: getattr(self, k).detach()[clone] for k in PARAMS}
        # Split: two samples drawn from the parent Gaussian, each 1.6x smaller.
        s = self.scales[split].repeat(2, 1)
        q = self.quats[split].repeat(2, 1)
        from .rasterizer import quat_to_rotmat
        offs = (quat_to_rotmat(q) @ (torch.randn_like(s) * s)[..., None])[..., 0]
        sp = {"means": self.means.detach()[split].repeat(2, 1) + offs,
              "log_scales": torch.log(s / 1.6), "quats": q,
              "opacity_logits": self.opacity_logits.detach()[split].repeat(2, 1),
              "sh_dc": self.sh_dc.detach()[split].repeat(2, 1, 1),
              "sh_rest": self.sh_rest.detach()[split].repeat(2, 1, 1)}
        self.append({k: torch.cat([new[k], sp[k]]) for k in PARAMS})
        n_split = int(split.sum())
        drop = torch.zeros(self.num, dtype=torch.bool, device=self.device)
        drop[:n0] = split
        drop |= self.opacity < min_opacity
        if max_screen:
            drop |= self.max_radii > max_screen
            drop |= self.scales.max(1).values > 0.1 * extent
        self.prune(~drop)
        self._reset_stats()
        return {"cloned": int(clone.sum()), "split": n_split, "pruned": int(drop.sum()) - n_split, "total": self.num}

    @torch.no_grad()
    def reset_opacity(self, value: float = 0.01) -> None:
        new = self.params()
        new = {k: v.detach() for k, v in new.items()}
        new["opacity_logits"] = torch.minimum(self.opacity_logits.detach(),
                                              inverse_sigmoid(torch.tensor(value, device=self.device)))
        self._replace(new, reset={"opacity_logits": new["opacity_logits"]})

    def normalized_quats(self) -> torch.Tensor:
        return F.normalize(self.quats.detach(), dim=-1)


def _read_ply(path) -> tuple[list[str], int, np.ndarray]:
    with open(path, "rb") as f:
        names, n = [], 0
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                if parts[1] not in ("float", "float32"):
                    raise ValueError(f"unsupported ply property type {parts[1]}")
                names.append(parts[-1])
            elif line == "end_header":
                break
            elif not line and f.tell() > 1 << 16:
                raise ValueError("not a ply file")
        raw = np.frombuffer(f.read(n * len(names) * 4), dtype="<f4").reshape(n, len(names))
    return names, n, raw
