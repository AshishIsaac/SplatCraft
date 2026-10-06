# Densification / compaction ablation (lego, quick schedule)

NeRF-synthetic lego, 7,000 iterations at 640x640, RTX 3050 Laptop GPU, all 200 test views.
Same seed and code; only the listed settings change.

| Run | `densify.grad_threshold` | Compaction (keep 60 % at 5.5k) | Gaussians | Test PSNR |
|---|---|---|---:|---:|
| baseline | 2e-4 (3DGS paper value) | yes | 44,588 | 31.57 |
| `ab_nocompact` | 2e-4 | no | 73,704 | 31.48 |
| `ab_thr1e4` | **1e-4** | no | 215,262 | **33.43** |

* Importance compaction removes 40 % of the Gaussians at no measurable cost (31.57 vs 31.48), so it
  stays on: smaller files and faster rendering for free.
* Halving the clone/split threshold gives +1.9 dB. With this rasterizer the accumulated NDC-space
  positional gradients come out smaller than the paper's threshold assumes, so too few Gaussians are
  densified. 1e-4 is the default from here on. On real captures (bonsai) the Gaussian cap
  (`train.max_gaussians`) binds first, so the threshold matters less there.

Reproduce: `splatcraft train lego -c quick --name ab_thr1e4 --set densify.grad_threshold=0.0001 compaction.at=[]`
followed by `splatcraft eval ab_thr1e4`.
