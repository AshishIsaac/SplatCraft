# SplatCraft benchmark

Generated 2026-10-06 by `splatcraft bench`. Quality is measured on held-out test views after decoding each file. FPS = 120 frames of a 360-degree orbit rendered back to back by the WebGPU viewer (at most 2 frames in flight, not capped by the display) / wall-clock time. On a fast discrete GPU this is bounded by the browser's per-frame submission cost; the GPU-time table below shows the actual GPU work.

- **lego**: 30,000 iterations (41.3 min on NVIDIA GeForce RTX 3050 6GB Laptop GPU), trained at 800x800, 200 test views, LPIPS-vgg
- **bonsai**: 30,000 iterations (123.5 min on NVIDIA GeForce RTX 3050 6GB Laptop GPU), trained at 800x533, 37 test views, LPIPS-vgg
- GPUs: `intel gen-12lp` = Intel Iris Xe (integrated, Chrome default adapter); `nvidia ampere` = NVIDIA GeForce RTX 3050 6GB Laptop (`--force_high_performance_gpu`)

| Scene | File | Size | vs .ply | Gaussians | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FPS intel gen-12lp @ 1280x720 | FPS intel gen-12lp @ 1920x1080 | FPS nvidia ampere @ 1280x720 | FPS nvidia ampere @ 1920x1080 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lego | `3dgs-ply` | 44.06 MB | 1.0x | 177,643 | 33.95 | 0.975 | 0.031 | – | – | – | – |
| lego | `splatc` | 2.94 MB | 15.0x | 177,546 | 33.34 | 0.972 | 0.035 | 16.4 | 13.8 | 299 | 386.5 |
| lego | `splatc-small` | 1.78 MB | 24.8x | 106,527 | 31.19 | 0.962 | 0.045 | 24.8 | 20 | 452.8 | 436.4 |
| lego | `splatc-tiny` | 1.05 MB | 42.1x | 62,141 | 26.63 | 0.931 | 0.072 | 36.9 | 28.8 | 483.9 | 419.1 |
| lego | `splatc-deflate` | 2.93 MB | 15.1x | 177,546 | – | – | – | – | – | – | – |
| lego | `splatc-no-entropy` | 3.57 MB | 12.3x | 177,546 | – | – | – | – | – | – | – |
| bonsai | `3dgs-ply` | 88.30 MB | 1.0x | 356,051 | 29.76 | 0.922 | 0.233 | – | – | – | – |
| bonsai | `splatc` | 5.57 MB | 15.8x | 354,126 | 29.00 | 0.906 | 0.254 | 12.3 | 8.5 | 331.1 | 257.8 |
| bonsai | `splatc-small` | 3.33 MB | 26.5x | 212,475 | 27.64 | 0.884 | 0.276 | 16.3 | 10.5 | 403 | 301 |
| bonsai | `splatc-tiny` | 1.94 MB | 45.5x | 123,944 | 24.29 | 0.815 | 0.329 | 21.6 | 12.6 | 425.4 | 319.7 |
| bonsai | `splatc-deflate` | 5.58 MB | 15.8x | 354,126 | – | – | – | – | – | – | – |
| bonsai | `splatc-no-entropy` | 6.92 MB | 12.8x | 354,126 | – | – | – | – | – | – | – |

FPS columns: `<GPU> @ <width>x<height>`. `splatc-deflate` / `splatc-no-entropy` decode to exactly the same scene as `splatc` (entropy-coder ablation: size only).

## Per-stage GPU time (ms, WebGPU timestamp queries)

| GPU | Scene | Resolution | preprocess | scan | emit | radix sort | ranges | raster | total |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| intel gen-12lp | lego-splatc | 1280x720 | 11.96 | 0.34 | 1.52 | 15.30 | 0.47 | 24.28 | 53.87 |
| intel gen-12lp | lego-splatc | 1920x1080 | 11.89 | 0.34 | 1.86 | 18.69 | 0.57 | 34.90 | 68.25 |
| intel gen-12lp | lego-splatc-small | 1280x720 | 7.36 | 0.23 | 0.99 | 10.39 | 0.33 | 16.72 | 36.02 |
| intel gen-12lp | lego-splatc-small | 1920x1080 | 7.39 | 0.24 | 1.30 | 12.97 | 0.41 | 22.14 | 44.45 |
| intel gen-12lp | lego-splatc-tiny | 1280x720 | 4.46 | 0.16 | 0.66 | 7.09 | 0.23 | 11.73 | 24.34 |
| intel gen-12lp | lego-splatc-tiny | 1920x1080 | 4.38 | 0.17 | 0.93 | 9.45 | 0.30 | 16.33 | 31.55 |
| intel gen-12lp | bonsai-splatc | 1280x720 | 12.94 | 0.62 | 4.20 | 23.04 | 0.65 | 40.40 | 81.85 |
| intel gen-12lp | bonsai-splatc | 1920x1080 | 13.64 | 0.61 | 8.21 | 34.00 | 0.99 | 65.05 | 122.51 |
| intel gen-12lp | bonsai-splatc-small | 1280x720 | 8.69 | 0.38 | 4.16 | 17.47 | 0.52 | 31.49 | 62.71 |
| intel gen-12lp | bonsai-splatc-small | 1920x1080 | 10.31 | 0.38 | 7.32 | 27.58 | 0.84 | 55.96 | 102.38 |
| intel gen-12lp | bonsai-splatc-tiny | 1280x720 | 5.83 | 0.26 | 3.87 | 13.40 | 0.41 | 25.61 | 49.38 |
| intel gen-12lp | bonsai-splatc-tiny | 1920x1080 | 9.09 | 0.26 | 11.35 | 24.35 | 0.73 | 46.04 | 91.82 |
| nvidia ampere | lego-splatc | 1280x720 | 0.23 | 0.02 | 0.11 | 0.52 | 0.02 | 0.96 | 1.85 |
| nvidia ampere | lego-splatc | 1920x1080 | 0.22 | 0.02 | 0.13 | 0.61 | 0.02 | 1.04 | 2.03 |
| nvidia ampere | lego-splatc-small | 1280x720 | 0.12 | 0.01 | 0.08 | 0.37 | 0.01 | 0.75 | 1.34 |
| nvidia ampere | lego-splatc-small | 1920x1080 | 0.12 | 0.01 | 0.12 | 0.43 | 0.01 | 0.74 | 1.44 |
| nvidia ampere | lego-splatc-tiny | 1280x720 | 0.07 | 0.01 | 0.06 | 0.26 | 0.01 | 0.49 | 0.90 |
| nvidia ampere | lego-splatc-tiny | 1920x1080 | 0.11 | 0.01 | 0.12 | 0.41 | 0.01 | 0.66 | 1.32 |
| nvidia ampere | bonsai-splatc | 1280x720 | 0.32 | 0.04 | 0.34 | 0.79 | 0.02 | 1.16 | 2.67 |
| nvidia ampere | bonsai-splatc | 1920x1080 | 0.40 | 0.04 | 0.69 | 1.09 | 0.03 | 1.85 | 4.10 |
| nvidia ampere | bonsai-splatc-small | 1280x720 | 0.24 | 0.02 | 0.36 | 0.59 | 0.02 | 0.95 | 2.18 |
| nvidia ampere | bonsai-splatc-small | 1920x1080 | 0.36 | 0.02 | 0.61 | 0.83 | 0.03 | 1.56 | 3.41 |
| nvidia ampere | bonsai-splatc-tiny | 1280x720 | 0.25 | 0.01 | 0.33 | 0.44 | 0.02 | 0.75 | 1.81 |
| nvidia ampere | bonsai-splatc-tiny | 1920x1080 | 0.33 | 0.01 | 0.65 | 0.76 | 0.03 | 1.31 | 3.09 |
