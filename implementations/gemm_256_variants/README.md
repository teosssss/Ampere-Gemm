# GEMM 256x128x32 Variants (Snapshot)

This folder contains a clean snapshot of the 4 tested kernels from `gemm.cu`, split one implementation per file:

- `reg_pingpong_256.cu`
- `reg_pingpong_256_mma.cu`
- `reg_pingpong_256_colb.cu`
- `reg_pingpong_256_colb_mma.cu`

Shared low-level PTX macros and async-copy primitives are isolated in:

- `ptx_primitives.cuh`

Shared helper routines (prefetch, fragment load/compute, stage rotation) are in:

- `gemm_256_common.cuh`

## Techniques Used (Short)

- Triple-buffered `cp.async` staging over `K_TILE=32`.
- CTA tile `256x128`, warp tile `64x64`.
- Register ping-pong over two 16-wide WMMA halves (`k_step=0/16`).
- Two B-layout families:
  - row-major B (`reg_pingpong_256`, `reg_pingpong_256_mma`)
  - pre-transposed/col-major B (`reg_pingpong_256_colb`, `reg_pingpong_256_colb_mma`)
- Two math backbones:
  - WMMA fragment API
  - inline PTX `ldmatrix` + `mma.sync` (`*_mma`)

## Benchmarks vs Baseline (`torch_mm`, Modal L4)

Source JSONs:

- `results/l4-tritonbench-20260408-111218.json` (k-means-like shapes)
- `results/l4-tritonbench-20260408-110716.json` (larger shapes)
- `results/l4-tritonbench-20260408-112823.json` (post-epilogue spot check)

### K-means-like Shapes (TFLOPS)

| Shape (M,N,K) | torch_mm | reg_pingpong_256 | reg_pingpong_256_mma | reg_pingpong_256_colb | reg_pingpong_256_colb_mma |
|---|---:|---:|---:|---:|---:|
| 16384x256x256 | 25.58 | 22.31 | 14.77 | 21.40 | 16.64 |
| 16384x512x256 | 36.16 | 33.55 | 20.07 | 33.03 | 23.83 |
| 16384x1024x256 | 44.86 | 41.53 | 23.56 | 40.33 | 27.32 |
| 32768x1024x256 | 45.71 | 43.69 | 22.70 | 42.58 | 27.55 |

### Larger Shapes (TFLOPS)

| Shape (M,N,K) | torch_mm | reg_pingpong_256 | reg_pingpong_256_mma | reg_pingpong_256_colb | reg_pingpong_256_colb_mma |
|---|---:|---:|---:|---:|---:|
| 2048x2048x2048 | 63.07 | 57.26 | 32.96 | 39.95 | 36.00 |
| 4096x4096x4096 | 59.34 | 63.22 | 37.85 | 45.12 | 46.49 |
| 8192x8192x8192 | 59.28 | 54.14 | 37.67 | 43.39 | 40.52 |
| 4096x8192x4096 | 66.48 | 57.38 | 39.11 | 44.38 | 44.79 |
| 8192x4096x4096 | 57.32 | 59.77 | 39.09 | 52.08 | 50.52 |

### Post-Change Spot Check (`_colb_mma` epilogue)

From `results/l4-tritonbench-20260408-112823.json`:

- Shape `16384x1024x256`
  - `reg_pingpong_256_colb`: `39.95`
  - `reg_pingpong_256_colb_mma`: `34.81`

## Plots

- `plots/kmeans_tflops.png`
- `plots/large_tflops.png`
- `plots/baseline_relative_tflops.png`

The baseline-relative plot shows each kernel as `TFLOPS / torch_mm` across the tested shapes, so values above `1.0` beat the baseline and values below `1.0` trail it.

Regenerate:

```bash
python implementations/gemm_256_variants/plot_benchmarks.py
```
