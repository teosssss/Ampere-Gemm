# GEMM Variants

This project contains Tensor Core GEMM implementations optimized for Ampere / Ada Lovelace CUDA architectures, with the goal of closing as much of the gap to cuBLAS as possible on practical GEMM shapes.

The implementations here use multi-stage shared-memory staging, register ping-pong, WMMA or inline MMA instructions, and alternative `B` operand layouts.

## Project Structure

- Main implementations:
  - `reg_pingpong_256.cu`
  - `reg_pingpong_256_mma.cu`
  - `reg_pingpong_256_colb.cu`
  - `reg_pingpong_256_colb_mma.cu`
- Shared PTX macros and async-copy primitives:
  - `ptx_primitives.cuh`
- Shared helpers:
  - `gemm_256_common.cuh`

## Techniques Used

- Triple-buffered `cp.async` staging over `K_TILE=32`.
- CTA tile `256x128`, warp tile `64x64`.
- Register ping-pong over two 16-wide WMMA halves (`k_step=0/16`).
- Two B-layout families:
  - row-major B for `reg_pingpong_256` and `reg_pingpong_256_mma`
  - pre-transposed col-major B for `reg_pingpong_256_colb` and `reg_pingpong_256_colb_mma`
- Two math backbones:
  - WMMA fragment API
  - inline PTX `ldmatrix` + `mma.sync` for the `*_mma` variants

## Benchmarks on Modal L4

The large-shape plot includes the custom kernels together with `torch_mm`, `torch_matmul`, and `cublas_gemm`.

The baseline-relative plot shows each backend as `TFLOPS / torch_mm` across the tested shapes, so values above `1.0` beat the baseline and values below `1.0` trail it.

The `colb` variants are measured with `B` pretransposed, so these figures isolate kernel runtime instead of layout conversion cost.

![Large-shape TFLOPS](../../plots/large_tflops.png)

![Baseline-relative TFLOPS](../../plots/baseline_relative_tflops.png)

Source JSONs:

- `results/l4-tritonbench-20260408-121555.json` (k-means-like shapes)
- `results/l4-tritonbench-20260408-121605.json` (larger shapes)

### K-means-like Shapes (TFLOPS)

| Shape (M,N,K) | torch_mm | torch_matmul | cublas_gemm | reg_pingpong_256 | reg_pingpong_256_mma | reg_pingpong_256_colb | reg_pingpong_256_colb_mma |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16384x256x256 | 24.67 | 24.67 | 24.67 | 22.31 | 14.78 | 22.08 | 20.76 |
| 16384x512x256 | 36.16 | 36.16 | 35.85 | 33.55 | 20.07 | 33.55 | 32.51 |
| 16384x1024x256 | 45.59 | 45.34 | 45.59 | 41.94 | 23.56 | 42.15 | 36.16 |
| 32768x1024x256 | 46.09 | 45.96 | 45.84 | 44.50 | 22.22 | 44.38 | 38.57 |

### Larger Shapes (TFLOPS)

| Shape (M,N,K) | torch_mm | torch_matmul | cublas_gemm | reg_pingpong_256 | reg_pingpong_256_mma | reg_pingpong_256_colb | reg_pingpong_256_colb_mma |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048x2048x2048 | 63.07 | 61.68 | 62.37 | 56.68 | 32.58 | 54.83 | 52.92 |
| 4096x4096x4096 | 63.34 | 59.36 | 56.90 | 57.36 | 36.48 | 56.78 | 59.71 |
| 8192x8192x8192 | 58.54 | 57.71 | 55.81 | 52.26 | 37.21 | 46.36 | 46.73 |
| 4096x8192x4096 | 61.68 | 64.20 | 64.20 | 59.00 | 38.15 | 53.38 | 54.66 |
| 8192x4096x4096 | 54.86 | 56.50 | 51.56 | 58.91 | 38.30 | 56.56 | 58.04 |

Regenerate:

```bash
python src/tensorcore_gemm/plot_benchmarks.py
```
