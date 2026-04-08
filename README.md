# tensorcore-gemm

Standalone CUDA Tensor Core GEMM repo with:

- a packaged PyTorch extension under `src/tensorcore_gemm`
- Modal + TritonBench benchmarking harnesses
- a clean snapshot of the 4 `256x128x32` implementation variants under `implementations/gemm_256_variants`

## Repo Structure

- [`src/tensorcore_gemm/gemm.cu`](./src/tensorcore_gemm/gemm.cu): canonical CUDA source used by the runtime wrapper
- [`src/tensorcore_gemm/gemm.py`](./src/tensorcore_gemm/gemm.py): Python API and mode dispatch
- [`src/tensorcore_gemm/cublas_gemm.cu`](./src/tensorcore_gemm/cublas_gemm.cu): cuBLAS comparison kernel
- [`src/tensorcore_gemm/cuda_compile.py`](./src/tensorcore_gemm/cuda_compile.py): JIT compile helper
- [`benchmark_tritonbench.py`](./benchmark_tritonbench.py): TritonBench harness
- [`modal_runner.py`](./modal_runner.py): Modal L4 runner
- [`implementations/gemm_256_variants`](./implementations/gemm_256_variants): commit-friendly extracted implementations and benchmark plots

## Kernel Modes

Available runtime modes:

- `wmma`
- `mma`
- `reg_pingpong_256`
- `reg_pingpong_256_mma`
- `reg_pingpong_256_colb`
- `reg_pingpong_256_colb_mma`

The `reg_pingpong_256*` kernels are the main optimized variants for the `256x128x32` tile family.

## Techniques Used

- CTA tile `256x128`, warp tile `64x64`
- triple-buffered `cp.async` staging on `K_TILE=32`
- register ping-pong between the two `k_step=0/16` halves
- shared-memory padding to reduce bank conflicts
- swizzled CTA traversal
- WMMA fragment path for the higher-level kernels
- `ldmatrix` + `mma.sync` PTX path for the lower-level MMA kernels
- optional pre-transposed / column-major `B` path for better operand layout

## Implementation Snapshot

The extracted implementations are here:

- [`implementations/gemm_256_variants/reg_pingpong_256.cu`](./implementations/gemm_256_variants/reg_pingpong_256.cu)
- [`implementations/gemm_256_variants/reg_pingpong_256_mma.cu`](./implementations/gemm_256_variants/reg_pingpong_256_mma.cu)
- [`implementations/gemm_256_variants/reg_pingpong_256_colb.cu`](./implementations/gemm_256_variants/reg_pingpong_256_colb.cu)
- [`implementations/gemm_256_variants/reg_pingpong_256_colb_mma.cu`](./implementations/gemm_256_variants/reg_pingpong_256_colb_mma.cu)

Shared pieces:

- [`implementations/gemm_256_variants/ptx_primitives.cuh`](./implementations/gemm_256_variants/ptx_primitives.cuh)
- [`implementations/gemm_256_variants/gemm_256_common.cuh`](./implementations/gemm_256_variants/gemm_256_common.cuh)
- [`implementations/gemm_256_variants/README.md`](./implementations/gemm_256_variants/README.md)

## Constraints

For the optimized `reg_pingpong_256*` path, the wrapper expects:

- `torch.float16`
- 2D contiguous inputs
- `M % 256 == 0`
- `N % 128 == 0`
- `K % 32 == 0`
- `K >= 64`

Outside those constraints, the Python wrapper falls back to `torch.matmul`.

## Local Usage

```bash
uv sync --extra cuda --extra bench
uv run python benchmark.py --m 4096 --n 4096 --k 4096
```

## TritonBench on Modal L4

```bash
uv run modal run modal_runner.py --action tritonbench --cases 4096x4096x4096 --warmup 20 --iters 50 --modes reg_pingpong_256,reg_pingpong_256_mma,reg_pingpong_256_colb,reg_pingpong_256_colb_mma
```

This stages the repo into Modal, runs the selected kernels on an NVIDIA L4, and saves the JSON summary in `results/`.

## Benchmarks

The concise benchmark tables and plots for the 4 `256` variants live in:

- [`implementations/gemm_256_variants/README.md`](./implementations/gemm_256_variants/README.md)

Generated plots:

- [`implementations/gemm_256_variants/plots/kmeans_tflops.png`](./implementations/gemm_256_variants/plots/kmeans_tflops.png)
- [`implementations/gemm_256_variants/plots/large_tflops.png`](./implementations/gemm_256_variants/plots/large_tflops.png)
