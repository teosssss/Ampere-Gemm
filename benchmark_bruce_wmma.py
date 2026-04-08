from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorcore_gemm.cublas_gemm import gemm as cublas_gemm
from tensorcore_gemm.bruce_wmma_async_stage3 import gemm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--cases", type=str, default="2048x2048x2048,4096x4096x4096,8192x8192x8192")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--modes", type=str, default="")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def parse_cases(spec: str) -> list[tuple[int, int, int]]:
    cases = []
    for raw_case in spec.split(","):
        m_str, n_str, k_str = raw_case.strip().lower().split("x")
        cases.append((int(m_str), int(n_str), int(k_str)))
    return cases


def benchmark_cuda_callable(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def tflops(m: int, n: int, k: int, ms: float) -> float:
    return (2.0 * m * n * k) / (ms / 1000.0) / 1.0e12


def run_case(m: int, n: int, k: int, dtype: torch.dtype, warmup: int, iters: int) -> dict[str, object]:
    A = torch.randn((m, k), device="cuda", dtype=dtype)
    B = torch.randn((k, n), device="cuda", dtype=dtype)

    matmul_out = torch.matmul(A, B)
    mm_out = torch.mm(A, B)
    cublas_out = cublas_gemm(A, B)
    kernel_out = gemm(A, B)
    torch.cuda.synchronize()

    matmul_ms = benchmark_cuda_callable(lambda: torch.matmul(A, B), warmup=warmup, iters=iters)
    mm_ms = benchmark_cuda_callable(lambda: torch.mm(A, B), warmup=warmup, iters=iters)
    cublas_ms = benchmark_cuda_callable(lambda: cublas_gemm(A, B), warmup=warmup, iters=iters)
    kernel_ms = benchmark_cuda_callable(lambda: gemm(A, B), warmup=warmup, iters=iters)

    return {
        "shape": {"m": m, "n": n, "k": k},
        "matmul_equals_mm": bool(torch.equal(matmul_out, mm_out)),
        "matmul_vs_mm_max_abs_diff": float((matmul_out - mm_out).abs().max().item()),
        "cublas_close_vs_mm": bool(torch.allclose(cublas_out, mm_out, atol=5e-2, rtol=5e-2)),
        "cublas_vs_mm_max_abs_diff": float((cublas_out - mm_out).abs().max().item()),
        "cublas_vs_mm_mean_abs_diff": float((cublas_out - mm_out).abs().mean().item()),
        "kernel_close_vs_mm": bool(torch.allclose(kernel_out, mm_out, atol=5e-2, rtol=5e-2)),
        "kernel_max_abs_err_vs_mm": float((kernel_out - mm_out).abs().max().item()),
        "kernel_mean_abs_err_vs_mm": float((kernel_out - mm_out).abs().mean().item()),
        "kernel_ms": float(kernel_ms),
        "matmul_ms": float(matmul_ms),
        "mm_ms": float(mm_ms),
        "cublas_ms": float(cublas_ms),
        "kernel_tflops": float(tflops(m, n, k, kernel_ms)),
        "matmul_tflops": float(tflops(m, n, k, matmul_ms)),
        "mm_tflops": float(tflops(m, n, k, mm_ms)),
        "cublas_tflops": float(tflops(m, n, k, cublas_ms)),
        "pct_of_torch_matmul_tflops": float(100.0 * tflops(m, n, k, kernel_ms) / tflops(m, n, k, matmul_ms)),
        "pct_of_torch_mm_tflops": float(100.0 * tflops(m, n, k, kernel_ms) / tflops(m, n, k, mm_ms)),
        "pct_of_cublas_tflops": float(100.0 * tflops(m, n, k, kernel_ms) / tflops(m, n, k, cublas_ms)),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is required"

    dtype = getattr(torch, args.dtype)
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "kernel": "Bruce-Lee-LY wmma_async_stage3",
        "cases": [run_case(m, n, k, dtype=dtype, warmup=args.warmup, iters=args.iters) for m, n, k in parse_cases(args.cases)],
    }

    print(json.dumps(results, indent=2))

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
