from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorcore_gemm import KERNEL_MODES, gemm, gemm_pretransposed
from tensorcore_gemm.cublas_gemm import gemm as cublas_gemm


COLB_MODES = {"reg_pingpong_256_colb", "reg_pingpong_256_colb_mma"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--modes", type=str, default="reg_pingpong_256,reg_pingpong_256_mma,reg_pingpong_256_colb,reg_pingpong_256_colb_mma")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def parse_cases(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    if not args.cases.strip():
        return [(args.m, args.n, args.k)]

    cases: list[tuple[int, int, int]] = []
    for raw_case in args.cases.split(","):
        case = raw_case.strip().lower()
        if not case:
            continue
        m_str, n_str, k_str = case.split("x")
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


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator


def measure_pytorch_baselines(A: torch.Tensor, B: torch.Tensor, warmup: int, iters: int) -> dict[str, float | bool]:
    matmul_out = torch.matmul(A, B)
    mm_out = torch.mm(A, B)
    cublas_out = cublas_gemm(A, B)
    torch.cuda.synchronize()

    matmul_ms = benchmark_cuda_callable(lambda: torch.matmul(A, B), warmup=warmup, iters=iters)
    mm_ms = benchmark_cuda_callable(lambda: torch.mm(A, B), warmup=warmup, iters=iters)
    cublas_ms = benchmark_cuda_callable(lambda: cublas_gemm(A, B), warmup=warmup, iters=iters)

    return {
        "matmul_ms": float(matmul_ms),
        "mm_ms": float(mm_ms),
        "cublas_ms": float(cublas_ms),
        "matmul_tflops": float(tflops(A.shape[0], B.shape[1], A.shape[1], matmul_ms)),
        "mm_tflops": float(tflops(A.shape[0], B.shape[1], A.shape[1], mm_ms)),
        "cublas_tflops": float(tflops(A.shape[0], B.shape[1], A.shape[1], cublas_ms)),
        "matmul_equals_mm": bool(torch.equal(matmul_out, mm_out)),
        "matmul_vs_mm_max_abs_diff": float((matmul_out - mm_out).abs().max().item()),
        "matmul_vs_mm_mean_abs_diff": float((matmul_out - mm_out).abs().mean().item()),
        "cublas_close_vs_mm": bool(torch.allclose(cublas_out, mm_out, atol=5e-2, rtol=5e-2)),
        "cublas_vs_mm_max_abs_diff": float((cublas_out - mm_out).abs().max().item()),
        "cublas_vs_mm_mean_abs_diff": float((cublas_out - mm_out).abs().mean().item()),
    }


def run_mode_case(
    mode: str,
    A: torch.Tensor,
    B: torch.Tensor,
    B_col_major: torch.Tensor,
    transpose_ms: float,
    baselines: dict[str, float | bool],
    warmup: int,
    iters: int,
) -> dict[str, float | bool]:
    reference = torch.mm(A, B)
    if mode in COLB_MODES:
        kernel_out = gemm_pretransposed(A, B_col_major, mode=mode)
        kernel_ms = benchmark_cuda_callable(
            lambda: gemm_pretransposed(A, B_col_major, mode=mode), warmup=warmup, iters=iters
        )
    else:
        kernel_out = gemm(A, B, mode=mode)
        kernel_ms = benchmark_cuda_callable(lambda: gemm(A, B, mode=mode), warmup=warmup, iters=iters)

    torch.cuda.synchronize()

    max_abs_err = (kernel_out - reference).abs().max().item()
    mean_abs_err = (kernel_out - reference).abs().mean().item()
    close = torch.allclose(kernel_out, reference, atol=5e-2, rtol=5e-2)
    kernel_tflops = tflops(A.shape[0], B.shape[1], A.shape[1], kernel_ms)
    matmul_ms = float(baselines["matmul_ms"])
    mm_ms = float(baselines["mm_ms"])
    cublas_ms = float(baselines["cublas_ms"])
    matmul_tflops = float(baselines["matmul_tflops"])
    mm_tflops = float(baselines["mm_tflops"])
    cublas_tflops = float(baselines["cublas_tflops"])

    result = {
        "mode": mode,
        "close_vs_torch_mm": bool(close),
        "max_abs_err_vs_torch_mm": float(max_abs_err),
        "mean_abs_err_vs_torch_mm": float(mean_abs_err),
        "kernel_ms": float(kernel_ms),
        "kernel_tflops": float(kernel_tflops),
        "pct_of_torch_matmul_tflops": float(pct(kernel_tflops, matmul_tflops)),
        "pct_of_torch_mm_tflops": float(pct(kernel_tflops, mm_tflops)),
        "latency_pct_of_torch_matmul": float(pct(kernel_ms, matmul_ms)),
        "latency_pct_of_torch_mm": float(pct(kernel_ms, mm_ms)),
        "latency_pct_of_cublas": float(pct(kernel_ms, cublas_ms)),
        "speedup_vs_torch_matmul": float(matmul_ms / kernel_ms),
        "speedup_vs_torch_mm": float(mm_ms / kernel_ms),
        "speedup_vs_cublas": float(cublas_ms / kernel_ms),
        "pct_of_cublas_tflops": float(pct(kernel_tflops, cublas_tflops)),
    }
    if mode in COLB_MODES:
        result["transpose_ms_separate"] = float(transpose_ms)
        result["kernel_plus_transpose_ms"] = float(kernel_ms + transpose_ms)
    return result


def run_shape_case(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    modes: list[str],
    warmup: int,
    iters: int,
) -> dict[str, object]:
    A = torch.randn((m, k), device="cuda", dtype=dtype)
    B = torch.randn((k, n), device="cuda", dtype=dtype)
    B_col_major = B.transpose(0, 1).contiguous()
    torch.cuda.synchronize()

    transpose_ms = benchmark_cuda_callable(lambda: B.transpose(0, 1).contiguous(), warmup=warmup, iters=iters)
    baselines = measure_pytorch_baselines(A, B, warmup=warmup, iters=iters)
    mode_results = [
        run_mode_case(
            mode,
            A,
            B,
            B_col_major,
            transpose_ms=transpose_ms,
            baselines=baselines,
            warmup=warmup,
            iters=iters,
        )
        for mode in modes
    ]

    return {
        "shape": {"m": m, "n": n, "k": k},
        "transpose_ms": float(transpose_ms),
        "baselines": baselines,
        "results": mode_results,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is required"

    dtype = getattr(torch, args.dtype)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    invalid = sorted(set(modes) - set(KERNEL_MODES))
    if invalid:
        raise ValueError(f"Unsupported modes: {invalid}. Expected a subset of {KERNEL_MODES}.")

    cases = parse_cases(args)
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "benchmark_type": "fair_pretransposed_for_colb",
        "cases": [
            run_shape_case(m, n, k, dtype=dtype, modes=modes, warmup=args.warmup, iters=args.iters)
            for m, n, k in cases
        ],
    }

    print(json.dumps(results, indent=2))

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
