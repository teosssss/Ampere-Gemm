from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Generator

import torch
from tritonbench.utils.triton_op import BenchmarkOperator, REGISTERED_X_VALS, override_args

from tensorcore_gemm import KERNEL_MODES, gemm, gemm_pretransposed
from tensorcore_gemm.cublas_gemm import gemm as cublas_gemm


DEFAULT_MODES = ",".join(KERNEL_MODES)
DEFAULT_METRICS = "latency,tflops,speedup,accuracy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--warmup", type=int, default=25, help="TritonBench warmup time in ms.")
    parser.add_argument("--rep", type=int, default=100, help="TritonBench rep time in ms.")
    parser.add_argument("--iters", type=int, default=None, help="Alias for --rep for compatibility with the existing Modal runner.")
    parser.add_argument("--modes", type=str, default=DEFAULT_MODES)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--metrics", type=str, default=DEFAULT_METRICS)
    parser.add_argument("--baseline", type=str, default="torch_mm")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def parse_cases(raw_cases: str, default_case: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    if not raw_cases.strip():
        return [default_case]

    cases: list[tuple[int, int, int]] = []
    for raw_case in raw_cases.split(","):
        case = raw_case.strip().lower()
        if not case:
            continue
        try:
            m_str, n_str, k_str = case.split("x")
            cases.append((int(m_str), int(n_str), int(k_str)))
        except ValueError as exc:
            raise ValueError(f"Invalid case {raw_case!r}. Expected format MxNxK.") from exc
    return cases


def parse_modes(raw_modes: str) -> list[str]:
    modes = [mode.strip() for mode in raw_modes.split(",") if mode.strip()]
    invalid = sorted(set(modes) - set(KERNEL_MODES))
    if invalid:
        raise ValueError(f"Unsupported modes: {invalid}. Expected a subset of {KERNEL_MODES}.")
    return modes


class TensorCoreGemmOperator(BenchmarkOperator):
    DEFAULT_PRECISION = "fp16"

    def __init__(
        self,
        tb_args: argparse.Namespace,
        extra_args: list[str] | None = None,
        *,
        cases: list[tuple[int, int, int]],
        modes: list[str],
        dtype: torch.dtype,
    ) -> None:
        self._cases = cases
        self._modes = modes
        self._input_dtype = dtype
        super().__init__(tb_args, extra_args)
        REGISTERED_X_VALS[self.name] = "(M, N, K)"

        self.add_benchmark("torch_mm", lambda a, b, _b_col: torch.mm(a, b), baseline=(tb_args.baseline == "torch_mm"))
        self.add_benchmark(
            "torch_matmul",
            lambda a, b, _b_col: torch.matmul(a, b),
            baseline=(tb_args.baseline == "torch_matmul"),
        )
        self.add_benchmark(
            "cublas_gemm",
            lambda a, b, _b_col: cublas_gemm(a, b),
            baseline=(tb_args.baseline == "cublas_gemm"),
        )
        for mode in self._modes:
            if mode in {"reg_pingpong_256_colb", "reg_pingpong_256_colb_mma"}:
                self.add_benchmark(
                    mode,
                    lambda a, _b, b_col, mode=mode: gemm_pretransposed(a, b_col, mode=mode),
                    baseline=(tb_args.baseline == mode),
                )
            else:
                self.add_benchmark(
                    mode,
                    lambda a, b, _b_col, mode=mode: gemm(a, b, mode=mode),
                    baseline=(tb_args.baseline == mode),
                )

    def get_input_iter(self) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        for m, n, k in self._cases:
            a = torch.randn((m, k), device=self.device, dtype=self._input_dtype)
            b = torch.randn((k, n), device=self.device, dtype=self._input_dtype)
            b_col_major = b.transpose(0, 1).contiguous()
            yield a, b, b_col_major

    def get_x_val(self, example_inputs: Any) -> tuple[int, int, int]:
        a, b, _b_col = example_inputs
        m, k = a.shape
        _, n = b.shape
        return (m, n, k)

    def flops(self, fn_name: str, example_inputs: Any, metrics: Any) -> float:
        a, b, _b_col = example_inputs
        m, k = a.shape
        _, n = b.shape
        return float(2 * m * n * k)


def latency_ms(latency: Any) -> float | None:
    if latency is None:
        return None
    if hasattr(latency, "p50"):
        return float(latency.p50)
    if hasattr(latency, "avg"):
        return float(latency.avg)
    if isinstance(latency, (int, float)):
        return float(latency)
    return None


def result_to_dict(output: Any, *, dtype: torch.dtype, baseline: str) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for x_val, backends in output.result:
        m, n, k = x_val
        case_results: list[dict[str, Any]] = []
        for backend_name, metrics in backends.items():
            case_results.append(
                {
                    "backend": backend_name,
                    "latency_ms": latency_ms(metrics.latency),
                    "tflops": None if metrics.tflops is None else float(metrics.tflops),
                    "speedup_vs_baseline": None if metrics.speedup is None else float(metrics.speedup),
                    "accuracy_vs_baseline": metrics.accuracy,
                    "error": metrics.error_msg,
                }
            )
        cases.append({"shape": {"m": m, "n": n, "k": k}, "results": case_results})

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype).replace("torch.", ""),
        "metrics": list(output.metrics),
        "baseline": baseline,
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required"

    dtype = getattr(torch, args.dtype)
    modes = parse_modes(args.modes)
    cases = parse_cases(args.cases, (args.m, args.n, args.k))
    rep = args.rep if args.iters is None else args.iters

    tb_args, extra_args = override_args(
        [
            "--device",
            "cuda",
            "--mode",
            "fwd_no_grad",
            "--warmup",
            str(args.warmup),
            "--rep",
            str(rep),
            "--metrics",
            args.metrics,
            "--benchmark-name",
            "tensorcore_gemm",
            "--baseline",
            args.baseline,
            "--atol",
            str(args.atol),
            "--rtol",
            str(args.rtol),
        ]
    )

    op = TensorCoreGemmOperator(tb_args, extra_args, cases=cases, modes=modes, dtype=dtype)
    op.run()

    results = result_to_dict(op.output, dtype=dtype, baseline=args.baseline)
    print(json.dumps(results, indent=2))

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
