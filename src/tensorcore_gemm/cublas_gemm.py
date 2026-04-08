from __future__ import annotations

from pathlib import Path

import torch

from .cuda_compile import compile_cuda

_SCRIPT_DIR = Path(__file__).resolve().parent
_CUDA_PATH = _SCRIPT_DIR / "cublas_gemm.cu"
_MODULE = None


def _load_cuda_src() -> str:
    return _CUDA_PATH.read_text()


def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = compile_cuda(
            _load_cuda_src(),
            "cublas_gemm_cuda",
            extra_ldflags=["-lcublas"],
        )
    return _MODULE


def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not (A.is_cuda and B.is_cuda):
        return torch.mm(A, B)
    if A.ndim != 2 or B.ndim != 2:
        return torch.matmul(A, B)
    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner dimensions must match.")
    if A.dtype != torch.float16 or B.dtype != torch.float16:
        return torch.mm(A, B)
    if not (A.is_contiguous() and B.is_contiguous()):
        A = A.contiguous()
        B = B.contiguous()

    mod = _get_module()
    return mod.cublas_gemm_cuda(A, B)
