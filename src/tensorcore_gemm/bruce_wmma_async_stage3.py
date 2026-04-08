from __future__ import annotations

from pathlib import Path

import torch

from .cuda_compile import compile_cuda

_SCRIPT_DIR = Path(__file__).resolve().parent
_CUDA_PATH = _SCRIPT_DIR / "bruce_wmma_async_stage3.cu"
_MODULE = None


def _load_cuda_src() -> str:
    return _CUDA_PATH.read_text()


def is_supported(A: torch.Tensor, B: torch.Tensor) -> bool:
    if not (A.is_cuda and B.is_cuda):
        return False
    if A.ndim != 2 or B.ndim != 2:
        return False
    if A.shape[1] != B.shape[0]:
        return False

    m, k = A.shape
    _, n = B.shape
    return (
        A.dtype == torch.float16
        and B.dtype == torch.float16
        and A.is_contiguous()
        and B.is_contiguous()
        and m % 256 == 0
        and n % 128 == 0
        and k % 32 == 0
    )


def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = compile_cuda(_load_cuda_src(), "bruce_wmma_async_stage3_cuda")
    return _MODULE


def gemm_pretransposed(A: torch.Tensor, B_col_major: torch.Tensor) -> torch.Tensor:
    mod = _get_module()
    return mod.bruce_wmma_async_stage3_cuda(A.contiguous(), B_col_major.contiguous())


def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not is_supported(A, B):
        return torch.mm(A, B)

    B_col_major = B.transpose(0, 1).contiguous()
    return gemm_pretransposed(A, B_col_major)
