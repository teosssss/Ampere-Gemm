from __future__ import annotations

from pathlib import Path

import torch

from .cuda_compile import compile_cuda

KERNEL_MODES = (
    "wmma",
    "mma",
    "reg_pingpong_256",
    "reg_pingpong_256_mma",
    "reg_pingpong_256_colb",
    "reg_pingpong_256_colb_mma",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_CUDA_PATH = _SCRIPT_DIR / "gemm.cu"
_MODULE = None
_ROOT_MODULE = None
_MODE_TO_ID = {
    "wmma": 0,
    "mma": 1,
    "reg_pingpong_256": 2,
    "reg_pingpong_256_mma": 3,
    "reg_pingpong_256_colb": 4,
    "reg_pingpong_256_colb_mma": 5,
}

_ROOT_WRAPPER = r"""
torch::Tensor root_gemm_cuda(torch::Tensor A, torch::Tensor B, int64_t mode) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(mode >= 2 && mode <= 5, "root mode must be one of reg_pingpong_256 variants");

    const bool use_colb = mode == 4 || mode == 5;
    TORCH_CHECK(
        use_colb ? (A.size(1) == B.size(1)) : (A.size(1) == B.size(0)),
        use_colb ? "Inner dimensions must match for pre-transposed B" : "Inner dimensions must match"
    );

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = use_colb ? static_cast<int>(B.size(0)) : static_cast<int>(B.size(1));

    constexpr int TILE_M = 256;
    constexpr int TILE_N = 128;
    constexpr int K_TILE = 32;
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int STAGE_COUNT = 3;
    constexpr int THREADS = 256;

    auto C = torch::empty({M, N}, A.options());

    dim3 grid_256(
        16,
        static_cast<unsigned int>((M + TILE_M - 1) / TILE_M),
        static_cast<unsigned int>(((N + TILE_N - 1) / TILE_N + 16 - 1) / 16)
    );
    dim3 block_256(THREADS);

    if (mode == 2) {
        constexpr int smem_bytes =
            STAGE_COUNT *
            (TILE_M * (K_TILE + PAD_A) + K_TILE * (TILE_N + PAD_B)) *
            static_cast<int>(sizeof(half));
        cudaError_t attr_err = cudaFuncSetAttribute(
            gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
        TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));

        gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256<<<grid_256, block_256, smem_bytes>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            M, N, K,
            1.0f, 0.0f
        );
    } else if (mode == 3) {
        constexpr int smem_bytes =
            STAGE_COUNT *
            (TILE_M * (K_TILE + PAD_A) + TILE_N * (K_TILE + PAD_B)) *
            static_cast<int>(sizeof(half)) +
            (K_TILE * (TILE_N + PAD_B)) *
            static_cast<int>(sizeof(half));
        cudaError_t attr_err = cudaFuncSetAttribute(
            gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256_MMA,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
        TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));

        gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256_MMA<<<grid_256, block_256, smem_bytes>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            M, N, K,
            1.0f, 0.0f
        );
    } else if (mode == 4) {
        constexpr int smem_bytes =
            STAGE_COUNT *
            (TILE_M * (K_TILE + PAD_A) + TILE_N * (K_TILE + PAD_B)) *
            static_cast<int>(sizeof(half));
        cudaError_t attr_err = cudaFuncSetAttribute(
            gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256_colb,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
        TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));

        gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256_colb<<<grid_256, block_256, smem_bytes>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            M, N, K,
            1.0f, 0.0f
        );
    } else {
        constexpr int smem_bytes =
            STAGE_COUNT *
            (TILE_M * (K_TILE + PAD_A) + TILE_N * (K_TILE + PAD_B)) *
            static_cast<int>(sizeof(half));
        cudaError_t attr_err = cudaFuncSetAttribute(
            gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256_colb_MMA,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
        TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));

        gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256_colb_MMA<<<grid_256, block_256, smem_bytes>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            M, N, K,
            1.0f, 0.0f
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "root_gemm launch failed: ", cudaGetErrorString(err));
    return C;
}
"""


def _load_cuda_src() -> str:
    return _CUDA_PATH.read_text()


def _load_root_cuda_src() -> str:
    prefix = "#include <torch/extension.h>\n#include <algorithm>\n"
    return prefix + _CUDA_PATH.read_text() + "\n" + _ROOT_WRAPPER


def _normalize_mode(mode: str) -> int:
    try:
        return _MODE_TO_ID[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode {mode!r}. Expected one of {KERNEL_MODES}.") from exc


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
        and k >= 64
    )


def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = compile_cuda(_load_cuda_src(), "gemm_cuda")
    return _MODULE


def _get_root_module():
    global _ROOT_MODULE
    if _ROOT_MODULE is None:
        _ROOT_MODULE = compile_cuda(_load_root_cuda_src(), "root_gemm_cuda")
    return _ROOT_MODULE


def gemm_pretransposed(A: torch.Tensor, B_col_major: torch.Tensor, mode: str = "wmma") -> torch.Tensor:
    mode_id = _normalize_mode(mode)
    if mode_id < 2:
        mod = _get_module()
        return mod.gemm_cuda(A.contiguous(), B_col_major.contiguous(), mode_id)

    mod = _get_root_module()
    return mod.root_gemm_cuda(A.contiguous(), B_col_major.contiguous(), mode_id)


def gemm(A: torch.Tensor, B: torch.Tensor, mode: str = "wmma") -> torch.Tensor:
    if not is_supported(A, B):
        return torch.matmul(A, B)

    if mode in {"reg_pingpong_256", "reg_pingpong_256_mma"}:
        mod = _get_root_module()
        return mod.root_gemm_cuda(A.contiguous(), B.contiguous(), _normalize_mode(mode))

    B_col_major = B.transpose(0, 1).contiguous()
    return gemm_pretransposed(A, B_col_major, mode=mode)
