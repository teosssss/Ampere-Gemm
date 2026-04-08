#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

torch::Tensor cublas_gemm_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasStatus_t create_status = cublasCreate(&handle);
        TORCH_CHECK(create_status == CUBLAS_STATUS_SUCCESS, "cublasCreate failed with status ", static_cast<int>(create_status));
        cublasStatus_t stream_status = cublasSetStream(handle, nullptr);
        TORCH_CHECK(stream_status == CUBLAS_STATUS_SUCCESS, "cublasSetStream failed with status ", static_cast<int>(stream_status));
        cublasStatus_t math_status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        TORCH_CHECK(math_status == CUBLAS_STATUS_SUCCESS, "cublasSetMathMode failed with status ", static_cast<int>(math_status));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS assumes column-major. Reinterpret the row-major tensors as:
    // A_row[M, K] -> A_col[K, M], B_row[K, N] -> B_col[N, K], C_row[M, N] -> C_col[N, M].
    // Then compute C_col = B_col * A_col with no transposes.
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(N),
        static_cast<int>(M),
        static_cast<int>(K),
        &alpha,
        B.data_ptr<at::Half>(),
        CUDA_R_16F,
        static_cast<int>(N),
        A.data_ptr<at::Half>(),
        CUDA_R_16F,
        static_cast<int>(K),
        &beta,
        C.data_ptr<at::Half>(),
        CUDA_R_16F,
        static_cast<int>(N),
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed with status ", static_cast<int>(status));
    return C;
}
