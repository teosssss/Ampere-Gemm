#include <torch/extension.h>

#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

namespace {

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#else
#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#endif

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define BLOCK_ROW_TILES 8
#define BLOCK_COL_TILES 16

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 4

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK 256

#define CHUNK_K 2

#define THREAD_COPY_BYTES 16
#define CHUNK_COPY_LINES_PER_WARP 8
#define CHUNK_COPY_LINE_LANES 4

#define SMEM_PADDING 8
#define AB_SMEM_STRIDE 40
#define C_SMEM_STRIDE 136
#define C_SMEM_OFFSET 64
#define BLOCK_STRIDE 16
#define K_STAGE 3

__global__ void bruce_wmma_async_stage3_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    size_t M,
    size_t N,
    size_t K
) {
    const size_t M_tiles = div_ceil(M, WMMA_M);
    const size_t N_tiles = div_ceil(N, WMMA_N);
    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t block_tile_i =
        (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * BLOCK_COL_TILES) : (blockIdx.y * BLOCK_COL_TILES);
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    extern __shared__ half smem[][AB_SMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    constexpr size_t B_smem_idx_off = BLOCK_ROWS;
    constexpr size_t smem_stage_off = BLOCK_ROWS + BLOCK_COLS;

    half* smem_warp_tile_ptr = &smem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SMEM_STRIDE * WARP_ROWS +
                               (warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET;

    half* smem_warp_stream_ptr = &smem[0][0] + warp_id * WMMA_M * 2 * C_SMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i + warp_id * 2) * WMMA_M * N + block_tile_j * WMMA_N;
    half* src_gmem_warp_stream_ptr = &C[gmem_idx];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag[WARP_COL_TILES][WARP_ROW_TILES];

    #pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            wmma::fill_fragment(C_frag[i][j], __float2half(0.0f));
        }
    }

    const half* A_warp_ptr = &A[block_tile_i * WMMA_M * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    const half* B_warp_ptr = &B[block_tile_j * WMMA_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

    size_t smem_store_idx = 0;
    size_t smem_load_idx = 0;
    size_t smem_store_off = 0;
    size_t smem_load_off = 0;

    size_t A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    int4* A_lane_ptr = (int4*)(A_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

    #pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        uint32_t A_smem_lane_addr =
            __cvta_generic_to_shared(&smem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);
        A_lane_ptr = (int4*)((half*)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    size_t B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    int4* B_lane_ptr = (int4*)(B_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

    #pragma unroll
    for (size_t i = 0; i < B_smem_iters; ++i) {
        uint32_t B_smem_lane_addr =
            __cvta_generic_to_shared(&smem[B_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);
        B_lane_ptr = (int4*)((half*)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();

    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;

    A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4*)(A_warp_ptr + CHUNK_K * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

    #pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        uint32_t A_smem_lane_addr =
            __cvta_generic_to_shared(&smem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);
        A_lane_ptr = (int4*)((half*)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4*)(B_warp_ptr + CHUNK_K * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

    #pragma unroll
    for (size_t i = 0; i < B_smem_iters; ++i) {
        uint32_t B_smem_lane_addr =
            __cvta_generic_to_shared(&smem[B_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);
        B_lane_ptr = (int4*)((half*)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(1);
    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag[2][WARP_COL_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag[2][WARP_ROW_TILES];

    size_t reg_store_idx = 0;
    size_t reg_load_idx = 1;

    #pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        size_t A_smem_idx_local = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
        const half* A_tile_ptr = &smem[A_smem_idx_local][0];
        wmma::load_matrix_sync(A_frag[reg_store_idx][i], A_tile_ptr, AB_SMEM_STRIDE);
    }

    #pragma unroll
    for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
        size_t B_smem_idx_local = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
        const half* B_tile_ptr = &smem[B_smem_idx_local][0];
        wmma::load_matrix_sync(B_frag[reg_store_idx][j], B_tile_ptr, AB_SMEM_STRIDE);
    }

    #pragma unroll
    for (size_t tile_k = CHUNK_K * (K_STAGE - 1); tile_k < K_tiles; tile_k += CHUNK_K) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx_local = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
            const half* A_tile_ptr = &smem[A_smem_idx_local][WMMA_K];
            wmma::load_matrix_sync(A_frag[reg_store_idx][i], A_tile_ptr, AB_SMEM_STRIDE);
        }

        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_smem_idx_local = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
            const half* B_tile_ptr = &smem[B_smem_idx_local][WMMA_K];
            wmma::load_matrix_sync(B_frag[reg_store_idx][j], B_tile_ptr, AB_SMEM_STRIDE);
        }

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;
                wmma::mma_sync(C_frag[i][j_s], A_frag[reg_load_idx][i], B_frag[reg_load_idx][j_s], C_frag[i][j_s]);
            }
        }

        smem_store_idx = (smem_store_idx + 1) % K_STAGE;
        smem_store_off = smem_store_idx * smem_stage_off;

        A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        A_lane_ptr = (int4*)(A_warp_ptr + tile_k * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

        #pragma unroll
        for (size_t i = 0; i < A_smem_iters / CHUNK_K; ++i) {
            uint32_t A_smem_lane_addr =
                __cvta_generic_to_shared(&smem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
            CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);
            A_lane_ptr = (int4*)((half*)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        B_lane_ptr = (int4*)(B_warp_ptr + tile_k * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

        #pragma unroll
        for (size_t i = 0; i < B_smem_iters / CHUNK_K; ++i) {
            uint32_t B_smem_lane_addr =
                __cvta_generic_to_shared(&smem[B_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
            CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);
            B_lane_ptr = (int4*)((half*)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        smem_load_idx = (smem_load_idx + 1) % K_STAGE;
        smem_load_off = smem_load_idx * smem_stage_off;

        #pragma unroll
        for (size_t i = (CHUNK_K - 1) * A_smem_iters / CHUNK_K; i < A_smem_iters; ++i) {
            uint32_t A_smem_lane_addr =
                __cvta_generic_to_shared(&smem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
            CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);
            A_lane_ptr = (int4*)((half*)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        #pragma unroll
        for (size_t i = (CHUNK_K - 1) * B_smem_iters / CHUNK_K; i < B_smem_iters; ++i) {
            uint32_t B_smem_lane_addr =
                __cvta_generic_to_shared(&smem[B_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
            CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);
            B_lane_ptr = (int4*)((half*)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(1);
        __syncthreads();

        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx_local = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
            const half* A_tile_ptr = &smem[A_smem_idx_local][0];
            wmma::load_matrix_sync(A_frag[reg_store_idx][i], A_tile_ptr, AB_SMEM_STRIDE);
        }

        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_smem_idx_local = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
            const half* B_tile_ptr = &smem[B_smem_idx_local][0];
            wmma::load_matrix_sync(B_frag[reg_store_idx][j], B_tile_ptr, AB_SMEM_STRIDE);
        }

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;
                wmma::mma_sync(C_frag[i][j_s], A_frag[reg_load_idx][i], B_frag[reg_load_idx][j_s], C_frag[i][j_s]);
            }
        }
    }

    #pragma unroll
    for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx_local = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
            const half* A_tile_ptr = &smem[A_smem_idx_local][((k_step + 1) % CHUNK_K) * WMMA_K];
            wmma::load_matrix_sync(A_frag[reg_store_idx][i], A_tile_ptr, AB_SMEM_STRIDE);
        }

        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_smem_idx_local = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
            const half* B_tile_ptr = &smem[B_smem_idx_local][((k_step + 1) % CHUNK_K) * WMMA_K];
            wmma::load_matrix_sync(B_frag[reg_store_idx][j], B_tile_ptr, AB_SMEM_STRIDE);
        }

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;
                wmma::mma_sync(C_frag[i][j_s], A_frag[reg_load_idx][i], B_frag[reg_load_idx][j_s], C_frag[i][j_s]);
            }
        }

        if (k_step + 2 == CHUNK_K) {
            smem_load_idx = (smem_load_idx + 1) % K_STAGE;
            smem_load_off = smem_load_idx * smem_stage_off;
            CP_ASYNC_WAIT_GROUP(0);
            __syncthreads();
        }
    }

    #pragma unroll
    for (size_t k_step = 1; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx_local = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
            const half* A_tile_ptr = &smem[A_smem_idx_local][k_step * WMMA_K];
            wmma::load_matrix_sync(A_frag[reg_store_idx][i], A_tile_ptr, AB_SMEM_STRIDE);
        }

        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_smem_idx_local = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
            const half* B_tile_ptr = &smem[B_smem_idx_local][k_step * WMMA_K];
            wmma::load_matrix_sync(B_frag[reg_store_idx][j], B_tile_ptr, AB_SMEM_STRIDE);
        }

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;
                wmma::mma_sync(C_frag[i][j_s], A_frag[reg_load_idx][i], B_frag[reg_load_idx][j_s], C_frag[i][j_s]);
            }
        }
    }

    #pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;
            wmma::mma_sync(C_frag[i][j_s], A_frag[reg_store_idx][i], B_frag[reg_store_idx][j_s], C_frag[i][j_s]);
        }
    }

    __syncthreads();

    #pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            half* C_tile_ptr = smem_warp_tile_ptr + i * C_SMEM_STRIDE * WMMA_M + j * WMMA_N;
            wmma::store_matrix_sync(C_tile_ptr, C_frag[i][j], C_SMEM_STRIDE, wmma::mem_row_major);
        }
    }

    __syncthreads();

    #pragma unroll
    for (size_t i = 0; i < WMMA_M; ++i) {
        *((int4*)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *((int4*)(smem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE) + lane_id % 16);
    }
}

}  // namespace

torch::Tensor bruce_wmma_async_stage3_cuda(torch::Tensor A, torch::Tensor B_col_major) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B_col_major.is_cuda(), "B_col_major must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B_col_major.dtype() == torch::kFloat16, "B_col_major must be float16");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B_col_major.dim() == 2, "B_col_major must be 2D");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B_col_major.is_contiguous(), "B_col_major must be contiguous");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B_col_major.size(0);

    TORCH_CHECK(B_col_major.size(1) == K, "B_col_major must have shape [N, K]");
    TORCH_CHECK(M % BLOCK_ROWS == 0, "M must be a multiple of 256");
    TORCH_CHECK(N % BLOCK_COLS == 0, "N must be a multiple of 128");
    TORCH_CHECK(K % (CHUNK_K * WMMA_K) == 0, "K must be a multiple of 32");

    auto C = torch::empty({M, N}, A.options());

    const size_t smem_max_size = std::max(
        static_cast<size_t>((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half) * K_STAGE),
        static_cast<size_t>(BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half))
    );

    cudaError_t attr_err = cudaFuncSetAttribute(
        bruce_wmma_async_stage3_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_max_size)
    );
    TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, div_ceil(static_cast<size_t>(M), static_cast<size_t>(BLOCK_ROWS)),
              div_ceil(static_cast<size_t>(N), static_cast<size_t>(BLOCK_COLS * BLOCK_STRIDE)));

    bruce_wmma_async_stage3_kernel<<<grid, block, smem_max_size>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B_col_major.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        static_cast<size_t>(M),
        static_cast<size_t>(N),
        static_cast<size_t>(K)
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Bruce wmma_async_stage3 launch failed: ", cudaGetErrorString(err));
    return C;
}
