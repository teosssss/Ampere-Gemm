#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

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

#define CHUNK_LINE_BYTES 64
#define CHUNK_COPY_LINES_PER_WARP 8
#define CHUNK_COPY_LINE_LANES 4

#define SMEM_PADDING 8

#define AB_SMEM_STRIDE 40
#define C_SMEM_STRIDE 136
#define C_SMEM_OFFSET 64

#define BLOCK_STRIDE 16
#define K_STAGE 3

using namespace nvcuda;

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "error: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fail(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

struct Options {
    int M = 16384;
    int N = 16384;
    int K = 16384;
    int iters = 1;
    int warmup = 0;
};

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind("--M=", 0) == 0) {
            options.M = std::stoi(arg.substr(4));
        } else if (arg.rfind("--N=", 0) == 0) {
            options.N = std::stoi(arg.substr(4));
        } else if (arg.rfind("--K=", 0) == 0) {
            options.K = std::stoi(arg.substr(4));
        } else if (arg.rfind("--iters=", 0) == 0) {
            options.iters = std::stoi(arg.substr(8));
        } else if (arg.rfind("--warmup=", 0) == 0) {
            options.warmup = std::stoi(arg.substr(9));
        } else {
            fail("unknown argument: " + arg);
        }
    }
    return options;
}

std::vector<half> make_random_half_data(int64_t count) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<half> out(static_cast<size_t>(count));
    for (half& value : out) {
        value = __float2half(dist(rng));
    }
    return out;
}

// Reference kernel adapted from:
// https://github.com/Bruce-Lee-LY/cuda_hgemm/blob/master/src/wmma/wmma_async_stage3.cu
__global__ void wmmaAsyncStage3Kernel(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
                                      size_t M, size_t N, size_t K) {
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
            wmma::fill_fragment(C_frag[i][j], 0.0);
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

    size_t A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id + lane_id / CHUNK_COPY_LINE_LANES;
    int4* A_lane_ptr =
        (int4*)(A_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);

    #pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        uint32_t A_smem_lane_addr =
            __cvta_generic_to_shared(&smem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);
        A_lane_ptr = (int4*)((half*)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    size_t B_smem_idx =
        smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id + lane_id / CHUNK_COPY_LINE_LANES;
    int4* B_lane_ptr =
        (int4*)(B_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);

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

    A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id + lane_id / CHUNK_COPY_LINE_LANES;
    A_lane_ptr = (int4*)(A_warp_ptr + CHUNK_K * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);

    #pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        uint32_t A_smem_lane_addr =
            __cvta_generic_to_shared(&smem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);
        A_lane_ptr = (int4*)((half*)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    B_smem_idx =
        smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id + lane_id / CHUNK_COPY_LINE_LANES;
    B_lane_ptr = (int4*)(B_warp_ptr + CHUNK_K * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);

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
        size_t idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[reg_store_idx][i], &smem[idx][0], AB_SMEM_STRIDE);
    }
    #pragma unroll
    for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
        size_t idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[reg_store_idx][j], &smem[idx][0], AB_SMEM_STRIDE);
    }

    #pragma unroll
    for (size_t tile_k = CHUNK_K * (K_STAGE - 1); tile_k < K_tiles; tile_k += CHUNK_K) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[reg_store_idx][i], &smem[idx][WMMA_K], AB_SMEM_STRIDE);
        }
        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[reg_store_idx][j], &smem[idx][WMMA_K], AB_SMEM_STRIDE);
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

        A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id + lane_id / CHUNK_COPY_LINE_LANES;
        A_lane_ptr = (int4*)(A_warp_ptr + tile_k * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        #pragma unroll
        for (size_t i = 0; i < A_smem_iters / CHUNK_K; ++i) {
            uint32_t addr =
                __cvta_generic_to_shared(&smem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
            CP_ASYNC_CG(addr, A_lane_ptr, THREAD_COPY_BYTES);
            A_lane_ptr = (int4*)((half*)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        B_smem_idx =
            smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id + lane_id / CHUNK_COPY_LINE_LANES;
        B_lane_ptr = (int4*)(B_warp_ptr + tile_k * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        #pragma unroll
        for (size_t i = 0; i < B_smem_iters / CHUNK_K; ++i) {
            uint32_t addr =
                __cvta_generic_to_shared(&smem[B_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
            CP_ASYNC_CG(addr, B_lane_ptr, THREAD_COPY_BYTES);
            B_lane_ptr = (int4*)((half*)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        smem_load_idx = (smem_load_idx + 1) % K_STAGE;
        smem_load_off = smem_load_idx * smem_stage_off;

        #pragma unroll
        for (size_t i = (CHUNK_K - 1) * A_smem_iters / CHUNK_K; i < A_smem_iters; ++i) {
            uint32_t addr =
                __cvta_generic_to_shared(&smem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
            CP_ASYNC_CG(addr, A_lane_ptr, THREAD_COPY_BYTES);
            A_lane_ptr = (int4*)((half*)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }
        #pragma unroll
        for (size_t i = (CHUNK_K - 1) * B_smem_iters / CHUNK_K; i < B_smem_iters; ++i) {
            uint32_t addr =
                __cvta_generic_to_shared(&smem[B_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;
            CP_ASYNC_CG(addr, B_lane_ptr, THREAD_COPY_BYTES);
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
            size_t idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[reg_store_idx][i], &smem[idx][0], AB_SMEM_STRIDE);
        }
        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[reg_store_idx][j], &smem[idx][0], AB_SMEM_STRIDE);
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
            size_t idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[reg_store_idx][i], &smem[idx][((k_step + 1) % CHUNK_K) * WMMA_K], AB_SMEM_STRIDE);
        }
        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[reg_store_idx][j], &smem[idx][((k_step + 1) % CHUNK_K) * WMMA_K], AB_SMEM_STRIDE);
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
            size_t idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[reg_store_idx][i], &smem[idx][k_step * WMMA_K], AB_SMEM_STRIDE);
        }
        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[reg_store_idx][j], &smem[idx][k_step * WMMA_K], AB_SMEM_STRIDE);
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

void launch_reference_wmma_async_stage3(half* A, half* B, half* C, size_t M, size_t N, size_t K) {
    const size_t smem_max_size = std::max(
        (BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half) * K_STAGE,
        BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half));
    check_cuda(
        cudaFuncSetAttribute(wmmaAsyncStage3Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size),
        "cudaFuncSetAttribute(reference)");

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));
    wmmaAsyncStage3Kernel<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_args(argc, argv);
    const int64_t a_elems = static_cast<int64_t>(options.M) * options.K;
    const int64_t b_elems = static_cast<int64_t>(options.K) * options.N;
    const int64_t c_elems = static_cast<int64_t>(options.M) * options.N;

    std::vector<half> a_host = make_random_half_data(a_elems);
    std::vector<half> b_host = make_random_half_data(b_elems);

    half* d_A = nullptr;
    half* d_B = nullptr;
    half* d_C = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_A), static_cast<size_t>(a_elems) * sizeof(half)), "cudaMalloc(A)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_B), static_cast<size_t>(b_elems) * sizeof(half)), "cudaMalloc(B)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_C), static_cast<size_t>(c_elems) * sizeof(half)), "cudaMalloc(C)");
    check_cuda(cudaMemcpy(d_A, a_host.data(), static_cast<size_t>(a_elems) * sizeof(half), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    check_cuda(cudaMemcpy(d_B, b_host.data(), static_cast<size_t>(b_elems) * sizeof(half), cudaMemcpyHostToDevice), "cudaMemcpy(B)");

    for (int i = 0; i < options.warmup; ++i) {
        launch_reference_wmma_async_stage3(d_A, d_B, d_C, options.M, options.N, options.K);
    }
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < options.iters; ++i) {
        launch_reference_wmma_async_stage3(d_A, d_B, d_C, options.M, options.N, options.K);
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    std::vector<half> c_host(static_cast<size_t>(c_elems));
    check_cuda(cudaMemcpy(c_host.data(), d_C, static_cast<size_t>(c_elems) * sizeof(half), cudaMemcpyDeviceToHost), "cudaMemcpy(C)");
    uint64_t checksum = 0;
    for (half v : c_host) {
        checksum += static_cast<uint16_t>(reinterpret_cast<uint16_t&>(v));
    }

    std::cout << std::fixed << std::setprecision(3)
              << "mode=reference_wmma_async_stage3"
              << " M=" << options.M
              << " N=" << options.N
              << " K=" << options.K
              << " avg_ms=" << (elapsed_ms / options.iters)
              << " checksum=" << checksum
              << '\n';

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);
    return 0;
}
