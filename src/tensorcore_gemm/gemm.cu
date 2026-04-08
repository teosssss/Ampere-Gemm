/*
"""
AutoKernel -- CUDA C++ GEMM kernel.

Current kernel: Tensor Core GEMM via wmma API with:
triple-buffered shared memory,
grid swizzling,
padding to avoid bank conflicts.
Target metric: throughput_tflops (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - nvcuda::wmma::mma_sync for tensor core acceleration (16x16x16 tiles)
  - triple-buffered shared memory shared memory to overlap loads with compute
  - Bank-conflict-free shared memory layout (padding)
  - Grid swizzling 
  - Vectorized float4 global memory loads for maximum bandwidth
  - __launch_bounds__ for register pressure control

The agent can change anything in this file:
  - Tile sizes, warp counts, buffer stages
  - Memory layout, swizzling, prefetch strategy
  - Accumulation precision, epilogue fusion
  - Any CUDA intrinsic or PTX instruction


*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda/barrier>
#include <cuda_pipeline_primitives.h>

using namespace nvcuda;

#ifndef AUTOKERNEL_ENABLE_PERSISTENT_GEMM
#define AUTOKERNEL_ENABLE_PERSISTENT_GEMM 0
#endif


// ------------- CONFIGURATION -------------
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64; // One block computes a 64 x 64 tile of the output matrix
constexpr int BLOCK_K = 16; // Accumulation step
constexpr int WARP_SIZE = 32;
constexpr int THREAD_COUNT = 128;
constexpr int WMMA = 16;
constexpr int SPLIT_K = 4;
#ifndef AUTOKERNEL_WS_PRODUCER_WARPS
#define AUTOKERNEL_WS_PRODUCER_WARPS 3
#endif
#ifndef AUTOKERNEL_WS_CONSUMER_WARPS
#define AUTOKERNEL_WS_CONSUMER_WARPS 4
#endif
constexpr int WARP_SPECIALIZED_PRODUCER_WARPS = AUTOKERNEL_WS_PRODUCER_WARPS;
constexpr int WARP_SPECIALIZED_CONSUMER_WARPS = AUTOKERNEL_WS_CONSUMER_WARPS;
constexpr int WARP_SPECIALIZED_THREAD_COUNT =
    (WARP_SPECIALIZED_PRODUCER_WARPS + WARP_SPECIALIZED_CONSUMER_WARPS) * WARP_SIZE;



#define LDMATRIX_X1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X2_TRANS(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define HMMA16816_F32(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3)                \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                                          \
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"                     \
                 : "=f"(RD0), "=f"(RD1), "=f"(RD2), "=f"(RD3)                                                  \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "f"(RC0), "f"(RC1), "f"(RC2),\
                   "f"(RC3))




#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
#define AUTOKERNEL_CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#else
#define AUTOKERNEL_CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#endif

__device__ __forceinline__ void cp_async_cg_16B(void* dst, const void* src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    AUTOKERNEL_CP_ASYNC_CG(smem_addr, src, 16);
}

__global__ void gemm_buffer_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {

    // ------------- INDEX CALCULATIONS -------------
    // Linear view for data loading: which worker out of 128 threads am I?
    int tid = threadIdx.x;

    // Global position: what tile of the output matrix am I calculating?
    int block_row_start = blockIdx.y * BLOCK_M;
    int block_col_start = blockIdx.x * BLOCK_N;

    // What warp am I in the 2x2 grid?
    int warp_id = tid / WARP_SIZE;
    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 32;


    // A tile: 64 x 16. Each row has 2 8-element vectors. 
    int row_A = tid / 2;       // 0 to 63
    int col_A = (tid % 2) * 8; // 0 or 8
    // B tile: 16 x 64. Each row has 8 8-element vectors. 
    int row_B = tid / 8;       // 0 to 15
    int col_B = (tid % 8) * 8; // 0, 8, 16, 24, 32, 40, 48, or 56
    // ----------------------------------------------


    // ------------- MEMORY INITIALIZATION ----------
    // Double Buffer: Shared Memory
    __shared__ half sA[2][BLOCK_M * BLOCK_K]; // 64 rows, 16 cols (K)
    __shared__ half sB[2][BLOCK_K * BLOCK_N]; // 16 rows (K), 64 cols

    // Declare fragments and initialize accumulator. 
    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> accum_frag[2][2];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(accum_frag[i][j], 0.0f);
        }
    }

    // Pipeline setup
    int stage = 0; // Alternates between 0 and 1
    // ----------------------------------------------


    // ------------- PROLOGUE -------------
    // Load the first tile. 
    {
        const half* src_A = A + (block_row_start + row_A) * K + (0 + col_A);
        half* dst_A = &sA[stage][row_A * BLOCK_K + col_A];

        const half* src_B = B + (0 + row_B) * N + (block_col_start + col_B);
        half* dst_B = &sB[stage][row_B * BLOCK_N + col_B];

        // Async copy. int4 is the size of 8 half elements
        __pipeline_memcpy_async(dst_A, src_A, sizeof(int4));
        __pipeline_memcpy_async(dst_B, src_B, sizeof(int4)); 

        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
    }
    // ------------------------------------

    // ------------- MAIN LOOP -------------
    #pragma unroll
    for (int k = 0; k < K; k += BLOCK_K) {

        int k_next = k + BLOCK_K;

        // 1. LOAD the next tile asynchronously
        if (k_next < K) {
            // Turns 1 into 0 or 0 into 1
            int next_stage = 1 - stage;

            const half* src_A = A + (block_row_start + row_A) * K + (k_next + col_A);
            half* dst_A = &sA[next_stage][row_A * BLOCK_K + col_A];
            
            const half* src_B = B + (k_next + row_B) * N + (block_col_start + col_B);
            half* dst_B = &sB[next_stage][row_B * BLOCK_N + col_B];

            __pipeline_memcpy_async(dst_A, src_A, sizeof(int4)); 
            __pipeline_memcpy_async(dst_B, src_B, sizeof(int4)); 

            __pipeline_commit();
        }

        // 2. MATH: process the current tile. Recall we have a 2 x 2 grid of 16 x 16 subtiles for each warp.
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                // Calculate pointer into shared memory for this sub-tile
                int smem_row = warp_row + (i * 16);
                int smem_col = warp_col + (j * 16);

                // Load fragments from shared memory
                half* tile_ptr_A = &sA[stage][smem_row * BLOCK_K];
                half* tile_ptr_B = &sB[stage][smem_col];

                wmma::load_matrix_sync(a_frag, tile_ptr_A, BLOCK_K);
                wmma::load_matrix_sync(b_frag, tile_ptr_B, BLOCK_N);

                // Multiply matrices and accumulate
                wmma::mma_sync(accum_frag[i][j], a_frag, b_frag, accum_frag[i][j]);
            }
        }

        // 3. WAIT for next tile
        if (k + BLOCK_K < K) {
            __pipeline_wait_prior(0);
            __syncthreads();
            stage = 1 - stage;
        }
    }
    // ------------------------------------

    __syncthreads(); // Since the syncthreads above won't execute on the last iteration
   
    // ------- EPILOGUE: Store C ----------
    // Size: 64 * 64 floats = 64 * 64 * 4 bytes = 16 KB. Fits easily in modern L1/Shared
    __shared__ float sC[BLOCK_M * BLOCK_N];

    // Warps dump their fragments to shared memory, one 16x16 subtile at a time.
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            float* subtile_ptr = sC + (warp_row + i * 16) * BLOCK_N + (warp_col + j * 16);
            wmma::store_matrix_sync(subtile_ptr, accum_frag[i][j], BLOCK_N, wmma::mem_row_major);
        }
    }

    // Wait for all threads to write to sC
    __syncthreads();

    #pragma unroll
    for (int i = tid * 8; i < BLOCK_M * BLOCK_N; i += THREAD_COUNT * 8) {
        int row = i / BLOCK_N;
        int col = i % BLOCK_N;

        int global_row = block_row_start + row;
        int global_col = block_col_start + col;

        half buffer[8];

        // Boundary check
        if (global_row < M && (global_col + 7) < N) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float val = alpha * sC[i + j];

                if (beta != 0.0f) {
                    float old_c = __half2float(C[global_row * N + global_col + j]);
                    val += beta * old_c;
                } 

                buffer[j] = __float2half(val);
            }
            
            // Vectorized store
            *(int4*)&C[global_row * N + global_col] = *(int4*)buffer;

        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (global_row < M && (global_col + j) < N) {
                    int out_idx = global_row * N + global_col + j;

                    float val = alpha * sC[i + j];

                    if (beta != 0.0f) {
                        float old_c = __half2float(C[out_idx]);
                        val += beta * old_c;
                    } 

                    C[out_idx] = __float2half(val);
                }
            }
        }
    }
}

__device__ __forceinline__ void rotate_stage_triplet_3(int& current_stage, int& next_stage, int& prefetch_stage) {
    int old_current = current_stage;
    current_stage = next_stage;
    next_stage = prefetch_stage;
    prefetch_stage = old_current;
}

__device__ __forceinline__ void gemm_rotate3_consume_stage_reg_pingpong(
    const half* sA_stage,
    const half* sB_stage,
    int warp_row,
    int warp_col,
    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> (&a_frag)[2][2],
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> (&b_frag)[2][2],
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> (&accum_frag)[2][2]
) {
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int K_TILE = 32;

    int reg_curr = 0;
    int reg_next = 1;

    // Preload the first WMMA k-slice before entering the software-pipelined loop.
    {
        constexpr int k_step = 0;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int smem_row = warp_row + (i * 16);
            const half* tile_ptr_A = sA_stage + smem_row * (K_TILE + PAD_A) + k_step;
            wmma::load_matrix_sync(a_frag[reg_curr][i], tile_ptr_A, K_TILE + PAD_A);
        }

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int smem_col = warp_col + (j * 16);
            const half* tile_ptr_B = sB_stage + k_step * (BLOCK_N + PAD_B) + smem_col;
            wmma::load_matrix_sync(b_frag[reg_curr][j], tile_ptr_B, BLOCK_N + PAD_B);
        }
    }

    #pragma unroll
    for (int k_step = WMMA; k_step < K_TILE; k_step += WMMA) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int smem_row = warp_row + (i * 16);
            const half* tile_ptr_A = sA_stage + smem_row * (K_TILE + PAD_A) + k_step;
            wmma::load_matrix_sync(a_frag[reg_next][i], tile_ptr_A, K_TILE + PAD_A);
        }

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int smem_col = warp_col + (j * 16);
            const half* tile_ptr_B = sB_stage + k_step * (BLOCK_N + PAD_B) + smem_col;
            wmma::load_matrix_sync(b_frag[reg_next][j], tile_ptr_B, BLOCK_N + PAD_B);
        }

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(
                    accum_frag[i][j],
                    a_frag[reg_curr][i],
                    b_frag[reg_curr][j],
                    accum_frag[i][j]
                );
            }
        }

        reg_curr ^= 1;
        reg_next ^= 1;
    }

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::mma_sync(
                accum_frag[i][j],
                a_frag[reg_curr][i],
                b_frag[reg_curr][j],
                accum_frag[i][j]
            );
        }
    }
}

__global__ void gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int K_TILE = 32;
    constexpr int total_pass = K_TILE/WMMA;

    const int swizzle_factor = 4;

    int idx_linear = blockIdx.y * gridDim.x + blockIdx.x;
    int grid_m_blocks = gridDim.y;
    int grid_n_blocks = gridDim.x;

    int swizzle_width = min(swizzle_factor, grid_n_blocks);
    int blocks_per_panel = swizzle_width * grid_m_blocks;
    int panel_number = idx_linear / blocks_per_panel;
    int idx_in_panel = idx_linear % blocks_per_panel;
    int block_row = idx_in_panel % grid_m_blocks;
    int block_col = panel_number * swizzle_width + (idx_in_panel / grid_m_blocks);

    if (block_row >= grid_m_blocks || block_col >= grid_n_blocks) return;

    int block_row_start = block_row * BLOCK_M;
    int block_col_start = block_col * BLOCK_N;

    int tid = threadIdx.x;

    int warp_id = tid / WARP_SIZE;
    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 32;

    int row_A = tid / 4;
    int col_A = (tid % 4) * 8;
    int row_B = tid / 8;
    int col_B = (tid % 8) * 8;

    __shared__ half sA[3][BLOCK_M * (K_TILE + PAD_A)];
    __shared__ half sB[3][K_TILE * (PAD_B + BLOCK_N)];

    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> a_frag[2][2];
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> b_frag[2][2];
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> accum_frag[2][2];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(accum_frag[i][j], 0.0f);
        }
    }

    int current_stage = 0;
    int next_stage = 1;
    int prefetch_stage = 2;

    {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int k_base = i * K_TILE;

            #pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                int k_pass = pass * WMMA;
                int row_pass = (k_pass / WMMA) * 32;

                const half* src_A = A + (block_row_start + row_pass + row_A) * K + (k_base + col_A);
                half* dst_A = &sA[i][(row_pass + row_A) * (K_TILE + PAD_A) + col_A];

                const half* src_B = B + (k_base + k_pass + row_B) * N + (block_col_start + col_B);
                half* dst_B = &sB[i][(k_pass + row_B) * (BLOCK_N + PAD_B) + col_B];
                __pipeline_memcpy_async(dst_A, src_A, sizeof(int4));
                __pipeline_memcpy_async(dst_B, src_B, sizeof(int4));
            }

            __pipeline_commit();
        }

        __pipeline_wait_prior(1);
        __syncthreads();
    }

    #pragma unroll
    for (int k = 0; k + 2 * K_TILE < K; k += K_TILE) {
        int k_next = k + 2 * K_TILE;
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            int k_pass = pass * WMMA;
            int row_pass = (k_pass / WMMA) * 32;
            const half* src_A = A + (block_row_start + row_pass + row_A) * K + (k_next + col_A);
            half* dst_A = &sA[prefetch_stage][(row_pass + row_A) * (K_TILE + PAD_A) + col_A];
            const half* src_B = B + (k_next + k_pass + row_B) * N + (block_col_start + col_B);
            half* dst_B = &sB[prefetch_stage][(k_pass + row_B) * (BLOCK_N + PAD_B) + col_B];
            __pipeline_memcpy_async(dst_A, src_A, sizeof(int4));
            __pipeline_memcpy_async(dst_B, src_B, sizeof(int4));
        }
        __pipeline_commit();

        gemm_rotate3_consume_stage_reg_pingpong(
            sA[current_stage],
            sB[current_stage],
            warp_row,
            warp_col,
            a_frag,
            b_frag,
            accum_frag
        );

        __pipeline_wait_prior(1);
        __syncthreads();
        rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);
    }

    gemm_rotate3_consume_stage_reg_pingpong(
        sA[current_stage],
        sB[current_stage],
        warp_row,
        warp_col,
        a_frag,
        b_frag,
        accum_frag
    );

    __pipeline_wait_prior(0);
    __syncthreads();
    rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

    gemm_rotate3_consume_stage_reg_pingpong(
        sA[current_stage],
        sB[current_stage],
        warp_row,
        warp_col,
        a_frag,
        b_frag,
        accum_frag
    );

    __shared__ float sC[BLOCK_M * BLOCK_N];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            float* subtile_ptr = sC + (warp_row + i * 16) * BLOCK_N + (warp_col + j * 16);
            wmma::store_matrix_sync(subtile_ptr, accum_frag[i][j], BLOCK_N, wmma::mem_row_major);
        }
    }

    __syncthreads();

    #pragma unroll
    for (int i = tid * 8; i < BLOCK_M * BLOCK_N; i += THREAD_COUNT * 8) {
        int row = i / BLOCK_N;
        int col = i % BLOCK_N;

        int global_row = block_row_start + row;
        int global_col = block_col_start + col;

        half buffer[8];

        if (global_row < M && (global_col + 7) < N) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float val = alpha * sC[i + j];

                if (beta != 0.0f) {
                    float old_c = __half2float(C[global_row * N + global_col + j]);
                    val += beta * old_c;
                }

                buffer[j] = __float2half(val);
            }

            *(int4*)&C[global_row * N + global_col] = *(int4*)buffer;

        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (global_row < M && (global_col + j) < N) {
                    int out_idx = global_row * N + global_col + j;

                    float val = alpha * sC[i + j];

                    if (beta != 0.0f) {
                        float old_c = __half2float(C[out_idx]);
                        val += beta * old_c;
                    }

                    C[out_idx] = __float2half(val);
                }
            }
        }
    }
}


__device__ __forceinline__ void gemm_rotate3_consume_stage_reg_pingpong_128(
    const half* sA_stage,
    const half* sB_stage,
    int warp_row,
    int warp_col,
    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> (&a_frag)[2][2],
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> (&b_frag)[2][4],
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> (&accum_frag)[2][4]
) {
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int K_TILE = 32;
    constexpr int TILE_N = 128;
    // This helper assumes the 128x128 CTA uses an 8-warp 4x2 warp grid.
    // Each warp therefore owns a 32x64 output region:
    //   128 rows / 4 warp rows = 32 rows per warp
    //   128 cols / 2 warp cols = 64 cols per warp
    // Since each WMMA op produces a 16x16 tile, one warp needs:
    //   2 tiles in M  -> i < 2
    //   4 tiles in N  -> j < 4

    int reg_curr = 0;
    int reg_next = 1;

    {
        constexpr int k_step = 0;

        // Load the four 16-row A fragments needed for a 64-row warp tile.
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int smem_row = warp_row + i * 16;
            const half* tile_ptr_A = sA_stage + smem_row * (K_TILE + PAD_A) + k_step;
            wmma::load_matrix_sync(a_frag[reg_curr][i], tile_ptr_A, K_TILE + PAD_A);
        }

        // Load the four 16-col B fragments needed for a 64-col warp tile.
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int smem_col = warp_col + j * 16;
            const half* tile_ptr_B = sB_stage + k_step * (TILE_N + PAD_B) + smem_col;
            wmma::load_matrix_sync(b_frag[reg_curr][j], tile_ptr_B, TILE_N + PAD_B);
        }
    }

    #pragma unroll
    for (int k_step = WMMA; k_step < K_TILE; k_step += WMMA) {
        // Same 4-by-4 fragment grid for the next K slice in the register ping-pong.
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int smem_row = warp_row + i * 16;
            const half* tile_ptr_A = sA_stage + smem_row * (K_TILE + PAD_A) + k_step;
            wmma::load_matrix_sync(a_frag[reg_next][i], tile_ptr_A, K_TILE + PAD_A);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int smem_col = warp_col + j * 16;
            const half* tile_ptr_B = sB_stage + k_step * (TILE_N + PAD_B) + smem_col;
            wmma::load_matrix_sync(b_frag[reg_next][j], tile_ptr_B, TILE_N + PAD_B);
        }

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                // Accumulate a 4x4 grid of 16x16 WMMA tiles = one 64x64 warp tile.
                wmma::mma_sync(
                    accum_frag[i][j],
                    a_frag[reg_curr][i],
                    b_frag[reg_curr][j],
                    accum_frag[i][j]
                );
            }
        }

        reg_curr ^= 1;
        reg_next ^= 1;
    }

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::mma_sync(
                accum_frag[i][j],
                a_frag[reg_curr][i],
                b_frag[reg_curr][j],
                accum_frag[i][j]
            );
        }
    }
}





__global__ void gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_128(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int K_TILE = 32;
    constexpr int total_pass = K_TILE/WMMA;
    // Larger CTA experiment:
    //   tile shape   = 128 x 128 x 32
    //   threads/CTA  = 256 (8 warps)
    //   warp layout  = 4 x 2
    // Keep these constants local so the 64x64 baseline kernels are untouched.
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int CTA_THREADS = 256;
    constexpr int WARP_LAYOUT_M = 4;
    constexpr int WARP_LAYOUT_N = 2;
    constexpr int A_STRIDE = K_TILE + PAD_A;
    constexpr int B_STRIDE = TILE_N + PAD_B;
    constexpr int A_STAGE_ELEMS = TILE_M * A_STRIDE;
    constexpr int B_STAGE_ELEMS = K_TILE * B_STRIDE;
    constexpr int STAGE_COUNT = 3;

    const int swizzle_factor = 4;

    int idx_linear = blockIdx.y * gridDim.x + blockIdx.x;
    int grid_m_blocks = gridDim.y;
    int grid_n_blocks = gridDim.x;

    int swizzle_width = min(swizzle_factor, grid_n_blocks);
    int blocks_per_panel = swizzle_width * grid_m_blocks;
    int panel_number = idx_linear / blocks_per_panel;
    int idx_in_panel = idx_linear % blocks_per_panel;
    int block_row = idx_in_panel % grid_m_blocks;
    int block_col = panel_number * swizzle_width + (idx_in_panel / grid_m_blocks);

    if (block_row >= grid_m_blocks || block_col >= grid_n_blocks) return;

    // Grid tiles the output in 128x128 CTA-sized chunks. The launcher in gemm.py uses the
    // same tile size when AUTOKERNEL_USE_REG_PINGPONG_128_GEMM=1 is selected.
    int block_row_start = block_row * TILE_M;
    int block_col_start = block_col * TILE_N;

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;

    // Eight warps are arranged as 4 warp-rows by 2 warp-cols.
    // Each warp therefore computes a 32x64 output tile.
    int warp_id = tid / WARP_SIZE;
    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 64;

    // A stage is 128x32 = 4096 halfs = 512 int4 loads. With 256 threads and 2 passes,
    // each pass covers 64 rows x 32 cols.
    int row_A = tid / 4; // 0..63
    int col_A = (tid % 4) * 8; // 0, 8, 16, 24
    // B stage is 32x128 = 4096 halfs = 512 int4 loads. With 256 threads and 2 passes,
    // each pass covers 16 K rows x 128 cols.
    int row_B = tid / 16;
    int col_B = (tid % 16) * 8;

    // Triple-buffered shared staging for the current, next, and prefetch K-slices.
    // Similar to the reference kernel, one dynamic shared allocation backs the staging area,
    // then we reinterpret it as row-strided array views for A and B.
    extern __shared__ half smem[];
    auto sA = reinterpret_cast<half (*)[A_STAGE_ELEMS]>(smem);
    auto sB = reinterpret_cast<half (*)[B_STAGE_ELEMS]>(smem + STAGE_COUNT * A_STAGE_ELEMS);

    // One warp computes a 32x64 output tile:
    //   A fragments     = 2 tiles in M
    //   B fragments     = 4 tiles in N
    //   accum fragments = 2 x 4 output tiles
    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> a_frag[2][2];
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> b_frag[2][4];
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> accum_frag[2][4];
    __shared__ float warp_store[8][WMMA * WMMA];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(accum_frag[i][j], 0.0f);
        }
    }

    int current_stage = 0;
    int next_stage = 1;
    int prefetch_stage = 2;

    {
        // Prologue: stage the first two K_TILE slices so the steady-state loop can overlap
        // prefetch with compute.
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int k_base = i * K_TILE;

            #pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                int k_pass = pass * WMMA;   // 0, 16
                int row_pass = (k_pass / WMMA) * 64; // 0, 64

                const half* src_A = A + (block_row_start + row_pass + row_A) * K + (k_base + col_A);
                half* dst_A = &sA[i][(row_pass + row_A) * A_STRIDE + col_A];

                const half* src_B = B + (k_base + k_pass + row_B) * N + (block_col_start + col_B);
                half* dst_B = &sB[i][(k_pass + row_B) * B_STRIDE + col_B];
                __pipeline_memcpy_async(dst_A, src_A, sizeof(int4));
                __pipeline_memcpy_async(dst_B, src_B, sizeof(int4));
            }

            __pipeline_commit();
        }

        __pipeline_wait_prior(1);
        __syncthreads();
    }

    #pragma unroll
    for (int k = 0; k + 2 * K_TILE < K; k += K_TILE) {
        int k_next = k + 2 * K_TILE;
        // Steady state: prefetch the next stage while consuming the current one.
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            int k_pass = pass * WMMA;
            int row_pass = (k_pass / WMMA) * 64;
            const half* src_A = A + (block_row_start + row_pass + row_A) * K + (k_next + col_A);
            half* dst_A = &sA[prefetch_stage][(row_pass + row_A) * A_STRIDE + col_A];
            const half* src_B = B + (k_next + k_pass + row_B) * N + (block_col_start + col_B);
            half* dst_B = &sB[prefetch_stage][(k_pass + row_B) * B_STRIDE + col_B];
            __pipeline_memcpy_async(dst_A, src_A, sizeof(int4));
            __pipeline_memcpy_async(dst_B, src_B, sizeof(int4));
        }
        __pipeline_commit();

        gemm_rotate3_consume_stage_reg_pingpong_128(
            sA[current_stage],
            sB[current_stage],
            warp_row,
            warp_col,
            a_frag,
            b_frag,
            accum_frag
        );

        __pipeline_wait_prior(1);
        __syncthreads();
        rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);
    }

    // Drain the last two prefetched stages after the main loop exits.
    gemm_rotate3_consume_stage_reg_pingpong_128(
        sA[current_stage],
        sB[current_stage],
        warp_row,
        warp_col,
        a_frag,
        b_frag,
        accum_frag
    );

    __pipeline_wait_prior(0);
    __syncthreads();
    rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

    gemm_rotate3_consume_stage_reg_pingpong_128(
        sA[current_stage],
        sB[current_stage],
        warp_row,
        warp_col,
        a_frag,
        b_frag,
        accum_frag
    );

    // Warp-distributed epilogue: each warp stores one 16x16 fragment to shared scratch,
    // then all 32 lanes cooperatively write it to global memory.
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int row_C = block_row_start + warp_row + i * 16;
            int col_C = block_col_start + warp_col + j * 16;

            wmma::store_matrix_sync(warp_store[warp_id], accum_frag[i][j], WMMA, wmma::mem_row_major);
            __syncwarp();

            #pragma unroll
            for (int idx = lane_id; idx < WMMA * WMMA; idx += WARP_SIZE) {
                int r = idx / WMMA;
                int c = idx % WMMA;
                int out_idx = (row_C + r) * N + (col_C + c);
                float val = alpha * warp_store[warp_id][idx];
                if (beta != 0.0f) {
                    val += beta * __half2float(C[out_idx]);
                }
                C[out_idx] = __float2half(val);
            }
            __syncwarp();
        }
    }
}





__device__ __forceinline__ void gemm_rotate3_load_stage_fragments_reg_pingpong_256(
    const half* sA_stage,
    const half* sB_stage,
    int warp_row,
    int warp_col,
    int reg_slot,
    int k_step,
    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> (&a_frag)[2][4],
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> (&b_frag)[2][4]
) {
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int TILE_N = 128;

    // Load one 16-wide K-slice of the 64x64 warp tile into the selected register slot.
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int smem_row = warp_row + i * 16;
        const half* tile_ptr_A = sA_stage + smem_row * (WMMA * 2 + PAD_A) + k_step;
        wmma::load_matrix_sync(a_frag[reg_slot][i], tile_ptr_A, WMMA * 2 + PAD_A);
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int smem_col = warp_col + j * 16;
        const half* tile_ptr_B = sB_stage + k_step * (TILE_N + PAD_B) + smem_col;
        wmma::load_matrix_sync(b_frag[reg_slot][j], tile_ptr_B, TILE_N + PAD_B);
    }
}

__device__ __forceinline__ void gemm_rotate3_mma_reg_pingpong_256(
    int reg_slot,
    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> (&a_frag)[2][4],
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> (&b_frag)[2][4],
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> (&accum_frag)[4][4]
) {
    // Accumulate one preloaded 16-wide K-slice across the full 64x64 warp tile.
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int j_s = (i % 2) ? (4 - j - 1) : j;
            wmma::mma_sync(
                accum_frag[i][j_s],
                a_frag[reg_slot][i],
                b_frag[reg_slot][j_s],
                accum_frag[i][j_s]
            );
        }
    }
}

__device__ __forceinline__ void gemm_rotate3_prefetch_stage_reg_pingpong_256(
    const half* A,
    const half* B,
    half* sA_stage,
    half* sB_stage,
    int block_row_start,
    int block_col_start,
    int row_A,
    int col_A,
    int row_B,
    int col_B,
    int K,
    int N,
    int k_next
) {

        /*
        #pragma unroll
        for (int pass = 0; pass < 4; pass++) {
            int row_pass = pass * 64; // 0, 64, 128, 192
            const half* src_A = A + (block_row_start + row_pass + row_A) * K + (k_next + col_A);
            half* dst_A = &sA[prefetch_stage][(row_pass + row_A) * A_STRIDE + col_A];
            cp_async_cg_16B(dst_A, src_A);
        }

        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            int k_pass = pass * WMMA;   // 0, 16
            const half* src_B = B + (k_next + k_pass + row_B) * N + (block_col_start + col_B);
            half* dst_B = &sB[prefetch_stage][(k_pass + row_B) * B_STRIDE + col_B];
            cp_async_cg_16B(dst_B, src_B);
        }
        */
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int K_TILE = 32;
    constexpr int A_PASS_ROWS = 64;
    constexpr int TILE_N = 128;
    constexpr int A_STRIDE = K_TILE + PAD_A;
    constexpr int B_STRIDE = TILE_N + PAD_B;

    const half* src_A_base = A + (block_row_start + row_A) * K + (k_next + col_A);
    half* dst_A_base = sA_stage + row_A * A_STRIDE + col_A;

    const half* src_B_base = B + (k_next + row_B) * N + (block_col_start + col_B);
    half* dst_B_base = sB_stage + row_B * B_STRIDE + col_B;

    cp_async_cg_16B(dst_A_base + 0 * A_PASS_ROWS * A_STRIDE, src_A_base + 0 * A_PASS_ROWS * K);
    cp_async_cg_16B(dst_B_base + 0 * WMMA * B_STRIDE, src_B_base + 0 * WMMA * N);

    cp_async_cg_16B(dst_A_base + 1 * A_PASS_ROWS * A_STRIDE, src_A_base + 1 * A_PASS_ROWS * K);
    cp_async_cg_16B(dst_B_base + 1 * WMMA * B_STRIDE, src_B_base + 1 * WMMA * N);

    cp_async_cg_16B(dst_A_base + 2 * A_PASS_ROWS * A_STRIDE, src_A_base + 2 * A_PASS_ROWS * K);
    cp_async_cg_16B(dst_A_base + 3 * A_PASS_ROWS * A_STRIDE, src_A_base + 3 * A_PASS_ROWS * K);
}

__device__ __forceinline__ void gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb(
    const half* sA_stage,
    const half* sB_stage,
    int warp_row,
    int warp_col,
    int reg_slot,
    int k_step,
    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> (&a_frag)[2][4],
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::col_major> (&b_frag)[2][4]
) {
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int K_TILE = 32;
    constexpr int A_STRIDE = K_TILE + PAD_A;
    constexpr int B_STRIDE = K_TILE + PAD_B;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int smem_row = warp_row + i * 16;
        const half* tile_ptr_A = sA_stage + smem_row * A_STRIDE + k_step;
        wmma::load_matrix_sync(a_frag[reg_slot][i], tile_ptr_A, A_STRIDE);
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int smem_col = warp_col + j * 16;
        const half* tile_ptr_B = sB_stage + smem_col * B_STRIDE + k_step;
        wmma::load_matrix_sync(b_frag[reg_slot][j], tile_ptr_B, B_STRIDE);
    }
}

__device__ __forceinline__ void gemm_rotate3_mma_reg_pingpong_256_colb(
    int reg_slot,
    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> (&a_frag)[2][4],
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::col_major> (&b_frag)[2][4],
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> (&accum_frag)[4][4]
) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int j_s = (i % 2) ? (4 - j - 1) : j;
            wmma::mma_sync(
                accum_frag[i][j_s],
                a_frag[reg_slot][i],
                b_frag[reg_slot][j_s],
                accum_frag[i][j_s]
            );
        }
    }
}

__device__ __forceinline__ void gemm_rotate3_prefetch_stage_reg_pingpong_256_colb(
    const half* A,
    const half* B_col_major,
    half* sA_stage,
    half* sB_stage,
    int block_row_start,
    int block_col_start,
    int row_A,
    int col_A,
    int row_B,
    int col_B,
    int K,
    int k_next
) {
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int K_TILE = 32;
    constexpr int A_PASS_ROWS = 64;
    constexpr int B_PASS_ROWS = 64;
    constexpr int A_STRIDE = K_TILE + PAD_A;
    constexpr int B_STRIDE = K_TILE + PAD_B;

    const half* src_A_base = A + (block_row_start + row_A) * K + (k_next + col_A);
    half* dst_A_base = sA_stage + row_A * A_STRIDE + col_A;

    const half* src_B_base = B_col_major + (block_col_start + row_B) * K + (k_next + col_B);
    half* dst_B_base = sB_stage + row_B * B_STRIDE + col_B;

    cp_async_cg_16B(dst_A_base + 0 * A_PASS_ROWS * A_STRIDE, src_A_base + 0 * A_PASS_ROWS * K);
    cp_async_cg_16B(dst_B_base + 0 * B_PASS_ROWS * B_STRIDE, src_B_base + 0 * B_PASS_ROWS * K);

    cp_async_cg_16B(dst_A_base + 1 * A_PASS_ROWS * A_STRIDE, src_A_base + 1 * A_PASS_ROWS * K);
    cp_async_cg_16B(dst_B_base + 1 * B_PASS_ROWS * B_STRIDE, src_B_base + 1 * B_PASS_ROWS * K);

    cp_async_cg_16B(dst_A_base + 2 * A_PASS_ROWS * A_STRIDE, src_A_base + 2 * A_PASS_ROWS * K);
    cp_async_cg_16B(dst_A_base + 3 * A_PASS_ROWS * A_STRIDE, src_A_base + 3 * A_PASS_ROWS * K);
}





__global__ void gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int K_TILE = 32;
    constexpr int total_pass = K_TILE/WMMA;
    // Larger CTA experiment:
    //   tile shape   = 256 x 128 x 32
    //   threads/CTA  = 256 (8 warps)
    //   warp layout  = 4 x 2.                                                                                                                               
    constexpr int TILE_M = 256;
    constexpr int TILE_N = 128;
    constexpr int CTA_THREADS = 256;
    constexpr int WARP_LAYOUT_M = 4;
    constexpr int WARP_LAYOUT_N = 2;
    constexpr int A_STRIDE = K_TILE + PAD_A;
    constexpr int B_STRIDE = TILE_N + PAD_B;
    constexpr int A_STAGE_ELEMS = TILE_M * A_STRIDE;
    constexpr int B_STAGE_ELEMS = K_TILE * B_STRIDE;
    constexpr int STAGE_COUNT = 3;

    const int m_tiles = (M + WMMA - 1) / WMMA;
    const int n_tiles = (N + WMMA - 1) / WMMA;
    const int block_tile_i = (blockIdx.z % 2)
        ? ((gridDim.y - blockIdx.y - 1) * (TILE_M / WMMA))
        : (blockIdx.y * (TILE_M / WMMA));
    const int block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * (TILE_N / WMMA);

    if (block_tile_i >= m_tiles || block_tile_j >= n_tiles) return;

    int block_row_start = block_tile_i * WMMA;
    int block_col_start = block_tile_j * WMMA;

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;

    // Eight warps are arranged as 4 warp-rows by 2 warp-cols.
    // Each warp therefore computes a 64x64 output tile.
    // Used in compute phase
    int warp_id = tid / WARP_SIZE; // 0,..., 7
    int warp_row = (warp_id / 2) * 64; // 0, 64, 128, 192 
    int warp_col = (warp_id % 2) * 64; // 0, 64

    // A stage is 256x32 = 8192 halfs = 1024 int4 loads. With 256 threads and 4 passes,
    // each pass covers 64 rows x 32 cols.
    // Used in load phase
    int row_A = tid / 4; // 0...63
    int col_A = (tid % 4) * 8; // 0, 8, 16, 24
    // B stage is 32x128 = 4096 halfs = 512 int4 loads. With 256 threads and 2 passes,
    // each pass covers 16 K rows x 128 cols.
    int row_B = tid / 16; //0...15
    int col_B = (tid % 16) * 8; // 0, 8, 16, 24, 32,..., 120

    // Triple-buffered shared staging for the current, next, and prefetch K-slices.
    // Similar to the reference kernel, one dynamic shared allocation backs the staging area,
    // then we reinterpret it as row-strided array views for A and B.
    extern __shared__ half smem[];
    auto sA = reinterpret_cast<half (*)[A_STAGE_ELEMS]>(smem);
    auto sB = reinterpret_cast<half (*)[B_STAGE_ELEMS]>(smem + STAGE_COUNT * A_STAGE_ELEMS);

    // One warp computes a 64x64 output tile:
    //   A fragments     = 4 tiles in M
    //   B fragments     = 4 tiles in N
    //   accum fragments = 4 x 4 output tiles
    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> a_frag[2][4];
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> b_frag[2][4];
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> accum_frag[4][4];
    __shared__ float warp_store[8][WMMA * WMMA];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(accum_frag[i][j], 0.0f);
        }
    }

    int current_stage = 0;
    int next_stage = 1;
    int prefetch_stage = 2;
    int reg_curr = 0;
    int reg_next = 1;

    {
        // Prologue: stage the first two K_TILE slices so the steady-state loop can overlap
        // prefetch with compute.
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int k_base = i * K_TILE;



            #pragma unroll
            for (int pass = 0; pass < 4; pass++) {

                int row_pass = pass * 64; // 0, 64, 128, 192

                const half* src_A = A + (block_row_start + row_pass + row_A) * K + (k_base + col_A);
                half* dst_A = &sA[i][(row_pass + row_A) * A_STRIDE + col_A];
                cp_async_cg_16B(dst_A, src_A);


            }

            #pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                int k_pass = pass * WMMA;   // 0, 16

                const half* src_B = B + (k_base + k_pass + row_B) * N + (block_col_start + col_B);
                half* dst_B = &sB[i][(k_pass + row_B) * B_STRIDE + col_B];
                cp_async_cg_16B(dst_B, src_B);
            }


           

            __pipeline_commit();
        }

        __pipeline_wait_prior(1);
        __syncthreads();
    }

    // Preload the first 16-wide K-slice of the current stage before entering the loop.
    gemm_rotate3_load_stage_fragments_reg_pingpong_256(
        sA[current_stage],
        sB[current_stage],
        warp_row,
        warp_col,
        reg_curr,
        0,
        a_frag,
        b_frag
    );

    #pragma unroll
    for (int k = 0; k + 2 * K_TILE < K; k += K_TILE) {
        int k_next = k + 2 * K_TILE;
        // Load the second K-slice of the current stage, then consume the first one.
        gemm_rotate3_load_stage_fragments_reg_pingpong_256(
            sA[current_stage],
            sB[current_stage],
            warp_row,
            warp_col,
            reg_next,
            WMMA,
            a_frag,
            b_frag
        );

        gemm_rotate3_mma_reg_pingpong_256(
            reg_curr,
            a_frag,
            b_frag,
            accum_frag
        );

    
        gemm_rotate3_prefetch_stage_reg_pingpong_256(
            A,
            B,
            sA[prefetch_stage],
            sB[prefetch_stage],
            block_row_start,
            block_col_start,
            row_A,
            col_A,
            row_B,
            col_B,
            K,
            N,
            k_next
        );

        __pipeline_commit();

        __pipeline_wait_prior(1);
        __syncthreads();
        rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

        // Preload the first K-slice of the next current stage, then finish the previous one.
        gemm_rotate3_load_stage_fragments_reg_pingpong_256(
            sA[current_stage],
            sB[current_stage],
            warp_row,
            warp_col,
            reg_curr,
            0,
            a_frag,
            b_frag
        );

        gemm_rotate3_mma_reg_pingpong_256(
            reg_next,
            a_frag,
            b_frag,
            accum_frag
        );
    }

    // Drain the second half of the current stage, then rotate to the final prefetched stage.
    gemm_rotate3_load_stage_fragments_reg_pingpong_256(
        sA[current_stage],
        sB[current_stage],
        warp_row,
        warp_col,
        reg_next,
        WMMA,
        a_frag,
        b_frag
    );

    gemm_rotate3_mma_reg_pingpong_256(
        reg_curr,
        a_frag,
        b_frag,
        accum_frag
    );

    __pipeline_wait_prior(0);
    __syncthreads();
    rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

    // Finish the previous stage while preloading the first K-slice of the final one.
    gemm_rotate3_load_stage_fragments_reg_pingpong_256(
        sA[current_stage],
        sB[current_stage],
        warp_row,
        warp_col,
        reg_curr,
        0,
        a_frag,
        b_frag
    );

    gemm_rotate3_mma_reg_pingpong_256(
        reg_next,
        a_frag,
        b_frag,
        accum_frag
    );

    // Finish the final stage.
    gemm_rotate3_load_stage_fragments_reg_pingpong_256(
        sA[current_stage],
        sB[current_stage],
        warp_row,
        warp_col,
        reg_next,
        WMMA,
        a_frag,
        b_frag
    );

    gemm_rotate3_mma_reg_pingpong_256(
        reg_curr,
        a_frag,
        b_frag,
        accum_frag
    );

    gemm_rotate3_mma_reg_pingpong_256(
        reg_next,
        a_frag,
        b_frag,
        accum_frag
    );

    // Warp-distributed epilogue: each fragment is first stored to a small shared scratch tile,
    // then all lanes participate in writing its 16x16 elements to global memory.
    // This path assumes the _256 kernel only runs on full tiles, which gemm.py should enforce.
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(warp_store[warp_id], accum_frag[i][j], WMMA, wmma::mem_row_major);
            __syncwarp();

            const int warp_row_base = block_row_start + warp_row + i * 16;
            const int warp_col_base = block_col_start + warp_col + j * 16;

            #pragma unroll
            for (int idx = lane_id; idx < WMMA * WMMA; idx += WARP_SIZE) {
                const int row = idx / WMMA;
                const int col = idx % WMMA;
                const int out_idx = (warp_row_base + row) * N + (warp_col_base + col);
                const float frag_val = warp_store[warp_id][idx];

                if (beta == 0.0f) {
                    C[out_idx] = __float2half(alpha * frag_val);
                } else {
                    float old_c = __half2float(C[out_idx]);
                    C[out_idx] = __float2half(alpha * frag_val + beta * old_c);
                }
            }

            __syncwarp();
        }
    }
}

__global__ void gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256_colb(const half* A, const half* B_col_major, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int K_TILE = 32;
    constexpr int TILE_M = 256;
    constexpr int TILE_N = 128;
    constexpr int A_STRIDE = K_TILE + PAD_A;
    constexpr int B_STRIDE = K_TILE + PAD_B;
    constexpr int A_STAGE_ELEMS = TILE_M * A_STRIDE;
    constexpr int B_STAGE_ELEMS = TILE_N * B_STRIDE;
    constexpr int STAGE_COUNT = 3;

    const int m_tiles = (M + WMMA - 1) / WMMA;
    const int n_tiles = (N + WMMA - 1) / WMMA;
    const int block_tile_i = (blockIdx.z % 2)
        ? ((gridDim.y - blockIdx.y - 1) * (TILE_M / WMMA))
        : (blockIdx.y * (TILE_M / WMMA));
    const int block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * (TILE_N / WMMA);
    if (block_tile_i >= m_tiles || block_tile_j >= n_tiles) return;

    int block_row_start = block_tile_i * WMMA;
    int block_col_start = block_tile_j * WMMA;

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warp_row = (warp_id / 2) * 64;
    int warp_col = (warp_id % 2) * 64;

    int row_A = tid / 4;
    int col_A = (tid % 4) * 8;
    int row_B = tid / 4;
    int col_B = (tid % 4) * 8;

    extern __shared__ half smem[];
    auto sA = reinterpret_cast<half (*)[A_STAGE_ELEMS]>(smem);
    auto sB = reinterpret_cast<half (*)[B_STAGE_ELEMS]>(smem + STAGE_COUNT * A_STAGE_ELEMS);

    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> a_frag[2][4];
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::col_major> b_frag[2][4];
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> accum_frag[4][4];
    __shared__ float warp_store[8][WMMA * WMMA];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(accum_frag[i][j], 0.0f);
        }
    }

    int current_stage = 0;
    int next_stage = 1;
    int prefetch_stage = 2;
    int reg_curr = 0;
    int reg_next = 1;

    {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int k_base = i * K_TILE;

            #pragma unroll
            for (int pass = 0; pass < 4; pass++) {
                int row_pass = pass * 64;
                const half* src_A = A + (block_row_start + row_pass + row_A) * K + (k_base + col_A);
                half* dst_A = &sA[i][(row_pass + row_A) * A_STRIDE + col_A];
                cp_async_cg_16B(dst_A, src_A);
            }

            #pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                int row_pass = pass * 64;
                const half* src_B = B_col_major + (block_col_start + row_pass + row_B) * K + (k_base + col_B);
                half* dst_B = &sB[i][(row_pass + row_B) * B_STRIDE + col_B];
                cp_async_cg_16B(dst_B, src_B);
            }

            __pipeline_commit();
        }

        __pipeline_wait_prior(1);
        __syncthreads();
    }

    gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb(
        sA[current_stage], sB[current_stage], warp_row, warp_col, reg_curr, 0, a_frag, b_frag
    );

    #pragma unroll
    for (int k = 0; k + 2 * K_TILE < K; k += K_TILE) {
        int k_next = k + 2 * K_TILE;

        gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb(
            sA[current_stage], sB[current_stage], warp_row, warp_col, reg_next, WMMA, a_frag, b_frag
        );

        gemm_rotate3_mma_reg_pingpong_256_colb(reg_curr, a_frag, b_frag, accum_frag);

        gemm_rotate3_prefetch_stage_reg_pingpong_256_colb(
            A, B_col_major, sA[prefetch_stage], sB[prefetch_stage],
            block_row_start, block_col_start, row_A, col_A, row_B, col_B, K, k_next
        );

        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

        gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb(
            sA[current_stage], sB[current_stage], warp_row, warp_col, reg_curr, 0, a_frag, b_frag
        );

        gemm_rotate3_mma_reg_pingpong_256_colb(reg_next, a_frag, b_frag, accum_frag);
    }

    gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb(
        sA[current_stage], sB[current_stage], warp_row, warp_col, reg_next, WMMA, a_frag, b_frag
    );
    gemm_rotate3_mma_reg_pingpong_256_colb(reg_curr, a_frag, b_frag, accum_frag);

    __pipeline_wait_prior(0);
    __syncthreads();
    rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

    gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb(
        sA[current_stage], sB[current_stage], warp_row, warp_col, reg_curr, 0, a_frag, b_frag
    );
    gemm_rotate3_mma_reg_pingpong_256_colb(reg_next, a_frag, b_frag, accum_frag);

    gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb(
        sA[current_stage], sB[current_stage], warp_row, warp_col, reg_next, WMMA, a_frag, b_frag
    );
    gemm_rotate3_mma_reg_pingpong_256_colb(reg_curr, a_frag, b_frag, accum_frag);
    gemm_rotate3_mma_reg_pingpong_256_colb(reg_next, a_frag, b_frag, accum_frag);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(warp_store[warp_id], accum_frag[i][j], WMMA, wmma::mem_row_major);
            __syncwarp();

            const int warp_row_base = block_row_start + warp_row + i * 16;
            const int warp_col_base = block_col_start + warp_col + j * 16;

            #pragma unroll
            for (int idx = lane_id; idx < WMMA * WMMA; idx += WARP_SIZE) {
                const int row = idx / WMMA;
                const int col = idx % WMMA;
                const int out_idx = (warp_row_base + row) * N + (warp_col_base + col);
                const float frag_val = warp_store[warp_id][idx];

                if (beta == 0.0f) {
                    C[out_idx] = __float2half(alpha * frag_val);
                } else {
                    float old_c = __half2float(C[out_idx]);
                    C[out_idx] = __float2half(alpha * frag_val + beta * old_c);
                }
            }

            __syncwarp();
        }
    }
}


__device__ __forceinline__ bool persistent_async_store_supported(
    int block_row_start,
    int block_col_start,
    int M,
    int N,
    float beta
) {
    return beta == 0.0f
        && (block_row_start + BLOCK_M) <= M
        && (block_col_start + BLOCK_N) <= N;
}



__device__ __forceinline__ void gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb_MMA(
    const half* sA_stage,
    const half* sB_stage,
    int warp_row,
    int warp_col,
    int lane_id,
    int reg_slot,
    int k_step,
    uint32_t (&a_regs)[2][4][4],
    uint32_t (&b_regs)[2][8][2]
) {
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int K_TILE = 32;
    constexpr int A_STRIDE = K_TILE + PAD_A;
    constexpr int B_STRIDE = K_TILE + PAD_B;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int smem_row = warp_row + i * 16;
        const half* tile_ptr_A =
            sA_stage + (smem_row + (lane_id % 16)) * A_STRIDE + k_step + (lane_id / 16) * 8;
        uint32_t A_smem_lane_addr = static_cast<uint32_t>(__cvta_generic_to_shared(tile_ptr_A));
        LDMATRIX_X4(
            a_regs[reg_slot][i][0],
            a_regs[reg_slot][i][1],
            a_regs[reg_slot][i][2],
            a_regs[reg_slot][i][3],
            A_smem_lane_addr
        );
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int smem_col = warp_col + j * 8;
        const half* tile_ptr_B =
            sB_stage + (smem_col + (lane_id % 8)) * B_STRIDE + k_step + ((lane_id / 8) % 2) * 8;
        uint32_t B_smem_lane_addr = static_cast<uint32_t>(__cvta_generic_to_shared(tile_ptr_B));
        LDMATRIX_X2(
            b_regs[reg_slot][j][0],
            b_regs[reg_slot][j][1],
            B_smem_lane_addr
        );
    }
}

__device__ __forceinline__ void gemm_rotate3_load_stage_fragments_reg_pingpong_256_MMA(
    const half* sA_stage,
    const half* sB_stage,
    int warp_row,
    int warp_col,
    int lane_id,
    int reg_slot,
    int k_step,
    uint32_t (&a_regs)[2][4][4],
    uint32_t (&b_regs)[2][8][2]
) {
    gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb_MMA(
        sA_stage, sB_stage, warp_row, warp_col, lane_id, reg_slot, k_step, a_regs, b_regs
    );
}

__device__ __forceinline__ void prefetch_a_stage_256x32_aligned_gemm(
    const half* A,
    half* sA_stage,
    int block_row_start,
    int row_A,
    int col_A,
    int K,
    int k_base
) {
    constexpr int PAD_A = 8;
    constexpr int A_STRIDE = 32 + PAD_A;

    #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int local_row = pass * 64 + row_A;
        half* dst = sA_stage + local_row * A_STRIDE + col_A;
        const half* src = A + (block_row_start + local_row) * K + (k_base + col_A);
        cp_async_cg_16B(dst, src);
    }
}

__device__ __forceinline__ void prefetch_b_stage_128x32_aligned_gemm(
    const half* B_col_major,
    half* sB_stage,
    int block_col_start,
    int row_B,
    int col_B,
    int K,
    int k_base
) {
    constexpr int PAD_B = 8;
    constexpr int B_STRIDE = 32 + PAD_B;

    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int local_row = pass * 64 + row_B;
        half* dst = sB_stage + local_row * B_STRIDE + col_B;
        const half* src = B_col_major + (block_col_start + local_row) * K + (k_base + col_B);
        cp_async_cg_16B(dst, src);
    }
}

__device__ __forceinline__ void prefetch_b_stage_128x32_rowmajor_async_gemm(
    const half* B,
    half* sB_row_stage,
    int block_col_start,
    int row_B,
    int col_B,
    int N,
    int k_base
) {
    constexpr int PAD_B_ROW = 8;
    constexpr int B_ROW_STRIDE = 128 + PAD_B_ROW;

    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int local_row = pass * 16 + row_B;
        half* dst = sB_row_stage + local_row * B_ROW_STRIDE + col_B;
        const half* src = B + (k_base + local_row) * N + (block_col_start + col_B);
        cp_async_cg_16B(dst, src);
    }
}

__device__ __forceinline__ void transpose_b_stage_128x32_row_to_col_gemm(
    const half* sB_row_stage,
    half* sB_col_stage,
    int row_B,
    int col_B
) {
    constexpr int PAD_B_ROW = 8;
    constexpr int PAD_B_COL = 8;
    constexpr int TILE_N = 128;
    constexpr int K_TILE = 32;
    constexpr int B_ROW_STRIDE = TILE_N + PAD_B_ROW;
    constexpr int B_COL_STRIDE = K_TILE + PAD_B_COL;

    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int local_row = pass * 16 + row_B;
        const half* src = sB_row_stage + local_row * B_ROW_STRIDE + col_B;
        int4 packed = *reinterpret_cast<const int4*>(src);
        const half* vals = reinterpret_cast<const half*>(&packed);

        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            sB_col_stage[(col_B + t) * B_COL_STRIDE + local_row] = vals[t];
        }
    }
}

__device__ __forceinline__ void gemm_rotate3_mma_reg_pingpong_256_MMA(
    int reg_slot,
    uint32_t (&a_regs)[2][4][4],
    uint32_t (&b_regs)[2][8][2],
    float (&c_regs)[4][8][4]
) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int j_s = (i % 2) ? (8 - j - 1) : j;
            float d0, d1, d2, d3;
            HMMA16816_F32(
                d0,
                d1,
                d2,
                d3,
                a_regs[reg_slot][i][0],
                a_regs[reg_slot][i][1],
                a_regs[reg_slot][i][2],
                a_regs[reg_slot][i][3],
                b_regs[reg_slot][j_s][0],
                b_regs[reg_slot][j_s][1],
                c_regs[i][j_s][0],
                c_regs[i][j_s][1],
                c_regs[i][j_s][2],
                c_regs[i][j_s][3]
            );
            c_regs[i][j_s][0] = d0;
            c_regs[i][j_s][1] = d1;
            c_regs[i][j_s][2] = d2;
            c_regs[i][j_s][3] = d3;
        }
    }
}

__device__ __forceinline__ void gemm_rotate3_prefetch_stage_reg_pingpong_256_mma(
    const half* A,
    const half* B,
    half* sA_stage,
    half* sB_row_stage,
    int block_row_start,
    int block_col_start,
    int row_A,
    int col_A,
    int row_B,
    int col_B,
    int K,
    int N,
    int k_next
) {
    constexpr int PAD_A = 8;
    constexpr int A_PASS_ROWS = 64;
    constexpr int A_STRIDE = 32 + PAD_A;

    const half* src_A_base = A + (block_row_start + row_A) * K + (k_next + col_A);
    half* dst_A_base = sA_stage + row_A * A_STRIDE + col_A;

    cp_async_cg_16B(dst_A_base + 0 * A_PASS_ROWS * A_STRIDE, src_A_base + 0 * A_PASS_ROWS * K);
    cp_async_cg_16B(dst_A_base + 1 * A_PASS_ROWS * A_STRIDE, src_A_base + 1 * A_PASS_ROWS * K);
    cp_async_cg_16B(dst_A_base + 2 * A_PASS_ROWS * A_STRIDE, src_A_base + 2 * A_PASS_ROWS * K);
    cp_async_cg_16B(dst_A_base + 3 * A_PASS_ROWS * A_STRIDE, src_A_base + 3 * A_PASS_ROWS * K);

    prefetch_b_stage_128x32_rowmajor_async_gemm(
        B, sB_row_stage, block_col_start, row_B, col_B, N, k_next
    );
}

__device__ __forceinline__ void gemm_rotate3_mma_reg_pingpong_256_colb_MMA(
    int reg_slot,
    uint32_t (&a_regs)[2][4][4],
    uint32_t (&b_regs)[2][8][2],
    float (&c_regs)[4][8][4]
) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int j_s = (i % 2) ? (8 - j - 1) : j;
            float d0, d1, d2, d3;
            HMMA16816_F32(
                d0,
                d1,
                d2,
                d3,
                a_regs[reg_slot][i][0],
                a_regs[reg_slot][i][1],
                a_regs[reg_slot][i][2],
                a_regs[reg_slot][i][3],
                b_regs[reg_slot][j_s][0],
                b_regs[reg_slot][j_s][1],
                c_regs[i][j_s][0],
                c_regs[i][j_s][1],
                c_regs[i][j_s][2],
                c_regs[i][j_s][3]
            );
            c_regs[i][j_s][0] = d0;
            c_regs[i][j_s][1] = d1;
            c_regs[i][j_s][2] = d2;
            c_regs[i][j_s][3] = d3;
        }
    }
}


__global__ void gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256_MMA(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int PAD_B_ROW = 8;
    constexpr int K_TILE = 32;
    constexpr int TILE_M = 256;
    constexpr int TILE_N = 128;
    constexpr int A_STRIDE = K_TILE + PAD_A;
    constexpr int B_STRIDE = K_TILE + PAD_B;
    constexpr int B_ROW_STRIDE = TILE_N + PAD_B_ROW;
    constexpr int A_STAGE_ELEMS = TILE_M * A_STRIDE;
    constexpr int B_STAGE_ELEMS = TILE_N * B_STRIDE;
    constexpr int B_ROW_STAGE_ELEMS = K_TILE * B_ROW_STRIDE;
    constexpr int STAGE_COUNT = 3;

    const int m_tiles = (M + WMMA - 1) / WMMA;
    const int n_tiles = (N + WMMA - 1) / WMMA;
    const int block_tile_i = (blockIdx.z % 2)
        ? ((gridDim.y - blockIdx.y - 1) * (TILE_M / WMMA))
        : (blockIdx.y * (TILE_M / WMMA));
    const int block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * (TILE_N / WMMA);
    if (block_tile_i >= m_tiles || block_tile_j >= n_tiles) return;

    int block_row_start = block_tile_i * WMMA;
    int block_col_start = block_tile_j * WMMA;

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warp_row = (warp_id / 2) * 64;
    int warp_col = (warp_id % 2) * 64;

    int row_A = tid / 4;
    int col_A = (tid % 4) * 8;
    int row_B = tid / 16;
    int col_B = (tid % 16) * 8;

    extern __shared__ half smem[];
    auto sA = reinterpret_cast<half (*)[A_STAGE_ELEMS]>(smem);
    auto sB = reinterpret_cast<half (*)[B_STAGE_ELEMS]>(smem + STAGE_COUNT * A_STAGE_ELEMS);
    half* sB_row = smem + STAGE_COUNT * A_STAGE_ELEMS + STAGE_COUNT * B_STAGE_ELEMS;

    uint32_t a_regs[2][4][4];
    uint32_t b_regs[2][8][2];
    float c_regs[4][8][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            c_regs[i][j][0] = 0.0f;
            c_regs[i][j][1] = 0.0f;
            c_regs[i][j][2] = 0.0f;
            c_regs[i][j][3] = 0.0f;
        }
    }

    int current_stage = 0;
    int next_stage = 1;
    int prefetch_stage = 2;
    int reg_curr = 0;
    int reg_next = 1;

    {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int k_base = i * K_TILE;
            prefetch_a_stage_256x32_aligned_gemm(
                A, sA[i], block_row_start, row_A, col_A, K, k_base
            );
            prefetch_b_stage_128x32_rowmajor_async_gemm(
                B, sB_row, block_col_start, row_B, col_B, N, k_base
            );

            __pipeline_commit();
            __pipeline_wait_prior(0);
            __syncthreads();
            transpose_b_stage_128x32_row_to_col_gemm(sB_row, sB[i], row_B, col_B);
            __syncthreads();
        }
    }

    gemm_rotate3_load_stage_fragments_reg_pingpong_256_MMA(
        sA[current_stage], sB[current_stage], warp_row, warp_col, lane_id, reg_curr, 0, a_regs, b_regs
    );

    #pragma unroll
    for (int k = 0; k + 2 * K_TILE < K; k += K_TILE) {
        int k_next = k + 2 * K_TILE;

        gemm_rotate3_load_stage_fragments_reg_pingpong_256_MMA(
            sA[current_stage], sB[current_stage], warp_row, warp_col, lane_id, reg_next, WMMA, a_regs, b_regs
        );

        gemm_rotate3_mma_reg_pingpong_256_MMA(reg_curr, a_regs, b_regs, c_regs);

        gemm_rotate3_prefetch_stage_reg_pingpong_256_mma(
            A, B, sA[prefetch_stage], sB_row,
            block_row_start, block_col_start, row_A, col_A, row_B, col_B, K, N, k_next
        );

        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        transpose_b_stage_128x32_row_to_col_gemm(sB_row, sB[prefetch_stage], row_B, col_B);
        __syncthreads();
        rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

        gemm_rotate3_load_stage_fragments_reg_pingpong_256_MMA(
            sA[current_stage], sB[current_stage], warp_row, warp_col, lane_id, reg_curr, 0, a_regs, b_regs
        );

        gemm_rotate3_mma_reg_pingpong_256_MMA(reg_next, a_regs, b_regs, c_regs);
    }

    gemm_rotate3_load_stage_fragments_reg_pingpong_256_MMA(
        sA[current_stage], sB[current_stage], warp_row, warp_col, lane_id, reg_next, WMMA, a_regs, b_regs
    );
    gemm_rotate3_mma_reg_pingpong_256_MMA(reg_curr, a_regs, b_regs, c_regs);

    __pipeline_wait_prior(0);
    __syncthreads();
    rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

    gemm_rotate3_load_stage_fragments_reg_pingpong_256_MMA(
        sA[current_stage], sB[current_stage], warp_row, warp_col, lane_id, reg_curr, 0, a_regs, b_regs
    );
    gemm_rotate3_mma_reg_pingpong_256_MMA(reg_next, a_regs, b_regs, c_regs);

    gemm_rotate3_load_stage_fragments_reg_pingpong_256_MMA(
        sA[current_stage], sB[current_stage], warp_row, warp_col, lane_id, reg_next, WMMA, a_regs, b_regs
    );
    gemm_rotate3_mma_reg_pingpong_256_MMA(reg_curr, a_regs, b_regs, c_regs);
    gemm_rotate3_mma_reg_pingpong_256_MMA(reg_next, a_regs, b_regs, c_regs);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int row0 = block_row_start + warp_row + i * 16 + lane_id / 4;
            const int row1 = row0 + 8;
            const int col0 = block_col_start + warp_col + j * 8 + (lane_id % 4) * 2;

            const float vals[4] = {c_regs[i][j][0], c_regs[i][j][1], c_regs[i][j][2], c_regs[i][j][3]};
            const int rows[4] = {row0, row0, row1, row1};
            const int cols[4] = {col0 + 0, col0 + 1, col0 + 0, col0 + 1};

            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int out_idx = rows[t] * N + cols[t];
                if (beta == 0.0f) {
                    C[out_idx] = __float2half(alpha * vals[t]);
                } else {
                    float old_c = __half2float(C[out_idx]);
                    C[out_idx] = __float2half(alpha * vals[t] + beta * old_c);
                }
            }
        }
    }
}

__global__ void gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256_colb_MMA(const half* A, const half* B_col_major, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int K_TILE = 32;
    constexpr int TILE_M = 256;
    constexpr int TILE_N = 128;
    constexpr int A_STRIDE = K_TILE + PAD_A;
    constexpr int B_STRIDE = K_TILE + PAD_B;
    constexpr int A_STAGE_ELEMS = TILE_M * A_STRIDE;
    constexpr int B_STAGE_ELEMS = TILE_N * B_STRIDE;
    constexpr int STAGE_COUNT = 3;

    const int m_tiles = (M + WMMA - 1) / WMMA;
    const int n_tiles = (N + WMMA - 1) / WMMA;
    const int block_tile_i = (blockIdx.z % 2)
        ? ((gridDim.y - blockIdx.y - 1) * (TILE_M / WMMA))
        : (blockIdx.y * (TILE_M / WMMA));
    const int block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * (TILE_N / WMMA);
    if (block_tile_i >= m_tiles || block_tile_j >= n_tiles) return;

    int block_row_start = block_tile_i * WMMA;
    int block_col_start = block_tile_j * WMMA;

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warp_row = (warp_id / 2) * 64;
    int warp_col = (warp_id % 2) * 64;

    int row_A = tid / 4;
    int col_A = (tid % 4) * 8;
    int row_B = tid / 4;
    int col_B = (tid % 4) * 8;

    extern __shared__ half smem[];
    auto sA = reinterpret_cast<half (*)[A_STAGE_ELEMS]>(smem);
    auto sB = reinterpret_cast<half (*)[B_STAGE_ELEMS]>(smem + STAGE_COUNT * A_STAGE_ELEMS);

    uint32_t a_regs[2][4][4];
    uint32_t b_regs[2][8][2];
    float c_regs[4][8][4];
    __shared__ float warp_store[8][WMMA * WMMA];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            c_regs[i][j][0] = 0.0f;
            c_regs[i][j][1] = 0.0f;
            c_regs[i][j][2] = 0.0f;
            c_regs[i][j][3] = 0.0f;
        }
    }

    int current_stage = 0;
    int next_stage = 1;
    int prefetch_stage = 2;
    int reg_curr = 0;
    int reg_next = 1;

    {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int k_base = i * K_TILE;

            prefetch_a_stage_256x32_aligned_gemm(
                A, sA[i], block_row_start, row_A, col_A, K, k_base
            );
            prefetch_b_stage_128x32_aligned_gemm(
                B_col_major, sB[i], block_col_start, row_B, col_B, K, k_base
            );

            __pipeline_commit();
        }

        __pipeline_wait_prior(1);
        __syncthreads();
    }

    gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb_MMA(
        sA[current_stage], sB[current_stage], warp_row, warp_col, lane_id, reg_curr, 0, a_regs, b_regs
    );

    #pragma unroll
    for (int k = 0; k + 2 * K_TILE < K; k += K_TILE) {
        int k_next = k + 2 * K_TILE;

        gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb_MMA(
            sA[current_stage], sB[current_stage], warp_row, warp_col, lane_id, reg_next, WMMA, a_regs, b_regs
        );

        gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_curr, a_regs, b_regs, c_regs);

        gemm_rotate3_prefetch_stage_reg_pingpong_256_colb(
            A, B_col_major, sA[prefetch_stage], sB[prefetch_stage],
            block_row_start, block_col_start, row_A, col_A, row_B, col_B, K, k_next
        );

        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

        gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb_MMA(
            sA[current_stage], sB[current_stage], warp_row, warp_col, lane_id, reg_curr, 0, a_regs, b_regs
        );

        gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_next, a_regs, b_regs, c_regs);
    }

    gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb_MMA(
        sA[current_stage], sB[current_stage], warp_row, warp_col, lane_id, reg_next, WMMA, a_regs, b_regs
    );
    gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_curr, a_regs, b_regs, c_regs);

    __pipeline_wait_prior(0);
    __syncthreads();
    rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

    gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb_MMA(
        sA[current_stage], sB[current_stage], warp_row, warp_col, lane_id, reg_curr, 0, a_regs, b_regs
    );
    gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_next, a_regs, b_regs, c_regs);

    gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb_MMA(
        sA[current_stage], sB[current_stage], warp_row, warp_col, lane_id, reg_next, WMMA, a_regs, b_regs
    );
    gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_curr, a_regs, b_regs, c_regs);
    gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_next, a_regs, b_regs, c_regs);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int local_row0 = lane_id / 4;
            const int local_row1 = local_row0 + 8;
            const int local_col0 = (lane_id % 4) * 2;

            warp_store[warp_id][local_row0 * 8 + local_col0 + 0] = c_regs[i][j][0];
            warp_store[warp_id][local_row0 * 8 + local_col0 + 1] = c_regs[i][j][1];
            warp_store[warp_id][local_row1 * 8 + local_col0 + 0] = c_regs[i][j][2];
            warp_store[warp_id][local_row1 * 8 + local_col0 + 1] = c_regs[i][j][3];
            __syncwarp();

            const int warp_row_base = block_row_start + warp_row + i * 16;
            const int warp_col_base = block_col_start + warp_col + j * 8;

            #pragma unroll
            for (int idx = lane_id; idx < 16 * 8; idx += WARP_SIZE) {
                const int row = idx / 8;
                const int col = idx % 8;
                const int out_idx = (warp_row_base + row) * N + (warp_col_base + col);
                const float frag_val = warp_store[warp_id][idx];

                if (beta == 0.0f) {
                    C[out_idx] = __float2half(alpha * frag_val);
                } else {
                    float old_c = __half2float(C[out_idx]);
                    C[out_idx] = __float2half(alpha * frag_val + beta * old_c);
                }
            }

            __syncwarp();
        }
    }
}













__global__ void gemm_3_stage_buffer_kernel_32_swizzled_rotate3(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;
    constexpr int K_TILE = 32;
    constexpr int total_pass = K_TILE/WMMA;

    const int swizzle_factor = 4;

    int idx_linear = blockIdx.y * gridDim.x + blockIdx.x;
    int grid_m_blocks = gridDim.y;
    int grid_n_blocks = gridDim.x;

    int swizzle_width = min(swizzle_factor, grid_n_blocks);
    int blocks_per_panel = swizzle_width * grid_m_blocks;
    int panel_number = idx_linear / blocks_per_panel;
    int idx_in_panel = idx_linear % blocks_per_panel;
    int block_row = idx_in_panel % grid_m_blocks;
    int block_col = panel_number * swizzle_width + (idx_in_panel / grid_m_blocks);

    if (block_row >= grid_m_blocks || block_col >= grid_n_blocks) return;

    int block_row_start = block_row * BLOCK_M;
    int block_col_start = block_col * BLOCK_N;

    int tid = threadIdx.x;

    int warp_id = tid / WARP_SIZE;
    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 32;

    int row_A = tid / 4;
    int col_A = (tid % 4) * 8;
    int row_B = tid / 8;
    int col_B = (tid % 8) * 8;

    __shared__ half sA[3][BLOCK_M * (K_TILE + PAD_A)];
    __shared__ half sB[3][K_TILE * (PAD_B + BLOCK_N)];

    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> accum_frag[2][2];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(accum_frag[i][j], 0.0f);
        }
    }

    int current_stage = 0;
    int next_stage = 1;
    int prefetch_stage = 2;

    {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int k_base = i * K_TILE;

            #pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                int k_pass = pass * WMMA;
                int row_pass = (k_pass / WMMA) * 32;

                const half* src_A = A + (block_row_start + row_pass + row_A) * K + (k_base + col_A);
                half* dst_A = &sA[i][(row_pass + row_A) * (K_TILE + PAD_A) + col_A];

                const half* src_B = B + (k_base + k_pass + row_B) * N + (block_col_start + col_B);
                half* dst_B = &sB[i][(k_pass + row_B) * (BLOCK_N + PAD_B) + col_B];
                __pipeline_memcpy_async(dst_A, src_A, sizeof(int4));
                __pipeline_memcpy_async(dst_B, src_B, sizeof(int4));
            }

            __pipeline_commit();
        }

        __pipeline_wait_prior(1);
        __syncthreads();
    }

    #pragma unroll
    for (int k = 0; k + 2 * K_TILE < K; k += K_TILE) {
        int k_next = k + 2 * K_TILE;
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            int k_pass = pass * WMMA;
            int row_pass = (k_pass / WMMA) * 32;
            const half* src_A = A + (block_row_start + row_pass + row_A) * K + (k_next + col_A);
            half* dst_A = &sA[prefetch_stage][(row_pass + row_A) * (K_TILE + PAD_A) + col_A];
            const half* src_B = B + (k_next + k_pass + row_B) * N + (block_col_start + col_B);
            half* dst_B = &sB[prefetch_stage][(k_pass + row_B) * (BLOCK_N + PAD_B) + col_B];
            __pipeline_memcpy_async(dst_A, src_A, sizeof(int4));
            __pipeline_memcpy_async(dst_B, src_B, sizeof(int4));
        }
        __pipeline_commit();

        #pragma unroll
        for (int k_step = 0; k_step < K_TILE; k_step += WMMA) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int smem_row = warp_row + (i * 16);
                half* tile_ptr_A = &sA[current_stage][smem_row * (K_TILE + PAD_A) + k_step];
                wmma::load_matrix_sync(a_frag[i], tile_ptr_A, K_TILE + PAD_A);
            }

            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int smem_col = warp_col + (j * 16);
                half* tile_ptr_B = &sB[current_stage][k_step * (BLOCK_N + PAD_B) + smem_col];
                wmma::load_matrix_sync(b_frag[j], tile_ptr_B, BLOCK_N + PAD_B);
            }

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    wmma::mma_sync(accum_frag[i][j], a_frag[i], b_frag[j], accum_frag[i][j]);
                }
            }
        }

        __pipeline_wait_prior(1);
        __syncthreads();
        rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);
    }

    #pragma unroll
    for (int k_step = 0; k_step < K_TILE; k_step += WMMA) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int smem_row = warp_row + (i * 16);
            half* tile_ptr_A = &sA[current_stage][smem_row * (K_TILE + PAD_A) + k_step];
            wmma::load_matrix_sync(a_frag[i], tile_ptr_A, K_TILE + PAD_A);
        }
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int smem_col = warp_col + (j * 16);
            half* tile_ptr_B = &sB[current_stage][k_step * (BLOCK_N + PAD_B) + smem_col];
            wmma::load_matrix_sync(b_frag[j], tile_ptr_B, BLOCK_N + PAD_B);
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(accum_frag[i][j], a_frag[i], b_frag[j], accum_frag[i][j]);
            }
        }
    }

    __pipeline_wait_prior(0);
    __syncthreads();
    rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

    #pragma unroll
    for (int k_step = 0; k_step < K_TILE; k_step += WMMA) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int smem_row = warp_row + (i * 16);
            half* tile_ptr_A = &sA[current_stage][smem_row * (K_TILE + PAD_A) + k_step];
            wmma::load_matrix_sync(a_frag[i], tile_ptr_A, K_TILE + PAD_A);
        }
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int smem_col = warp_col + (j * 16);
            half* tile_ptr_B = &sB[current_stage][k_step * (BLOCK_N + PAD_B) + smem_col];
            wmma::load_matrix_sync(b_frag[j], tile_ptr_B, BLOCK_N + PAD_B);
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(accum_frag[i][j], a_frag[i], b_frag[j], accum_frag[i][j]);
            }
        }
    }

    __shared__ float sC[BLOCK_M * BLOCK_N];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            float* subtile_ptr = sC + (warp_row + i * 16) * BLOCK_N + (warp_col + j * 16);
            wmma::store_matrix_sync(subtile_ptr, accum_frag[i][j], BLOCK_N, wmma::mem_row_major);
        }
    }

    __syncthreads();

    #pragma unroll
    for (int i = tid * 8; i < BLOCK_M * BLOCK_N; i += THREAD_COUNT * 8) {
        int row = i / BLOCK_N;
        int col = i % BLOCK_N;

        int global_row = block_row_start + row;
        int global_col = block_col_start + col;

        half buffer[8];

        if (global_row < M && (global_col + 7) < N) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float val = alpha * sC[i + j];

                if (beta != 0.0f) {
                    float old_c = __half2float(C[global_row * N + global_col + j]);
                    val += beta * old_c;
                }

                buffer[j] = __float2half(val);
            }

            *(int4*)&C[global_row * N + global_col] = *(int4*)buffer;

        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (global_row < M && (global_col + j) < N) {
                    int out_idx = global_row * N + global_col + j;

                    float val = alpha * sC[i + j];

                    if (beta != 0.0f) {
                        float old_c = __half2float(C[out_idx]);
                        val += beta * old_c;
                    }

                    C[out_idx] = __float2half(val);
                }
            }
        }
    }
}
