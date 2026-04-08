#include "gemm_256_common.cuh"

__global__ void gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256_colb_MMA(
    const half* A,
    const half* B_col_major,
    half* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
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

    const int block_row_start = block_tile_i * WMMA;
    const int block_col_start = block_tile_j * WMMA;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int warp_row = (warp_id / 2) * 64;
    const int warp_col = (warp_id % 2) * 64;
    const int row_A = tid / 4;
    const int col_A = (tid % 4) * 8;
    const int row_B = tid / 4;
    const int col_B = (tid % 4) * 8;

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
            const int k_base = i * K_TILE;
            prefetch_a_stage_256x32_aligned_gemm(A, sA[i], block_row_start, row_A, col_A, K, k_base);
            prefetch_b_stage_128x32_aligned_gemm(B_col_major, sB[i], block_col_start, row_B, col_B, K, k_base);
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
        const int k_next = k + 2 * K_TILE;

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
                    const float old_c = __half2float(C[out_idx]);
                    C[out_idx] = __float2half(alpha * frag_val + beta * old_c);
                }
            }

            __syncwarp();
        }
    }
}
