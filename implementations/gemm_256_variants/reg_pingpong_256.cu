#include "gemm_256_common.cuh"

__global__ void gemm_3_stage_buffer_kernel_32_swizzled_rotate3_reg_pingpong_256(
    const half* A,
    const half* B,
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

    const int block_row_start = block_tile_i * WMMA;
    const int block_col_start = block_tile_j * WMMA;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int warp_row = (warp_id / 2) * 64;
    const int warp_col = (warp_id % 2) * 64;
    const int row_A = tid / 4;
    const int col_A = (tid % 4) * 8;
    const int row_B = tid / 16;
    const int col_B = (tid % 16) * 8;

    extern __shared__ half smem[];
    auto sA = reinterpret_cast<half (*)[A_STAGE_ELEMS]>(smem);
    auto sB = reinterpret_cast<half (*)[B_STAGE_ELEMS]>(smem + STAGE_COUNT * A_STAGE_ELEMS);

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
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int k_base = i * K_TILE;

            #pragma unroll
            for (int pass = 0; pass < 4; pass++) {
                const int row_pass = pass * 64;
                const half* src_A = A + (block_row_start + row_pass + row_A) * K + (k_base + col_A);
                half* dst_A = &sA[i][(row_pass + row_A) * A_STRIDE + col_A];
                cp_async_cg_16B(dst_A, src_A);
            }

            #pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                const int k_pass = pass * WMMA;
                const half* src_B = B + (k_base + k_pass + row_B) * N + (block_col_start + col_B);
                half* dst_B = &sB[i][(k_pass + row_B) * B_STRIDE + col_B];
                cp_async_cg_16B(dst_B, src_B);
            }

            __pipeline_commit();
        }

        __pipeline_wait_prior(1);
        __syncthreads();
    }

    gemm_rotate3_load_stage_fragments_reg_pingpong_256(
        sA[current_stage], sB[current_stage], warp_row, warp_col, reg_curr, 0, a_frag, b_frag
    );

    #pragma unroll
    for (int k = 0; k + 2 * K_TILE < K; k += K_TILE) {
        const int k_next = k + 2 * K_TILE;

        gemm_rotate3_load_stage_fragments_reg_pingpong_256(
            sA[current_stage], sB[current_stage], warp_row, warp_col, reg_next, WMMA, a_frag, b_frag
        );
        gemm_rotate3_mma_reg_pingpong_256(reg_curr, a_frag, b_frag, accum_frag);

        gemm_rotate3_prefetch_stage_reg_pingpong_256(
            A, B, sA[prefetch_stage], sB[prefetch_stage],
            block_row_start, block_col_start, row_A, col_A, row_B, col_B, K, N, k_next
        );

        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

        gemm_rotate3_load_stage_fragments_reg_pingpong_256(
            sA[current_stage], sB[current_stage], warp_row, warp_col, reg_curr, 0, a_frag, b_frag
        );
        gemm_rotate3_mma_reg_pingpong_256(reg_next, a_frag, b_frag, accum_frag);
    }

    gemm_rotate3_load_stage_fragments_reg_pingpong_256(
        sA[current_stage], sB[current_stage], warp_row, warp_col, reg_next, WMMA, a_frag, b_frag
    );
    gemm_rotate3_mma_reg_pingpong_256(reg_curr, a_frag, b_frag, accum_frag);

    __pipeline_wait_prior(0);
    __syncthreads();
    rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

    gemm_rotate3_load_stage_fragments_reg_pingpong_256(
        sA[current_stage], sB[current_stage], warp_row, warp_col, reg_curr, 0, a_frag, b_frag
    );
    gemm_rotate3_mma_reg_pingpong_256(reg_next, a_frag, b_frag, accum_frag);

    gemm_rotate3_load_stage_fragments_reg_pingpong_256(
        sA[current_stage], sB[current_stage], warp_row, warp_col, reg_next, WMMA, a_frag, b_frag
    );
    gemm_rotate3_mma_reg_pingpong_256(reg_curr, a_frag, b_frag, accum_frag);
    gemm_rotate3_mma_reg_pingpong_256(reg_next, a_frag, b_frag, accum_frag);

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
                    const float old_c = __half2float(C[out_idx]);
                    C[out_idx] = __float2half(alpha * frag_val + beta * old_c);
                }
            }

            __syncwarp();
        }
    }
}
