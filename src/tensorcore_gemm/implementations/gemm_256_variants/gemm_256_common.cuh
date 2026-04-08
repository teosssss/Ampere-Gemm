#pragma once

#include <cuda/barrier>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "ptx_primitives.cuh"

using namespace nvcuda;

constexpr int WARP_SIZE = 32;
constexpr int WMMA = 16;

__device__ __forceinline__ void rotate_stage_triplet_3(int& current_stage, int& next_stage, int& prefetch_stage) {
    const int old_current = current_stage;
    current_stage = next_stage;
    next_stage = prefetch_stage;
    prefetch_stage = old_current;
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

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int smem_row = warp_row + i * 16;
        const half* tile_ptr_A = sA_stage + smem_row * (WMMA * 2 + PAD_A) + k_step;
        wmma::load_matrix_sync(a_frag[reg_slot][i], tile_ptr_A, WMMA * 2 + PAD_A);
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int smem_col = warp_col + j * 16;
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
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int j_s = (i % 2) ? (4 - j - 1) : j;
            wmma::mma_sync(accum_frag[i][j_s], a_frag[reg_slot][i], b_frag[reg_slot][j_s], accum_frag[i][j_s]);
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
        const int smem_row = warp_row + i * 16;
        const half* tile_ptr_A = sA_stage + smem_row * A_STRIDE + k_step;
        wmma::load_matrix_sync(a_frag[reg_slot][i], tile_ptr_A, A_STRIDE);
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int smem_col = warp_col + j * 16;
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
            const int j_s = (i % 2) ? (4 - j - 1) : j;
            wmma::mma_sync(accum_frag[i][j_s], a_frag[reg_slot][i], b_frag[reg_slot][j_s], accum_frag[i][j_s]);
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
        const int smem_row = warp_row + i * 16;
        const half* tile_ptr_A =
            sA_stage + (smem_row + (lane_id % 16)) * A_STRIDE + k_step + (lane_id / 16) * 8;
        const uint32_t A_smem_lane_addr = static_cast<uint32_t>(__cvta_generic_to_shared(tile_ptr_A));
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
        const int smem_col = warp_col + j * 8;
        const half* tile_ptr_B =
            sB_stage + (smem_col + (lane_id % 8)) * B_STRIDE + k_step + ((lane_id / 8) % 2) * 8;
        const uint32_t B_smem_lane_addr = static_cast<uint32_t>(__cvta_generic_to_shared(tile_ptr_B));
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
        const int4 packed = *reinterpret_cast<const int4*>(src);
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
            const int j_s = (i % 2) ? (8 - j - 1) : j;
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
    prefetch_b_stage_128x32_rowmajor_async_gemm(B, sB_row_stage, block_col_start, row_B, col_B, N, k_next);
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
            const int j_s = (i % 2) ? (8 - j - 1) : j;
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
