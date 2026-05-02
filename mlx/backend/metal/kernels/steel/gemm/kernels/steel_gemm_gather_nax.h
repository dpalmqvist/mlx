// Copyright © 2024 Apple Inc.

using namespace mlx::steel;

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    typename AccumType = float,
    class NAXFrag_ = mlx::steel::BaseNAXFrag>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
gather_mm_rhs_nax(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device uint32_t* rhs_indices [[buffer(2)]],
    device T* C [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / NAXFrag_::kFragRows;
  constexpr short TN = SN / NAXFrag_::kFragCols;
  static_assert(SM % NAXFrag_::kFragRows == 0,
                "SM must be a multiple of NAXFrag_::kFragRows");
  static_assert(SN % NAXFrag_::kFragCols == 0,
                "SN must be a multiple of NAXFrag_::kFragCols");

  if (params->tiles_n <= static_cast<int>(tid.x) ||
      params->tiles_m <= static_cast<int>(tid.y)) {
    return;
  }

  // Find the block in A, B, C
  const int c_row = tid.y * BM;
  const int c_col = tid.x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  C += c_row_long * params->ldd + c_col_long;
  rhs_indices += c_row;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const int sgp_sm_int =
      align_M ? int(SM) : min(int(SM), params->M - (c_row + tm));
  const short sgp_sm = short(sgp_sm_int);
  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);

  const int sgp_sn_int =
      align_N ? int(SN) : min(int(SN), params->N - (c_col + tn));
  const short sgp_sn = short(sgp_sn_int);
  const bool is_unaligned_sn = align_N ? false : (sgp_sn != SN);

  A += transpose_a ? tm : (tm * params->lda);
  B += transpose_b ? (tn * params->ldb) : tn;
  C += tm * params->ldd + tn;
  rhs_indices += tm;

  // Threadgroup scratch for the NAXFrag32 (kPacking==1) path. Mirrors the
  // splitk_nax/fused_nax kernels. Doubled per simdgroup so Atile and Btile
  // can stage in parallel (Phase 8 prototype B). For BaseNAXFrag
  // (kPacking==2), size 1 keeps the array non-zero (Metal rejects zero-sized
  // threadgroup arrays); the BaseNAXFrag path never reads scratch_buf.
  constexpr int kFragArea = NAXFrag_::kFragRows * NAXFrag_::kFragCols;
  constexpr int kScratchSize = (NAXFrag_::kPacking == 1)
      ? (WM * WN * 2 * kFragArea)
      : 1;
  threadgroup AccumType scratch_buf[kScratchSize];
  threadgroup AccumType* sg_scratch_a =
      (NAXFrag_::kPacking == 1)
          ? (scratch_buf + simd_group_id * 2 * kFragArea)
          : nullptr;
  threadgroup AccumType* sg_scratch_b =
      (NAXFrag_::kPacking == 1)
          ? (scratch_buf + simd_group_id * 2 * kFragArea + kFragArea)
          : nullptr;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = rhs_indices[0];
  short offset_next = 0;
  int n = 0;
  while (n < sgp_sm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = sgp_sm;
    for (; n < sgp_sm; n++) {
      if (rhs_indices[n] != index) {
        offset_next = n;
        index_next = rhs_indices[n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    NAXTile<AccumType, TM, TN, NAXFrag_> Ctile;

    dispatch_bool(align_K, [&](auto kAlignedK) {
      dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
        dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
          auto do_gemm = gemm_loop< // Matmul for partial BM, full BN and full K
              T,
              SM,
              SN,
              SK,
              BK,
              transpose_a,
              transpose_b,
              kAlignedM.value,
              kAlignedN.value,
              kAlignedK.value,
              AccumType,
              NAXFrag_>;
          Ctile = do_gemm(
              A,
              B + index * params->batch_stride_b,
              params->lda,
              params->ldb,
              params->K,
              params->gemm_k_iterations_aligned,
              sgp_sm,
              sgp_sn,
              (threadgroup T*)sg_scratch_a,
              (threadgroup T*)sg_scratch_b);

          if constexpr (kAlignedN.value) {
            if (offset_next - offset == SM) {
              Ctile.store(C, int(params->ldd), sg_scratch_a);
            } else {
              Ctile.store_slice(
                  C,
                  int(params->ldd),
                  short2(0, offset),
                  short2(SN, offset_next),
                  sg_scratch_a);
            }
          } else {
            Ctile.store_slice(
                C,
                int(params->ldd),
                short2(0, offset),
                short2(sgp_sn, offset_next),
                sg_scratch_a);
          }
        });
      });
    });
  }
}
