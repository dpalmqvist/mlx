// Copyright © 2026 Apple Inc.

using namespace mlx::steel;

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];

///////////////////////////////////////////////////////////////////////////////
// NAX Split-K GEMM kernel
///////////////////////////////////////////////////////////////////////////////

// clang-format off
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
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gemm_splitk_nax(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device AccumType* C [[buffer(2)]],
    const constant GEMMSpiltKParams* params [[buffer(3)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on

  const int linear_tid = tid.x;

  // Compute swizzled tile dimensions
  const int tn_swizzled = params->tiles_n << params->swizzle_log;
  const int tm_swizzled =
      (params->tiles_m + (1 << params->swizzle_log) - 1) >> params->swizzle_log;
  const int tiles_per_partition = tn_swizzled * tm_swizzled;

  const int tid_z = linear_tid / tiles_per_partition;
  const int xy_flat = linear_tid % tiles_per_partition;

  // Decode 2D grid coordinates in swizzled space
  const int grid_x = xy_flat % tn_swizzled;
  const int grid_y = xy_flat / tn_swizzled;

  // Apply X-Y swizzle
  const int tid_y = (grid_y << params->swizzle_log) +
      (grid_x & ((1 << params->swizzle_log) - 1));
  const int tid_x = grid_x >> params->swizzle_log;

  // Exit early
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // Calculate partition bounds
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int k_start = params->split_k_partition_size * tid_z;
  const int k_end = min(k_start + params->split_k_partition_size, params->K);

  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);
  const size_t k_start_long = size_t(k_start);

  // Adjust pointers for split-K partition
  A += transpose_a ? (c_row_long + k_start_long * params->lda)
                   : (k_start_long + c_row_long * params->lda);
  B += transpose_b ? (k_start_long + c_col_long * params->ldb)
                   : (c_col_long + k_start_long * params->ldb);
  C += (size_t(params->split_k_partition_stride) * tid_z) +
      (c_row_long * params->ldc + c_col_long);

  // NAX tile configuration
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / NAXFrag_::kFragRows;
  constexpr short TN = SN / NAXFrag_::kFragCols;
  static_assert(SM % NAXFrag_::kFragRows == 0,
                "SM must be a multiple of NAXFrag_::kFragRows");
  static_assert(SN % NAXFrag_::kFragCols == 0,
                "SN must be a multiple of NAXFrag_::kFragCols");

  // Calculate simdgroup offsets and alignment
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
  C += tm * params->ldc + tn;

  NAXTile<AccumType, TM, TN, NAXFrag_> Dtile;

  // Threadgroup scratch for the NAXFrag32 (kPacking==1) path. Mirrors the
  // fused_nax kernel: each simdgroup gets its own kFragRows*kFragCols slice
  // used by NAXTile's safe/rows methods to stage device <-> register through
  // the contiguous-tg cooperative-tensor load. For BaseNAXFrag (kPacking==2),
  // size 1 keeps the array non-zero (Metal rejects zero-sized tg arrays); the
  // BaseNAXFrag path never reads scratch_buf.
  constexpr int kFragArea = NAXFrag_::kFragRows * NAXFrag_::kFragCols;
  constexpr int kScratchSize = (NAXFrag_::kPacking == 1)
      ? (WM * WN * 2 * kFragArea)  // 2x for A/B double-buffering
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

  // gemm_loop through the partition
  // Check K-alignment at runtime (partition-specific)
  const int partition_k_size = k_end - k_start;
  const int partition_k_iters = partition_k_size / BK;
  const bool partition_k_aligned = (partition_k_size % BK) == 0;

  dispatch_bool(partition_k_aligned, [&](auto kAlignedK) {
    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
        Dtile = gemm_loop<
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
            NAXFrag_>(
            A,
            B,
            params->lda,
            params->ldb,
            partition_k_size,
            partition_k_iters,
            sgp_sm,
            sgp_sn,
            (threadgroup T*)sg_scratch_a,
            (threadgroup T*)sg_scratch_b);
      });
    });
  });

  // Store result
  dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
      if constexpr (kAlignedM && kAlignedN) {
        Dtile.store(C, int(params->ldc), sg_scratch_a);
      } else {
        Dtile.store_safe(C, int(params->ldc), short2(sgp_sn, sgp_sm), sg_scratch_a);
      }
    });
  });
}
