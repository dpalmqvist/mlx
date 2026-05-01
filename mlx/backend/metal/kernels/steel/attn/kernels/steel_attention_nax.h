// Copyright © 2024-25 Apple Inc.

#include "mlx/backend/metal/kernels/steel/attn/nax.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/kernels/steel/attn/transforms.h"
#include "mlx/backend/metal/kernels/steel/utils.h"

using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) : scale(scale_) {}

  METAL_FUNC T apply(T x) const {
    return scale * x;
  }
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float,
    class NAXFrag_ = mlx::steel::BaseNAXFrag>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;
  (void)simd_lane_id;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  // kU adapts to whichever frag class is used: 16 for BaseNAXFrag, 32 for NAXFrag32
  constexpr short kU = NAXFrag_::kFragRows;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  using otile_t = NAXTile<AccumType, TQ, TD, NAXFrag_>;
  otile_t Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short tm = kU * TQ * simd_group_id;
  Q += tm * int(params->Q_strides[2]);

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Threadgroup scratch for the NAXFrag32 (kPacking==1) path. Each simdgroup
  // gets its own kFragRows*kFragCols region used by NAXTile's safe/rows methods
  // to stage device <-> register through the contiguous-tg cooperative-tensor
  // load. For BaseNAXFrag (kPacking==2), `1` keeps the array non-zero-sized
  // (Metal rejects zero-sized threadgroup arrays); the BaseNAXFrag path never
  // reads scratch_buf, so the 4-byte cost is negligible.
  constexpr int kScratchSize = (NAXFrag_::kPacking == 1)
      ? (WM * WN * NAXFrag_::kFragRows * NAXFrag_::kFragCols)
      : 1;
  threadgroup AccumType scratch_buf[kScratchSize];
  threadgroup AccumType* sg_scratch =
      (NAXFrag_::kPacking == 1)
          ? (scratch_buf + simd_group_id * (NAXFrag_::kFragRows * NAXFrag_::kFragCols))
          : nullptr;

  // Init row reduction variables
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  const bool is_last_bq = int(tid.x) == (params->NQ_aligned);
  // const bool is_last_tq = int(simd_group_id) >= (params->qL_rem / UQ);
  const bool is_last_q = is_last_bq;

  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    // Do S = Q @ K.T
    using stile_t = NAXTile<AccumType, TQ, TK, NAXFrag_>;
    stile_t Stile;

    Stile.clear();

    if constexpr (NAXFrag_::kPacking == 2) {
      // BaseNAXFrag path: (16,32,16) descriptor produces 2 B frags per mma.
      // Iterate ik in steps of 2 to match the fused-pair mma overload.
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik += 2) {
          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; id++) {
            NAXTile<T, 1, 1> Qtile;
            NAXTile<T, 2, 1> Ktile;

            const int Q_load_off = iq * kU * int(params->Q_strides[2]) + id * kU;
            const int K_load_off = ik * kU * int(params->K_strides[2]) + id * kU;

            if (!align_Q && is_last_q) {
              Qtile.load_rows(
                  Q + Q_load_off,
                  int(params->Q_strides[2]),
                  lim_rows_q - iq * kU);
            } else {
              Qtile.load(Q + Q_load_off, int(params->Q_strides[2]));
            }

            if (!align_K && is_last_k) {
              Ktile.load_rows(
                  K + K_load_off,
                  int(params->K_strides[2]),
                  lim_rows_k - ik * kU);
            } else {
              Ktile.load(K + K_load_off, int(params->K_strides[2]));
            }

            stile_t::NAXFrag_t::mma(
                Stile.frag_at(iq, ik),
                Stile.frag_at(iq, ik + 1),
                Qtile.frag_at(0, 0),
                metal::false_type{},
                Ktile.frag_at(0, 0),
                Ktile.frag_at(1, 0),
                metal::true_type{});
          }
        }
      }
    } else {
      // NAXFrag32 path: (32,32,32) descriptor produces 1 frag per mma.
      // Iterate ik in steps of 1; Ktile is 1x1, K loads kFragRows rows at a time.
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; id++) {
            NAXTile<T, 1, 1, NAXFrag_> Qtile;
            NAXTile<T, 1, 1, NAXFrag_> Ktile;

            const int Q_load_off = iq * kU * int(params->Q_strides[2]) + id * kU;
            const int K_load_off = ik * kU * int(params->K_strides[2]) + id * kU;

            if (!align_Q && is_last_q) {
              Qtile.template load_rows<Role::Left, false>(
                  Q + Q_load_off,
                  int(params->Q_strides[2]),
                  lim_rows_q - iq * kU,
                  (threadgroup T*)sg_scratch);
            } else {
              Qtile.template load<Role::Left, false>(
                  Q + Q_load_off,
                  int(params->Q_strides[2]),
                  (threadgroup T*)sg_scratch);
            }

            if (!align_K && is_last_k) {
              Ktile.template load_rows<Role::Right, true>(
                  K + K_load_off,
                  int(params->K_strides[2]),
                  lim_rows_k - ik * kU,
                  (threadgroup T*)sg_scratch);
            } else {
              Ktile.template load<Role::Right, true>(
                  K + K_load_off,
                  int(params->K_strides[2]),
                  (threadgroup T*)sg_scratch);
            }

            stile_t::NAXFrag_t::mma(
                Stile.frag_at(iq, ik),
                Qtile.frag_at(0, 0),
                metal::false_type{},
                Ktile.frag_at(0, 0),
                metal::true_type{});
          }
        }
      }
    }

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;

          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              // For NAXFrag32 (kPacking==1) the per-thread column layout is
              // non-contiguous: element jj has offset dc_table[jj] within the
              // 32-column frag, not simply jj. BaseNAXFrag (kPacking==2) has
              // contiguous 4-element rows so col offset == jj.
              short col_off;
              if constexpr (NAXFrag_::kPacking == 1) {
                col_off = stile_t::NAXFrag_t::dr_dc(loc).x;
              } else {
                col_off = jj;
              }
              fg[loc] = ((col_pos + col_off) < params->kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              // For NAXFrag32 (kPacking==1) the per-thread layout is
              // non-contiguous: element loc has row offset dr_table[loc/8] and
              // col offset dc_table[loc%8]. BaseNAXFrag (kPacking==2) has
              // contiguous rows (row stride = kFragRowsJump, col stride = 1).
              short row_off, col_off;
              if constexpr (NAXFrag_::kPacking == 1) {
                const auto coords = stile_t::NAXFrag_t::dr_dc(loc);
                row_off = coords.y;
                col_off = coords.x;
              } else {
                row_off = ii * stile_t::kFragRowsJump;
                col_off = jj;
              }
              const auto r = base_row + iq * kU + row_off + sm;
              const auto c = base_col + ik * kU + col_off + sn;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = NAXTile<melem_t, TQ, TK, NAXFrag_>;
      using mfrag_t = typename mtile_t::frag_type;

      if constexpr (NAXFrag_::kPacking == 1) {
        // NAXFrag32 mask load: use per-element device reads with dr_dc offsets.
        //
        // The score tile Stile holds MMA output (C/destination) cooperative
        // tensor elements. The per-thread element mapping for C is given by
        // get_coord() + dr_dc(jj). The left-input (A) cooperative tensor uses
        // a different per-thread layout, so load<Role::Left> stages through
        // scratch and yields mfrag[jj] at the A layout — NOT the C layout.
        // Applying mfrag[jj] += mask at the C layout positions would map the
        // wrong mask value to the wrong score element.
        //
        // Fix: directly read mask[q_row][k_col] per element using get_coord()
        // and dr_dc() to compute the exact (row, col) address each thread owns.
        // This avoids cooperative tensors for the mask load entirely.
        //
        // M_strides[2] is 0 when the mask Q dim is 1 (shape-aware stride from
        // the dispatcher); in that case the mask row offset is 0 for all warps
        // (broadcast semantics: all Q positions see the same mask row).
        const short2 coord = stile_t::NAXFrag_t::get_coord();
        // row_base of this warp in the global attention sequence
        const int warp_row_base = base_row;

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = warp_row_base + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;

            // Aligned path: all (row_pos + dr, col_pos + dc) are in bounds.
            const bool tile_in_bounds =
                (row_pos + kU <= params->qL) && (col_pos + kU <= params->kL);

            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < NAXFrag_::kElemsPerFrag; jj++) {
              const short2 delta = stile_t::NAXFrag_t::dr_dc(jj);
              // Absolute query position: row_pos + lane's row_base + dr_table offset.
              // int64_t arithmetic: M_strides[2] may be 0 (broadcast) or kL.
              const int r = row_pos + coord.y + delta.y;
              const int c = col_pos + coord.x + delta.x;
              if (tile_in_bounds) {
                mfrag[jj] = static_cast<melem_t>(
                    mask[int64_t(r) * mask_params->M_strides[2] + c]);
              } else {
                // Partial tile: bounds-check each element.
                if (r < params->qL && c < params->kL) {
                  mfrag[jj] = static_cast<melem_t>(
                      mask[int64_t(r) * mask_params->M_strides[2] + c]);
                } else {
                  mfrag[jj] = melem_t(0);
                }
              }
            }

            thread auto& fg = Stile.frag_at(iq, ik);

            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      } else {
        // BaseNAXFrag path: unchanged — use NAXTile load/load_safe API
        // which is correct for the 16×16 simdgroup matrix layout.
        if (base_row + BQ <= params->qL && base_col + BK <= params->kL) {
          for (short iq = 0; iq < TQ; iq++) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
              const int row_pos = base_row + iq * kU;
              const int col_pos = base_col + ik * kU;

              mfrag_t mfrag;
              mtile_t::NAXFrag_t::load(
                  mfrag,
                  mask,
                  int64_t(mask_params->M_strides[2]),
                  Int<1>{},
                  row_pos,
                  col_pos);

              thread auto& fg = Stile.frag_at(iq, ik);

              STEEL_PRAGMA_UNROLL
              for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
                if constexpr (is_bool) {
                  fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
                } else {
                  fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
                }
              }
            }
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short iq = 0; iq < TQ; iq++) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
              const int row_pos = base_row + iq * kU;
              const int col_pos = base_col + ik * kU;

              mfrag_t mfrag;
              mtile_t::NAXFrag_t::load_safe(
                  mfrag,
                  mask,
                  int64_t(mask_params->M_strides[2]),
                  Int<1>{},
                  params->qL,
                  params->kL,
                  row_pos,
                  col_pos);

              thread auto& fg = Stile.frag_at(iq, ik);

              STEEL_PRAGMA_UNROLL
              for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
                if constexpr (is_bool) {
                  fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
                } else {
                  fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
                }
              }
            }
          }
        }
      }
    }

    // Do softmax

    // Temp variables
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    // Row Sum
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<SumOp>(sum_score);

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // Do O = P @ V
    if constexpr (NAXFrag_::kPacking == 2) {
      // BaseNAXFrag path: (16,32,16) descriptor produces 2 B frags per mma.
      // Iterate id in steps of 2 to match the fused-pair mma overload.
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id += 2) {
          if constexpr (BD == 128) {
            if (id == 4) {
              threadgroup_barrier(mem_flags::mem_none);
            }
          }

          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            NAXTile<T, 1, 2> Vtile;

            const int V_load_off = ik * kU * int(params->V_strides[2]) + id * kU;

            if (!align_K && is_last_k) {
              Vtile.load_rows(
                  V + V_load_off,
                  int(params->V_strides[2]),
                  lim_rows_k - ik * kU);
            } else {
              Vtile.load(V + V_load_off, int(params->V_strides[2]));
            }

            otile_t::NAXFrag_t::mma(
                Otile.frag_at(iq, id),
                Otile.frag_at(iq, id + 1),
                Stile.frag_at(iq, ik),
                metal::false_type{},
                Vtile.frag_at(0, 0),
                Vtile.frag_at(0, 1),
                metal::false_type{});
          }
        }
      }
    } else {
      // NAXFrag32 path: (32,32,32) descriptor produces 1 frag per mma.
      // Iterate id in steps of 1; Vtile is 1x1, V loads kFragRows rows at a time.
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          if constexpr (BD == 128) {
            if (id == 2) {
              threadgroup_barrier(mem_flags::mem_none);
            }
          }

          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            NAXTile<T, 1, 1, NAXFrag_> Vtile;

            const int V_load_off = ik * kU * int(params->V_strides[2]) + id * kU;

            if (!align_K && is_last_k) {
              Vtile.template load_rows<Role::Right, false>(
                  V + V_load_off,
                  int(params->V_strides[2]),
                  lim_rows_k - ik * kU,
                  (threadgroup T*)sg_scratch);
            } else {
              Vtile.template load<Role::Right, false>(
                  V + V_load_off,
                  int(params->V_strides[2]),
                  (threadgroup T*)sg_scratch);
            }

            otile_t::NAXFrag_t::mma(
                Otile.frag_at(iq, id),
                Stile.frag_at(iq, ik),
                metal::false_type{},
                Vtile.frag_at(0, 0),
                metal::false_type{});
          }
        }
      }
    }

    // Prepare for next iteration
    K += BK * int(params->K_strides[2]);
    V += BK * int(params->V_strides[2]);
  }

  // Normalize output

  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  // Store results
  O += tm * int(params->O_strides[2]);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;

    Otile.store_rows(O, int(params->O_strides[2]), lim_rows_q, sg_scratch);
  } else {
    Otile.store(O, int(params->O_strides[2]), sg_scratch);
  }
}
