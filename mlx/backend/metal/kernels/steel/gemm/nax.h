// Copyright © 2025 Apple Inc.

#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// NAX Steel with new tiles
///////////////////////////////////////////////////////////////////////////////

struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;

  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST short kElemRows = 2;
  STEEL_CONST short kElemCols = 4;

  STEEL_CONST short kElemRowsJump = 8;

  // Number of frags produced per MMA along N. The (16, 32, 16) descriptor
  // produces a 16x32 output = 2 frags wide. tile_matmad_nax dispatches on this.
  STEEL_CONST short kPacking = 2;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
    return short2{fn, fm};
  }

  METAL_FUNC static short2 get_coord(short idx) {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3)) + (idx >> 2) * 8;
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4 + idx % 4;
    return short2{fn, fm};
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        }
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j)]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] =
                static_cast<T>(src[r * str_x + (c + j) * str_y]);
          }
        }

      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((r < lx) && ((c + j) < ly)) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_rows(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + (c + j) * str_y] =
                static_cast<U>(src[i * kElemCols + j]);
          }
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_safe(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if (r < lx && (c + j) < ly) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename StartX,
      typename StopX,
      typename StartY,
      typename StopY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_slice(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      StartX start_x,
      StopX stop_x,
      StartY start_y,
      StopY stop_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();

    const_for_loop<0, kElemRows, 1>([&](auto idx_row) {
      const auto r = off_x + idx_row * Int<kElemRowsJump>{};
      if (r >= stop_x - sc.y || r < start_x - sc.y) {
        return;
      }

      const_for_loop<0, kElemCols, 1>([&](auto idx_col) {
        const auto c = off_y + idx_col;
        if (c >= stop_y - sc.x || c < start_y - sc.x) {
          return;
        }

        const auto src_idx = idx_row * Int<kElemCols>{} + idx_col;
        dst[(r + sc.y) * str_x + (c + sc.x) * str_y] =
            static_cast<U>(src[src_idx]);
      });
    });
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals,
      thread T* reduced_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      T thr_reduce = Op::apply(
          Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
          Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));

      T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

      T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

      reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_bin_op(
      thread dtype_frag_t<T>& inp_vals,
      thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cn0,
      thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0,
      const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        decltype(ct_a),
        decltype(ct_b),
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = A[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = Bn0[i];
      ct_b[kElemsPerFrag + i] = Bn1[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cn0[i];
      ct_c[kElemsPerFrag + i] = Cn1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cn0[i] = ct_c[i];
      Cn1[i] = ct_c[kElemsPerFrag + i];
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cm0,
      thread dtype_frag_t<CType>& Cm1,
      const thread dtype_frag_t<AType>& Am0,
      const thread dtype_frag_t<AType>& Am1,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& B,
      metal::bool_constant<transpose_b>) {
    // Create Matmul descriptor
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        decltype(ct_a),
        decltype(ct_b),
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = Am0[i];
      ct_a[kElemsPerFrag + i] = Am1[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = B[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cm0[i];
      ct_c[kElemsPerFrag + i] = Cm1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cm0[i] = ct_c[i];
      Cm1[i] = ct_c[kElemsPerFrag + i];
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// NAXFrag32 — 32x32 frag, single-frag MMA, layout-agnostic I/O via
// metal::tensor_inline. Used on g16 (M4 Pro), where matmul2d is only correct
// for the (32, 32, 32) descriptor.
//
// I/O strategy (chosen during Task 1.5 of the NAX g16 fix):
//   The Metal SDK exposes only `tensor_inline`, `tensor_handle`, and
//   `tensor_offset` access kinds — no `tensor_padded` or strided/bounded
//   variants. Cooperative tensors have no masked-load entry points. So
//   "Strategy A" (route safe/rows variants through an MPP-bounded view) is
//   infeasible on this SDK. We use "Strategy B": the safe/rows variants
//   stage through a 32x32 threadgroup scratch buffer with a cooperative
//   bounds-checked copy, then route through the contiguous load/store.
//
//   Metal forbids declaring `threadgroup T buf[N]` inside non-kernel
//   device functions. So load_safe/load_rows/store_safe/store_rows take an
//   extra `threadgroup T* scratch` parameter — callers must allocate
//   `threadgroup T scratch[32 * 32]` at kernel scope and pass it in.
//
//   This is a real signature divergence from BaseNAXFrag. NAXTile's
//   wrappers (load_rows, load_safe, store_rows, store_safe, store_slice)
//   currently follow BaseNAXFrag's convention; Phase 2 / Task 2.2 must
//   redesign that dispatch before any NAXTile<..., NAXFrag32> instantiation
//   compiles. See docs/superpowers/specs/2026-04-27-nax-g16-fix-plan.md.
///////////////////////////////////////////////////////////////////////////////

struct NAXFrag32 {
  STEEL_CONST short kFragRows = 32;
  STEEL_CONST short kFragCols = 32;

  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;  // 32

  STEEL_CONST short kElemRows = 8;
  STEEL_CONST short kElemCols = 4;

  // NAXFrag32's per-thread row layout is non-uniform (dr_table =
  // {0,1,8,9,16,17,24,25}) — there is no valid constant stride. We set this
  // to 1 only because NAXTile propagates the field as kFragRowsJump; do NOT
  // use it for stride arithmetic. Callers that need per-element offsets
  // must call dr_dc() explicitly. See Phase 5 (SDPA) of
  // docs/superpowers/specs/2026-04-27-nax-g16-fix-plan.md for the
  // attention path that currently uses this stride and will need rework.
  STEEL_CONST short kElemRowsJump = 1;

  // One MMA produces one full 32x32 frag, i.e. unpacked.
  STEEL_CONST short kPacking = 1;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "NAXFrag32 shape is not consistent with its size");

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  // Per-thread base coordinate in the 32x32 frag. Element i lives at
  //   (b + dc[i % 8], a + dr[i / 8])  // {col, row} — matches short2 / dr_dc()
  // with
  //   a (row base, 8 values in {0..7}) = ((lane & 2) >> 1) | ((lane & 4) >> 1) | ((lane & 16) >> 2)
  //   b (col base, 4 values in {0,2,4,6}) = ((lane & 1) << 1) | ((lane & 8) >> 1)
  //   dr = {0, 8, 16, 24}                          // 4 row offsets
  //   dc = {0, 1, 8, 9, 16, 17, 24, 25}            // 8 col offsets
  // Source: tools/probe_nax_descriptor.py layout-dump variant on g16s.
  // (Note: spec's dr/dc labels were swapped vs the actual cooperative-tensor layout — corrected in commit at Task 1.6b. The numerical values are unchanged.)
  METAL_FUNC static short2 get_coord() {
    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    const short a = ((lane & 2) >> 1) | ((lane & 4) >> 1) | ((lane & 16) >> 2);  // 8 values — row base
    const short b = ((lane & 1) << 1) | ((lane & 8) >> 1);                       // 4 values — col base
    return short2{b, a};  // short2 is {x = col, y = row}
  }

  METAL_FUNC static short2 dr_dc(short i) {
    constexpr short dr_table[4] = {0, 8, 16, 24};                  // 4 row offsets
    constexpr short dc_table[8] = {0, 1, 8, 9, 16, 17, 24, 25};   // 8 col offsets
    return short2{dc_table[i % 8], dr_table[i / 8]};               // {col_off, row_off}
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& C,
      const thread dtype_frag_t<AType>& A,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& B,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        32,
        32,
        32,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        decltype(ct_a),
        decltype(ct_b),
        CType>();

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = A[i];
      ct_b[i] = B[i];
      ct_c[i] = C[i];
    }

    gemm_op.run(ct_a, ct_b, ct_c);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      C[i] = ct_c[i];
    }
  }

  // Load the entire 32x32 frag from a contiguous threadgroup region.
  // Materializes a tensor_inline view over the threadgroup pointer and uses
  // the cooperative-tensor load that the MPP runtime handles. Avoids
  // hard-coding (dr, dc) per-thread offsets — same code path as
  // tools/probe_nax_descriptor.py's tg-staged variant, which empirically
  // verifies max|err| = 0 for the (32, 32, 32) descriptor on g16s.
  template <typename T, typename U>
  METAL_FUNC static void load(
      thread dtype_frag_t<T>& dst,
      const threadgroup U* src,
      const short ld) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, 32,
        /*transpose_left=*/false,
        /*transpose_right=*/false,
        /*relaxed_precision=*/true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct = op.template get_left_input_cooperative_tensor<T, T, T>();

    metal::dextents<int32_t, 2> ext(32, ld);
    metal::tensor<threadgroup const U, metal::dextents<int32_t, 2>, metal::tensor_inline>
        view((threadgroup const U*)src, ext);
    ct.load(view);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      dst[i] = static_cast<T>(ct[i]);
    }
  }

  template <typename T, typename U>
  METAL_FUNC static void store(
      const thread dtype_frag_t<T>& src,
      threadgroup U* dst,
      const short ld) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, 32,
        /*transpose_left=*/false,
        /*transpose_right=*/false,
        /*relaxed_precision=*/true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct = op.template get_destination_cooperative_tensor<
        decltype(op.template get_left_input_cooperative_tensor<T, T, T>()),
        decltype(op.template get_right_input_cooperative_tensor<T, T, T>()),
        T>();

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct[i] = static_cast<T>(src[i]);
    }

    metal::dextents<int32_t, 2> ext(32, ld);
    metal::tensor<threadgroup U, metal::dextents<int32_t, 2>, metal::tensor_inline>
        view((threadgroup U*)dst, ext);
    ct.store(view);
  }

  // Load a 32x32 frag from device memory, zero-fill out-of-bounds elements.
  // Strategy B: caller provides a threadgroup scratch buffer (32*32 elements).
  // Each lane fills its slice of the staging buffer with bounds-checked copies
  // from device, then calls the cooperative-tensor load on the staged data.
  // Avoids depending on MPP partial-tile support that does not exist on this
  // SDK (only tensor_inline/tensor_handle/tensor_offset exist — no padded or
  // strided bounded-view types); per-thread layout still owned by the
  // cooperative tensor inside load().
  // Note: threadgroup scratch cannot be declared inside a device function in
  // Metal; callers must allocate `threadgroup T scratch[32*32]` and pass it.
  template <typename T, typename U>
  METAL_FUNC static void load_safe(
      thread dtype_frag_t<T>& dst,
      const device U* src,
      const int ld,
      const short row_lim,
      const short col_lim,
      threadgroup T* scratch) {
    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; ++i) {
      const short flat = lane * kElemsPerFrag + i;
      const short r = flat / 32;
      const short c = flat % 32;
      scratch[r * 32 + c] =
          (r < row_lim && c < col_lim) ? static_cast<T>(src[r * ld + c]) : T(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load(dst, scratch, /*ld=*/short(32));
  }

  // Load a 32x32 frag from device memory with a row bound only; columns are
  // always treated as full 32. Zero-fills rows at or beyond row_lim.
  // Caller must allocate `threadgroup T scratch[32*32]` and pass it.
  template <typename T, typename U>
  METAL_FUNC static void load_rows(
      thread dtype_frag_t<T>& dst,
      const device U* src,
      const int ld,
      const short row_lim,
      threadgroup T* scratch) {
    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; ++i) {
      const short flat = lane * kElemsPerFrag + i;
      const short r = flat / 32;
      const short c = flat % 32;
      scratch[r * 32 + c] =
          (r < row_lim) ? static_cast<T>(src[r * ld + c]) : T(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load(dst, scratch, /*ld=*/short(32));
  }

  // Store a 32x32 frag to device memory, skipping elements out-of-bounds.
  // Caller must allocate `threadgroup T scratch[32*32]` and pass it.
  template <typename T, typename U>
  METAL_FUNC static void store_safe(
      const thread dtype_frag_t<T>& src,
      device U* dst,
      const int ld,
      const short row_lim,
      const short col_lim,
      threadgroup T* scratch) {
    store(src, scratch, /*ld=*/short(32));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; ++i) {
      const short flat = lane * kElemsPerFrag + i;
      const short r = flat / 32;
      const short c = flat % 32;
      if (r < row_lim && c < col_lim) {
        dst[r * ld + c] = static_cast<U>(scratch[r * 32 + c]);
      }
    }
  }

  // Store a 32x32 frag to device memory with a row bound only; skips rows at
  // or beyond row_lim. Columns always written for valid rows.
  // Caller must allocate `threadgroup T scratch[32*32]` and pass it.
  template <typename T, typename U>
  METAL_FUNC static void store_rows(
      const thread dtype_frag_t<T>& src,
      device U* dst,
      const int ld,
      const short row_lim,
      threadgroup T* scratch) {
    store(src, scratch, /*ld=*/short(32));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; ++i) {
      const short flat = lane * kElemsPerFrag + i;
      const short r = flat / 32;
      const short c = flat % 32;
      if (r < row_lim) {
        dst[r * ld + c] = static_cast<U>(scratch[r * 32 + c]);
      }
    }
  }

  // store_slice is only used by the SDPA path (Phase 5). Leaving it out for
  // now so callers who reach for it on g16s fail at compile time rather than
  // silently produce wrong output. Variadic over all arguments so any caller
  // shape is intercepted by the static_assert (rather than a more confusing
  // template-deduction error earlier in the chain).
  // Note: Metal forbids forwarding references (Args&&) without explicit address
  // space qualifiers, so we use value-parameter variadic (Args...) here.
  template <typename T, typename... Args>
  METAL_FUNC static void store_slice(
      const thread dtype_frag_t<T>&,
      Args...) {
    static_assert(
        sizeof(T) < 0,
        "NAXFrag32::store_slice not yet implemented; Phase 5 SDPA needs this");
  }
};

template <
    typename T,
    short kTileRows_,
    short kTileCols_,
    class NAXFrag_ = BaseNAXFrag>
struct NAXTile {
  using NAXFrag_t = NAXFrag_;
  using elem_type = T;

  STEEL_CONST short kFragRows = NAXFrag_t::kFragRows;
  STEEL_CONST short kFragCols = NAXFrag_t::kFragCols;
  STEEL_CONST short kElemsPerFrag = NAXFrag_t::kElemsPerFrag;

  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;

  STEEL_CONST short kRows = kTileRows * kFragRows;
  STEEL_CONST short kCols = kTileCols * kFragCols;

  STEEL_CONST short kNumFrags = kTileRows * kTileCols;
  STEEL_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;

  STEEL_CONST short kFragThrRows = NAXFrag_t::kElemRows;
  STEEL_CONST short kFragThrCols = NAXFrag_t::kElemCols;
  STEEL_CONST short kFragRowsJump = NAXFrag_t::kElemRowsJump;

  STEEL_CONST short kRowsPerThread = kTileRows * NAXFrag_t::kElemRows;
  STEEL_CONST short kColsPerThread = kTileCols * NAXFrag_t::kElemCols;

  typedef typename NAXFrag_t::template dtype_frag_t<T> frag_type;

  frag_type val_frags[kNumFrags]; // = {frag_type(0)};

  METAL_FUNC NAXTile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr thread frag_type& frag_at() {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr const thread frag_type& frag_at() const {
    return val_frags[i * kTileCols + j];
  }

  template <bool transpose>
  METAL_FUNC constexpr thread frag_type&
  frag_at(const short i, const short j, metal::bool_constant<transpose>) {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }

  template <bool transpose>
  METAL_FUNC constexpr const thread frag_type&
  frag_at(const short i, const short j, metal::bool_constant<transpose>) const {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }

  template <int i, int j, bool transpose>
  METAL_FUNC constexpr thread frag_type& frag_at() {
    if constexpr (transpose) {
      return frag_at<j, i>();
    } else {
      return frag_at<i, j>();
    }
  }

  template <int i, int j, bool transpose>
  METAL_FUNC constexpr const thread frag_type& frag_at() const {
    if constexpr (transpose) {
      return frag_at<j, i>();
    } else {
      return frag_at<i, j>();
    }
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  METAL_FUNC const thread elem_type* elems() const {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread metal::vec<T, kRowsPerThread>& vals) const {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread metal::vec<T, kRowsPerThread>& vals) {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void store(threadgroup U* dst) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void
  load_rows(const device U* src, const int ld, const short n_rows) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        // TODO(nax-g16-fix): NAXFrag32's load_safe/load_rows/store_safe/store_rows
        // require an extra `threadgroup T* scratch` parameter (see NAXFrag32
        // struct header). Phase 2 / Task 2.2 must redesign this dispatch.
        NAXFrag_t::load_rows(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            n_rows,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void
  load_safe(const device U* src, const int ld, const short2 src_tile_dims) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        // TODO(nax-g16-fix): NAXFrag32's load_safe/load_rows/store_safe/store_rows
        // require an extra `threadgroup T* scratch` parameter (see NAXFrag32
        // struct header). Phase 2 / Task 2.2 must redesign this dispatch.
        NAXFrag_t::load_safe(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            src_tile_dims.y,
            src_tile_dims.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_rows(device U* dst, const int ld, const short n_rows)
      const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        // TODO(nax-g16-fix): NAXFrag32's load_safe/load_rows/store_safe/store_rows
        // require an extra `threadgroup T* scratch` parameter (see NAXFrag32
        // struct header). Phase 2 / Task 2.2 must redesign this dispatch.
        NAXFrag_t::store_rows(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            n_rows,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void
  store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        // TODO(nax-g16-fix): NAXFrag32's load_safe/load_rows/store_safe/store_rows
        // require an extra `threadgroup T* scratch` parameter (see NAXFrag32
        // struct header). Phase 2 / Task 2.2 must redesign this dispatch.
        NAXFrag_t::store_safe(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_slice(
      device U* dst,
      const int ld,
      const short2 start,
      const short2 stop) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        // TODO(nax-g16-fix): NAXFrag32's load_safe/load_rows/store_safe/store_rows
        // require an extra `threadgroup T* scratch` parameter (see NAXFrag32
        // struct header). Phase 5 must redesign this dispatch.
        NAXFrag_t::store_slice(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            start.y,
            stop.y,
            start.x,
            stop.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }
};

template <
    class CTile,
    class ATile,
    class BTile,
    bool transpose_a,
    bool transpose_b>
METAL_FUNC void tile_matmad_nax(
    thread CTile& C,
    thread ATile& A,
    metal::bool_constant<transpose_a>,
    thread BTile& B,
    metal::bool_constant<transpose_b>) {
  // Static checks
  constexpr short TMa = transpose_a ? ATile::kTileCols : ATile::kTileRows;
  constexpr short TM = CTile::kTileRows;
  static_assert(TMa == TM, "MXU tile matmul: M dimensions do not match");

  constexpr short TNb = transpose_b ? BTile::kTileRows : BTile::kTileCols;
  constexpr short TN = CTile::kTileCols;
  static_assert(TNb == TN, "MXU tile matmul: N dimensions do not match");

  constexpr short TKa = transpose_a ? ATile::kTileRows : ATile::kTileCols;
  constexpr short TK = transpose_b ? BTile::kTileCols : BTile::kTileRows;
  static_assert(TKa == TK, "MXU tile matmul: K dimensions do not match");

  constexpr auto ta = metal::bool_constant<transpose_a>{};
  constexpr auto tb = metal::bool_constant<transpose_b>{};

  if constexpr (TN == 1 && TM % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; mm += 2) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; ++nn) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm + 1, nn),
              A.frag_at(mm, kk, ta),
              A.frag_at(mm + 1, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  } else if constexpr (TN % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; ++mm) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; nn += 2) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm, nn + 1),
              A.frag_at(mm, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              B.frag_at(kk, nn + 1, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  }
}

} // namespace steel
} // namespace mlx
