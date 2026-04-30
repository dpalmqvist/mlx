"""
Diagnostic probe for NAXFrag32 transpose correctness on g16.

Exercises mma with all four (transpose_a, transpose_b) combinations using two
different load strategies, to isolate which layer of NAXFrag32 is responsible
for the transposed-matmul regression introduced in Phase 2.

Three plausible fault layers:
  Layer 1 (load):  NAXFrag32::load hardcodes transpose_left=false / transpose_right=false.
                   If MPP's per-thread element layout depends on (role, transpose),
                   the loaded register values may be in wrong positions for mma.
  Layer 2 (mma):   NAXFrag32::mma's element-routing has a transpose bug.
  Layer 3 (desc):  The (32,32,32) descriptor on g16 has unexpected transpose semantics.

Sub-probe 1 — test_mma_all_transposes:
  Uses the existing NAXFrag32::load (hardcodes false/false). Loads A and B
  through threadgroup staging, then calls mma<ta, tb>. Compares against numpy.
  Expected on buggy code: (F,F) passes, transposed cases fail.

Sub-probe 2 — test_mma_with_role_aware_load:
  Uses a new naxfrag32_load_role_aware<ta, tb, is_right> helper that passes the
  runtime transpose flags into the load descriptor and dispatches on role
  (left vs right). If sub-probe 2 passes all 4 cases → Layer 1 confirmed.
  If sub-probe 2 still fails → Layer 2 or 3.

NOTE: This probe is DIAGNOSTIC, not RED/GREEN gated. Both sub-probes run
regardless of results. The script exits 0 always. Results are summarised
at the top of __main__.

Results summary (recorded after running on g16 / M4 Pro, 2026-04-30):
  Sub-probe 1 — test_mma_all_transposes (existing NAXFrag32::load):
    (F,F): OK   max|err|=0.0
    (F,T): FAIL max|err|=103.0
    (T,F): FAIL max|err|=102.0
    (T,T): FAIL max|err|=117.0

  Sub-probe 2 — test_mma_with_role_aware_load (role-aware descriptor):
    (F,F): OK   max|err|=0.0
    (F,T): OK   max|err|=0.0
    (T,F): OK   max|err|=0.0
    (T,T): OK   max|err|=0.0

  Layer determination: LAYER 1 CONFIRMED.
    Sub-probe 2 passes all four cases once the load descriptor passes the
    correct transpose flags and role (left vs right input). The bug is in
    NAXFrag32::load, which hardcodes transpose_left=false / transpose_right=false
    and always uses get_left_input_cooperative_tensor, regardless of which
    operand is being loaded. Task 2a (fix NAXFrag32::load) is the correct next step.

Pattern: follows probe_naxfrag32_store_slice.py (_INT_HELPER + _NAX_STRUCT_MARKER split).
IF YOU CHANGE NAXFrag32 IN nax.h, UPDATE THE HEADER COPY BELOW.
"""
import sys

import mlx.core as mx
import numpy as np

# fmt: off
HEADER = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

// ----------------------------------------------------------------------------
// NAXFrag32 — INLINED COPY from mlx/backend/metal/kernels/steel/gemm/nax.h.
// mx.fast.metal_kernel does not expose the source-tree include path, so this
// probe inlines the struct rather than #include'ing the upstream header.
// IF YOU CHANGE NAXFrag32 IN nax.h, UPDATE THIS COPY.
// ----------------------------------------------------------------------------

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

  STEEL_CONST short kElemRows = 4;
  STEEL_CONST short kElemCols = 8;

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
    // MPP cooperative_tensor::load requires the view's element type to match
    // the cooperative tensor's element type exactly (no const qualifier).
    // Casting away const internally preserves the const-correct API.
    metal::tensor<threadgroup U, metal::dextents<int32_t, 2>, metal::tensor_inline>
        view(const_cast<threadgroup U*>(src), ext);
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

  // Device-memory variant of the contiguous load. Cooperative tensor is
  // parameterized on U so MPP's tensor_inline view (also U) accepts
  // ct.load(view). Loaded values are static_cast<T>(ct[i]) into the thread
  // frag, mirroring the threadgroup overload's pattern. Validated by
  // tools/probe_nax_frag32.py::test_mma_via_dv_load_store.
  template <typename T, typename U>
  METAL_FUNC static void load(
      thread dtype_frag_t<T>& dst,
      const device U* src,
      const int ld) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, 32,
        /*transpose_left=*/false,
        /*transpose_right=*/false,
        /*relaxed_precision=*/true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct = op.template get_left_input_cooperative_tensor<U, U, U>();

    metal::dextents<int32_t, 2> ext(32, ld);
    metal::tensor<device U, metal::dextents<int32_t, 2>, metal::tensor_inline>
        view(const_cast<device U*>(src), ext);
    ct.load(view);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      dst[i] = static_cast<T>(ct[i]);
    }
  }

  // Device-memory variant of the contiguous store. Same body as the
  // threadgroup overload with the address space swapped, plus support for
  // T != U: the destination cooperative tensor is parameterized on the
  // device pointer type U so MPP's tensor_inline view (also U) accepts
  // ct.store(view). Frag elements are cast static_cast<U>(src[i]) before
  // being written into the cooperative tensor.
  // Validated by tools/probe_nax_frag32.py::test_mma_via_dv_load_store and
  // ::test_store_device_mixed_precision.
  template <typename T, typename U>
  METAL_FUNC static void store(
      const thread dtype_frag_t<T>& src,
      device U* dst,
      const int ld) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, 32,
        /*transpose_left=*/false,
        /*transpose_right=*/false,
        /*relaxed_precision=*/true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct = op.template get_destination_cooperative_tensor<
        decltype(op.template get_left_input_cooperative_tensor<U, U, U>()),
        decltype(op.template get_right_input_cooperative_tensor<U, U, U>()),
        U>();

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct[i] = static_cast<U>(src[i]);
    }

    metal::dextents<int32_t, 2> ext(32, ld);
    metal::tensor<device U, metal::dextents<int32_t, 2>, metal::tensor_inline>
        view((device U*)dst, ext);
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

  // Row reduction: combine all 32 column elements in each of the thread's
  // 4 owned rows. Each row is shared by 4 lanes (those with the same
  // row_base, varying col_base). Lane bits 0 and 3 vary across these 4
  // lanes; bits 1, 2, 4 are constant. Within each thread, 8 col elements
  // per row at indices [row*8 .. row*8+7].
  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals,
      thread T* reduced_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      // Within-thread reduction: combine the 8 col elements for this row.
      T thr_reduce = inp_vals[i * kElemCols + 0];
      STEEL_PRAGMA_UNROLL
      for (short j = 1; j < kElemCols; j++) {
        thr_reduce = Op::apply(thr_reduce, inp_vals[i * kElemCols + j]);
      }

      // Cross-lane: combine 4 lanes sharing this row_base. Lanes differ
      // only in bits 0 and 3 of the simd lane id.
      T shuf1 = simd_shuffle_xor(thr_reduce, ushort(1));
      thr_reduce = Op::apply(thr_reduce, shuf1);
      T shuf8 = simd_shuffle_xor(thr_reduce, ushort(8));
      thr_reduce = Op::apply(thr_reduce, shuf8);

      reduced_vals[i] = Op::apply(reduced_vals[i], thr_reduce);
    }
  }

  // Apply a binary op between each row's elements and a per-row scalar.
  // dtype_frag_t element ordering: row i lives at indices [i*8 .. i*8+7].
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

  // Store a 32x32 frag to device memory, writing only elements inside the
  // rectangular slice [start_x, stop_x) x [start_y, stop_y). Out-of-bounds
  // device positions are not touched. Caller must allocate
  // `threadgroup T scratch[32*32]` and pass it. The signature mirrors
  // BaseNAXFrag::store_slice so NAXTile can dispatch with a single call site
  // (gated on kPacking).
  template <typename T, typename U, typename StrX, typename StrY,
            typename StartX, typename StopX, typename StartY, typename StopY,
            typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static void store_slice(
      const thread dtype_frag_t<T>& src,
      device U* dst,
      StrX str_x,
      StrY str_y,
      StartX start_x,
      StopX stop_x,
      StartY start_y,
      StopY stop_y,
      threadgroup T* scratch,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    // Stage the 32x32 frag to threadgroup scratch via the contiguous store.
    store(src, scratch, /*ld=*/short(32));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread bounded copy: each lane owns kElemsPerFrag (=32) elements.
    // Address them by flat index: flat = lane * 32 + i, r = flat / 32,
    // c = flat % 32. The slice bounds are expressed in NAXTile coordinates:
    // off_x/off_y are the frag's top-left in tile coords (set by NAXTile's
    // multi-frag dispatch), so a frag-local (r, c) maps to tile-local
    // (r + off_x, c + off_y).
    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; ++i) {
      const short flat = lane * kElemsPerFrag + i;
      const short r = flat / 32;
      const short c = flat % 32;
      const short tile_r = r + short(off_x);
      const short tile_c = c + short(off_y);
      if (tile_r >= short(start_x) && tile_r < short(stop_x) &&
          tile_c >= short(start_y) && tile_c < short(stop_y)) {
        dst[r * str_x + c * str_y] = static_cast<U>(scratch[r * 32 + c]);
      }
    }
  }
};

"""
# fmt: on

# ---------------------------------------------------------------------------
# _INT_HELPER: must appear before NAXFrag32 (store_slice uses Int<0> default args).
# _NAX_STRUCT_MARKER: split point used to insert _INT_HELPER before the struct.
# ---------------------------------------------------------------------------
_INT_HELPER = """
// Minimal Int<N> helper (mirrors steel/utils.h Int<N>).
template <short N>
struct Int {
  static constant constexpr const short value = N;
  METAL_FUNC constexpr operator short() const { return N; }
};
"""

# ---------------------------------------------------------------------------
# Role-aware load helper: uses the correct (transpose_left, transpose_right)
# in the descriptor AND dispatches on is_right_input to call get_left vs
# get_right cooperative tensor. This is the "correct" load for sub-probe 2.
# ---------------------------------------------------------------------------
_ROLE_AWARE_LOAD_HELPER = """
// naxfrag32_load_role_aware: load a 32x32 frag from threadgroup memory with
// the correct descriptor transpose flags and input role (left vs right).
// This is the corrected load that sub-probe 2 uses to test Layer 1 isolation.
template <bool transpose_left, bool transpose_right, bool is_right_input,
          typename T, typename U>
METAL_FUNC void naxfrag32_load_role_aware(
    thread NAXFrag32::dtype_frag_t<T>& dst,
    const threadgroup U* src,
    const short ld) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      32, 32, 32,
      transpose_left, transpose_right,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  // Use if constexpr to dispatch on is_right_input — Metal may not allow
  // ternary with auto type deduction across both branches.
  if constexpr (is_right_input) {
    auto ct = op.template get_right_input_cooperative_tensor<T, T, T>();
    metal::dextents<int32_t, 2> ext(32, ld);
    metal::tensor<threadgroup U, metal::dextents<int32_t, 2>, metal::tensor_inline>
        view(const_cast<threadgroup U*>(src), ext);
    ct.load(view);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < NAXFrag32::kElemsPerFrag; i++) {
      dst[i] = static_cast<T>(ct[i]);
    }
  } else {
    auto ct = op.template get_left_input_cooperative_tensor<T, T, T>();
    metal::dextents<int32_t, 2> ext(32, ld);
    metal::tensor<threadgroup U, metal::dextents<int32_t, 2>, metal::tensor_inline>
        view(const_cast<threadgroup U*>(src), ext);
    ct.load(view);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < NAXFrag32::kElemsPerFrag; i++) {
      dst[i] = static_cast<T>(ct[i]);
    }
  }
}
"""

_NAX_STRUCT_MARKER = "///////////////////////////////////////////////////////////////////////////////\n// NAXFrag32"
_header_pre, _header_post = HEADER.split(_NAX_STRUCT_MARKER, 1)
# Insert _INT_HELPER before the NAXFrag32 struct, then append the role-aware
# helper after the closing brace of the struct (i.e. after HEADER ends).
FULL_HEADER = (
    _header_pre
    + _INT_HELPER
    + "\n"
    + _NAX_STRUCT_MARKER
    + _header_post
    + _ROLE_AWARE_LOAD_HELPER
)

# ---------------------------------------------------------------------------
# Sub-probe 1: test_mma_all_transposes
# Uses existing NAXFrag32::load (hardcodes transpose_left=false/transpose_right=false).
# Expected: (F,F) passes; transposed cases fail on buggy code.
# ---------------------------------------------------------------------------

_KERNEL_SRC_EXISTING_LOAD = """
using Frag = NAXFrag32;

threadgroup float A_tg[32 * 32];
threadgroup float B_tg[32 * 32];
threadgroup float C_tg[32 * 32];

const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
STEEL_PRAGMA_UNROLL
for (short i = 0; i < 32; ++i) {{
    const short flat = lane * 32 + i;
    A_tg[flat] = A[flat];
    B_tg[flat] = B[flat];
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;
Frag::load(a_frag, (const threadgroup float*)A_tg, (short)32);
Frag::load(b_frag, (const threadgroup float*)B_tg, (short)32);

STEEL_PRAGMA_UNROLL
for (short i = 0; i < 32; ++i) {{
    c_frag[i] = 0.0f;
}}

Frag::mma(c_frag,
          a_frag, metal::bool_constant<{ta}>{{}},
          b_frag, metal::bool_constant<{tb}>{{}});

Frag::store(c_frag, (threadgroup float*)C_tg, (short)32);
threadgroup_barrier(mem_flags::mem_threadgroup);

STEEL_PRAGMA_UNROLL
for (short i = 0; i < 32; ++i) {{
    const short flat = lane * 32 + i;
    C[flat] = C_tg[flat];
}}
"""

# ---------------------------------------------------------------------------
# Sub-probe 2: test_mma_with_role_aware_load
# Uses naxfrag32_load_role_aware<ta, tb, is_right>(...) which passes the
# correct transpose flags and role to the load descriptor.
# If this passes all 4 cases → Layer 1 is the bug.
# ---------------------------------------------------------------------------

_KERNEL_SRC_ROLE_AWARE_LOAD = """
using Frag = NAXFrag32;

threadgroup float A_tg[32 * 32];
threadgroup float B_tg[32 * 32];
threadgroup float C_tg[32 * 32];

const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
STEEL_PRAGMA_UNROLL
for (short i = 0; i < 32; ++i) {{
    const short flat = lane * 32 + i;
    A_tg[flat] = A[flat];
    B_tg[flat] = B[flat];
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;
naxfrag32_load_role_aware<{ta}, {tb}, false>(
    a_frag, (const threadgroup float*)A_tg, (short)32);
naxfrag32_load_role_aware<{ta}, {tb}, true>(
    b_frag, (const threadgroup float*)B_tg, (short)32);

STEEL_PRAGMA_UNROLL
for (short i = 0; i < 32; ++i) {{
    c_frag[i] = 0.0f;
}}

Frag::mma(c_frag,
          a_frag, metal::bool_constant<{ta}>{{}},
          b_frag, metal::bool_constant<{tb}>{{}});

Frag::store(c_frag, (threadgroup float*)C_tg, (short)32);
threadgroup_barrier(mem_flags::mem_threadgroup);

STEEL_PRAGMA_UNROLL
for (short i = 0; i < 32; ++i) {{
    const short flat = lane * 32 + i;
    C[flat] = C_tg[flat];
}}
"""

# Four (transpose_a, transpose_b) cases.
_TRANSPOSE_CASES = [
    (False, False, "FF"),
    (False, True,  "FT"),
    (True,  False, "TF"),
    (True,  True,  "TT"),
]


def _metal_bool(b: bool) -> str:
    return "true" if b else "false"


def _numpy_ref(A_np: np.ndarray, B_np: np.ndarray, ta: bool, tb: bool) -> np.ndarray:
    """Compute numpy reference for mma<ta, tb>(A, B).
    mma<ta=false, tb=false>: C += A @ B
    mma<ta=true,  tb=false>: C += A.T @ B
    mma<ta=false, tb=true>:  C += A @ B.T
    mma<ta=true,  tb=true>:  C += A.T @ B.T
    """
    a = A_np.T if ta else A_np
    b = B_np.T if tb else B_np
    return a @ b


def _run_kernel(kernel_src_template: str, ta: bool, tb: bool, name: str) -> np.ndarray:
    src = kernel_src_template.format(
        ta=_metal_bool(ta),
        tb=_metal_bool(tb),
    )
    k = mx.fast.metal_kernel(
        name=name,
        input_names=["A", "B"],
        output_names=["C"],
        source=src,
        header=FULL_HEADER,
    )
    rng = np.random.RandomState(42)
    A_np = rng.randint(-3, 4, (32, 32)).astype(np.float32)
    B_np = rng.randint(-3, 4, (32, 32)).astype(np.float32)
    A = mx.array(A_np)
    B = mx.array(B_np)
    out = k(
        inputs=[A, B],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(32, 32)],
        output_dtypes=[mx.float32],
    )
    mx.eval(out[0])
    result = np.asarray(out[0])
    ref = _numpy_ref(A_np, B_np, ta, tb)
    return result, ref, A_np, B_np


def test_mma_all_transposes():
    """Sub-probe 1: existing NAXFrag32::load (hardcodes false/false in descriptor).
    Expected on buggy code: (F,F) OK, transposed cases FAIL.
    """
    print("Sub-probe 1: test_mma_all_transposes (existing NAXFrag32::load)")
    results = []
    for ta, tb, tag in _TRANSPOSE_CASES:
        name = f"probe_transpose_existing_load_{tag}"
        try:
            result, ref, A_np, B_np = _run_kernel(
                _KERNEL_SRC_EXISTING_LOAD, ta, tb, name
            )
            err = float(np.max(np.abs(result - ref)))
            if err >= 1e-3:
                status = "FAIL"
            else:
                status = "OK"
        except Exception as e:
            err = float("nan")
            status = f"ERROR: {e}"
        label = f"(ta={ta}, tb={tb})"
        print(f"  {status:4s}  {label}  max|err|={err}")
        results.append((ta, tb, status, err))
    return results


def test_mma_with_role_aware_load():
    """Sub-probe 2: role-aware load that passes correct transpose flags and
    dispatches on left vs right input role.
    If all 4 cases pass → Layer 1 confirmed. If fails → Layer 2 or 3.
    """
    print("Sub-probe 2: test_mma_with_role_aware_load (role-aware descriptor)")
    results = []
    for ta, tb, tag in _TRANSPOSE_CASES:
        name = f"probe_transpose_role_aware_{tag}"
        try:
            result, ref, A_np, B_np = _run_kernel(
                _KERNEL_SRC_ROLE_AWARE_LOAD, ta, tb, name
            )
            err = float(np.max(np.abs(result - ref)))
            if err >= 1e-3:
                status = "FAIL"
            else:
                status = "OK"
        except Exception as e:
            err = float("nan")
            status = f"ERROR: {e}"
        label = f"(ta={ta}, tb={tb})"
        print(f"  {status:4s}  {label}  max|err|={err}")
        results.append((ta, tb, status, err))
    return results


if __name__ == "__main__":
    # Results summary (see module docstring for recorded results):
    #
    # Sub-probe 1 — existing NAXFrag32::load (transpose flags hardcoded false/false):
    #   (F,F): OK   max|err|=0.0
    #   (F,T): FAIL max|err|=103.0
    #   (T,F): FAIL max|err|=102.0
    #   (T,T): FAIL max|err|=117.0
    #
    # Sub-probe 2 — role-aware load (correct flags + left/right dispatch):
    #   (F,F): OK   max|err|=0.0
    #   (F,T): OK   max|err|=0.0
    #   (T,F): OK   max|err|=0.0
    #   (T,T): OK   max|err|=0.0
    #
    # Layer determination: LAYER 1 CONFIRMED.
    #   The role-aware load fixes all four transpose cases, proving the bug is in
    #   NAXFrag32::load's hardcoded (false, false) descriptor and its exclusive use
    #   of get_left_input_cooperative_tensor regardless of which operand is loaded.
    #   Next step: Task 2a — fix NAXFrag32::load to accept transpose flags and
    #   a role (left vs right) parameter, matching the approach validated here.
    #
    print("=" * 60)
    r1 = test_mma_all_transposes()
    print()
    r2 = test_mma_with_role_aware_load()
    print("=" * 60)

    # Determine layer from results (for automated summary, does not gate exit).
    sp1_transposed_fail = any(
        s != "OK" for ta, tb, s, _ in r1 if ta or tb
    )
    sp1_ff_ok = all(s == "OK" for ta, tb, s, _ in r1 if not ta and not tb)
    sp2_all_ok = all(s == "OK" for _, _, s, _ in r2)

    if sp1_ff_ok and sp1_transposed_fail and sp2_all_ok:
        print("Layer determination: LAYER 1 CONFIRMED (load descriptor bug).")
        print("Next: Task 2a — fix NAXFrag32::load transpose + role dispatch.")
    elif sp1_ff_ok and sp1_transposed_fail and not sp2_all_ok:
        print("Layer determination: LAYER 2 or 3 (mma element-routing or descriptor).")
        print("Role-aware load does not fix the issue; inspect mma or descriptor.")
    else:
        print("Layer determination: AMBIGUOUS — unexpected result pattern.")
        print("Manual inspection required.")

    # Always exit 0 — this is a diagnostic probe, not a RED/GREEN gate.
    sys.exit(0)
