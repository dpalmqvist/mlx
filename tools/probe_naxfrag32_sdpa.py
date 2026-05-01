"""
Regression probe for NAXFrag32 SDPA call shapes on g16.

Exercises the two matmuls SDPA performs:
- Sub-probe 1: Q @ K^T (transpose_b=true)
  Q is left input, non-transposed.
  K is right input, transposed.
- Sub-probe 2: softmax(QK) @ V (transpose_b=false)
  S (score tile) is left input, non-transposed.
  V is right input, non-transposed.

Both use 32x32 fp32 inputs with small integer values [-3, 3] so
fp32 reductions are exact (no numerical slop).

This probe is a SDPA-flavored mirror of probe_naxfrag32_transpose.
The latter is the comprehensive transpose probe; this one labels
its cases for SDPA so failures are clearer about which production
path is affected.

Results summary (recorded after Phase 6 + Task 2 dedup, g16 / M4 Pro, 2026-05-01):
  Sub-probe 1 — Q @ K^T (transpose_b=true):
    qkT: OK   max|err|=0.0
  Sub-probe 2 — softmax(QK) @ V (transpose_b=false):
    sv:  OK   max|err|=0.0

Pattern: follows probe_naxfrag32_transpose.py (_INT_HELPER + _NAX_STRUCT_MARKER split).
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
  // role and transpose must match the descriptor used by the consumer mma —
  // MPP's cooperative-tensor per-thread layout depends on (role, transpose).
  // Materializes a tensor_inline view over the threadgroup pointer and uses
  // the cooperative-tensor load that the MPP runtime handles. Avoids
  // hard-coding (dr, dc) per-thread offsets — same code path as
  // tools/probe_nax_descriptor.py's tg-staged variant, which empirically
  // verifies max|err| = 0 for the (32, 32, 32) descriptor on g16s.
  template <Role role, bool transpose, typename T, typename U>
  METAL_FUNC static void load(
      thread dtype_frag_t<T>& dst,
      const threadgroup U* src,
      const short ld) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, 32,
        /*transpose_left=*/  (role == Role::Left)  ? transpose : false,
        /*transpose_right=*/ (role == Role::Right) ? transpose : false,
        /*relaxed_precision=*/true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    metal::dextents<int32_t, 2> ext(32, ld);
    // MPP cooperative_tensor::load requires the view's element type to match
    // the cooperative tensor's element type exactly (no const qualifier).
    // Casting away const internally preserves the const-correct API.
    metal::tensor<threadgroup U, metal::dextents<int32_t, 2>, metal::tensor_inline>
        view(const_cast<threadgroup U*>(src), ext);

    // Use if constexpr — Metal may not allow ternary across cooperative_tensor
    // types, since each branch returns a distinct type.
    if constexpr (role == Role::Right) {
      auto ct = op.template get_right_input_cooperative_tensor<T, T, T>();
      ct.load(view);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kElemsPerFrag; i++) dst[i] = static_cast<T>(ct[i]);
    } else {
      auto ct = op.template get_left_input_cooperative_tensor<T, T, T>();
      ct.load(view);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kElemsPerFrag; i++) dst[i] = static_cast<T>(ct[i]);
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

  // Device-memory variant of the contiguous load. role and transpose must
  // match the consumer mma's descriptor (same reason as the threadgroup
  // overload). Cooperative tensor is parameterized on U so MPP's
  // tensor_inline view (also U) accepts ct.load(view). Loaded values are
  // static_cast<T>(ct[i]) into the thread frag, mirroring the threadgroup
  // overload's pattern. Validated by
  // tools/probe_nax_frag32.py::test_mma_via_dv_load_store.
  template <Role role, bool transpose, typename T, typename U>
  METAL_FUNC static void load(
      thread dtype_frag_t<T>& dst,
      const device U* src,
      const int ld) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, 32,
        /*transpose_left=*/  (role == Role::Left)  ? transpose : false,
        /*transpose_right=*/ (role == Role::Right) ? transpose : false,
        /*relaxed_precision=*/true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    metal::dextents<int32_t, 2> ext(32, ld);
    metal::tensor<device U, metal::dextents<int32_t, 2>, metal::tensor_inline>
        view(const_cast<device U*>(src), ext);

    if constexpr (role == Role::Right) {
      auto ct = op.template get_right_input_cooperative_tensor<U, U, U>();
      ct.load(view);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kElemsPerFrag; i++) dst[i] = static_cast<T>(ct[i]);
    } else {
      auto ct = op.template get_left_input_cooperative_tensor<U, U, U>();
      ct.load(view);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kElemsPerFrag; i++) dst[i] = static_cast<T>(ct[i]);
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
  // role and transpose are forwarded to load() so the cooperative-tensor
  // descriptor matches the consumer mma.
  // Avoids depending on MPP partial-tile support that does not exist on this
  // SDK (only tensor_inline/tensor_handle/tensor_offset exist — no padded or
  // strided bounded-view types); per-thread layout still owned by the
  // cooperative tensor inside load().
  // Note: threadgroup scratch cannot be declared inside a device function in
  // Metal; callers must allocate `threadgroup T scratch[32*32]` and pass it.
  template <Role role, bool transpose, typename T, typename U>
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
    load<role, transpose>(dst, scratch, /*ld=*/short(32));
  }

  // Load a 32x32 frag from device memory with a row bound only; columns are
  // always treated as full 32. Zero-fills rows at or beyond row_lim.
  // role and transpose are forwarded to load() so the cooperative-tensor
  // descriptor matches the consumer mma.
  // Caller must allocate `threadgroup T scratch[32*32]` and pass it.
  template <Role role, bool transpose, typename T, typename U>
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
    load<role, transpose>(dst, scratch, /*ld=*/short(32));
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

// Role of an operand in a matmul2d descriptor (mirrors mlx::steel::Role).
enum class Role { Left, Right };
"""

_NAX_STRUCT_MARKER = "///////////////////////////////////////////////////////////////////////////////\n// NAXFrag32"
_header_pre, _header_post = HEADER.split(_NAX_STRUCT_MARKER, 1)
FULL_HEADER = (
    _header_pre
    + _INT_HELPER
    + "\n"
    + _NAX_STRUCT_MARKER
    + _header_post
)

# ---------------------------------------------------------------------------
# Kernel body template: stage A and B from device into threadgroup buffers,
# load with role-aware (Role, transpose) template params, run mma, store.
# ---------------------------------------------------------------------------
_BODY_TEMPLATE = """
using Frag = NAXFrag32;

threadgroup float A_buf[32 * 32];
threadgroup float B_buf[32 * 32];

const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
STEEL_PRAGMA_UNROLL
for (short i = 0; i < 32; ++i) {{
    A_buf[lane * 32 + i] = a_in[lane * 32 + i];
    B_buf[lane * 32 + i] = b_in[lane * 32 + i];
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

Frag::dtype_frag_t<float> A_frag, B_frag, C_frag;
for (short i = 0; i < Frag::kElemsPerFrag; ++i) C_frag[i] = 0.0f;

Frag::load<Role::Left,  {ta}>(A_frag, (const threadgroup float*)A_buf, short(32));
Frag::load<Role::Right, {tb}>(B_frag, (const threadgroup float*)B_buf, short(32));

Frag::mma<float, float, float, {ta}, {tb}>(
    C_frag, A_frag, metal::bool_constant<{ta}>{{}}, B_frag, metal::bool_constant<{tb}>{{}});

threadgroup float C_buf[32 * 32];
Frag::store(C_frag, (threadgroup float*)C_buf, short(32));
threadgroup_barrier(mem_flags::mem_threadgroup);

STEEL_PRAGMA_UNROLL
for (short i = 0; i < 32; ++i) {{
    out_buf[lane * 32 + i] = C_buf[lane * 32 + i];
}}
"""


def run_qk_t(name: str) -> bool:
    """Q @ K^T sub-probe: Q (Left, transpose_a=false), K (Right, transpose_b=true).

    SDPA first matmul: scores = Q @ K^T.
    """
    np.random.seed(0)
    A = np.random.randint(-3, 4, size=(32, 32)).astype(np.float32)  # Q
    B = np.random.randint(-3, 4, size=(32, 32)).astype(np.float32)  # K
    expected = A @ B.T  # transpose_b=true

    kernel = mx.fast.metal_kernel(
        name=f"probe_sdpa_{name}",
        input_names=["a_in", "b_in"],
        output_names=["out_buf"],
        header=FULL_HEADER,
        source=_BODY_TEMPLATE.format(ta="false", tb="true"),
    )
    out = kernel(
        inputs=[mx.array(A), mx.array(B)],
        output_shapes=[(32, 32)],
        output_dtypes=[mx.float32],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
    )[0]
    mx.eval(out)
    got = np.array(out)
    err = float(np.max(np.abs(got - expected)))
    ok = err < 0.5
    status = "OK" if ok else "FAIL"
    print(f"  {status}  sub-probe Q@K.T  max|err|={err:.4e}")
    return ok


def run_sv(name: str) -> bool:
    """softmax @ V sub-probe: S (Left, transpose_a=false), V (Right, transpose_b=false).

    SDPA second matmul: out = softmax(scores) @ V.
    """
    np.random.seed(1)
    A = np.random.randint(-3, 4, size=(32, 32)).astype(np.float32)  # S (score tile)
    B = np.random.randint(-3, 4, size=(32, 32)).astype(np.float32)  # V
    expected = A @ B  # transpose_b=false

    kernel = mx.fast.metal_kernel(
        name=f"probe_sdpa_{name}",
        input_names=["a_in", "b_in"],
        output_names=["out_buf"],
        header=FULL_HEADER,
        source=_BODY_TEMPLATE.format(ta="false", tb="false"),
    )
    out = kernel(
        inputs=[mx.array(A), mx.array(B)],
        output_shapes=[(32, 32)],
        output_dtypes=[mx.float32],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
    )[0]
    mx.eval(out)
    got = np.array(out)
    err = float(np.max(np.abs(got - expected)))
    ok = err < 0.5
    status = "OK" if ok else "FAIL"
    print(f"  {status}  sub-probe S@V     max|err|={err:.4e}")
    return ok


if __name__ == "__main__":
    print("Probing NAXFrag32 SDPA call shapes")

    failed = 0
    if not run_qk_t("qkT"):
        failed += 1
    if not run_sv("sv"):
        failed += 1

    if failed:
        print(f"FAILED: {failed}/2 sub-probes")
        sys.exit(1)
    print("All 2/2 sub-probes passed")
