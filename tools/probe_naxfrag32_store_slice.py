"""
Standalone correctness probe for NAXFrag32::store_slice.

Exercises store_slice on rectangular slice patterns — verifying that
in-bounds positions receive the expected value (r*32+c) and out-of-bounds
positions retain the sentinel (-9999.0).

RED state: NAXFrag32::store_slice at nax.h:943 is currently a
static_assert stub. This probe MUST fail to compile until Phase 5 Task 2
implements the method, at which point it should go GREEN.

Pattern: same "inline NAXFrag32 source into the probe" approach as
tools/probe_nax_frag32.py (mx.fast.metal_kernel cannot include source-tree
headers). The HEADER block is a verbatim copy of the one in that file.
IF YOU CHANGE NAXFrag32 IN nax.h, UPDATE THAT FILE'S COPY (and this one).
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
# Kernel source template for one store_slice test case.
# Sentinel initialisation uses Pattern A: all 32 threads collaboratively
# fill the 32x32 output buffer with -9999.0f before store_slice is called.
# ---------------------------------------------------------------------------

_KERNEL_SRC = """
using Frag = NAXFrag32;

// ----- Step 1: fill output buffer with sentinel -----
// Each of the 32 threads fills its 32 contiguous elements (lane * 32 .. +32).
const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
STEEL_PRAGMA_UNROLL
for (short i = 0; i < 32; ++i) {{
    out_buf[lane * 32 + i] = -9999.0f;
}}
threadgroup_barrier(mem_flags::mem_device);

// ----- Step 2: build the frag with value r*32+c -----
// Element i of this lane's frag owns position
//   r = get_coord().y + dr_dc(i).y
//   c = get_coord().x + dr_dc(i).x
Frag::dtype_frag_t<float> src;
STEEL_PRAGMA_UNROLL
for (short i = 0; i < Frag::kElemsPerFrag; ++i) {{
    short2 base = Frag::get_coord();   // {{col_base, row_base}}
    short2 d    = Frag::dr_dc(i);      // {{col_off,  row_off}}
    short r = base.y + d.y;
    short c = base.x + d.x;
    src[i] = (float)(r * 32 + c);
}}

// ----- Step 3: call store_slice -----
threadgroup float scratch[32 * 32];
NAXFrag32::store_slice(
    src,
    (device float*)out_buf,
    int(32),         // str_x = ld
    Int<1>{{}},       // str_y
    short({start_x}),
    short({stop_x}),
    short({start_y}),
    short({stop_y}),
    scratch);
"""

# We need Int<N> in the header for store_slice; it's part of steel/utils.h in
# the real tree. Define a minimal version here so the probe is self-contained.
_INT_HELPER = """
// Minimal Int<N> helper (mirrors steel/utils.h Int<N>).
template <short N>
struct Int {
  static constant constexpr const short value = N;
  METAL_FUNC constexpr operator short() const { return N; }
};
"""

# _INT_HELPER must come before the NAXFrag32 struct (which uses Int<0> in
# store_slice default template args), so we split HEADER at the struct boundary.
_NAX_STRUCT_MARKER = "///////////////////////////////////////////////////////////////////////////////\n// NAXFrag32"
_header_pre, _header_post = HEADER.split(_NAX_STRUCT_MARKER, 1)
FULL_HEADER = _header_pre + _INT_HELPER + "\n" + _NAX_STRUCT_MARKER + _header_post


def _make_kernel(name: str, start_x: int, stop_x: int, start_y: int, stop_y: int):
    src = _KERNEL_SRC.format(
        start_x=start_x, stop_x=stop_x, start_y=start_y, stop_y=stop_y
    )
    return mx.fast.metal_kernel(
        name=name,
        input_names=[],
        output_names=["out_buf"],
        source=src,
        header=FULL_HEADER,
    )


def _run_case(name: str, start_x: int, stop_x: int, start_y: int, stop_y: int):
    k = _make_kernel(name, start_x, stop_x, start_y, stop_y)
    res = k(
        inputs=[],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(32, 32)],
        output_dtypes=[mx.float32],
    )
    mx.eval(res[0])
    return np.asarray(res[0])


def _check(
    result: np.ndarray,
    start_x: int,
    stop_x: int,
    start_y: int,
    stop_y: int,
    label: str,
):
    """Verify in-bounds == r*32+c, out-of-bounds == -9999."""
    errors = []
    for r in range(32):
        for c in range(32):
            in_bounds = (start_x <= r < stop_x) and (start_y <= c < stop_y)
            expected = float(r * 32 + c) if in_bounds else -9999.0
            got = float(result[r, c])
            if abs(got - expected) >= 0.5:
                errors.append(
                    f"  [{r},{c}] expected {expected:.0f} got {got:.0f}"
                )
    if errors:
        raise AssertionError(
            f"{label}: {len(errors)} mismatches (first 5):\n"
            + "\n".join(errors[:5])
        )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# Convention: (start_x, stop_x) are ROW bounds, (start_y, stop_y) are COL bounds.
# Yes, x→row, y→col — NAXTile passes start.y/stop.y here. See spec Risk #1.
CASES = [
    # (name_suffix, start_x, stop_x, start_y, stop_y, description)
    ("full",        0,  32,  0, 32, "Full frag — matches store"),
    ("col_clip",    0,  32,  8, 24, "Cols 8..23 only"),
    ("row_clip",    4,  28,  0, 32, "Rows 4..27 only"),
    ("both_clip",   4,  28,  8, 24, "Both axes clipped"),
    ("empty",      10,  10, 10, 10, "Empty slice — no writes"),
]


def run_all():
    failed = 0
    for suffix, sx, ex, sy, ey, desc in CASES:
        name = f"probe_store_slice_{suffix}"
        try:
            result = _run_case(name, sx, ex, sy, ey)
            _check(result, sx, ex, sy, ey, desc)
            print(f"  OK  {name}  ({desc})")
        except Exception as e:
            print(f"  FAIL {name}  ({desc}): {e}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run_all())
