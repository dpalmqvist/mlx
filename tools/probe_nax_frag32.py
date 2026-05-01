"""
Standalone correctness probe for NAXFrag32 (32x32 frag, (32,32,32) descriptor).

Runs a microkernel that exercises NAXFrag32 directly — without going through
the steel gemm pipeline — so any divergence is in the frag itself, not in
tile staging or the matmul wrapper. Compare against a numpy reference with
integer inputs in [-3, 3] so fp32 reductions are exact.

Each test is gated: it raises SystemExit(1) if max|err| >= 1e-3.
"""
import sys

import mlx.core as mx
import numpy as np

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

"""


def test_compile_smoke():
    """Trivial kernel that touches NAXFrag32 — proves the include resolves
    and basic constants/types are visible from inside mx.fast.metal_kernel."""
    src = """
    using Frag = NAXFrag32;
    out[0] = (float)Frag::kFragRows;
    out[1] = (float)Frag::kFragCols;
    out[2] = (float)Frag::kElemsPerFrag;
    out[3] = (float)Frag::kPacking;
    """
    k = mx.fast.metal_kernel(
        name="probe_naxfrag32_smoke",
        input_names=[],
        output_names=["out"],
        source=src,
        header=HEADER,
    )
    res = k(
        inputs=[],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(4,)],
        output_dtypes=[mx.float32],
    )
    mx.eval(res[0])
    arr = np.asarray(res[0])
    expected = np.array([32.0, 32.0, 32.0, 1.0])
    if not np.allclose(arr, expected):
        raise AssertionError(f"Constants mismatch: got {arr.tolist()}, expected {expected.tolist()}")


def test_mma_register_only():
    """Register-only mma correctness: explicit dr/dc loads → NAXFrag32::mma →
    explicit dr/dc stores. Isolates mma itself from the load/store I/O path.

    Per spec, dtype_frag_t element i lives at (a + dr[i % 8], b + dc[i / 8])
    where (b, a) is get_coord() (returned as short2{col, row}).

    NOTE: FT/TF/TT register-only variants were attempted during Phase 6 Task 3
    and failed (~1000+ bad cells each). This confirms that MPP's per-thread
    register layout is NOT invariant under the transpose descriptor variant —
    the dr/dc table probed in tools/probe_nax_descriptor.py is specific to the
    (false, false) descriptor. Transpose correctness is instead covered by the
    load+mma path in tools/probe_naxfrag32_transpose.py, which exercises the
    correct load→mma→store sequence for each transpose configuration.
    """
    src = """
    constexpr int M = 32, N = 32, K = 32;
    using Frag = NAXFrag32;

    short2 base = Frag::get_coord();              // {col_base, row_base}
    Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;

    // Explicit dr/dc load — no NAXFrag32::load here.
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 32; ++i) {
        short2 d = Frag::dr_dc(i);                // {col_off, row_off}
        short row = base.y + d.y;
        short col = base.x + d.x;
        a_frag[i] = A[row * K + col];
        b_frag[i] = B[row * N + col];
        c_frag[i] = 0.0f;
    }

    Frag::mma(c_frag,
              a_frag, metal::bool_constant<false>{},
              b_frag, metal::bool_constant<false>{});

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 32; ++i) {
        short2 d = Frag::dr_dc(i);
        short row = base.y + d.y;
        short col = base.x + d.x;
        C[row * N + col] = c_frag[i];
    }
    """
    k = mx.fast.metal_kernel(
        name="probe_naxfrag32_mma_register_only",
        input_names=["A", "B"],
        output_names=["C"],
        source=src,
        header=HEADER,
    )
    A_np = np.random.RandomState(0).randint(-3, 4, (32, 32)).astype(np.float32)
    B_np = np.random.RandomState(1).randint(-3, 4, (32, 32)).astype(np.float32)
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
    ref = A_np @ B_np
    err = float(np.max(np.abs(np.asarray(out[0]) - ref)))
    if err >= 1e-3:
        # Print which positions differ — helps debug layout vs mma issues.
        diff = np.abs(np.asarray(out[0]) - ref)
        bad = np.argwhere(diff >= 1e-3)
        sample = bad[:5].tolist() if len(bad) > 0 else []
        raise AssertionError(
            f"max|err|={err:.4g} (>= 1e-3); {len(bad)} bad cells; "
            f"first few: {sample}"
        )


def test_mma_via_tg_load_store():
    """End-to-end mma via NAXFrag32::load (tensor_inline over threadgroup) →
    mma → NAXFrag32::store. Validates the layout-agnostic I/O path that
    production kernels will use. With Task 1.6b's register-only test passing,
    a failure here points to NAXFrag32::load or store, not mma itself.
    """
    src = """
    constexpr int M = 32, N = 32, K = 32;
    using Frag = NAXFrag32;

    threadgroup float A_tg[32 * 32];
    threadgroup float B_tg[32 * 32];
    threadgroup float C_tg[32 * 32];

    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 32; ++i) {
        const short flat = lane * 32 + i;
        A_tg[flat] = A[flat];
        B_tg[flat] = B[flat];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;
    Frag::load(a_frag, (const threadgroup float*)A_tg, (short)32);
    Frag::load(b_frag, (const threadgroup float*)B_tg, (short)32);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 32; ++i) {
        c_frag[i] = 0.0f;
    }

    Frag::mma(c_frag,
              a_frag, metal::bool_constant<false>{},
              b_frag, metal::bool_constant<false>{});

    Frag::store(c_frag, (threadgroup float*)C_tg, (short)32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 32; ++i) {
        const short flat = lane * 32 + i;
        C[flat] = C_tg[flat];
    }
    """
    k = mx.fast.metal_kernel(
        name="probe_naxfrag32_tg_load_store",
        input_names=["A", "B"],
        output_names=["C"],
        source=src,
        header=HEADER,
    )
    A_np = np.random.RandomState(0).randint(-3, 4, (32, 32)).astype(np.float32)
    B_np = np.random.RandomState(1).randint(-3, 4, (32, 32)).astype(np.float32)
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
    ref = A_np @ B_np
    err = float(np.max(np.abs(np.asarray(out[0]) - ref)))
    if err >= 1e-3:
        diff = np.abs(np.asarray(out[0]) - ref)
        bad = np.argwhere(diff >= 1e-3)
        sample = bad[:5].tolist() if len(bad) > 0 else []
        raise AssertionError(
            f"max|err|={err:.4g} (>= 1e-3); {len(bad)} bad cells; "
            f"first few: {sample}"
        )


def test_mma_via_dv_load_store():
    """End-to-end mma via NAXFrag32::load (tensor_inline over device) →
    mma → NAXFrag32::store (tensor_inline over device). Validates the
    device-pointer I/O path added in Task 2.0 that NAXTile's aligned
    load/store and gemm_epilogue's aligned C-load will use.
    """
    src = """
    using Frag = NAXFrag32;

    Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;
    Frag::load(a_frag, (const device float*)A, (short)32);
    Frag::load(b_frag, (const device float*)B, (short)32);

    STEEL_PRAGMA_UNROLL
    for (uint16_t i = 0; i < Frag::kElemsPerFrag; ++i) c_frag[i] = 0.0f;

    Frag::mma(c_frag,
              a_frag, metal::bool_constant<false>{},
              b_frag, metal::bool_constant<false>{});

    Frag::store(c_frag, (device float*)C, (short)32);
    """
    k = mx.fast.metal_kernel(
        name="probe_naxfrag32_dv_load_store",
        input_names=["A", "B"],
        output_names=["C"],
        source=src,
        header=HEADER,
    )
    A_np = np.random.RandomState(0).randint(-3, 4, (32, 32)).astype(np.float32)
    B_np = np.random.RandomState(1).randint(-3, 4, (32, 32)).astype(np.float32)
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
    ref = A_np @ B_np
    err = float(np.max(np.abs(np.asarray(out[0]) - ref)))
    if err >= 1e-3:
        diff = np.abs(np.asarray(out[0]) - ref)
        bad = np.argwhere(diff >= 1e-3)
        sample = bad[:5].tolist() if len(bad) > 0 else []
        raise AssertionError(
            f"max|err|={err:.4g} (>= 1e-3); {len(bad)} bad cells; "
            f"first few: {sample}"
        )


def test_store_device_mixed_precision():
    """NAXFrag32::store<float, half> via device pointer.
    Loads A and B as floats, runs mma with float accumulator, stores to
    half via the device-pointer store<float, half> overload.
    Compares against numpy: (A @ B).astype(np.float16).
    Validates the T != U path needed by steel_gemm_fused_nax_g16 which
    stores a float accumulator (AccumType=float) to a half output buffer.
    """
    src = """
    using Frag = NAXFrag32;

    Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;
    Frag::load(a_frag, (const device float*)A, (short)32);
    Frag::load(b_frag, (const device float*)B, (short)32);

    STEEL_PRAGMA_UNROLL
    for (uint16_t i = 0; i < Frag::kElemsPerFrag; ++i) c_frag[i] = 0.0f;

    Frag::mma(c_frag,
              a_frag, metal::bool_constant<false>{},
              b_frag, metal::bool_constant<false>{});

    Frag::store(c_frag, (device half*)C, (short)32);
    """
    k = mx.fast.metal_kernel(
        name="probe_naxfrag32_store_mixed_precision",
        input_names=["A", "B"],
        output_names=["C"],
        source=src,
        header=HEADER,
    )
    A_np = np.random.RandomState(2).randint(-3, 4, (32, 32)).astype(np.float32)
    B_np = np.random.RandomState(3).randint(-3, 4, (32, 32)).astype(np.float32)
    A = mx.array(A_np)
    B = mx.array(B_np)
    out = k(
        inputs=[A, B],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(32, 32)],
        output_dtypes=[mx.float16],
    )
    mx.eval(out[0])
    ref = (A_np @ B_np).astype(np.float16)
    err = float(np.max(np.abs(np.asarray(out[0]).astype(np.float32) - ref.astype(np.float32))))
    if err >= 1e-2:
        diff = np.abs(np.asarray(out[0]).astype(np.float32) - ref.astype(np.float32))
        bad = np.argwhere(diff >= 1e-2)
        sample = bad[:5].tolist() if len(bad) > 0 else []
        raise AssertionError(
            f"max|err|={err:.4g} (>= 1e-2); {len(bad)} bad cells; "
            f"first few: {sample}"
        )


def test_load_safe_zero_fill():
    """Validate NAXFrag32::load_safe: load a 32x32 matrix with bounds
    (row_lim=20, col_lim=24). Elements outside the bounds should be zero-filled.
    With B = identity, the result C should equal A masked to the (20, 24)
    top-left rectangle, with zeros elsewhere.
    """
    src = """
    using Frag = NAXFrag32;
    threadgroup float scratch[32 * 32];
    Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;
    Frag::load_safe(a_frag, (const device float*)A, 32, (short)20, (short)24, scratch);
    Frag::load(b_frag, (const device float*)B, (short)32);
    STEEL_PRAGMA_UNROLL
    for (uint16_t i = 0; i < Frag::kElemsPerFrag; ++i) c_frag[i] = 0.0f;
    Frag::mma(c_frag,
              a_frag, metal::bool_constant<false>{},
              b_frag, metal::bool_constant<false>{});
    Frag::store(c_frag, (device float*)C, (short)32);
    """
    k = mx.fast.metal_kernel(
        name="probe_naxfrag32_load_safe_zero_fill",
        input_names=["A", "B"],
        output_names=["C"],
        source=src,
        header=HEADER,
    )
    A_np = np.random.RandomState(0).randint(-3, 4, (32, 32)).astype(np.float32)
    B_np = np.eye(32, dtype=np.float32)
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
    # Reference: A masked to top-left (20, 24), zeros outside.
    ref = A_np.copy()
    ref[20:, :] = 0.0
    ref[:, 24:] = 0.0
    err = float(np.max(np.abs(np.asarray(out[0]) - ref)))
    if err >= 1e-3:
        diff = np.abs(np.asarray(out[0]) - ref)
        bad = np.argwhere(diff >= 1e-3)
        sample = bad[:5].tolist() if len(bad) > 0 else []
        raise AssertionError(
            f"max|err|={err:.4g} (>= 1e-3); {len(bad)} bad cells; "
            f"first few: {sample}"
        )


def test_load_rows_zero_fill():
    """Validate NAXFrag32::load_rows: load a 32x32 matrix with row_lim=20.
    Rows 20..31 should be zero-filled; all 32 columns are kept.
    With B = identity, the result C should equal A with rows 20..31 zeroed.
    """
    src = """
    using Frag = NAXFrag32;
    threadgroup float scratch[32 * 32];
    Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;
    Frag::load_rows(a_frag, (const device float*)A, 32, (short)20, scratch);
    Frag::load(b_frag, (const device float*)B, (short)32);
    STEEL_PRAGMA_UNROLL
    for (uint16_t i = 0; i < Frag::kElemsPerFrag; ++i) c_frag[i] = 0.0f;
    Frag::mma(c_frag,
              a_frag, metal::bool_constant<false>{},
              b_frag, metal::bool_constant<false>{});
    Frag::store(c_frag, (device float*)C, (short)32);
    """
    k = mx.fast.metal_kernel(
        name="probe_naxfrag32_load_rows_zero_fill",
        input_names=["A", "B"],
        output_names=["C"],
        source=src,
        header=HEADER,
    )
    A_np = np.random.RandomState(0).randint(-3, 4, (32, 32)).astype(np.float32)
    B_np = np.eye(32, dtype=np.float32)
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
    # Reference: A with rows 20..31 zeroed.
    ref = A_np.copy()
    ref[20:, :] = 0.0
    err = float(np.max(np.abs(np.asarray(out[0]) - ref)))
    if err >= 1e-3:
        diff = np.abs(np.asarray(out[0]) - ref)
        bad = np.argwhere(diff >= 1e-3)
        sample = bad[:5].tolist() if len(bad) > 0 else []
        raise AssertionError(
            f"max|err|={err:.4g} (>= 1e-3); {len(bad)} bad cells; "
            f"first few: {sample}"
        )


def test_store_safe_skip_oob():
    """Validate NAXFrag32::store_safe: store a matmul result with bounds
    (row_lim=20, col_lim=24). Out-of-bounds elements should retain the
    pre-filled sentinel value of -1.0. C[:20, :24] should match A[:20, :24]
    (since B = identity), and C[20:, :] / C[:, 24:] should retain -1.0.
    """
    src = """
    using Frag = NAXFrag32;

    // Copy Cinit into C before the matmul so OOB regions start as -1.0.
    uint tid = thread_position_in_threadgroup.x;
    for (uint i = tid; i < 32*32; i += 32) ((device float*)C)[i] = ((const device float*)Cinit)[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float scratch[32 * 32];
    Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;
    Frag::load(a_frag, (const device float*)A, (short)32);
    Frag::load(b_frag, (const device float*)B, (short)32);
    STEEL_PRAGMA_UNROLL
    for (uint16_t i = 0; i < Frag::kElemsPerFrag; ++i) c_frag[i] = 0.0f;
    Frag::mma(c_frag,
              a_frag, metal::bool_constant<false>{},
              b_frag, metal::bool_constant<false>{});
    Frag::store_safe(c_frag, (device float*)C, 32, (short)20, (short)24, scratch);
    """
    k = mx.fast.metal_kernel(
        name="probe_naxfrag32_store_safe_skip_oob",
        input_names=["A", "B", "Cinit"],
        output_names=["C"],
        source=src,
        header=HEADER,
    )
    A_np = np.random.RandomState(0).randint(-3, 4, (32, 32)).astype(np.float32)
    B_np = np.eye(32, dtype=np.float32)
    Cinit_np = np.full((32, 32), -1.0, dtype=np.float32)
    A = mx.array(A_np)
    B = mx.array(B_np)
    Cinit = mx.array(Cinit_np)
    out = k(
        inputs=[A, B, Cinit],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(32, 32)],
        output_dtypes=[mx.float32],
    )
    mx.eval(out[0])
    result = np.asarray(out[0])

    # Build reference: A @ B = A (identity B), store only in [:20, :24].
    # The rest should remain -1.0.
    ref = Cinit_np.copy()
    ref[:20, :24] = A_np[:20, :24]  # A @ I = A

    err = float(np.max(np.abs(result - ref)))
    if err >= 1e-3:
        diff = np.abs(result - ref)
        bad = np.argwhere(diff >= 1e-3)
        sample = bad[:5].tolist() if len(bad) > 0 else []
        # Extra diagnostics: check if OOB sentinel was overwritten.
        oob_rows_changed = np.any(result[20:, :] != -1.0)
        oob_cols_changed = np.any(result[:, 24:] != -1.0)
        raise AssertionError(
            f"max|err|={err:.4g} (>= 1e-3); {len(bad)} bad cells; "
            f"first few: {sample}; "
            f"OOB rows[20:] overwritten={oob_rows_changed}; "
            f"OOB cols[24:] overwritten={oob_cols_changed}"
        )


def test_store_rows_skip_oob():
    """Validate NAXFrag32::store_rows: store a matmul result with row_lim=20.
    Rows 20..31 should retain the pre-filled sentinel value of -1.0.
    C[:20, :] should match A[:20, :] (since B = identity).
    """
    src = """
    using Frag = NAXFrag32;

    // Copy Cinit into C before the matmul so OOB regions start as -1.0.
    uint tid = thread_position_in_threadgroup.x;
    for (uint i = tid; i < 32*32; i += 32) ((device float*)C)[i] = ((const device float*)Cinit)[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float scratch[32 * 32];
    Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;
    Frag::load(a_frag, (const device float*)A, (short)32);
    Frag::load(b_frag, (const device float*)B, (short)32);
    STEEL_PRAGMA_UNROLL
    for (uint16_t i = 0; i < Frag::kElemsPerFrag; ++i) c_frag[i] = 0.0f;
    Frag::mma(c_frag,
              a_frag, metal::bool_constant<false>{},
              b_frag, metal::bool_constant<false>{});
    Frag::store_rows(c_frag, (device float*)C, 32, (short)20, scratch);
    """
    k = mx.fast.metal_kernel(
        name="probe_naxfrag32_store_rows_skip_oob",
        input_names=["A", "B", "Cinit"],
        output_names=["C"],
        source=src,
        header=HEADER,
    )
    A_np = np.random.RandomState(0).randint(-3, 4, (32, 32)).astype(np.float32)
    B_np = np.eye(32, dtype=np.float32)
    Cinit_np = np.full((32, 32), -1.0, dtype=np.float32)
    A = mx.array(A_np)
    B = mx.array(B_np)
    Cinit = mx.array(Cinit_np)
    out = k(
        inputs=[A, B, Cinit],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(32, 32)],
        output_dtypes=[mx.float32],
    )
    mx.eval(out[0])
    result = np.asarray(out[0])

    # Build reference: A @ B = A (identity B), store only rows [:20].
    # Rows [20:] should remain -1.0.
    ref = Cinit_np.copy()
    ref[:20, :] = A_np[:20, :]  # A @ I = A

    err = float(np.max(np.abs(result - ref)))
    if err >= 1e-3:
        diff = np.abs(result - ref)
        bad = np.argwhere(diff >= 1e-3)
        sample = bad[:5].tolist() if len(bad) > 0 else []
        # Extra diagnostics: check if OOB sentinel was overwritten.
        oob_rows_changed = np.any(result[20:, :] != -1.0)
        raise AssertionError(
            f"max|err|={err:.4g} (>= 1e-3); {len(bad)} bad cells; "
            f"first few: {sample}; "
            f"OOB rows[20:] overwritten={oob_rows_changed}"
        )


def test_row_reduce_sum():
    """Validate NAXFrag32::row_reduce<Sum>: fill a 32x32 frag (in registers
    via explicit dr/dc loads) with v(r,c) = r*100 + c. Each row's sum should
    be 32*(r*100) + sum(0..31) = 3200*r + 496.
    """
    src = """
    using Frag = NAXFrag32;
    using T = float;

    short2 base = Frag::get_coord();
    Frag::dtype_frag_t<T> inp;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 32; ++i) {
        short2 d = Frag::dr_dc(i);
        short row = base.y + d.y;
        short col = base.x + d.x;
        inp[i] = (T)(row * 100 + col);
    }

    struct Sum { static T apply(T a, T b) { return a + b; } };
    T row_sums[4] = {0, 0, 0, 0};
    Frag::row_reduce<Sum>(inp, row_sums);

    constexpr short dr_table[4] = {0, 8, 16, 24};
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 4; ++r) {
        short abs_row = base.y + dr_table[r];
        out[abs_row] = row_sums[r];
    }
    """
    k = mx.fast.metal_kernel(
        name="probe_naxfrag32_row_reduce_sum",
        input_names=[],
        output_names=["out"],
        source=src,
        header=HEADER,
    )
    res = k(
        inputs=[],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(32,)],
        output_dtypes=[mx.float32],
    )
    mx.eval(res[0])
    arr = np.asarray(res[0])
    expected = np.array([3200.0 * r + 496.0 for r in range(32)])
    err = float(np.max(np.abs(arr - expected)))
    if err >= 1e-3:
        diff = np.abs(arr - expected)
        bad = np.argwhere(diff >= 1e-3)
        sample = bad[:5].tolist() if len(bad) > 0 else []
        raise AssertionError(
            f"max|err|={err:.4g}; {len(bad)} bad rows; first few: {sample}; "
            f"got[:4]={arr[:4].tolist()}, expected[:4]={expected[:4].tolist()}"
        )


def main():
    tests = [
        test_compile_smoke,
        test_mma_register_only,
        test_mma_via_tg_load_store,
        test_mma_via_dv_load_store,
        test_store_device_mixed_precision,
        test_load_safe_zero_fill,
        test_load_rows_zero_fill,
        test_store_safe_skip_oob,
        test_store_rows_skip_oob,
        test_row_reduce_sum,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
        except NotImplementedError as e:
            print(f"  PEND {t.__name__}: {e}")
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
