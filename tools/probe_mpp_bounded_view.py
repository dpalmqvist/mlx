"""
Probe whether MPP's `tensor_inline` with extents smaller than the underlying
buffer acts as a bounded view for `cooperative_tensor.load` / `.store`.

Why this matters: NAXFrag32's safe/rows variants need a way to do partial-tile
I/O without hard-coding the per-thread (dr, dc) layout. If `tensor_inline`
honors its extents (zero-fills rows past extent on load, leaves untouched
rows past extent on store), Phase 2 can avoid threadgroup-scratch staging
entirely.

Three tests on the (32, 32, 32) descriptor:

  (T1) aligned baseline: extents = (32, 32). Sanity check.
  (T2) truncated load extents: A buffer 32x32, rows 20..31 contain a sentinel
       (99.0). Build tensor_inline over A with extents (20, 32). Expect:
       result[:20] == A[:20] @ B (sentinel rows 20-31 of A are NOT used).
  (T3) truncated store extents: C buffer 32x32 pre-filled with sentinel
       (-1.0). Build tensor_inline over C with extents (20, 32). cT.store(...).
       Expect: C[:20] == matmul, C[20:] == -1.0 (untouched).

Outcomes:
  - Both T2 and T3 OK → no scratch plumbing needed; safe variants can route
    through tensor_inline with truncated extents.
  - Either fails → tensor_inline does NOT bounds-check; staging required.
"""
import sys

import mlx.core as mx
import numpy as np

HEADER = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

THREADGROUP_SIZE = 32


def make_source_truncated_load(load_rows: int):
    """A is 32x32 in device memory; staged into a 32x32 threadgroup buffer
    so we control the sentinel pattern exactly. tensor_inline over the
    threadgroup buffer with extents (load_rows, 32). B is full extents.
    """
    return f"""
constexpr int M=32, N=32, K=32;
constexpr int LR={load_rows};
constexpr auto desc = matmul2d_descriptor(
    M, N, K,
    /*transpose_left=*/false,
    /*transpose_right=*/false,
    /*relaxed_precision=*/false,
    matmul2d_descriptor::mode::multiply);

threadgroup float tgA[M*K];
threadgroup float tgB[K*N];

uint tid = thread_position_in_threadgroup.x;
for (uint i = tid; i < M*K; i += {THREADGROUP_SIZE}) tgA[i] = ((device float*)A)[i];
for (uint i = tid; i < K*N; i += {THREADGROUP_SIZE}) tgB[i] = ((device float*)B)[i];
threadgroup_barrier(mem_flags::mem_threadgroup);

matmul2d<desc, execution_simdgroup> op;
auto ctA = op.get_left_input_cooperative_tensor<float, float, float>();
auto ctB = op.get_right_input_cooperative_tensor<float, float, float>();
auto ctC = op.get_destination_cooperative_tensor<decltype(ctA), decltype(ctB), float>();

dextents<int32_t, 2> extA(LR, K);  // truncated rows
dextents<int32_t, 2> extB(K, N);
dextents<int32_t, 2> extC(M, N);
tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tA(tgA, extA);
tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tB(tgB, extB);
tensor<device      float, dextents<int32_t, 2>, tensor_inline> tC(C, extC);

ctA.load(tA);
ctB.load(tB);
for (uint16_t i=0; i<ctC.get_capacity(); ++i) ctC[i] = 0.0f;
op.run(ctA, ctB, ctC);
ctC.store(tC);
"""


def make_source_truncated_load_default_tag(load_rows: int):
    """Same as make_source_truncated_load but builds the tensor view WITHOUT
    the explicit `tensor_inline` tag — uses the default tensor type, which
    according to MPP docs is `tensor_handle` (the bounds-tracked view).
    """
    return f"""
constexpr int M=32, N=32, K=32;
constexpr int LR={load_rows};
constexpr auto desc = matmul2d_descriptor(
    M, N, K,
    /*transpose_left=*/false,
    /*transpose_right=*/false,
    /*relaxed_precision=*/false,
    matmul2d_descriptor::mode::multiply);

threadgroup float tgA[M*K];
threadgroup float tgB[K*N];

uint tid = thread_position_in_threadgroup.x;
for (uint i = tid; i < M*K; i += {THREADGROUP_SIZE}) tgA[i] = ((device float*)A)[i];
for (uint i = tid; i < K*N; i += {THREADGROUP_SIZE}) tgB[i] = ((device float*)B)[i];
threadgroup_barrier(mem_flags::mem_threadgroup);

matmul2d<desc, execution_simdgroup> op;
auto ctA = op.get_left_input_cooperative_tensor<float, float, float>();
auto ctB = op.get_right_input_cooperative_tensor<float, float, float>();
auto ctC = op.get_destination_cooperative_tensor<decltype(ctA), decltype(ctB), float>();

dextents<int32_t, 2> extA(LR, K);  // truncated rows
dextents<int32_t, 2> extB(K, N);
dextents<int32_t, 2> extC(M, N);
// NOTE: no third template arg — should default to tensor_handle.
tensor<threadgroup float, dextents<int32_t, 2>> tA(tgA, extA);
tensor<threadgroup float, dextents<int32_t, 2>> tB(tgB, extB);
tensor<device      float, dextents<int32_t, 2>> tC(C, extC);

ctA.load(tA);
ctB.load(tB);
for (uint16_t i=0; i<ctC.get_capacity(); ++i) ctC[i] = 0.0f;
op.run(ctA, ctB, ctC);
ctC.store(tC);
"""


def make_source_truncated_store(store_rows: int):
    """A and B are full 32x32. Output tensor_inline has extents (store_rows, 32)
    over a 32x32 device buffer. C is pre-filled with sentinel by the caller;
    we expect rows store_rows..31 to remain at the sentinel value.
    """
    return f"""
constexpr int M=32, N=32, K=32;
constexpr int SR={store_rows};
constexpr auto desc = matmul2d_descriptor(
    M, N, K,
    /*transpose_left=*/false,
    /*transpose_right=*/false,
    /*relaxed_precision=*/false,
    matmul2d_descriptor::mode::multiply);

threadgroup float tgA[M*K];
threadgroup float tgB[K*N];

uint tid = thread_position_in_threadgroup.x;
for (uint i = tid; i < M*K; i += {THREADGROUP_SIZE}) tgA[i] = ((device float*)A)[i];
for (uint i = tid; i < K*N; i += {THREADGROUP_SIZE}) tgB[i] = ((device float*)B)[i];
threadgroup_barrier(mem_flags::mem_threadgroup);

matmul2d<desc, execution_simdgroup> op;
auto ctA = op.get_left_input_cooperative_tensor<float, float, float>();
auto ctB = op.get_right_input_cooperative_tensor<float, float, float>();
auto ctC = op.get_destination_cooperative_tensor<decltype(ctA), decltype(ctB), float>();

dextents<int32_t, 2> extA(M, K);
dextents<int32_t, 2> extB(K, N);
dextents<int32_t, 2> extC(SR, N);  // truncated rows for the store
tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tA(tgA, extA);
tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tB(tgB, extB);
tensor<device      float, dextents<int32_t, 2>, tensor_inline> tC(C, extC);

ctA.load(tA);
ctB.load(tB);
for (uint16_t i=0; i<ctC.get_capacity(); ++i) ctC[i] = 0.0f;
op.run(ctA, ctB, ctC);
ctC.store(tC);
"""


def test_baseline():
    """T1: aligned 32x32 baseline through truncated_load with LR=32."""
    src = make_source_truncated_load(32)
    name = "bounded_view_T1_baseline"
    k = mx.fast.metal_kernel(
        name=name, input_names=["A", "B"], output_names=["C"],
        source=src, header=HEADER,
    )
    A_np = np.random.RandomState(0).randint(-3, 4, (32, 32)).astype(np.float32)
    B_np = np.random.RandomState(1).randint(-3, 4, (32, 32)).astype(np.float32)
    A = mx.array(A_np); B = mx.array(B_np)
    out = k(inputs=[A, B], grid=(THREADGROUP_SIZE, 1, 1),
            threadgroup=(THREADGROUP_SIZE, 1, 1),
            output_shapes=[(32, 32)], output_dtypes=[mx.float32])
    mx.eval(out[0])
    ref = A_np @ B_np
    err = float(np.max(np.abs(np.asarray(out[0]) - ref)))
    return err, "OK" if err < 1e-3 else "WRONG"


def test_truncated_load(load_rows: int):
    """T2: A rows >= load_rows contain a SENTINEL value. If tensor_inline
    honors extents, those rows are NOT read by cT.load, so the result of
    A_truncated @ B equals (A[:load_rows] padded with anything) @ B for
    the first `load_rows` of the output. We pad A with the sentinel, then
    compare result[:load_rows] against A[:load_rows] @ B.
    """
    SENTINEL = 99.0
    src = make_source_truncated_load(load_rows)
    name = f"bounded_view_T2_load_{load_rows}"
    k = mx.fast.metal_kernel(
        name=name, input_names=["A", "B"], output_names=["C"],
        source=src, header=HEADER,
    )
    A_np = np.random.RandomState(0).randint(-3, 4, (32, 32)).astype(np.float32)
    A_np[load_rows:, :] = SENTINEL
    B_np = np.random.RandomState(1).randint(-3, 4, (32, 32)).astype(np.float32)
    A = mx.array(A_np); B = mx.array(B_np)
    out = k(inputs=[A, B], grid=(THREADGROUP_SIZE, 1, 1),
            threadgroup=(THREADGROUP_SIZE, 1, 1),
            output_shapes=[(32, 32)], output_dtypes=[mx.float32])
    mx.eval(out[0])
    out_np = np.asarray(out[0])
    ref_top = A_np[:load_rows] @ B_np
    err_top = float(np.max(np.abs(out_np[:load_rows] - ref_top)))
    # Extra diagnostic: was the sentinel actually used? Compare full A @ B
    # against output. If err_top is small but err_full is large, that means
    # tensor_inline DID truncate — only the first `load_rows` results match
    # the truncated input, while rows >= load_rows are something else.
    full_ref = A_np @ B_np
    err_full = float(np.max(np.abs(out_np - full_ref)))
    if err_top < 1e-3:
        if err_full < 1e-3:
            verdict = "AMBIGUOUS (sentinel-row read but happened to give same answer)"
        else:
            verdict = "BOUNDED (top rows match truncated, full disagrees)"
    else:
        verdict = "UNBOUNDED (top rows do NOT match A[:LR] @ B)"
    return err_top, err_full, verdict


def test_truncated_load_default_tag(load_rows: int):
    """Like test_truncated_load but uses the default tensor type."""
    SENTINEL = 99.0
    src = make_source_truncated_load_default_tag(load_rows)
    name = f"bounded_view_T2b_load_default_{load_rows}"
    k = mx.fast.metal_kernel(
        name=name, input_names=["A", "B"], output_names=["C"],
        source=src, header=HEADER,
    )
    A_np = np.random.RandomState(0).randint(-3, 4, (32, 32)).astype(np.float32)
    A_np[load_rows:, :] = SENTINEL
    B_np = np.random.RandomState(1).randint(-3, 4, (32, 32)).astype(np.float32)
    A = mx.array(A_np); B = mx.array(B_np)
    out = k(inputs=[A, B], grid=(THREADGROUP_SIZE, 1, 1),
            threadgroup=(THREADGROUP_SIZE, 1, 1),
            output_shapes=[(32, 32)], output_dtypes=[mx.float32])
    mx.eval(out[0])
    out_np = np.asarray(out[0])
    ref_top = A_np[:load_rows] @ B_np
    err_top = float(np.max(np.abs(out_np[:load_rows] - ref_top)))
    full_ref = A_np @ B_np
    err_full = float(np.max(np.abs(out_np - full_ref)))
    if err_top < 1e-3:
        if err_full < 1e-3:
            verdict = "AMBIGUOUS"
        else:
            verdict = "BOUNDED (top rows match truncated, full disagrees)"
    else:
        verdict = "UNBOUNDED (top rows do NOT match A[:LR] @ B)"
    return err_top, err_full, verdict


def test_truncated_store(store_rows: int):
    """T3: C is pre-filled with -1.0. tensor_inline over C with extents
    (store_rows, 32). Expect rows >= store_rows of C remain at -1.0.
    """
    SENTINEL = -1.0
    src = make_source_truncated_store(store_rows)
    name = f"bounded_view_T3_store_{store_rows}"
    k = mx.fast.metal_kernel(
        name=name, input_names=["A", "B", "Cinit"], output_names=["C"],
        source="""
            // copy Cinit to C as initial state, then run the matmul kernel
            uint tid = thread_position_in_threadgroup.x;
            for (uint i = tid; i < 32*32; i += 32) C[i] = ((device float*)Cinit)[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        """ + src,
        header=HEADER,
    )
    A_np = np.random.RandomState(0).randint(-3, 4, (32, 32)).astype(np.float32)
    B_np = np.random.RandomState(1).randint(-3, 4, (32, 32)).astype(np.float32)
    Cinit_np = np.full((32, 32), SENTINEL, dtype=np.float32)
    out = k(inputs=[mx.array(A_np), mx.array(B_np), mx.array(Cinit_np)],
            grid=(THREADGROUP_SIZE, 1, 1),
            threadgroup=(THREADGROUP_SIZE, 1, 1),
            output_shapes=[(32, 32)], output_dtypes=[mx.float32])
    mx.eval(out[0])
    out_np = np.asarray(out[0])
    ref_top = A_np @ B_np
    err_top = float(np.max(np.abs(out_np[:store_rows] - ref_top[:store_rows])))
    # Rows >= store_rows: we want them to still be -1.0
    bottom = out_np[store_rows:]
    bottom_unchanged = np.all(bottom == SENTINEL)
    bottom_max_diff_from_sentinel = float(np.max(np.abs(bottom - SENTINEL)))
    if err_top < 1e-3 and bottom_unchanged:
        verdict = "BOUNDED (top rows match, bottom rows untouched)"
    elif err_top < 1e-3 and not bottom_unchanged:
        verdict = f"UNBOUNDED-STORE (bottom rows changed; max|diff_from_sentinel|={bottom_max_diff_from_sentinel:.3g})"
    else:
        verdict = f"WRONG (top rows mismatch, err_top={err_top:.3g})"
    return err_top, bottom_max_diff_from_sentinel, verdict


def main():
    print("MPP tensor_inline bounded-view probe")
    print("=" * 70)
    print()

    print("[T1] Aligned 32x32 baseline (tensor_inline, full extents)")
    try:
        err, verdict = test_baseline()
        print(f"     max|err| = {err:.4g}  -> {verdict}")
    except Exception as e:
        print(f"     ERR: {str(e).splitlines()[0][:120]}")
        return 1
    print()

    for lr in [20, 8, 1]:
        print(f"[T2] Truncated load: tensor_inline extents=({lr}, 32), sentinel rows {lr}..31")
        try:
            err_top, err_full, verdict = test_truncated_load(lr)
            print(f"     err_top(rows[:{lr}] vs A[:{lr}]@B) = {err_top:.4g}")
            print(f"     err_full(all 32 rows vs full A@B)   = {err_full:.4g}")
            print(f"     -> {verdict}")
        except Exception as e:
            print(f"     ERR: {str(e).splitlines()[0][:120]}")
        print()

    for lr in [20, 8, 1]:
        print(f"[T2b] Truncated load with DEFAULT tag (likely tensor_handle), extents=({lr}, 32)")
        try:
            err_top, err_full, verdict = test_truncated_load_default_tag(lr)
            print(f"     err_top = {err_top:.4g}    err_full = {err_full:.4g}")
            print(f"     -> {verdict}")
        except Exception as e:
            print(f"     ERR: {str(e).splitlines()[0][:120]}")
        print()

    for sr in [20, 8, 1]:
        print(f"[T3] Truncated store: tensor_inline extents=({sr}, 32), C rows {sr}..31 sentinel=-1")
        try:
            err_top, bot_diff, verdict = test_truncated_store(sr)
            print(f"     err_top(rows[:{sr}] vs A@B[:{sr}]) = {err_top:.4g}")
            print(f"     max|C[{sr}:] - sentinel|              = {bot_diff:.4g}")
            print(f"     -> {verdict}")
        except Exception as e:
            print(f"     ERR: {str(e).splitlines()[0][:120]}")
        print()


if __name__ == "__main__":
    sys.exit(main() or 0)
