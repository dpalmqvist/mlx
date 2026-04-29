"""
Multi-simdgroup probe for the K>=129 failure in NAXFrag32 gemm_loop.

This probe directly mimics what gemm_loop does for the partial K block (psk=1):
- 4 simdgroups (matching BN=64, WM=2, WN=2 g16s kernel)
- Each simdgroup loads a 32x32 A frag with col_lim=1 (partial K block)
- Then loads a full 32x32 B frag
- Runs NAXFrag32::mma
- Stores result via store_rows

The probe tests whether partial-column Atile load + full Btile load + mma works
correctly when multiple simdgroups run concurrently with shared scratch_buf.

Simdgroup 0,1: M rows 0..31 (valid, sgp_sm=32)
Simdgroup 2,3: M rows 32..63 (dead, sgp_sm=0)
"""
import sys

import mlx.core as mx
import numpy as np

HEADER = open(
    "/Users/daniel/.config/superpowers/worktrees/mlx/nax-g16-phase2/tools/probe_nax_frag32.py"
).read()
# Extract just the header part (the HEADER variable value)
import re
m = re.search(r'^HEADER = """(.+?)"""', HEADER, re.DOTALL | re.MULTILINE)
NAX_HEADER = m.group(1)

KERNEL_HEADER = NAX_HEADER + """

// Minimal NAXTile-like wrapper for this probe
// Only implements load_safe and store_rows for kPacking==1

"""

def test_multisg_partial_k_load():
    """
    4 simdgroups run concurrently.
    Each loads a 32x32 A frag with col_lim=1 (partial K block psk=1).
    B is identity. C = A[:, :1] * I (for col 0 only).
    Simdgroups 0,1 own M rows 0..31; simdgroups 2,3 have dead rows (sgp_sm=0).
    Expected: C[m, n] = A[m, 0] if n==0 else 0 for simdgroups 0,1.
    """
    # 4 simdgroups, 32 lanes each = 128 threads total
    # grid=(128,1,1), threadgroup=(128,1,1)
    src = """
    using Frag = NAXFrag32;
    constexpr short kFragRows = 32;
    constexpr short kFragCols = 32;

    const uint sg_id = thread_position_in_threadgroup.x / 32;
    const uint lane  = thread_position_in_threadgroup.x % 32;

    // Shared scratch: 4 simdgroups * 1024 floats each
    threadgroup float scratch_buf[4 * 32 * 32];
    threadgroup float* sg_scratch = scratch_buf + sg_id * (32 * 32);

    // Simdgroups 0,1: valid (sgp_sm=32, sgp_sn=32)
    // Simdgroups 2,3: dead (sgp_sm=0, sgp_sn=32)
    // sg_id / 2 = wm index; sg_id % 2 = wn index
    const short tm = (sg_id / 2) * 32;  // sg0,sg1: tm=0; sg2,sg3: tm=32
    const short tn = (sg_id % 2) * 32;  // sg0,sg2: tn=0; sg1,sg3: tn=32
    const short sgp_sm = max((short)0, (short)min((int)32, (int)(32 - tm)));  // sg0,sg1: 32; sg2,sg3: 0
    const short sgp_sn = max((short)0, (short)min((int)32, (int)(64 - tn)));  // all: 32

    // Partial K block: col_lim=1, row_lim=sgp_sm
    // Atile load_safe
    Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;
    {
        const device float* a_src = A + tm * 32;  // A is [64, 32] shaped, row stride 32
        const short rl = sgp_sm;   // 32 for sg0,1; 0 for sg2,3
        const short cl = 1;        // partial K, col_lim=1
        Frag::load_safe(a_frag, a_src, (int)32, rl, cl, sg_scratch);
    }

    // Btile load (B is identity [32x32]), full load
    // All simdgroups load same B (no tn offset for simplicity since identity)
    {
        const device float* b_src = B + tn * 32;  // B is [64, 32], row stride 32
        const short rl = 32;
        const short cl = sgp_sn;
        Frag::load_safe(b_frag, b_src, (int)32, rl, cl, sg_scratch);
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 32; ++i) c_frag[i] = 0.0f;

    Frag::mma(c_frag,
              a_frag, metal::bool_constant<false>{},
              b_frag, metal::bool_constant<false>{});

    // Store result for valid simdgroups only
    if (sgp_sm > 0 && sgp_sn > 0) {
        Frag::store_safe(c_frag, C + tm * 32 + tn, (int)32, sgp_sm, sgp_sn, sg_scratch);
    }
    """
    k = mx.fast.metal_kernel(
        name="probe_multisg_partial_k",
        input_names=["A", "B"],
        output_names=["C"],
        source=src,
        header=NAX_HEADER,
    )
    # A is [64, 32] (M=64 rows, K=32 cols but only col 0 matters for psk=1)
    A_np = np.random.RandomState(42).randint(-3, 4, (64, 32)).astype(np.float32)
    # B is [64, 32] split into 2 simdgroup regions of [32, 32]:
    # sg0 (tn=0): B[0:32, 0:32] = identity; sg1 (tn=32): B[32:64, 0:32] = identity
    B_np = np.zeros((64, 32), dtype=np.float32)
    B_np[:32, :32] = np.eye(32, dtype=np.float32)  # sg0's B portion
    B_np[32:64, :32] = np.eye(32, dtype=np.float32)  # sg1's B portion

    A = mx.array(A_np)
    B = mx.array(B_np)
    out = k(
        inputs=[A, B],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(64, 32)],  # 64 rows (only 0..31 valid), 32 cols (only 0..31 valid)
        output_dtypes=[mx.float32],
        init_value=0,
    )
    mx.eval(out[0])
    result = np.asarray(out[0])

    # Expected: C[m, n] = sum_k A_masked[m,k] * B[k,n]
    # A_masked[:, 1:] = 0 (col_lim=1), A_masked[:32, :] = A[:32, :] (row_lim=32 for sg0,1)
    # A_masked[32:, :] = 0 (dead simdgroups)
    # B = identity (for each simdgroup's B tile)
    # But we need to account for tn offset: sg0 and sg1 each output to tn=0 and tn=32
    # C[:32, :32] = A[:32, :32] with col 0 filled, others 0 (sg0)
    # Wait, the MMA is A[32x32] @ B[32x32] -> C[32x32]
    # A for sg0: rows 0..31, cols 0..31 with only col 0 valid → A_masked[:32, 0:1]
    # B for sg0: rows 0..31, cols 0..31 = identity
    # C[m, n] = sum_k A_masked[m,k] * I[k,n] = A_masked[m,n]
    # But A_masked has only col 0 filled: C[m, n] = A[m, 0] if n==0, else 0

    ref = np.zeros((64, 32), dtype=np.float32)
    # sg0: rows 0..31, cols 0..31
    ref[:32, 0] = A_np[:32, 0]  # Only col 0 from A[:32, col=0] * I[col=0, n=0]

    # sg1 (tn=32): B for sg1 is identity, but stored at C[0:32, 32:64]... but our C is [64,32]
    # Hmm, this layout is getting complicated. Let me simplify.
    # Actually the tn=32 simdgroup writes to C[tm:tm+32, tn:tn+32] = C[0:32, 32:64]
    # but C is only [64, 32], so tn+32 = 64 > 32...
    # This probe setup is getting messy. Let me simplify to just test the partial load issue.

    err = float(np.max(np.abs(result[:32, :32] - ref[:32, :32])))
    print(f"  result[:32, :5] =\n{result[:32, :5]}")
    print(f"  ref[:32, :5] =\n{ref[:32, :5]}")
    if err >= 1e-3:
        bad = np.argwhere(np.abs(result[:32, :32] - ref[:32, :32]) >= 1e-3)
        raise AssertionError(f"max|err|={err:.4g}; {len(bad)} bad cells; first few: {bad[:5].tolist()}")
    print(f"  PASS max|err|={err:.6g}")


def test_single_sg_partial_k():
    """Single simdgroup baseline: partial-K load with col_lim=1, B=identity."""
    src = """
    using Frag = NAXFrag32;
    threadgroup float scratch[32 * 32];
    Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;
    Frag::load_safe(a_frag, (const device float*)A, 32, (short)32, (short)1, scratch);
    Frag::load(b_frag, (const device float*)B, (short)32);
    STEEL_PRAGMA_UNROLL
    for (uint16_t i = 0; i < 32; ++i) c_frag[i] = 0.0f;
    Frag::mma(c_frag,
              a_frag, metal::bool_constant<false>{},
              b_frag, metal::bool_constant<false>{});
    Frag::store(c_frag, (device float*)C, (short)32);
    """
    k = mx.fast.metal_kernel(
        name="probe_single_sg_partial_k",
        input_names=["A", "B"],
        output_names=["C"],
        source=src,
        header=NAX_HEADER,
    )
    A_np = np.random.RandomState(42).randint(-3, 4, (32, 32)).astype(np.float32)
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
    ref = np.zeros((32, 32), dtype=np.float32)
    ref[:, 0] = A_np[:, 0]  # Only col 0 valid, B=identity → C[:,n] = A[:,0]*I[0,n] = A[:,0] if n=0
    err = float(np.max(np.abs(np.asarray(out[0]) - ref)))
    if err >= 1e-3:
        bad = np.argwhere(np.abs(np.asarray(out[0]) - ref) >= 1e-3)
        raise AssertionError(f"max|err|={err:.4g}; {len(bad)} bad; first: {bad[:3].tolist()}")
    print(f"  PASS single-sg partial-K (col_lim=1): max|err|={err:.6g}")


def test_multisg_4sg_partial_k_simple():
    """
    4 simdgroups, ALL valid (sgp_sm=32 for all).
    Each SG loads a 32x32 A frag with col_lim=1.
    B = identity.
    Expected: C[m, n] = A[m, 0] if n==0 else 0.

    This tests the multi-simdgroup load_safe with partial col interleaved.
    If this fails where single-SG passes, the bug is in multi-SG threadgroup_barrier
    interaction with cooperative tensor load.
    """
    src = """
    using Frag = NAXFrag32;
    const uint sg_id = thread_position_in_threadgroup.x / 32;

    // 4 simdgroups, each with their own 32x32 scratch region
    threadgroup float scratch_buf[4 * 32 * 32];
    threadgroup float* sg_scratch = scratch_buf + sg_id * (32 * 32);

    Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;

    // All SGs load the same A (each sees the same 32x32 matrix)
    Frag::load_safe(a_frag, (const device float*)A, (int)32, (short)32, (short)1, sg_scratch);

    // Load B = identity
    Frag::load(b_frag, (const device float*)B, (short)32);

    STEEL_PRAGMA_UNROLL
    for (uint16_t i = 0; i < 32; ++i) c_frag[i] = 0.0f;

    Frag::mma(c_frag,
              a_frag, metal::bool_constant<false>{},
              b_frag, metal::bool_constant<false>{});

    // Store result to per-SG output region (sg_id * 32*32 offset)
    Frag::store(c_frag, (device float*)(C + sg_id * 32 * 32), (int)32);
    """
    k = mx.fast.metal_kernel(
        name="probe_multisg_4sg_partial_k_simple",
        input_names=["A", "B"],
        output_names=["C"],
        source=src,
        header=NAX_HEADER,
    )
    A_np = np.random.RandomState(42).randint(-3, 4, (32, 32)).astype(np.float32)
    B_np = np.eye(32, dtype=np.float32)
    A = mx.array(A_np)
    B = mx.array(B_np)
    out = k(
        inputs=[A, B],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(4 * 32 * 32,)],
        output_dtypes=[mx.float32],
        init_value=0,
    )
    mx.eval(out[0])
    result = np.asarray(out[0]).reshape(4, 32, 32)

    ref = np.zeros((32, 32), dtype=np.float32)
    ref[:, 0] = A_np[:, 0]

    failed = 0
    for sg in range(4):
        err = float(np.max(np.abs(result[sg] - ref)))
        if err >= 1e-3:
            bad = np.argwhere(np.abs(result[sg] - ref) >= 1e-3)
            print(f"  FAIL sg={sg} max|err|={err:.4g}; {len(bad)} bad; first: {bad[:3].tolist()}")
            failed += 1
        else:
            print(f"  PASS sg={sg} max|err|={err:.6g}")
    if failed:
        raise AssertionError(f"{failed}/4 simdgroups failed")


def test_multisg_8sg_partial_k_with_dead():
    """
    8 simdgroups: sg0..3 valid (load real A), sg4..7 dead (load_safe with rl=0).
    Tests whether dead simdgroups' zero-fill scratch writes corrupt the barrier
    ordering for valid simdgroups.

    This matches the g16s kernel with BM=64, BN=128, WM=2, WN=4 (8 simdgroups)
    where M=32 makes simdgroups 4..7 dead.
    """
    src = """
    using Frag = NAXFrag32;
    const uint sg_id = thread_position_in_threadgroup.x / 32;

    // 8 simdgroups
    threadgroup float scratch_buf[8 * 32 * 32];
    threadgroup float* sg_scratch = scratch_buf + sg_id * (32 * 32);

    // sg0..3: valid (sm=32); sg4..7: dead (sm=0)
    // (mimics WM=2, WN=4: sg0..3 have tm=0, sg4..7 have tm=32 with M=32)
    const short sgp_sm = (sg_id < 4) ? (short)32 : (short)0;

    Frag::dtype_frag_t<float> a_frag, b_frag, c_frag;

    // A load: col_lim=1, row_lim=sgp_sm
    // Dead SGs write zeros (rl=0), valid SGs write A column 0
    Frag::load_safe(a_frag, (const device float*)A, (int)32, sgp_sm, (short)1, sg_scratch);

    // B = identity, all SGs load it (full)
    Frag::load(b_frag, (const device float*)B, (short)32);

    STEEL_PRAGMA_UNROLL
    for (uint16_t i = 0; i < 32; ++i) c_frag[i] = 0.0f;

    Frag::mma(c_frag,
              a_frag, metal::bool_constant<false>{},
              b_frag, metal::bool_constant<false>{});

    // Valid SGs store to output
    if (sgp_sm > 0) {
        Frag::store(c_frag, (device float*)(C + sg_id * 32 * 32), (int)32);
    }
    """
    k = mx.fast.metal_kernel(
        name="probe_multisg_8sg_partial_k_dead",
        input_names=["A", "B"],
        output_names=["C"],
        source=src,
        header=NAX_HEADER,
    )
    A_np = np.random.RandomState(42).randint(-3, 4, (32, 32)).astype(np.float32)
    B_np = np.eye(32, dtype=np.float32)
    A = mx.array(A_np)
    B = mx.array(B_np)
    out = k(
        inputs=[A, B],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(8 * 32 * 32,)],
        output_dtypes=[mx.float32],
        init_value=0,
    )
    mx.eval(out[0])
    result = np.asarray(out[0]).reshape(8, 32, 32)

    ref = np.zeros((32, 32), dtype=np.float32)
    ref[:, 0] = A_np[:, 0]

    failed = 0
    for sg in range(4):  # only valid SGs
        err = float(np.max(np.abs(result[sg] - ref)))
        if err >= 1e-3:
            bad = np.argwhere(np.abs(result[sg] - ref) >= 1e-3)
            print(f"  FAIL sg={sg} (valid) max|err|={err:.4g}; {len(bad)} bad; first: {bad[:3].tolist()}")
            failed += 1
        else:
            print(f"  PASS sg={sg} (valid) max|err|={err:.6g}")
    if failed:
        raise AssertionError(f"{failed}/4 valid simdgroups failed")


def test_multisg_full_then_partial_k():
    """
    4 simdgroups, each running a sequence of:
    - N full K-block iterations (load_safe with full cl=32)
    - 1 partial K-block iteration (load_safe with cl=1)
    - Accumulate MMA result

    This most closely mimics the K=129 BK=256 scenario.
    K=129 = 4 full SK=32 iterations + 1 partial (psk=1).
    """
    src = """
    using Frag = NAXFrag32;
    const uint sg_id = thread_position_in_threadgroup.x / 32;

    // 4 simdgroups
    threadgroup float scratch_buf[4 * 32 * 32];
    threadgroup float* sg_scratch = scratch_buf + sg_id * (32 * 32);

    Frag::dtype_frag_t<float> c_frag;
    STEEL_PRAGMA_UNROLL
    for (uint16_t i = 0; i < 32; ++i) c_frag[i] = 0.0f;

    // 4 full iterations (simulate kk1=0,32,64,96 for K=129 with SK=32)
    for (int kk = 0; kk < 4; ++kk) {
        Frag::dtype_frag_t<float> a_frag, b_frag;
        const device float* a_src = A + kk * 32 * 32;   // A is [5*32, 32]: 5 K-blocks
        const device float* b_src = B + kk * 32 * 32;   // B is [5*32, 32]: 5 K-blocks
        Frag::load_safe(a_frag, a_src, (int)32, (short)32, (short)32, sg_scratch); // full
        Frag::load_safe(b_frag, b_src, (int)32, (short)32, (short)32, sg_scratch); // full
        Frag::mma(c_frag,
                  a_frag, metal::bool_constant<false>{},
                  b_frag, metal::bool_constant<false>{});
    }

    // 1 partial iteration (kk1=128, psk=1)
    {
        Frag::dtype_frag_t<float> a_frag, b_frag;
        const device float* a_src = A + 4 * 32 * 32;   // last block of A
        const device float* b_src = B + 4 * 32 * 32;   // last block of B
        Frag::load_safe(a_frag, a_src, (int)32, (short)32, (short)1, sg_scratch);  // partial!
        Frag::load_safe(b_frag, b_src, (int)32, (short)1, (short)32, sg_scratch);  // partial!
        Frag::mma(c_frag,
                  a_frag, metal::bool_constant<false>{},
                  b_frag, metal::bool_constant<false>{});
    }

    // Store to per-SG output
    Frag::store(c_frag, (device float*)(C + sg_id * 32 * 32), (int)32);
    """
    k = mx.fast.metal_kernel(
        name="probe_multisg_full_then_partial_k",
        input_names=["A", "B"],
        output_names=["C"],
        source=src,
        header=NAX_HEADER,
    )
    # A and B are each [5*32, 32] = [160, 32] (5 K-blocks of 32x32)
    rng = np.random.RandomState(7)
    A_np = rng.randint(-3, 4, (5 * 32, 32)).astype(np.float32)
    B_np = rng.randint(-3, 4, (5 * 32, 32)).astype(np.float32)

    # Reference: C = sum over 5 blocks, but last block partial (col 0 of A, row 0 of B)
    ref = np.zeros((32, 32), dtype=np.float32)
    for kk in range(4):
        ref += A_np[kk*32:(kk+1)*32] @ B_np[kk*32:(kk+1)*32]
    # Partial last block: A with cl=1 → only col 0, B with rl=1 → only row 0
    A_partial = A_np[4*32:5*32].copy()
    A_partial[:, 1:] = 0.0  # col_lim=1
    B_partial = B_np[4*32:5*32].copy()
    B_partial[1:, :] = 0.0  # row_lim=1
    ref += A_partial @ B_partial

    A = mx.array(A_np)
    B = mx.array(B_np)
    out = k(
        inputs=[A, B],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(4 * 32 * 32,)],
        output_dtypes=[mx.float32],
        init_value=0,
    )
    mx.eval(out[0])
    result = np.asarray(out[0]).reshape(4, 32, 32)

    failed = 0
    for sg in range(4):
        err = float(np.max(np.abs(result[sg] - ref)))
        if err >= 1e-3:
            diff = np.abs(result[sg] - ref)
            bad = np.argwhere(diff >= 1e-3)
            rows_wrong = np.unique(bad[:, 0]).tolist()
            print(f"  FAIL sg={sg} max|err|={err:.4g}; {len(bad)} bad cells; wrong rows: {rows_wrong}")
            # Show which rows are wrong vs ref
            for r in rows_wrong[:4]:
                print(f"    row {r}: got {result[sg][r, :4]}, ref {ref[r, :4]}")
            failed += 1
        else:
            print(f"  PASS sg={sg} max|err|={err:.6g}")
    if failed:
        raise AssertionError(f"{failed}/4 simdgroups failed")


def main():
    tests = [
        test_single_sg_partial_k,
        test_multisg_4sg_partial_k_simple,
        test_multisg_8sg_partial_k_with_dead,
        test_multisg_full_then_partial_k,
    ]
    failed = 0
    for t in tests:
        print(f"Running {t.__name__}...")
        try:
            t()
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
