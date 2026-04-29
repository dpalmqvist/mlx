"""
Mimics gemm_loop's K-accumulation pattern with a single simdgroup to isolate
whether the K-multiple-of-bk failure is per-simdgroup or multi-simdgroup.

Loads A (32 x K), B (K x 32) from device, runs N_iter inner iterations of
NAXFrag32 mma accumulating into Dtile, stores 32x32 result back. K = 32 * N_iter.

If this passes for N_iter=8 (K=256) but fails for N_iter=16 (K=512), the bug
is in the per-simdgroup mma accumulation. If both pass, the bug is in
multi-simdgroup interactions or kernel infrastructure.
"""
import sys

import mlx.core as mx
import numpy as np

THREADGROUP_SIZE = 32

HEADER = open(
    "/Users/daniel/.config/superpowers/worktrees/mlx/nax-g16-phase2/tools/probe_nax_frag32.py"
).read()
import re
m = re.search(r'^HEADER = """(.+?)"""', HEADER, re.DOTALL | re.MULTILINE)
NAX_HEADER = m.group(1)


def make_source(n_iter: int):
    """Stage A and B into threadgroup scratch one 32x32 block at a time, run
    NAXFrag32::mma on each block, accumulate into Dframe.
    """
    return f"""
constexpr short N_ITER = {n_iter};
constexpr short K = N_ITER * 32;

threadgroup float scratch[32 * 32];

NAXFrag32::dtype_frag_t<float> a_frag, b_frag, d_frag;

// Init Dframe to zero
for (uint16_t i = 0; i < NAXFrag32::kElemsPerFrag; ++i) d_frag[i] = 0.0f;

// Loop over K in 32-wide blocks
for (short kk = 0; kk < K; kk += 32) {{
    // Load Atile block (rows 0..31, cols kk..kk+32) via load_rows
    NAXFrag32::load_rows(a_frag, A + kk, K, /*row_lim=*/(short)32, scratch);
    // load_rows ends with a barrier and cT.load — Atile in registers now.

    // Load Btile block (rows kk..kk+32, cols 0..31) via load_rows
    NAXFrag32::load_rows(b_frag, B + kk * 32, 32, /*row_lim=*/(short)32, scratch);

    // Accumulate into d_frag
    NAXFrag32::mma(
        d_frag, a_frag, metal::bool_constant<false>{{}},
        b_frag, metal::bool_constant<false>{{}});
}}

// Store result
NAXFrag32::store_rows(d_frag, C, /*ld=*/(int)32, /*row_lim=*/(short)32, scratch);
"""


def test_n_iter(n_iter: int):
    K = 32 * n_iter
    src = make_source(n_iter)
    name = f"multik_accum_{n_iter}"
    k = mx.fast.metal_kernel(
        name=name, input_names=["A", "B"], output_names=["C"],
        source=src, header=NAX_HEADER,
    )
    np.random.seed(0)
    A_np = np.random.randint(-3, 4, (32, K)).astype(np.float32)
    B_np = np.random.randint(-3, 4, (K, 32)).astype(np.float32)
    A = mx.array(A_np); B = mx.array(B_np)
    out = k(inputs=[A, B], grid=(32, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[(32, 32)], output_dtypes=[mx.float32])
    mx.eval(out[0])
    out_np = np.asarray(out[0])
    ref = A_np @ B_np  # exact for small ints
    err = float(np.max(np.abs(out_np - ref)))
    return n_iter, K, err


def main():
    print("Multi-K accumulation probe (single simdgroup)")
    print("=" * 60)
    failed = 0
    for n in [1, 2, 4, 8, 9, 12, 16, 24, 32]:
        try:
            n_iter, K, err = test_n_iter(n)
            verdict = "OK" if err < 1e-3 else "FAIL"
            print(f"  n_iter={n_iter:2d}  K={K:4d}  max|err|={err:>10.4g}  {verdict}")
            if err >= 1e-3:
                failed += 1
        except Exception as e:
            print(f"  n_iter={n}  ERR: {str(e).splitlines()[0][:120]}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
