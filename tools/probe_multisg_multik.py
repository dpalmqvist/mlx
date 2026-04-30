"""
Multi-simdgroup × multi-K-iteration probe.

Mimics the production gemm_loop pattern: N_SG simdgroups, each running its own
K-accumulation loop. Each simdgroup uses a per-simdgroup scratch slice from a
shared scratch_buf. Tests whether the K-multiple-of-bk failure observed in
production is reproducible at this minimal level.

Setup:
- N_SG = 8 simdgroups (matches wm=2, wn=4 of production g16 kernel).
- Each simdgroup processes a 32x32 output tile.
- N_iter inner iterations of NAXFrag32 mma.
- Total K = 32 * N_iter.
- All simdgroups have valid (full 32x32) bounds — no out-of-bounds dance.

If this fails for n_iter=16 (K=512), the bug reproduces at this level —
fundamentally a multi-simdgroup multi-K issue with shared scratch_buf.
"""
import sys
import re

import mlx.core as mx
import numpy as np


import os as _os
_HEADER_FILE = open(
    _os.path.join(_os.path.dirname(__file__), "probe_nax_frag32.py")
).read()
m = re.search(r'^HEADER = """(.+?)"""', _HEADER_FILE, re.DOTALL | re.MULTILINE)
NAX_HEADER = m.group(1)


def make_source(n_sg: int, n_iter: int):
    """Each simdgroup i gets its own 32-row block of A (rows [i*32, (i+1)*32))
    and computes its 32x32 output. K is fixed (= 32 * n_iter).

    Layout: A is (n_sg*32, K), B is (K, 32), C is (n_sg*32, 32). Each simdgroup
    handles output rows [sg*32, (sg+1)*32). All N_SG simdgroups read the same
    B but different A slices.
    """
    return f"""
constexpr short N_SG = {n_sg};
constexpr short N_ITER = {n_iter};
constexpr short K = N_ITER * 32;

// Per-simdgroup 32x32 = 1024 floats of scratch
threadgroup float scratch_buf[N_SG * 32 * 32];
const uint sg_id = thread_position_in_threadgroup.x / 32;
threadgroup float* scratch = scratch_buf + sg_id * (32 * 32);

NAXFrag32::dtype_frag_t<float> a_frag, b_frag, d_frag;

for (uint16_t i = 0; i < NAXFrag32::kElemsPerFrag; ++i) d_frag[i] = 0.0f;

// Each simdgroup's A pointer offset
const device float* sg_A = A + (uint(sg_id) * 32u) * uint(K);

for (short kk = 0; kk < K; kk += 32) {{
    // A block: rows [sg*32, sg*32+32), cols [kk, kk+32)
    NAXFrag32::load_rows(a_frag, sg_A + kk, K, /*row_lim=*/(short)32, scratch);

    // B block: rows [kk, kk+32), cols [0, 32). Same for all simdgroups.
    NAXFrag32::load_rows(b_frag, B + kk * 32, 32, /*row_lim=*/(short)32, scratch);

    NAXFrag32::mma(
        d_frag, a_frag, metal::bool_constant<false>{{}},
        b_frag, metal::bool_constant<false>{{}});
}}

// Store: rows [sg*32, sg*32+32), cols [0, 32)
device float* sg_C = C + (uint(sg_id) * 32u) * 32u;
NAXFrag32::store_rows(d_frag, sg_C, /*ld=*/(int)32, /*row_lim=*/(short)32, scratch);
"""


def test(n_sg: int, n_iter: int):
    K = 32 * n_iter
    M = n_sg * 32
    src = make_source(n_sg, n_iter)
    name = f"multisg_multik_sg{n_sg}_n{n_iter}"
    k = mx.fast.metal_kernel(
        name=name, input_names=["A", "B"], output_names=["C"],
        source=src, header=NAX_HEADER,
    )
    np.random.seed(0)
    A_np = np.random.randint(-3, 4, (M, K)).astype(np.float32)
    B_np = np.random.randint(-3, 4, (K, 32)).astype(np.float32)
    A = mx.array(A_np); B = mx.array(B_np)
    out = k(inputs=[A, B], grid=(n_sg * 32, 1, 1), threadgroup=(n_sg * 32, 1, 1),
            output_shapes=[(M, 32)], output_dtypes=[mx.float32])
    mx.eval(out[0])
    out_np = np.asarray(out[0])
    ref = A_np @ B_np
    err_per_sg = []
    for s in range(n_sg):
        e = float(np.max(np.abs(out_np[s*32:(s+1)*32] - ref[s*32:(s+1)*32])))
        err_per_sg.append(e)
    err = float(np.max(np.abs(out_np - ref)))
    return err, err_per_sg


def main():
    print("Multi-simdgroup × multi-K-iter probe")
    print("=" * 70)
    failed = 0
    cases = [
        (1, 8), (1, 16),    # baseline single sg
        (2, 8), (2, 16),    # 2 sg
        (4, 8), (4, 16),    # 4 sg
        (8, 8), (8, 16),    # 8 sg (matches production wm=2, wn=4)
    ]
    for n_sg, n_iter in cases:
        try:
            err, per_sg = test(n_sg, n_iter)
            verdict = "OK" if err < 1e-3 else "FAIL"
            print(f"  n_sg={n_sg}  n_iter={n_iter:2d}  K={32*n_iter:4d}  max|err|={err:>9.4g}  {verdict}")
            if err >= 1e-3:
                failed += 1
                # Print per-sg error
                print(f"      per-sg errors: {[f'{e:.3g}' for e in per_sg]}")
        except Exception as e:
            print(f"  n_sg={n_sg}  n_iter={n_iter}  ERR: {str(e).splitlines()[0][:100]}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
