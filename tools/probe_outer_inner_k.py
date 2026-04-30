"""
Mimics gemm_loop's nested outer + inner K-loop pattern with pointer advancement
between outer iterations. The production kernel:

  for (kk0 = 0; kk0 < gemm_k_iterations_; kk0++) {
    for (kk1 = 0; kk1 < BK; kk1 += SK) { load, mma }
    A += BK;
    B += BK * ldb;
  }

Tests whether the K-multiple-of-bk failure reproduces with this exact pattern.
"""
import sys
import re

import mlx.core as mx
import numpy as np


_HF = open("/Users/daniel/.config/superpowers/worktrees/mlx/nax-g16-phase2/tools/probe_nax_frag32.py").read()
NAX_HEADER = re.search(r'^HEADER = """(.+?)"""', _HF, re.DOTALL | re.MULTILINE).group(1)


def make_source(n_sg: int, gemm_k_iters: int, bk: int):
    """A is (n_sg*32, K), B is (K, 32), C is (n_sg*32, 32). Inner SK=32.
    Each simdgroup handles its 32-row slice. K = gemm_k_iters * bk.
    """
    return f"""
constexpr short N_SG = {n_sg};
constexpr short GEMM_K_ITERS = {gemm_k_iters};
constexpr short BK = {bk};
constexpr short SK = 32;
constexpr short K = GEMM_K_ITERS * BK;

threadgroup float scratch_buf[N_SG * 32 * 32];
const uint sg_id = thread_position_in_threadgroup.x / 32;
threadgroup float* scratch = scratch_buf + sg_id * (32 * 32);

NAXFrag32::dtype_frag_t<float> a_frag, b_frag, d_frag;
for (uint16_t i = 0; i < NAXFrag32::kElemsPerFrag; ++i) d_frag[i] = 0.0f;

// Per-simdgroup A pointer (M-row offset)
const device float* A_ptr = A + (uint(sg_id) * 32u) * uint(K);
const device float* B_ptr = B;  // shared across simdgroups

const int lda = K;
const int ldb = 32;

// Outer K loop
for (short kk0 = 0; kk0 < GEMM_K_ITERS; kk0++) {{
    threadgroup_barrier(mem_flags::mem_none);

    // Inner K loop
    for (short kk1 = 0; kk1 < BK; kk1 += SK) {{
        const int A_off = kk1;
        const int B_off = kk1 * ldb;

        NAXFrag32::load_rows(a_frag, A_ptr + A_off, lda, /*row_lim=*/(short)32, scratch);
        NAXFrag32::load_rows(b_frag, B_ptr + B_off, ldb, /*row_lim=*/(short)32, scratch);

        NAXFrag32::mma(
            d_frag, a_frag, metal::bool_constant<false>{{}},
            b_frag, metal::bool_constant<false>{{}});
    }}

    A_ptr += BK;
    B_ptr += BK * ldb;
}}

device float* sg_C = C + (uint(sg_id) * 32u) * 32u;
NAXFrag32::store_rows(d_frag, sg_C, /*ld=*/(int)32, /*row_lim=*/(short)32, scratch);
"""


def test(n_sg: int, gemm_k_iters: int, bk: int):
    K = gemm_k_iters * bk
    M = n_sg * 32
    src = make_source(n_sg, gemm_k_iters, bk)
    name = f"outer_inner_sg{n_sg}_iters{gemm_k_iters}_bk{bk}"
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
    err = float(np.max(np.abs(out_np - ref)))
    return err


def main():
    print("Outer/inner K-loop probe (mimics gemm_loop pattern)")
    print("=" * 70)
    cases = [
        # (n_sg, gemm_k_iters, bk) — K = iters * bk
        (1, 1, 256),    # K=256, single iter, single sg
        (1, 2, 256),    # K=512, two outer iters
        (8, 1, 256),    # K=256, single iter, 8 sg (matches 128x128x256, PASSES in production)
        (8, 2, 256),    # K=512, two outer iters, 8 sg (matches 128x128x512, FAILS in production)
        (8, 3, 256),    # K=768
        (8, 4, 256),    # K=1024
    ]
    failed = 0
    for n_sg, gki, bk in cases:
        try:
            err = test(n_sg, gki, bk)
            verdict = "OK" if err < 1e-3 else "FAIL"
            print(f"  n_sg={n_sg}  gemm_k_iters={gki}  bk={bk}  K={gki*bk:4d}  max|err|={err:>9.4g}  {verdict}")
            if err >= 1e-3:
                failed += 1
        except Exception as e:
            print(f"  n_sg={n_sg}  gki={gki}  bk={bk}  ERR: {str(e).splitlines()[0][:100]}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
