"""
Regression sweep across gather_mm bm dispatch paths on g16.

Exercises:
  bm=64 (M/E > 48)  — NAX (NAXFrag32 on g16)
  bm=32 (24 < M/E <= 48) — NAX (NAXFrag32 on g16)
  bm=16 (M/E <= 24) — falls through to non-NAX gather_mm_rhs on g16

Compares against the per-row reference `a[i] @ b[rhs[i]]` at fp32 rtol/atol 1e-4.
"""
import sys
import mlx.core as mx
import numpy as np  # only for printing nicer error magnitudes if needed


def run(M, N, K, E, name):
    """Run a single shape; return True on pass, False on fail."""
    a = mx.random.normal((M, 1, K), dtype=mx.float32)
    b = mx.random.normal((E, K, N), dtype=mx.float32)
    rhs = mx.sort(mx.random.randint(0, E, shape=(M,)))

    c_ref = a @ b[rhs]
    c_test = mx.gather_mm(a, b, rhs_indices=rhs, sorted_indices=True)

    ok = mx.allclose(c_ref, c_test, rtol=1e-4, atol=1e-4).item()
    if not ok:
        err = mx.max(mx.abs(c_ref - c_test)).item()
        ref_max = mx.max(mx.abs(c_ref)).item()
        rel = err / max(ref_max, 1e-9)
        print(f"  FAIL {name}: M={M}, N={N}, K={K}, E={E}, M/E={M/E:.1f}, "
              f"max|err|={err:.4e}, rel={rel:.4e}")
        return False
    print(f"  OK   {name}: M={M}, N={N}, K={K}, E={E}, M/E={M/E:.1f}")
    return True


if __name__ == "__main__":
    print("Gather sweep across bm paths on g16")
    mx.random.seed(0)
    cases = [
        # (M, N, K, E, name)
        (512, 128, 128, 8, "bm64_balanced"),
        (512, 128, 256, 4, "bm64_large_k"),
        (256, 128, 128, 8, "bm32_boundary"),
        (200, 128, 128, 8, "bm32_lower"),
        (128, 128, 128, 8, "bm16_decode"),
        (64,  128, 128, 8, "bm16_small"),
        (8,   128, 128, 8, "bm16_minimal"),
    ]
    failed = 0
    for M, N, K, E, name in cases:
        if not run(M, N, K, E, name):
            failed += 1
    if failed:
        print(f"FAILED: {failed}/{len(cases)} cases")
        sys.exit(1)
    print(f"All {len(cases)} cases passed")
