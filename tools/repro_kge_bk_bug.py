"""K>=bk repro driver — hits production NAX gemm via mx.matmul.

Sweeps K for fixed M=N=128 to isolate which branches of gemm_loop are at fault:

  K=128  -> gemm_k_iters_aligned=0, remainder=128  (only !kAlignedK branch)
  K=256  -> 1 outer iter,           remainder=0    (only kAlignedK branch, 1 iter)
  K=257  -> 1 outer iter,           remainder=1    (both branches)
  K=384  -> 1 outer iter,           remainder=128  (both branches)
  K=512  -> 2 outer iters,          remainder=0    (only kAlignedK branch, 2 iters)
  K=640  -> 2 outer iters,          remainder=128  (both branches, 2 iters)
  K=768  -> 3 outer iters,          remainder=0
  K=1024 -> 4 outer iters,          remainder=0
"""
import sys
import mlx.core as mx
import numpy as np


BM, BN, BK = 64, 128, 256


def run(M, N, K, seed=0):
    np.random.seed(seed)
    A_np = np.random.randint(-3, 4, (M, K)).astype(np.float32)
    B_np = np.random.randint(-3, 4, (K, N)).astype(np.float32)
    A = mx.array(A_np)
    B = mx.array(B_np)
    out = A @ B
    mx.eval(out)
    err = float(np.max(np.abs(np.asarray(out) - (A_np @ B_np))))
    return err


def sweep(M, N, Ks):
    print(f"M={M}  N={N}  (BM={BM}, BN={BN}, BK={BK})")
    print(f"  {'K':>5}  {'iters':>5}  {'rem':>4}  {'kAlignedK':>9}  {'max|err|':>10}  verdict")
    print("-" * 78)
    failed = 0
    for K in Ks:
        iters = K // BK
        rem = K - iters * BK
        aligned_k = (rem == 0)
        try:
            err = run(M, N, K)
            verdict = "OK" if err < 1e-3 else "FAIL"
            print(f"  {K:>5}  {iters:>5}  {rem:>4}  {str(aligned_k):>9}  {err:>10.4g}  {verdict}")
            if err >= 1e-3:
                failed += 1
        except Exception as e:
            print(f"  {K:>5}  ERR: {str(e).splitlines()[0][:100]}")
            failed += 1
    return failed


def main():
    Ks = [128, 256, 257, 384, 512, 640, 768, 1024]
    failed = 0
    for (M, N) in [(128, 128), (256, 256), (64, 128), (128, 256)]:
        print("=" * 78)
        failed += sweep(M, N, Ks)
        print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
