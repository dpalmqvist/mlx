"""
Probe whether the NAX quantized-matmul path produces correct results on
the current GPU (M4 Pro / g16s, in this case).

Method: for each shape, compute the same quantized matmul twice —
  - NAX path enabled  (MLX_QMM_NAX_M_THRESHOLD=1, so any M triggers NAX)
  - NAX path disabled (MLX_QMM_NAX_M_THRESHOLD=10**9)
plus a dequantized fp32 reference. Report max-abs and relative errors.

Run (one shape at a time, since the threshold is read at process start):

    MLX_QMM_NAX_M_THRESHOLD=1     python tools/check_qmm_nax_correctness.py nax
    MLX_QMM_NAX_M_THRESHOLD=999999 python tools/check_qmm_nax_correctness.py ref

The driver at the bottom invokes both children and diffs the saved arrays.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path("/tmp/mlx_qmm_nax_probe")
ARTIFACT_DIR.mkdir(exist_ok=True)


# (M, N, K, group_size, bits, dtype) — all hit the NAX gate:
#   K % 64 == 0, transpose=True, M >= 1.
SHAPES = [
    (256,  4096, 4096, 64, 4, "float16"),
    (512,  4096, 4096, 64, 4, "float16"),
    (256, 11008, 4096, 64, 4, "float16"),
    (256,  4096, 4096, 64, 4, "float32"),  # tf32 path
    (256,  4096, 4096, 32, 4, "float16"),
    (256,  4096, 4096, 64, 8, "float16"),
]


def run_child(tag: str) -> None:
    import mlx.core as mx

    rng = np.random.default_rng(0)
    results = {}

    for shape_idx, (M, N, K, gs, bits, dtype_name) in enumerate(SHAPES):
        dtype = getattr(mx, dtype_name)
        x_np = rng.standard_normal((M, K)).astype(np.float32) * 0.1
        w_np = rng.standard_normal((N, K)).astype(np.float32) * 0.1

        x = mx.array(x_np).astype(dtype)
        w = mx.array(w_np).astype(dtype)

        w_q, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
        mx.eval(w_q, scales, biases)

        # Quantized matmul (transpose=True, the NAX-eligible mode).
        out = mx.quantized_matmul(
            x, w_q, scales=scales, biases=biases,
            transpose=True, group_size=gs, bits=bits,
        )
        mx.eval(out)
        results[f"out_{shape_idx}"] = np.asarray(out).astype(np.float32)

        # Dequantized reference, computed in numpy in fp64 — independent of
        # any MLX matmul kernel (which is the thing we're auditing).
        if tag == "nax":  # only need to save it once
            w_dq = mx.dequantize(w_q, scales=scales, biases=biases,
                                 group_size=gs, bits=bits)
            mx.eval(w_dq)
            w_dq_np = np.asarray(w_dq).astype(np.float64)
            x_np_d = np.asarray(x).astype(np.float64)
            ref = (x_np_d @ w_dq_np.T).astype(np.float32)
            results[f"ref_{shape_idx}"] = ref

    np.savez(ARTIFACT_DIR / f"{tag}.npz", **results)
    print(f"[{tag}] wrote {len(results)} arrays")


def diff() -> int:
    nax = np.load(ARTIFACT_DIR / "nax.npz")
    ref_path = ARTIFACT_DIR / "ref.npz"
    nonax = np.load(ref_path)

    print(f"{'shape':<35}  {'max|nax-nonax|':>15}  {'max|nax-deq|':>15}  "
          f"{'max|nonax-deq|':>15}  verdict")
    print("-" * 110)
    fail = 0
    for shape_idx, shape in enumerate(SHAPES):
        a = nax[f"out_{shape_idx}"]
        b = nonax[f"out_{shape_idx}"]
        ref = nax[f"ref_{shape_idx}"]
        d_ab = float(np.max(np.abs(a - b)))
        d_ar = float(np.max(np.abs(a - ref)))
        d_br = float(np.max(np.abs(b - ref)))

        # Tolerance: nonax-vs-deq is the quantization-noise floor; NAX should
        # not exceed it by more than a small slack (call it 4x).
        tol = max(4 * d_br, 1e-3)
        ok = d_ar <= tol
        if not ok:
            fail += 1
        verdict = "OK " if ok else "FAIL"
        print(f"{str(shape):<35}  {d_ab:>15.4g}  {d_ar:>15.4g}  "
              f"{d_br:>15.4g}  {verdict}")
    print("-" * 110)
    print(f"{'failures':<35}  {fail}")
    return fail


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("nax", "ref"):
        run_child(sys.argv[1])
        return

    py = sys.executable
    env_nax = {**os.environ, "MLX_QMM_NAX_M_THRESHOLD": "1"}
    env_ref = {**os.environ, "MLX_QMM_NAX_M_THRESHOLD": "999999999"}

    print("== running NAX-enabled child ==")
    subprocess.run([py, __file__, "nax"], env=env_nax, check=True)
    print("== running NAX-disabled child ==")
    subprocess.run([py, __file__, "ref"], env=env_ref, check=True)
    print("== diff ==")
    sys.exit(diff())


if __name__ == "__main__":
    main()
