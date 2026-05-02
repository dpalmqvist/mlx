"""Phase 8 prototype-A correctness probe: register-only NAXFrag32 load.

This probe runs end-to-end through the standard MLX matmul dispatch. After
load_direct is wired into NAXTile::load fast path (Step 5), this probe
will exercise it via the gemm path. If the wired-in load_direct produces
incorrect math, this probe will catch it via the checksum.
"""
import json
import sys
import mlx.core as mx
import numpy as np


def main() -> int:
    if not mx.metal.is_nax_available():
        print(json.dumps({"skip": "NAX not available"}))
        return 0
    flavor = mx.metal.nax_arch_flavor()
    if flavor != "g16":
        print(json.dumps({"skip": f"NAX flavor is {flavor}"}))
        return 0

    # Aligned, non-transposed matmul that hits the fast path.
    a = mx.random.normal((128, 128)).astype(mx.float16)
    b = mx.random.normal((128, 128)).astype(mx.float16)
    c = a @ b
    mx.eval(c)

    a_np = np.array(a, copy=True).astype(np.float32)
    b_np = np.array(b, copy=True).astype(np.float32)
    c_ref = a_np @ b_np

    err = float(np.max(np.abs(np.array(c).astype(np.float32) - c_ref)))
    rel = err / max(1e-9, float(np.max(np.abs(c_ref))))
    ok = rel < 5e-3

    print(json.dumps({
        "probe": "naxfrag32_direct_load",
        "max_abs_err": err,
        "max_rel_err": rel,
        "ok": ok,
    }))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
