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
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
using namespace metal;
using namespace mlx::steel;
"""


def test_mma_basic():
    """Single-frag MMA: C(32x32) = A(32x32) @ B(32x32), all in registers."""
    src = """
    using Frag = NAXFrag32;
    Frag::dtype_frag_t<float> a, b, c;

    // TODO(phase1): load A from device into `a`, B into `b`, init `c` to zero,
    // call Frag::mma(c, a, b), store `c` back to device.
    // Placeholder until Phase 1 lands the load/store/mma helpers:
    for (uint16_t i = 0; i < 32; ++i) {
        c[i] = 0.0f;
    }
    """
    raise NotImplementedError(
        "test_mma_basic: implement after NAXFrag32::load/mma/store land "
        "in Phase 1, Task 1.6"
    )


def main():
    tests = [test_mma_basic]
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
