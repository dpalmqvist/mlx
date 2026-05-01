"""Probe: MLX_DISABLE_NAX runtime gate.

Run this twice:
  $ /Users/daniel/mlx/.venv/bin/python tools/probe_nax_disable_gate.py
  $ MLX_DISABLE_NAX=1 /Users/daniel/mlx/.venv/bin/python tools/probe_nax_disable_gate.py

Each run prints a JSON line with:
  - MLX_DISABLE_NAX_env: the env var as the harness set it
  - nax_available: what is_nax_available() returns (bool)
  - nax_flavor: what nax_arch_flavor() returns (one of "none","g16","g17plus")
  - checksum: sum of a fixed-seed 128x128 fp16 matmul

Expected behavior with the gate working on g16 hardware:
  - unset / "" / "0":  nax_available=true,  nax_flavor in {"g16","g17plus"}
  - "1" / anything else: nax_available=false, nax_flavor="none"
  - checksums agree to fp16 tolerance across both arms (proves the
    non-NAX path produces the same answer as the NAX path).

If both arms print the same nax_available, the gate did not flip and any
downstream timing harness will be measuring noise.
"""

import json
import os

import mlx.core as mx


def main():
    env = os.environ.get("MLX_DISABLE_NAX", "")
    mx.random.seed(42)
    a = mx.random.normal((128, 128)).astype(mx.float16)
    b = mx.random.normal((128, 128)).astype(mx.float16)
    c = a @ b
    mx.eval(c)
    checksum = float(c.astype(mx.float32).sum().item())

    print(json.dumps({
        "MLX_DISABLE_NAX_env": env,
        "nax_available": mx.metal.is_nax_available(),
        "nax_flavor": mx.metal.nax_arch_flavor(),
        "checksum": checksum,
    }))


if __name__ == "__main__":
    main()
