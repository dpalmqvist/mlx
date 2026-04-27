# Copyright © 2026 Apple Inc.
"""Phase 1 follow-up: programmatic Metal frame capture of the quantized vector
SDPA worst cell.

Writes a .gputrace file containing a single quantized SDPA dispatch (after
warmup), which can be opened in Xcode for counter inspection per §5 of
docs/superpowers/reports/2026-04-27-sdpa-quantized-profile.md.

Requires MLX built with -DMLX_METAL_DEBUG=ON and the env var
MTL_CAPTURE_ENABLED=1 to allow programmatic capture.

Default target is the worst cell from §3 of that report:
qwen-gqa @ ctx=32768 (q4/fp16 = 2.79x).
"""

import argparse
import math
import os
import sys

import mlx.core as mx

from sdpa_vector_quantized_bench import (
    BITS,
    GROUP_SIZE,
    MODELS,
    dq_then_sdpa,
    fp16_sdpa,
    make_inputs,
    q4_sdpa,
)

VARIANTS = ("q4_sdpa", "fp16_sdpa", "dq_then_sdpa")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen-gqa", choices=sorted(MODELS))
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Iterations executed inside the capture window. Matches the "
        "MLX metal-debugger example (>1 reliably produces a non-empty "
        "trace bundle).",
    )
    p.add_argument(
        "--variant",
        default="q4_sdpa",
        choices=VARIANTS,
        help="Which kernel to capture.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output .gputrace path. Defaults to /tmp/sdpa_<variant>_<model>_<ctx>.gputrace. Must not already exist.",
    )
    args = p.parse_args()
    if args.out is None:
        args.out = (
            f"/tmp/sdpa_{args.variant}_{args.model.replace('-', '_')}"
            f"_{args.ctx}.gputrace"
        )

    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        sys.exit(
            "MTL_CAPTURE_ENABLED=1 must be set in the environment for "
            "programmatic capture. Re-run as:\n"
            "  MTL_CAPTURE_ENABLED=1 python "
            "benchmarks/python/sdpa_vector_quantized_capture.py ..."
        )
    if os.path.exists(args.out):
        sys.exit(
            f"refusing to overwrite existing capture at {args.out}; "
            "remove it or pass --out elsewhere"
        )

    shape = MODELS[args.model]
    print(
        f"capturing {args.variant} on {args.model} (Hq={shape.Hq} "
        f"Hk={shape.Hk} D={shape.D}) ctx={args.ctx} bits={BITS} "
        f"group_size={GROUP_SIZE}",
        flush=True,
    )

    q, k, v, qk_pack, qv_pack = make_inputs(shape, args.ctx)
    scale = 1.0 / math.sqrt(shape.D)

    if args.variant == "q4_sdpa":
        run = lambda: q4_sdpa(q, qk_pack, qv_pack, scale)
    elif args.variant == "fp16_sdpa":
        run = lambda: fp16_sdpa(q, k, v, scale)
    elif args.variant == "dq_then_sdpa":
        run = lambda: dq_then_sdpa(q, qk_pack, qv_pack, scale)
    else:
        sys.exit(f"unknown variant: {args.variant}")

    print(f"warming up ({args.warmup} iters) ...", flush=True)
    for _ in range(args.warmup):
        mx.eval(run())

    print(
        f"starting capture -> {args.out} ({args.iters} iters in capture)",
        flush=True,
    )
    mx.metal.start_capture(args.out)
    for _ in range(args.iters):
        mx.eval(run())
    mx.metal.stop_capture()
    print(f"wrote {args.out}", flush=True)
    print(
        f"open it with: open -a Xcode {args.out}\n"
        "Then read the counters listed in §5 of the Phase 1 report on "
        "the relevant SDPA dispatch.",
        flush=True,
    )


if __name__ == "__main__":
    main()
