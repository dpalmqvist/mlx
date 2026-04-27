# Copyright © 2026 Apple Inc.
"""Dequant-throughput microbenchmark for Phase 3 of the quantized vector SDPA
optimization workstream.

Times mx.dequantize on the same K/V shapes the SDPA kernel processes per call,
plus a memory-bandwidth lower bound (pure-read sum) and an ALU-bandwidth
reference (pure-multiply). The point is to attribute the dequant cost in
isolation from any SDPA-kernel structure.

If dequant runs ≈ at the pure-read time, dequant ALU is essentially free and
the SDPA gap is not from the dequant FMAs. If it runs ≈ at the multiply time,
dequant ALU IS the cost. Far from both → a third bottleneck (bit-extract
pattern, or scale/bias lookup).

See docs/superpowers/specs/2026-04-27-phase3-measurement-design.md §M2.
"""

import argparse
import math
import statistics
import time
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class Shape:
    name: str
    Hk: int
    ctx: int
    D: int


SHAPES = {
    # qwen-gqa worst-cell (Phase 1 §3 worst q4/fp16 ratio)
    "qwen-gqa-32768": Shape("qwen-gqa-32768", Hk=4, ctx=32768, D=128),
    # llama3-8b worst-cell
    "llama3-8b-32768": Shape("llama3-8b-32768", Hk=8, ctx=32768, D=128),
    # smaller ctx for sanity
    "qwen-gqa-8192": Shape("qwen-gqa-8192", Hk=4, ctx=8192, D=128),
}

DTYPE = mx.float16
GROUP_SIZE = 64
BITS = 4


def make_inputs(shape: Shape, seed: int = 0):
    mx.random.seed(seed)
    fp16 = mx.random.normal((1, shape.Hk, shape.ctx, shape.D)).astype(DTYPE)
    packed, scales, biases = mx.quantize(fp16, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(fp16, packed, scales, biases)
    return fp16, packed, scales, biases


def time_op(fn, warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    samples = []
    for _ in range(iters):
        tic = time.perf_counter()
        out = fn()
        mx.eval(out)
        samples.append((time.perf_counter() - tic) * 1e3)
    samples.sort()
    median = statistics.median(samples)
    p10 = samples[max(0, int(0.10 * iters) - 1)]
    p90 = samples[min(iters - 1, int(0.90 * iters) - 1)]
    return median, p10, p90


def bytes_for_shape(shape: Shape) -> dict[str, float]:
    """Theoretical bytes touched per op, in MB."""
    n_elements = 1 * shape.Hk * shape.ctx * shape.D
    fp16_bytes = n_elements * 2
    q4_bytes = (n_elements * BITS) // 8
    n_groups = (1 * shape.Hk * shape.ctx * shape.D) // GROUP_SIZE
    scales_bytes = n_groups * 2  # fp16
    biases_bytes = n_groups * 2  # fp16
    return {
        "fp16_total_mb": fp16_bytes / (1024 * 1024),
        "q4_packed_mb": q4_bytes / (1024 * 1024),
        "q4_meta_mb": (scales_bytes + biases_bytes) / (1024 * 1024),
        "q4_total_mb": (q4_bytes + scales_bytes + biases_bytes) / (1024 * 1024),
    }


def run_one(shape: Shape, warmup: int, iters: int):
    fp16, packed, scales, biases = make_inputs(shape)

    def dequant():
        return mx.dequantize(packed, scales, biases, group_size=GROUP_SIZE, bits=BITS)

    def pure_read():
        # touch all bytes once, return a small reduction so the read isn't dropped
        return mx.sum(fp16)

    def pure_multiply():
        # one fp16 multiply per element — pure ALU bandwidth reference
        return fp16 * mx.array(1.5, dtype=DTYPE)

    correctness_ref = mx.dequantize(packed, scales, biases, group_size=GROUP_SIZE, bits=BITS)
    correctness_test = dequant()
    mx.eval(correctness_ref, correctness_test)
    if not mx.allclose(correctness_ref, correctness_test, atol=1e-6, rtol=1e-6).item():
        max_err = mx.max(mx.abs(correctness_ref - correctness_test)).item()
        raise SystemExit(f"correctness gate failed: max abs err {max_err}")

    rows = {}
    for label, fn in [
        ("dequant", dequant),
        ("pure_read", pure_read),
        ("pure_multiply", pure_multiply),
    ]:
        median, p10, p90 = time_op(fn, warmup, iters)
        rows[label] = (median, p10, p90)
    return rows


def print_table(name: str, shape: Shape, rows: dict, byte_info: dict):
    print()
    print(
        f"shape={name}  (Hk={shape.Hk}, ctx={shape.ctx}, D={shape.D}; "
        f"q4_total={byte_info['q4_total_mb']:.1f} MB, "
        f"fp16_total={byte_info['fp16_total_mb']:.1f} MB)"
    )
    print(f"  {'op':<14}  {'median ms':>9}  {'p10':>7}  {'p90':>7}  {'effective GB/s':>14}")
    for label, (median, p10, p90) in rows.items():
        # Effective bandwidth: bytes touched / time. Approximate.
        if label == "dequant":
            bytes_touched = byte_info["q4_total_mb"] * 1024 * 1024  # read q4 inputs
            bytes_touched += byte_info["fp16_total_mb"] * 1024 * 1024  # write fp16 output
        else:
            bytes_touched = byte_info["fp16_total_mb"] * 1024 * 1024
        gb_s = (bytes_touched / 1e9) / (median / 1e3)
        print(f"  {label:<14}  {median:>9.4f}  {p10:>7.4f}  {p90:>7.4f}  {gb_s:>13.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", default=",".join(SHAPES))
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    shape_names = [s.strip() for s in args.shapes.split(",") if s.strip()]
    for name in shape_names:
        if name not in SHAPES:
            raise SystemExit(f"unknown shape: {name}; choices: {sorted(SHAPES)}")

    print("Dequant-throughput microbench. Phase 3 §M2.")
    print(f"iters={args.iters} warmup={args.warmup} group_size={GROUP_SIZE} bits={BITS}")
    for name in shape_names:
        shape = SHAPES[name]
        byte_info = bytes_for_shape(shape)
        rows = run_one(shape, args.warmup, args.iters)
        print_table(name, shape, rows, byte_info)


if __name__ == "__main__":
    main()
