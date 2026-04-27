# Copyright © 2026 Apple Inc.
"""Phase 5 §6.3 prefill SDPA bench.

Times the full / multi-query prefill SDPA path (NOT the vector decode path).
The Phase 1 parent doc's Tier 1.5 §B (NAX in SDPA prefill) was written
before MLX commit 54f1cc6e added Neural Accelerator support to the steel
attention kernels. This bench checks whether NAX prefill is actually
operative on M4 Pro and whether there's remaining headroom.

Compares two variants per shape:
  - fp16_prefill: mx.fast.scaled_dot_product_attention on fp16 K/V (NAX path).
  - q4_dq_prefill: mx.dequantize -> fp16 SDPA, the only quantized prefill
    flow available today (the QuantizedScaledDotProductAttention primitive
    falls back when query_sequence_length != 1).

Two shape regimes:
  - Llama-3 8B style: Hq=32, Hk=8, D=128.
  - Qwen-style GQA:   Hq=28, Hk=4, D=128.

Prefill lengths: 512, 1024, 2048, 4096, 8192. 100 measured + 10 warmup
iters per cell, median + p10/p90.
"""

import argparse
import csv
import math
import statistics
import time
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class Shape:
    name: str
    Hq: int
    Hk: int
    D: int


MODELS = {
    "llama3-8b": Shape("llama3-8b", Hq=32, Hk=8, D=128),
    "qwen-gqa": Shape("qwen-gqa", Hq=28, Hk=4, D=128),
}

DEFAULT_QL = [512, 1024, 2048, 4096, 8192]
DTYPE = mx.float16
GROUP_SIZE = 64
BITS = 4


def make_inputs(shape: Shape, qL: int, seed: int = 0):
    """qL = prefill length; K/V are at the same length (self-attention prefill)."""
    mx.random.seed(seed)
    q = mx.random.normal((1, shape.Hq, qL, shape.D)).astype(DTYPE)
    k = mx.random.normal((1, shape.Hk, qL, shape.D)).astype(DTYPE)
    v = mx.random.normal((1, shape.Hk, qL, shape.D)).astype(DTYPE)
    qk, ks, kb = mx.quantize(k, group_size=GROUP_SIZE, bits=BITS)
    qv, vs, vb = mx.quantize(v, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(q, k, v, qk, ks, kb, qv, vs, vb)
    return q, k, v, (qk, ks, kb), (qv, vs, vb)


def fp16_prefill(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


def q4_dq_prefill(q, qk_pack, qv_pack, scale):
    qk, ks, kb = qk_pack
    qv, vs, vb = qv_pack
    k_dq = mx.dequantize(qk, ks, kb, group_size=GROUP_SIZE, bits=BITS).astype(DTYPE)
    v_dq = mx.dequantize(qv, vs, vb, group_size=GROUP_SIZE, bits=BITS).astype(DTYPE)
    return mx.fast.scaled_dot_product_attention(q, k_dq, v_dq, scale=scale)


def time_op(fn, warmup: int, iters: int):
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


def run_cell(shape: Shape, qL: int, warmup: int, iters: int):
    q, k, v, qk_pack, qv_pack = make_inputs(shape, qL)
    scale = 1.0 / math.sqrt(shape.D)
    rows = {}
    for label, fn in [
        ("fp16_prefill", lambda: fp16_prefill(q, k, v, scale)),
        ("q4_dq_prefill", lambda: q4_dq_prefill(q, qk_pack, qv_pack, scale)),
    ]:
        median, p10, p90 = time_op(fn, warmup, iters)
        rows[label] = (median, p10, p90)
    return rows


def print_table(name: str, shape: Shape, results: dict):
    print()
    print(f"model={name}  Hq={shape.Hq} Hk={shape.Hk} D={shape.D}")
    print(f"  {'qL':>6}  {'fp16_prefill':>12}  {'q4_dq_prefill':>13}  "
          f"{'q4_dq/fp16':>10}  {'fp16 TFLOPS':>11}")
    for qL in sorted(results):
        r = results[qL]
        m_fp16 = r["fp16_prefill"][0]
        m_q4dq = r["q4_dq_prefill"][0]
        ratio = m_q4dq / m_fp16
        # Approx FLOPS: 4 * B * H * qL^2 * D for QK^T + softmax + AV
        flops = 4 * shape.Hq * qL * qL * shape.D
        tflops = (flops / 1e12) / (m_fp16 / 1e3)
        print(f"  {qL:>6}  {m_fp16:>12.4f}  {m_q4dq:>13.4f}  "
              f"{ratio:>9.2f}x  {tflops:>10.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", help="optional CSV output path")
    p.add_argument("--models", default="llama3-8b,qwen-gqa")
    p.add_argument("--qL", default=",".join(str(c) for c in DEFAULT_QL))
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    qLs = [int(c) for c in args.qL.split(",") if c.strip()]

    all_rows = []
    for name in model_names:
        if name not in MODELS:
            raise SystemExit(f"unknown model: {name}; choices: {sorted(MODELS)}")
        shape = MODELS[name]
        per_qL = {}
        for qL in qLs:
            print(f"  running {name} qL={qL} ...", flush=True)
            per_qL[qL] = run_cell(shape, qL, args.warmup, args.iters)
            for variant, (m, p10, p90) in per_qL[qL].items():
                all_rows.append((shape, qL, variant, m, p10, p90))
        print_table(name, shape, per_qL)

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "Hq", "Hk", "D", "qL", "variant",
                        "median_ms", "p10_ms", "p90_ms"])
            for shape, qL, variant, m, p10, p90 in all_rows:
                w.writerow([shape.name, shape.Hq, shape.Hk, shape.D,
                            qL, variant,
                            f"{m:.6f}", f"{p10:.6f}", f"{p90:.6f}"])
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
