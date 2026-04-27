# Copyright © 2026 Apple Inc.
"""Phase 1 sweep: 4-bit quantized vector SDPA vs fp16 vector SDPA.

See docs/superpowers/specs/2026-04-27-profile-first-optimization-design.md.
"""

import argparse
import csv
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx


@dataclass(frozen=True)
class ModelShape:
    name: str
    Hq: int
    Hk: int
    D: int
    V: int


MODELS = {
    "llama3-8b": ModelShape("llama3-8b", Hq=32, Hk=8, D=128, V=128),
    "qwen-gqa": ModelShape("qwen-gqa", Hq=28, Hk=4, D=128, V=128),
}

DEFAULT_CTX = [1024, 2048, 4096, 8192, 16384, 32768]
DTYPE = mx.float16
GROUP_SIZE = 64
BITS = 4

# Tolerance for bits=4 / fp16 path, taken from
# test_quantized_sdpa_vector_matches_dequantized in
# python/tests/test_fast_sdpa.py:731-732.
ATOL = 5e-3
RTOL = 5e-3


def make_inputs(shape: ModelShape, ctx: int, seed: int = 0):
    mx.random.seed(seed)
    q = mx.random.normal((1, shape.Hq, 1, shape.D)).astype(DTYPE)
    k = mx.random.normal((1, shape.Hk, ctx, shape.D)).astype(DTYPE)
    v = mx.random.normal((1, shape.Hk, ctx, shape.V)).astype(DTYPE)
    qk, ks, kb = mx.quantize(k, group_size=GROUP_SIZE, bits=BITS)
    qv, vs, vb = mx.quantize(v, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(q, k, v, qk, ks, kb, qv, vs, vb)
    return q, k, v, (qk, ks, kb), (qv, vs, vb)


def fp16_sdpa(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


def q4_sdpa(q, qk_pack, qv_pack, scale):
    qk, ks, kb = qk_pack
    qv, vs, vb = qv_pack
    return mx.fast.quantized_scaled_dot_product_attention(
        q, qk, ks, kb, qv, vs, vb,
        scale=scale, group_size=GROUP_SIZE, bits=BITS,
    )


def dq_then_sdpa(q, qk_pack, qv_pack, scale):
    qk, ks, kb = qk_pack
    qv, vs, vb = qv_pack
    k_dq = mx.dequantize(qk, ks, kb, group_size=GROUP_SIZE, bits=BITS).astype(DTYPE)
    v_dq = mx.dequantize(qv, vs, vb, group_size=GROUP_SIZE, bits=BITS).astype(DTYPE)
    return mx.fast.scaled_dot_product_attention(q, k_dq, v_dq, scale=scale)


def time_op(fn: Callable, warmup: int, iters: int):
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


def assert_close(label: str, a, b):
    a32 = a.astype(mx.float32)
    b32 = b.astype(mx.float32)
    if not mx.allclose(a32, b32, atol=ATOL, rtol=RTOL).item():
        max_err = mx.max(mx.abs(a32 - b32)).item()
        raise SystemExit(
            f"correctness gate failed for {label}: max abs err {max_err:.3e} "
            f"(atol={ATOL}, rtol={RTOL})"
        )


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def run_cell(shape: ModelShape, ctx: int, warmup: int, iters: int):
    q, k, v, qk_pack, qv_pack = make_inputs(shape, ctx)
    scale = 1.0 / math.sqrt(shape.D)

    out_q4 = q4_sdpa(q, qk_pack, qv_pack, scale)
    out_dq = dq_then_sdpa(q, qk_pack, qv_pack, scale)
    mx.eval(out_q4, out_dq)
    assert_close(f"{shape.name} ctx={ctx}", out_q4, out_dq)

    rows = []
    for variant, fn in [
        ("fp16_sdpa", lambda: fp16_sdpa(q, k, v, scale)),
        ("q4_sdpa", lambda: q4_sdpa(q, qk_pack, qv_pack, scale)),
        ("dq_then_sdpa", lambda: dq_then_sdpa(q, qk_pack, qv_pack, scale)),
    ]:
        median, p10, p90 = time_op(fn, warmup, iters)
        rows.append((variant, median, p10, p90))
    return rows


def print_table(model_name: str, shape: ModelShape, results: dict):
    print()
    print(f"model={model_name}  Hq={shape.Hq} Hk={shape.Hk} D={shape.D}")
    header = f"{'ctx':>6}  {'fp16_sdpa':>10}  {'q4_sdpa':>10}  {'dq_then_sdpa':>14}  {'q4/fp16':>8}  {'q4/dq':>8}"
    print(header)
    for ctx in sorted(results):
        m = {v: t for v, (t, _, _) in results[ctx].items()}
        ratio_fp = m["q4_sdpa"] / m["fp16_sdpa"]
        ratio_dq = m["q4_sdpa"] / m["dq_then_sdpa"]
        print(
            f"{ctx:>6}  {m['fp16_sdpa']:>10.4f}  {m['q4_sdpa']:>10.4f}  "
            f"{m['dq_then_sdpa']:>14.4f}  {ratio_fp:>7.2f}x  {ratio_dq:>7.2f}x"
        )


def write_csv(path: str, all_rows, device: str, sha: str, iters: int):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "Hq", "Hk", "D", "V", "ctx", "variant", "iters",
            "median_ms", "p10_ms", "p90_ms", "git_sha", "device",
        ])
        for shape, ctx, variant, median, p10, p90 in all_rows:
            w.writerow([
                shape.name, shape.Hq, shape.Hk, shape.D, shape.V,
                ctx, variant, iters,
                f"{median:.6f}", f"{p10:.6f}", f"{p90:.6f}", sha, device,
            ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", help="optional CSV output path")
    p.add_argument("--models", default="llama3-8b,qwen-gqa")
    p.add_argument("--ctx", default=",".join(str(c) for c in DEFAULT_CTX))
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="M4Pro")
    args = p.parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    ctxs = [int(c) for c in args.ctx.split(",") if c.strip()]
    sha = git_sha()

    all_rows = []
    for name in model_names:
        if name not in MODELS:
            sys.exit(f"unknown model: {name}; choices: {sorted(MODELS)}")
        shape = MODELS[name]
        per_ctx = {}
        for ctx in ctxs:
            print(f"  running {name} ctx={ctx} ...", flush=True)
            rows = run_cell(shape, ctx, args.warmup, args.iters)
            per_ctx[ctx] = {v: (m, p10, p90) for v, m, p10, p90 in rows}
            for variant, median, p10, p90 in rows:
                all_rows.append((shape, ctx, variant, median, p10, p90))
        print_table(name, shape, per_ctx)

    if args.out:
        write_csv(args.out, all_rows, args.device, sha, args.iters)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
