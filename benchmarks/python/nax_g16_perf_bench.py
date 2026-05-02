# Copyright © 2026 Apple Inc.
"""NAX-on-g16 perf baseline harness.

Measures NAX-on vs NAX-off wall time for a fixed list of cases on Apple
gen-16 GPUs (M3/M4 family). The two halves of the A/B are run in separate
subprocesses because is_nax_available() is statically cached on first call.

Usage:
  /Users/daniel/mlx/.venv/bin/python benchmarks/python/nax_g16_perf_bench.py
      run both subprocesses, print table, write JSON

  /Users/daniel/mlx/.venv/bin/python benchmarks/python/nax_g16_perf_bench.py --list
      print the case list and exit

  /Users/daniel/mlx/.venv/bin/python benchmarks/python/nax_g16_perf_bench.py --child on
      child mode (used internally by the orchestrator)

Spec: docs/superpowers/specs/2026-05-01-nax-g16-perf-baseline-design.md
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import mlx.core as mx


# ---- Case definitions ------------------------------------------------------
# Cases are (kernel_label, shape_dict, builder, work_estimator). builder
# returns a zero-arg callable that runs the op; work_estimator returns
# float-ops for the shape so we can report TFLOPS. Tasks 3 and 4 fill in
# the full grid; Task 2 ships with a single trivial case so the harness
# can be tested end-to-end.

@dataclass
class Case:
    kernel_label: str
    shape: dict
    build: Callable[[], Callable[[], "mx.array"]]
    flops: Callable[[dict], float]

    @property
    def shape_str(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.shape.items())


def _gemm_fused(M: int, N: int, K: int) -> Case:
    def build():
        a = mx.random.normal((M, K)).astype(mx.float16)
        b = mx.random.normal((K, N)).astype(mx.float16)
        mx.eval(a, b)
        def run():
            return a @ b
        return run
    def flops(s):
        return 2 * s["M"] * s["N"] * s["K"]
    return Case("gemm_fused", {"M": M, "N": N, "K": K}, build, flops)


def _gemm_splitk(M: int, N: int, K: int) -> Case:
    """Same op as gemm_fused; small-MN-large-K shape regime where the
    dispatcher selects steel_gemm_splitk on g16."""
    def build():
        a = mx.random.normal((M, K)).astype(mx.float16)
        b = mx.random.normal((K, N)).astype(mx.float16)
        mx.eval(a, b)
        def run():
            return a @ b
        return run
    def flops(s):
        return 2 * s["M"] * s["N"] * s["K"]
    return Case("gemm_splitk", {"M": M, "N": N, "K": K}, build, flops)


def _gemm_segmented(B: int, M: int, N: int, K: int) -> Case:
    """Batched matmul; dispatcher picks the segmented kernel on batched
    input."""
    def build():
        a = mx.random.normal((B, M, K)).astype(mx.float16)
        b = mx.random.normal((B, K, N)).astype(mx.float16)
        mx.eval(a, b)
        def run():
            return a @ b
        return run
    def flops(s):
        return 2 * s["B"] * s["M"] * s["N"] * s["K"]
    return Case("gemm_segmented",
                {"B": B, "M": M, "N": N, "K": K}, build, flops)


def _gather(tokens: int, E: int, hidden: int, expert_hidden: int,
            top_k: int) -> Case:
    """MoE-style gather_mm. tokens > 32 keeps us on the bm>=32 NAX path on g16
    (bm=16 falls back to non-NAX). One expert-projection layer routed via
    rhs_indices over E experts:
        x:   (tokens*top_k, 1, hidden)        # per-token (1 x hidden) operands
        w:   (E, hidden, expert_hidden)
        out: (tokens*top_k, 1, expert_hidden)
    """
    def build():
        x = mx.random.normal((tokens, hidden)).astype(mx.float16)
        # Weight matrix per expert: (E, hidden, expert_hidden)
        w = mx.random.normal((E, hidden, expert_hidden)).astype(mx.float16)
        # Each token routes to top_k experts. For simplicity in benching
        # we just generate random rhs_indices over [0, E) of length
        # tokens * top_k and reshape x accordingly.
        rhs_idx = mx.random.randint(low=0, high=E,
                                    shape=(tokens * top_k,)).astype(mx.uint32)
        # Reshape x to (tokens*top_k, 1, hidden) so that gather_mm treats
        # each token as a batched (1 x hidden) operand, selecting one expert
        # weight matrix per token via rhs_indices.
        # Output: (tokens*top_k, 1, expert_hidden).
        x_rep = mx.repeat(x, repeats=top_k, axis=0)  # (tokens*top_k, hidden)
        x_rep = x_rep.reshape(tokens * top_k, 1, hidden)
        mx.eval(x, w, rhs_idx, x_rep)
        def run():
            return mx.gather_mm(x_rep, w, rhs_indices=rhs_idx)
        return run
    def flops(s):
        return (2 * s["tokens"] * s["top_k"]
                * s["hidden"] * s["expert_hidden"])
    return Case(
        "gather",
        {"tokens": tokens, "E": E, "hidden": hidden,
         "expert_hidden": expert_hidden, "top_k": top_k},
        build, flops)


def _sdpa_prefill(B: int, H: int, kL: int, hd: int) -> Case:
    """Self-attention prefill: q,k,v all at length kL. fp16 inputs, no mask.
    On g16 this routes through attention_nax_g16 with NAXFrag32 + wm=2.
    """
    def build():
        q = mx.random.normal((B, H, kL, hd)).astype(mx.float16)
        k = mx.random.normal((B, H, kL, hd)).astype(mx.float16)
        v = mx.random.normal((B, H, kL, hd)).astype(mx.float16)
        scale = 1.0 / math.sqrt(hd)
        mx.eval(q, k, v)
        def run():
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        return run
    def flops(s):
        # 4 * B * H * kL * kL * hd: Q@K^T (2BH kL^2 hd) + S@V (2BH kL^2 hd).
        return 4 * s["B"] * s["H"] * s["kL"] * s["kL"] * s["hd"]
    return Case(
        "sdpa_prefill",
        {"B": B, "H": H, "kL": kL, "hd": hd},
        build, flops)


def all_cases() -> list[Case]:
    return [
        # Matmul (Task 3).
        _gemm_fused(M=2048, N=4096, K=4096),
        _gemm_fused(M=2048, N=11008, K=4096),
        _gemm_fused(M=512, N=4096, K=4096),
        _gemm_splitk(M=64, N=64, K=8192),
        _gemm_splitk(M=128, N=128, K=4096),
        _gemm_segmented(B=8, M=512, N=4096, K=4096),
        _gemm_segmented(B=32, M=128, N=128, K=128),
        # Gather (MoE prefill, bm>=32).
        _gather(tokens=2048, E=8, hidden=4096, expert_hidden=14336, top_k=2),
        _gather(tokens=512, E=8, hidden=4096, expert_hidden=4096, top_k=2),
        # SDPA prefill (fp16, no bool mask).
        _sdpa_prefill(B=1, H=32, kL=2048, hd=128),
        _sdpa_prefill(B=1, H=32, kL=8192, hd=128),
        _sdpa_prefill(B=1, H=32, kL=512, hd=128),
        _sdpa_prefill(B=1, H=32, kL=2048, hd=64),
    ]


# ---- Timing ---------------------------------------------------------------

def time_case(case: Case, warmup: int = 3, iters: int = 10,
              capture_path: str | None = None) -> dict:
    run = case.build()
    for _ in range(warmup):
        mx.eval(run())
    mx.synchronize()

    if capture_path is not None:
        mx.metal.start_capture(capture_path)
        mx.eval(run())
        mx.synchronize()
        mx.metal.stop_capture()

    samples = []
    for _ in range(iters):
        tic = time.perf_counter()
        out = run()
        mx.eval(out)
        samples.append((time.perf_counter() - tic) * 1e3)
    mx.synchronize()
    samples.sort()
    return {
        "kernel_label": case.kernel_label,
        "shape": case.shape,
        "median_ms": statistics.median(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


# ---- Child mode -----------------------------------------------------------

def child_main(label: str) -> int:
    print(json.dumps({
        "_header": True,
        "arm": label,
        "nax_available": mx.metal.is_nax_available(),
        "nax_flavor": mx.metal.nax_arch_flavor(),
    }), flush=True)
    capture_dir = os.environ.get("MLX_NAX_BENCH_CAPTURE_DIR")
    for case in all_cases():
        capture_path = None
        if capture_dir is not None:
            shape_slug = "_".join(f"{k}{v}" for k, v in case.shape.items())
            capture_path = os.path.join(
                capture_dir,
                f"{label}__{case.kernel_label}__{shape_slug}.gputrace",
            )
        rec = time_case(case, capture_path=capture_path)
        rec["arm"] = label
        print(json.dumps(rec), flush=True)
    return 0


# ---- Orchestrator ---------------------------------------------------------

def run_child(arm: str, disable_nax: bool, capture_dir: str | None = None) -> tuple[dict, list[dict]]:
    env = dict(os.environ)
    if disable_nax:
        env["MLX_DISABLE_NAX"] = "1"
    else:
        env.pop("MLX_DISABLE_NAX", None)
    if capture_dir is not None:
        env["MLX_NAX_BENCH_CAPTURE_DIR"] = capture_dir
    else:
        env.pop("MLX_NAX_BENCH_CAPTURE_DIR", None)
    cmd = [sys.executable, __file__, "--child", arm]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"child arm={arm} failed (exit {proc.returncode})")
    header = {}
    cases = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("_header"):
            header = rec
        else:
            cases.append(rec)
    return header, cases


def join(on_recs: list[dict], off_recs: list[dict]) -> list[dict]:
    def key(r): return (r["kernel_label"], json.dumps(r["shape"], sort_keys=True))
    on_by_key = {key(r): r for r in on_recs}
    off_by_key = {key(r): r for r in off_recs}
    keys = list(dict.fromkeys(list(on_by_key) + list(off_by_key)))
    rows = []
    for k in keys:
        on = on_by_key.get(k)
        off = off_by_key.get(k)
        if on is None or off is None:
            sys.stderr.write(f"WARN: missing arm for case {k}\n")
            continue
        nax_on = on["median_ms"]
        nax_off = off["median_ms"]
        speedup = nax_off / nax_on if nax_on > 0 else float("nan")
        rows.append({
            "kernel_label": on["kernel_label"],
            "shape": on["shape"],
            "nax_on_ms": nax_on,
            "nax_off_ms": nax_off,
            "speedup": speedup,
        })
    return rows


def annotate_tflops(rows: list[dict], cases: list[Case]) -> None:
    by_key = {(c.kernel_label, json.dumps(c.shape, sort_keys=True)): c
              for c in cases}
    for r in rows:
        c = by_key.get((r["kernel_label"], json.dumps(r["shape"], sort_keys=True)))
        if c is None:
            continue
        f = c.flops(r["shape"])
        if r["nax_on_ms"] > 0:
            r["nax_on_tflops"] = (f / 1e12) / (r["nax_on_ms"] / 1e3)


def print_table(rows: list[dict]) -> None:
    print()
    print(f"{'kernel':<14}  {'shape':<48}  {'nax_on':>9}  {'nax_off':>9}  "
          f"{'speedup':>8}  {'TFLOPS':>7}")
    for r in rows:
        shape_str = " ".join(f"{k}={v}" for k, v in r["shape"].items())
        tfl = r.get("nax_on_tflops")
        tfl_str = f"{tfl:7.2f}" if tfl is not None else "      -"
        print(f"{r['kernel_label']:<14}  {shape_str:<48}  "
              f"{r['nax_on_ms']:>7.3f}ms  {r['nax_off_ms']:>7.3f}ms  "
              f"{r['speedup']:>7.2f}x  {tfl_str}")


def preflight(on_header: dict, off_header: dict) -> None:
    """Abort if the gate clearly didn't flip.

    Each child reports nax_available and nax_flavor at startup. If both
    arms agree on availability, the gate isn't doing anything and the
    rest of the run is meaningless — abort before any timing is reported.
    """
    on_avail = on_header.get("nax_available")
    off_avail = off_header.get("nax_available")
    on_flavor = on_header.get("nax_flavor")
    off_flavor = off_header.get("nax_flavor")
    if on_avail is None or off_avail is None:
        raise SystemExit(
            "preflight: child header missing nax_available; rebuild may be stale")
    if on_avail == off_avail:
        raise SystemExit(
            f"preflight: NAX availability did not flip across arms "
            f"(on={on_avail}/{on_flavor}, off={off_avail}/{off_flavor}). "
            f"The MLX_DISABLE_NAX gate is not effective on this build. "
            f"Run tools/probe_nax_disable_gate.py to verify.")
    print(f"preflight: NAX-on arm: available={on_avail} flavor={on_flavor}", flush=True)
    print(f"preflight: NAX-off arm: available={off_avail} flavor={off_flavor}", flush=True)


# ---- Entry point ----------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", choices=["on", "off"],
                   help="child mode (internal use)")
    p.add_argument("--list", action="store_true", help="list cases and exit")
    p.add_argument("--out", help="JSON output path (default: nax_g16_perf_<ts>.json)")
    p.add_argument(
        "--capture", default=None, metavar="DIR",
        help=("If set, run one capture iteration per case and write "
              "<arm>__<kernel>__<shape>.gputrace files to DIR. The "
              "regular timing loop still runs after the capture. "
              "Pass an existing directory; this script does not create it."))
    args = p.parse_args()

    if args.list:
        for c in all_cases():
            print(f"{c.kernel_label:<14}  {c.shape_str}")
        return 0

    if args.child:
        return child_main(args.child)

    # Orchestrator
    print("running NAX-on arm ...", flush=True)
    on_header, on_recs = run_child("on", disable_nax=False, capture_dir=args.capture)
    print("running NAX-off arm ...", flush=True)
    off_header, off_recs = run_child("off", disable_nax=True, capture_dir=args.capture)
    preflight(on_header, off_header)
    rows = join(on_recs, off_recs)
    annotate_tflops(rows, all_cases())
    print_table(rows)

    out_path = args.out or f"nax_g16_perf_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump({
            "headers": {"on": on_header, "off": off_header},
            "rows": rows,
        }, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
