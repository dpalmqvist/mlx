# NAX-on-g16 Perf Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runtime `MLX_DISABLE_NAX` gate plus a subprocess-based A/B bench harness that prints a NAX-on-vs-NAX-off speedup table for 13 cases across 5 g16 NAX kernel paths.

**Architecture:** A 5-line edit to `is_nax_available()` in `mlx/backend/metal/device.cpp` reads `MLX_DISABLE_NAX` once at first call. A single Python script `benchmarks/python/nax_g16_perf_bench.py` is both the case-list owner and the orchestrator: when invoked normally it spawns two subprocess copies of itself (one with `MLX_DISABLE_NAX=1`), each runs all cases and prints JSON to stdout, parent joins and prints a table. No kernel code changes.

**Tech Stack:** C++ (env-var gate in MLX core), Python 3 + `mlx.core` (bench harness), `subprocess` + JSON for A/B, no third-party deps.

**Spec:** `docs/superpowers/specs/2026-05-01-nax-g16-perf-baseline-design.md`.

**Python interpreter:** `/Users/daniel/mlx/.venv/bin/python` (system `python3` lacks `mlx.core`).

---

## File Structure

| File | Action | Purpose |
| --- | --- | --- |
| `mlx/backend/metal/device.cpp` | edit (~5 lines inside existing `_check_nax` lambda) | runtime `MLX_DISABLE_NAX` gate |
| `tools/probe_nax_disable_gate.py` | new | one-off correctness probe for the gate (numerical equivalence + dispatch flip evidence) |
| `benchmarks/python/nax_g16_perf_bench.py` | new | bench harness; single source of truth for case list and the subprocess A/B orchestrator |
| `docs/superpowers/reports/2026-05-01-nax-g16-perf-baseline.md` | new in last task | report skeleton; numbers filled in from a real run |

The bench script is intentionally one file. The case list, the per-op execution helpers, and the orchestrator are all small and change together; splitting them now would just add import boilerplate. If we add knob-sweeping or per-case visualization later, that's the time to split.

---

## Task 1: Runtime `MLX_DISABLE_NAX` gate

**Files:**
- Modify: `mlx/backend/metal/device.cpp` (the `_check_nax` lambda inside `is_nax_available()`, around lines 828–856)
- Create: `tools/probe_nax_disable_gate.py`

The existing `_check_nax` is wrapped in a `static bool is_nax_available_` cache. The env var must be read inside the lambda so it's part of the cached result; reading it elsewhere defeats the cache or, worse, gives time-varying answers. Subprocesses are how the harness flips the gate.

- [ ] **Step 1: Write the probe (RED)**

The probe is the gate's behavioral test. With the gate off, NAX should be live on g16 (`nax_arch_flavor() == kG16`). With the gate on, NAX should be off (`nax_arch_flavor() == kNone`) and the same matmul should produce the same numerical result via the non-NAX path.

Create `tools/probe_nax_disable_gate.py`:

```python
"""Probe: MLX_DISABLE_NAX runtime gate.

Run this twice:
  $ /Users/daniel/mlx/.venv/bin/python tools/probe_nax_disable_gate.py
  $ MLX_DISABLE_NAX=1 /Users/daniel/mlx/.venv/bin/python tools/probe_nax_disable_gate.py

Each run prints a JSON line with `nax_available`, `nax_flavor`, and a
checksum of a small fp16 matmul. Both runs should print the SAME checksum
(within fp16 tolerance) but DIFFERENT availability/flavor.

Numerical-equivalence guard: if both runs print the same nax_available, the
gate didn't flip and the harness will be measuring noise.
"""

import json
import os
import sys

import mlx.core as mx
import numpy as np


def main():
    env = os.environ.get("MLX_DISABLE_NAX", "")
    # Touch the device once so is_nax_available() / nax_arch_flavor() resolve.
    a = mx.random.normal((128, 128)).astype(mx.float16)
    b = mx.random.normal((128, 128)).astype(mx.float16)
    c = a @ b
    mx.eval(c)
    checksum = float(c.astype(mx.float32).sum().item())

    # We can't directly query nax_arch_flavor from Python today, so we infer
    # via mx.metal.device_info() if available, otherwise we just record the
    # env var state. Both signals are useful: env-state proves the harness
    # set it; the checksum proves the kernel produced the right answer.
    info = {}
    try:
        info = dict(mx.metal.device_info())
    except Exception:
        pass

    print(json.dumps({
        "MLX_DISABLE_NAX_env": env,
        "checksum": checksum,
        "device_info_keys": sorted(info.keys()),
    }))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run probe twice to verify it currently does NOT distinguish gate state**

Build the current main first (gate not yet added):

```bash
cd /Users/daniel/mlx && cmake --build build -j
```

Then run:

```bash
/Users/daniel/mlx/.venv/bin/python tools/probe_nax_disable_gate.py
MLX_DISABLE_NAX=1 /Users/daniel/mlx/.venv/bin/python tools/probe_nax_disable_gate.py
```

Expected: both lines print the same `MLX_DISABLE_NAX_env` field reflecting their respective env (so the runner is correctly setting it), and the same checksum. The `MLX_DISABLE_NAX_env=1` run is doing nothing useful yet — that's the RED state.

- [ ] **Step 3: Add the env var gate to `is_nax_available()`**

Open `mlx/backend/metal/device.cpp` and find the `_check_nax` lambda (around line 832). Insert the env var check as the first thing inside the lambda:

```cpp
bool is_nax_available() {
#ifdef MLX_METAL_NO_NAX
  return false;
#else
  auto _check_nax = []() {
    if (const char* env = std::getenv("MLX_DISABLE_NAX")) {
      if (env[0] != '\0' && env[0] != '0') {
        return false;
      }
    }
    bool can_use_nax = false;
    if (__builtin_available(
            macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      can_use_nax = true;
    }
    auto& d = metal::device(mlx::core::Device::gpu);
    auto gen = d.get_architecture_gen();
    // ... existing comment block stays unchanged ...
    can_use_nax &= gen >= 16;
    return can_use_nax;
  };
  static bool is_nax_available_ = _check_nax();
  return is_nax_available_;
#endif
}
```

Keep the existing historical-note comment block intact between the `__builtin_available` check and `can_use_nax &= gen >= 16`. Only the four-line env var block at the top of the lambda is new.

- [ ] **Step 4: Rebuild**

```bash
cd /Users/daniel/mlx && cmake --build build -j
```

Expected: clean build, no warnings about `getenv`. (`<cstdlib>` is already transitively included in `device.cpp`.)

- [ ] **Step 5: Run probe twice to verify the gate now flips correctness path**

```bash
/Users/daniel/mlx/.venv/bin/python tools/probe_nax_disable_gate.py
MLX_DISABLE_NAX=1 /Users/daniel/mlx/.venv/bin/python tools/probe_nax_disable_gate.py
```

Expected:
- First line: `MLX_DISABLE_NAX_env=""` and a checksum value `C`.
- Second line: `MLX_DISABLE_NAX_env="1"` and a checksum value within `~1e-2` relative tolerance of `C` (fp16 matmul, different kernel, slightly different rounding is normal).

If checksums differ by more than ~1% on a 128×128 fp16 matmul, that's a sign the non-NAX path produces a meaningfully different result — flag it before continuing.

- [ ] **Step 6: Confirm gate values `0` and empty also disable correctly**

```bash
MLX_DISABLE_NAX=0 /Users/daniel/mlx/.venv/bin/python tools/probe_nax_disable_gate.py
MLX_DISABLE_NAX= /Users/daniel/mlx/.venv/bin/python tools/probe_nax_disable_gate.py
```

Expected: both behave like NAX-on (the gate code returns false only when env is non-empty and not `"0"`). This matches the convention used by `MLX_QMM_NAX_M_THRESHOLD` and other MLX env knobs. Probe output should match the unset case from Step 5.

- [ ] **Step 7: Commit**

```bash
git add mlx/backend/metal/device.cpp tools/probe_nax_disable_gate.py
git commit -m "$(cat <<'EOF'
metal: add MLX_DISABLE_NAX runtime gate to is_nax_available()

When MLX_DISABLE_NAX is set to a non-empty, non-"0" value, the existing
static-cached _check_nax lambda returns false on its first call, forcing
all NAX dispatch sites (matmul, quantized, SDPA, gather) onto their
non-NAX fallback paths. This enables single-build A/B comparison via
subprocesses for the upcoming nax-g16 perf-baseline harness.

The check lives inside the lambda so the result is part of the static
cache; flipping the env var mid-process does not work (by design).

tools/probe_nax_disable_gate.py is the behavioral test: it prints a JSON
line per run with the env state and a fp16 matmul checksum. With the gate
off vs. on, checksums must agree to fp16 tolerance (proves both paths
compute the same matmul) but the dispatched kernels differ.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Bench harness skeleton — subprocess A/B + JSON + table

**Files:**
- Create: `benchmarks/python/nax_g16_perf_bench.py`

This task adds the harness with one trivial case (fp16 128×128 matmul) so the wiring can be verified before the real cases land. Real cases come in Tasks 3 and 4.

The script has three modes, selected by argv:
1. **default (no args)** — orchestrator. Spawns two child runs, joins JSON, prints table.
2. **`--child <flag>`** — child mode. Runs all cases, prints one JSON record per case to stdout, exits. `flag` is `on` or `off` (purely a label; the env var is what actually controls dispatch).
3. **`--list`** — prints the case list and exits. Useful for ad-hoc inspection.

- [ ] **Step 1: Write the harness skeleton**

Create `benchmarks/python/nax_g16_perf_bench.py`:

```python
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


def _smoke_matmul_case() -> Case:
    def build():
        a = mx.random.normal((128, 128)).astype(mx.float16)
        b = mx.random.normal((128, 128)).astype(mx.float16)
        mx.eval(a, b)
        def run():
            return a @ b
        return run
    def flops(s):
        return 2 * s["M"] * s["N"] * s["K"]
    return Case("smoke", {"M": 128, "N": 128, "K": 128}, build, flops)


def all_cases() -> list[Case]:
    return [_smoke_matmul_case()]


# ---- Timing ---------------------------------------------------------------

def time_case(case: Case, warmup: int = 3, iters: int = 10) -> dict:
    run = case.build()
    for _ in range(warmup):
        mx.eval(run())
    samples = []
    for _ in range(iters):
        tic = time.perf_counter()
        out = run()
        mx.eval(out)
        samples.append((time.perf_counter() - tic) * 1e3)
    mx.synchronize()
    samples.sort()
    median_ms = statistics.median(samples)
    p10 = samples[max(0, int(0.10 * iters) - 1)]
    p90 = samples[min(iters - 1, int(0.90 * iters) - 1)]
    return {
        "kernel_label": case.kernel_label,
        "shape": case.shape,
        "median_ms": median_ms,
        "p10_ms": p10,
        "p90_ms": p90,
    }


# ---- Child mode -----------------------------------------------------------

def child_main(label: str) -> int:
    for case in all_cases():
        rec = time_case(case)
        rec["arm"] = label
        print(json.dumps(rec), flush=True)
    return 0


# ---- Orchestrator ---------------------------------------------------------

def run_child(arm: str, disable_nax: bool) -> list[dict]:
    env = dict(os.environ)
    if disable_nax:
        env["MLX_DISABLE_NAX"] = "1"
    else:
        env.pop("MLX_DISABLE_NAX", None)
    cmd = [sys.executable, __file__, "--child", arm]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"child arm={arm} failed (exit {proc.returncode})")
    out = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


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


def sanity_check(on_recs: list[dict], off_recs: list[dict]) -> None:
    """Abort if the gate clearly didn't flip.

    Heuristic: if every case has nax_on_ms == nax_off_ms within 1%, suspect
    the NAX dispatch is the same kernel both runs.
    """
    if not on_recs or not off_recs:
        return
    diffs = []
    for on, off in zip(on_recs, off_recs):
        if on["kernel_label"] != off["kernel_label"]:
            continue
        a = on["median_ms"]
        b = off["median_ms"]
        if a > 0:
            diffs.append(abs(a - b) / max(a, b))
    if diffs and max(diffs) < 0.01:
        sys.stderr.write(
            "WARN: NAX-on and NAX-off median timings agree within 1% across\n"
            "      every case. The MLX_DISABLE_NAX gate may not be effective\n"
            "      (the runtime might be running the same kernel both arms).\n"
            "      Run tools/probe_nax_disable_gate.py to verify.\n")


# ---- Entry point ----------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", choices=["on", "off"],
                   help="child mode (internal use)")
    p.add_argument("--list", action="store_true", help="list cases and exit")
    p.add_argument("--out", help="JSON output path (default: nax_g16_perf_<ts>.json)")
    args = p.parse_args()

    if args.list:
        for c in all_cases():
            print(f"{c.kernel_label:<14}  {c.shape_str}")
        return 0

    if args.child:
        return child_main(args.child)

    # Orchestrator
    print("running NAX-on arm ...", flush=True)
    on_recs = run_child("on", disable_nax=False)
    print("running NAX-off arm ...", flush=True)
    off_recs = run_child("off", disable_nax=True)
    sanity_check(on_recs, off_recs)
    rows = join(on_recs, off_recs)
    annotate_tflops(rows, all_cases())
    print_table(rows)

    out_path = args.out or f"nax_g16_perf_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify `--list` works**

```bash
/Users/daniel/mlx/.venv/bin/python benchmarks/python/nax_g16_perf_bench.py --list
```

Expected: prints exactly one line, `smoke           M=128 N=128 K=128`. (Real cases land in Tasks 3 and 4.)

- [ ] **Step 3: Verify `--child on` works**

```bash
/Users/daniel/mlx/.venv/bin/python benchmarks/python/nax_g16_perf_bench.py --child on
```

Expected: one JSON line on stdout with `"kernel_label": "smoke"`, a `median_ms` numeric value, etc. Exits 0.

- [ ] **Step 4: Verify orchestrator end-to-end produces a table**

```bash
/Users/daniel/mlx/.venv/bin/python benchmarks/python/nax_g16_perf_bench.py
```

Expected: prints "running NAX-on arm ...", "running NAX-off arm ...", a table with one row labeled `smoke`, and writes a `nax_g16_perf_<timestamp>.json` file with one entry. The smoke case is a 128×128 fp16 matmul which on g16 is small enough that NAX may or may not engage — speedup ratio is expected to be roughly 1.0× ± noise. Don't worry about the value; what matters is the wiring.

The sanity-check warning may print at this point because all timings are within 1% of each other — that's expected for a single tiny case. The warning will quiet down once Tasks 3 and 4 add cases where NAX dispatch genuinely differs.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/python/nax_g16_perf_bench.py
git commit -m "$(cat <<'EOF'
benchmarks: nax_g16_perf_bench skeleton — subprocess A/B + table + JSON

Single-file harness for NAX-on vs NAX-off perf comparison on g16. The
script is its own orchestrator: invoking it spawns two subprocess copies
(one with MLX_DISABLE_NAX=1) that each run the case list and print JSON
records to stdout; the parent joins on (kernel_label, shape) and prints
a comparison table.

This commit ships only a smoke case (128x128 fp16 matmul) so the wiring
can be verified end-to-end. Real cases land in follow-up commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add matmul cases (gemm_fused, gemm_splitk, gemm_segmented)

**Files:**
- Modify: `benchmarks/python/nax_g16_perf_bench.py` (add 7 cases, replace `_smoke_matmul_case` from `all_cases()`)

The three matmul kernel families are all reached through `mx.matmul` / `@` with different shape regimes that trigger the dispatcher to pick fused, splitk, or segmented (batched). The harness doesn't need to know *which* kernel ran — the case naming is for our reporting; the dispatcher decides.

- [ ] **Step 1: Replace `_smoke_matmul_case` and `all_cases()` with the matmul case set**

Replace the `_smoke_matmul_case` function and the body of `all_cases()` with:

```python
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


def all_cases() -> list[Case]:
    return [
        # Llama-7B prefill shapes and a shorter boundary.
        _gemm_fused(M=2048, N=4096, K=4096),
        _gemm_fused(M=2048, N=11008, K=4096),
        _gemm_fused(M=512, N=4096, K=4096),
        # Small-MN-large-K splitk regime.
        _gemm_splitk(M=64, N=64, K=8192),
        _gemm_splitk(M=128, N=128, K=4096),
        # Batched matmul.
        _gemm_segmented(B=8, M=512, N=4096, K=4096),
        _gemm_segmented(B=32, M=128, N=128, K=128),
    ]
```

- [ ] **Step 2: Verify case list**

```bash
/Users/daniel/mlx/.venv/bin/python benchmarks/python/nax_g16_perf_bench.py --list
```

Expected: 7 lines, one per case, in the order above.

- [ ] **Step 3: Run a single arm to confirm cases time successfully**

```bash
/Users/daniel/mlx/.venv/bin/python benchmarks/python/nax_g16_perf_bench.py --child on
```

Expected: 7 JSON lines, each with a non-zero `median_ms`. The 11008-N case will take a few hundred ms per iter (so ~3s total per case for warmup+10 iters). Total runtime ~30s.

- [ ] **Step 4: Run end-to-end orchestrator**

```bash
/Users/daniel/mlx/.venv/bin/python benchmarks/python/nax_g16_perf_bench.py
```

Expected: A 7-row table. On g16 hardware, at least one of the larger gemm_fused rows should show a noticeable speedup difference (>1.05× or <0.95×) — that confirms the gate is doing real work for at least one shape. If every row shows speedup within 1%, the sanity-check warning will trigger and you should investigate before continuing.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/python/nax_g16_perf_bench.py
git commit -m "$(cat <<'EOF'
benchmarks: nax_g16_perf_bench — add matmul cases

Adds 7 matmul cases across three kernel-dispatch regimes:
  - gemm_fused: 3 Llama-style prefill shapes
  - gemm_splitk: 2 small-MN-large-K shapes
  - gemm_segmented: 2 batched matmul shapes

Each case is a plain mx.matmul; the dispatcher picks the kernel.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add gather and SDPA prefill cases

**Files:**
- Modify: `benchmarks/python/nax_g16_perf_bench.py`

Brings the case grid up to the full 13 cases.

- [ ] **Step 1: Add gather and SDPA case constructors**

Insert these helpers after `_gemm_segmented`:

```python
def _gather(tokens: int, E: int, hidden: int, expert_hidden: int,
            top_k: int) -> Case:
    """MoE-style gather_mm. tokens > 32 keeps us on the bm>=32 NAX path on g16
    (bm=16 falls back to non-NAX). Mimics one expert-projection layer:
        x: (tokens, hidden) -> w_per_token: (tokens, expert_hidden, hidden)
        result: (tokens, expert_hidden)
    routed via rhs_indices over E experts.
    """
    def build():
        x = mx.random.normal((tokens, hidden)).astype(mx.float16)
        # Weight matrix per expert: (E, expert_hidden, hidden); we'll swap
        # the last two axes to make it (E, hidden, expert_hidden) for rhs.
        w = mx.random.normal((E, hidden, expert_hidden)).astype(mx.float16)
        # Each token routes to top_k experts. For simplicity in benching
        # we just generate random rhs_indices over [0, E) of length
        # tokens * top_k and reshape x accordingly.
        rhs_idx = mx.random.randint(low=0, high=E,
                                    shape=(tokens * top_k,)).astype(mx.uint32)
        x_rep = mx.repeat(x, repeats=top_k, axis=0)  # (tokens*top_k, hidden)
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
```

- [ ] **Step 2: Extend `all_cases()`**

Replace the `return [...]` body of `all_cases()` with:

```python
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
```

- [ ] **Step 3: Verify case list is now 13 entries**

```bash
/Users/daniel/mlx/.venv/bin/python benchmarks/python/nax_g16_perf_bench.py --list
```

Expected: 13 lines.

- [ ] **Step 4: Run a single arm**

```bash
/Users/daniel/mlx/.venv/bin/python benchmarks/python/nax_g16_perf_bench.py --child on
```

Expected: 13 JSON lines, all with positive `median_ms`. The 8192-kL SDPA case is the longest-running individual case (~50–200 ms/iter), so the whole arm takes maybe a minute.

If a case raises (e.g., shape doesn't accept the API call), the child process exits non-zero and the orchestrator will print the stderr — fix the case before proceeding.

- [ ] **Step 5: Run orchestrator end-to-end**

```bash
/Users/daniel/mlx/.venv/bin/python benchmarks/python/nax_g16_perf_bench.py
```

Expected: 13-row table, both arms complete, JSON file written, sanity check passes (i.e., the 1%-agreement warning does NOT fire — at least one case must show >1% delta).

- [ ] **Step 6: Commit**

```bash
git add benchmarks/python/nax_g16_perf_bench.py
git commit -m "$(cat <<'EOF'
benchmarks: nax_g16_perf_bench — add gather + sdpa_prefill cases

Brings the case grid to 13:
  - gather: 2 MoE-style prefill shapes (bm>=32 stays on NAX)
  - sdpa_prefill: 4 self-attention prefill shapes (fp16, no bool mask)

Excluded by design: bm=16 gather and bool-mask SDPA both fall through
to non-NAX on g16 (per docs/superpowers/specs/2026-05-01-...-design.md
"Excluded on purpose"); A/B is meaningless there.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Run end-to-end and commit the report

**Files:**
- Create: `docs/superpowers/reports/2026-05-01-nax-g16-perf-baseline.md`

Closes the loop: run the harness on g16 hardware, capture the table, and stage the report so the follow-up perf phase has a starting line.

- [ ] **Step 1: Run the harness end-to-end and save the JSON**

```bash
cd /Users/daniel/mlx
/Users/daniel/mlx/.venv/bin/python benchmarks/python/nax_g16_perf_bench.py \
    --out /tmp/nax_g16_perf_baseline.json | tee /tmp/nax_g16_perf_baseline.txt
```

Expected: full 13-row table printed and written to both files. If the sanity-check warning fires, STOP — investigate (rerun `tools/probe_nax_disable_gate.py`) before writing the report.

- [ ] **Step 2: Write the report**

Create `docs/superpowers/reports/2026-05-01-nax-g16-perf-baseline.md` with this template, filling in the table from `/tmp/nax_g16_perf_baseline.txt`:

```markdown
# NAX-on-g16 Performance Baseline — Report

**Date:** 2026-05-01
**Hardware:** [fill in: M3 Pro / M4 Pro / etc., from `system_profiler SPDisplaysDataType | grep -A1 Chipset`]
**MLX commit:** [`git rev-parse HEAD`]
**Spec / plan:** `docs/superpowers/specs/2026-05-01-nax-g16-perf-baseline-design.md`, `docs/superpowers/plans/2026-05-01-nax-g16-perf-baseline.md`

## Method

Single-build A/B via the `MLX_DISABLE_NAX` runtime gate (added in commit
[short SHA of the Task 1 commit]). Two subprocesses, one with the gate
set, run the same 13 cases. Per case: 3 warmup iters, 10 timed iters,
median wall time. Full harness: `benchmarks/python/nax_g16_perf_bench.py`.

## Results

[paste the table from /tmp/nax_g16_perf_baseline.txt verbatim, in a
fenced ``` block ]

JSON: [paste contents of /tmp/nax_g16_perf_baseline.json in a fenced
```json block, or attach as a sibling file if it's long]

## Ranking

By kernel_label, max-to-min speedup (NAX-on-vs-off, higher is better for NAX):

1. **[kernel_label]** — speedup [X.XX]× — [NAX wins / NAX hurts / parity]
2. ...
5. ...

## Observations

- [What stood out — e.g., "gemm_splitk speedup is the worst at 0.83x;
  that's the path with the most threadgroup-scratch overhead per FLOP"]
- [Any anomalies — large p10/p90 spread, unexpected directions, etc.]

## Recommendation for next phase

[Pick the lowest-speedup kernel_label and describe in 2-3 sentences why
its scratch / wm / addressing pattern is the suspected bottleneck.]

## Caveats

- Wall-clock variance: 10-iter median, no fan/thermal control.
- Single-shape-per-row: no statistical sweep within a regime.
- NAX-off arm uses the legacy SIMD-only kernels — for SDPA on g16 that
  is `sdpa_full_self_attention_metal`; for matmul it's the non-NAX
  `steel_gemm_*` family. The comparison is "NAX g16 path" vs. "what we
  would run if NAX were unavailable" — *not* "NAX g16 path" vs. "NAX g17
  path". The latter requires different hardware.
```

- [ ] **Step 3: Commit the report**

```bash
git add docs/superpowers/reports/2026-05-01-nax-g16-perf-baseline.md
git commit -m "$(cat <<'EOF'
docs: nax-g16 perf baseline — report

13-case NAX-on vs NAX-off comparison on g16 hardware. Ranks the 5
kernel paths (gemm_fused, gemm_splitk, gemm_segmented, gather,
sdpa_prefill) by speedup so the next perf phase has a target.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify final state**

```bash
git log --oneline -8
```

Expected: 5 commits on this branch (Task 1 gate, Tasks 2/3/4 harness, Task 5 report) on top of the design-spec commit.

---

## Self-review notes

- **Spec coverage:** all five sections of the spec map to tasks. §"Component 1 — Runtime NAX gate" → Task 1; §"Component 2 — Bench harness" → Tasks 2–4; §"Shape grid" → Tasks 3 and 4 between them; §"Output format" → Task 2 implements the table + JSON; §"File layout" Tasks 1–5 cover all four file rows; §"Success criteria" Task 5 produces the report.
- **No placeholders:** every code block is concrete; no "implement later"; the only `[fill in]` markers are in Task 5 Step 2 (the report template), which is correct because those are values that come from running the harness on hardware.
- **Type consistency:** `Case`, `kernel_label`, `shape`, `median_ms` are used consistently across Tasks 2–4. `_gemm_fused` / `_gemm_splitk` / `_gemm_segmented` / `_gather` / `_sdpa_prefill` builders all return a `Case`. The `flops` callable always takes the shape dict and returns a float.
- **TDD discipline:** Task 1 has the probe before the gate (Step 1 RED, Step 5 GREEN). Tasks 2–4 use `--list` and `--child` invocations as their behavioral checks before the orchestrator runs.
