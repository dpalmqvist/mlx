# Profile-First Optimization (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sweep harness that compares 4-bit quantized vector SDPA against fp16 vector SDPA on M4 Pro, run it, and write a report identifying where the gap lives.

**Architecture:** One standalone benchmark script under `benchmarks/python/` (matching the repo's per-op convention) plus a markdown report under `docs/superpowers/reports/`. No kernel changes. The script enforces a correctness gate (q4 vs dequant→SDPA) before timing each cell, so we never ship timings of a misbehaving kernel.

**Tech Stack:** Python, MLX (`mx.fast.scaled_dot_product_attention`, `mx.fast.quantized_scaled_dot_product_attention`, `mx.quantize`, `mx.dequantize`), `time.perf_counter`, `csv`. Run on macOS / Apple M4 Pro.

**Spec:** `docs/superpowers/specs/2026-04-27-profile-first-optimization-design.md`

**Branch:** `kv4-sdpa-profile` (already created and currently checked out; spec already committed as `bf873cd5`).

---

## File Structure

- **Create:** `benchmarks/python/sdpa_vector_quantized_bench.py` — the entire sweep harness as one self-contained script.
- **Create:** `docs/superpowers/reports/2026-04-27-sdpa-quantized-profile.md` — the Phase 1 report. Sections 1–5 filled in from a real run; sections 6–7 ship as labelled placeholders.

One file per deliverable. No shared harness module — Phase 2's needs are unknown, so introducing reusable abstractions now would be premature.

A note on TDD: this is a benchmark with a baked-in correctness gate, not application code. There is no separate test file. The "tests" in the steps below are smoke-runs of the script with reduced parameters. The correctness gate inside the script is the actual behavioral check.

---

## Task 1: Write the bench script

**Files:**
- Create: `benchmarks/python/sdpa_vector_quantized_bench.py`

- [ ] **Step 1: Write the script in full**

Create `benchmarks/python/sdpa_vector_quantized_bench.py` with exactly this content:

```python
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
```

- [ ] **Step 2: Smoke-run the script**

Run:

```bash
python benchmarks/python/sdpa_vector_quantized_bench.py \
    --ctx 1024 --iters 5 --warmup 2 --models llama3-8b
```

Expected: prints `running llama3-8b ctx=1024 ...` then a table with one ctx row. All three median times under 5 ms. No `correctness gate failed` error. Exit code 0.

If the correctness gate fires: the kernel under test (`q4_sdpa`) disagrees with the dequant fallback by more than 5e-3 absolute / 5e-3 relative. Stop and investigate before continuing — timing data is meaningless if the gate fails.

- [ ] **Step 3: Smoke-run with CSV output**

Run:

```bash
python benchmarks/python/sdpa_vector_quantized_bench.py \
    --ctx 1024,2048 --iters 5 --warmup 2 --models llama3-8b \
    --out /tmp/sdpa_smoke.csv
```

Expected: prints two rows in the table; `wrote /tmp/sdpa_smoke.csv`. Inspect the CSV:

```bash
column -s, -t /tmp/sdpa_smoke.csv | head -20
```

Expected columns: `model Hq Hk D V ctx variant iters median_ms p10_ms p90_ms git_sha device`. Six data rows total (2 ctx × 3 variants). `git_sha` matches `git rev-parse --short HEAD`.

- [ ] **Step 4: Commit the script**

```bash
git add benchmarks/python/sdpa_vector_quantized_bench.py
git commit -m "$(cat <<'EOF'
bench: add quantized vector SDPA sweep

Phase 1 of profile-first optimization (see
docs/superpowers/specs/2026-04-27-profile-first-optimization-design.md).

Sweeps two model shapes (Llama-3 8B style, Qwen-style GQA) across decode
context lengths 1024..32768 and times three variants per cell: fp16 vector
SDPA, 4-bit quantized vector SDPA, and dequant->fp16 SDPA. Reports median
plus p10/p90 latency. Each cell runs a correctness gate
(q4_sdpa vs dequant+sdpa) before timing so timing data is never collected
on a misbehaving kernel. CSV output is long-form so future runs can be
concatenated and diffed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Run the full sweep

**Files:**
- Create (transient): `/tmp/sdpa_phase1.csv`

- [ ] **Step 1: Run the full default sweep**

```bash
python benchmarks/python/sdpa_vector_quantized_bench.py \
    --out /tmp/sdpa_phase1.csv 2>&1 | tee /tmp/sdpa_phase1.stdout
```

Expected: completes in under ~5 minutes on M4 Pro. Two tables printed (one per model), six ctx rows each. No correctness-gate failures.

If the run exceeds 10 minutes or any cell's `q4_sdpa` median exceeds ~50 ms, something is off (likely the GPU is contended or the kernel routed off the vector path). Stop and investigate before writing the report.

- [ ] **Step 2: Sanity-check the captured output**

```bash
wc -l /tmp/sdpa_phase1.csv         # expect 1 header + 36 data rows = 37
column -s, -t /tmp/sdpa_phase1.csv | head -10
grep -c "fp16_sdpa\|q4_sdpa\|dq_then_sdpa" /tmp/sdpa_phase1.csv  # expect 36
```

Confirm the stdout tee captured the per-model tables (they are what gets pasted into the report).

---

## Task 3: Write the report

**Files:**
- Create: `docs/superpowers/reports/2026-04-27-sdpa-quantized-profile.md`

- [ ] **Step 1: Capture environment metadata**

Run these and keep the output to paste into §1 of the report:

```bash
git rev-parse --short HEAD
sw_vers
xcodebuild -version 2>/dev/null || echo "Xcode not on PATH"
sysctl -n machdep.cpu.brand_string
```

- [ ] **Step 2: Write the report file**

Create `docs/superpowers/reports/2026-04-27-sdpa-quantized-profile.md` using this exact skeleton, filling in the bracketed placeholders from the run output and the metadata captured above:

```markdown
# Phase 1 Report — Quantized Vector SDPA Profile

**Date:** 2026-04-27
**Spec:** [`../specs/2026-04-27-profile-first-optimization-design.md`](../specs/2026-04-27-profile-first-optimization-design.md)
**Status:** Phase 1 complete (data collected). Sections 6–7 to be filled in after manual GPU frame capture.

## 1. Setup

- **Device:** Apple M4 Pro ([CPU brand string from `sysctl`]).
- **NAX:** enabled per `bf760f92 Enable NAX selectively on M4 Pro`.
- **MLX commit:** [`git rev-parse --short HEAD` output] (branch `kv4-sdpa-profile`).
- **macOS:** [`sw_vers -productVersion` output].
- **Xcode:** [`xcodebuild -version` output, or "not installed" if absent].
- **Command:**

  ```bash
  python benchmarks/python/sdpa_vector_quantized_bench.py --out /tmp/sdpa_phase1.csv
  ```

- **Iterations:** 100 measured + 10 warmup per cell. Median reported; p10/p90 in CSV.

## 2. Results

### llama3-8b (Hq=32, Hk=8, D=128)

```
[paste the llama3-8b table block from /tmp/sdpa_phase1.stdout, including the header row]
```

### qwen-gqa (Hq=28, Hk=4, D=128)

```
[paste the qwen-gqa table block from /tmp/sdpa_phase1.stdout]
```

CSV: `/tmp/sdpa_phase1.csv` (not committed — regenerate via the command in §1).

## 3. Where the gap lives

- **Worst (model, ctx) cell by `q4/fp16` ratio:** [model, ctx, ratio].
- **Trend with ctx:** [growing | flat | shrinking] for [each model]. Interpretation:
  - growing → bandwidth/dequant tax dominates at long context.
  - flat → fixed per-call overhead (kernel launch, encoder, dispatch).
  - shrinking → fp16 path saturates first; quantized path scales better.
- [One sentence per model summarizing the shape of the gap.]

## 4. Sanity check (q4 vs dequant→SDPA)

- [Cells where `q4_sdpa` is faster than `dq_then_sdpa` — expected.]
- [Cells (if any) where `q4_sdpa` is slower than `dq_then_sdpa` — call them out; this would be a correctness-of-purpose failure for the new kernel.]

## 5. Frame-capture playbook

Use this when Phase 2 starts (or whenever the worst cell from §3 needs diagnosis).

1. **Pick the target cell** — the (model, ctx) with the highest `q4/fp16` ratio in §3.
2. **Build MLX with debug shaders.** Verify the actual flag in `mlx/backend/metal/CMakeLists.txt` before running; do not commit a guessed flag name. The build env is otherwise the standard MLX dev build.
3. **Reduce the workload to a single dispatch.**

   ```bash
   python benchmarks/python/sdpa_vector_quantized_bench.py \
       --models <worst-model> --ctx <worst-ctx> \
       --iters 1 --warmup 0
   ```

4. **Capture in Xcode.** Open Xcode → Debug → Capture GPU Frame, attach to the Python process, trigger one frame.
5. **Inspect the `sdpa_vector_quantized` dispatch.** Read these counters/views and write the value into §6:
   - **Occupancy** — low (<25%) usually means register or threadgroup-memory pressure.
   - **ALU active vs memory wait** — ratio identifies bound class. >70% memory wait → bandwidth-bound. >70% ALU → compute-bound.
   - **L1/L2 cache hit rate on K/V loads** — low hit rate suggests the dequant unpack pattern is cache-unfriendly.
   - **Register pressure / spills** — any spills mean the kernel is ALU-bound for the wrong reason and a register-pressure fix is the first thing to try.

## 6. Findings

_To be filled in after frame capture. Bulleted observations only — no conclusions._

## 7. Phase 2 candidates

_To be written after §6 is filled in. References §6._
```

- [ ] **Step 3: Verify the report renders sanely**

```bash
wc -l docs/superpowers/reports/2026-04-27-sdpa-quantized-profile.md
grep -n "\[paste\|\[CPU brand\|\[git rev\|\[sw_vers\|\[xcodebuild\|\[model, ctx\|\[Cells " docs/superpowers/reports/2026-04-27-sdpa-quantized-profile.md
```

Expected: `grep` returns nothing — every bracketed placeholder has been replaced. If any remain, fill them in before committing.

- [ ] **Step 4: Commit the report**

```bash
git add docs/superpowers/reports/2026-04-27-sdpa-quantized-profile.md
git commit -m "$(cat <<'EOF'
docs: add Phase 1 SDPA quantized profile report

Captures the q4/fp16 vector-SDPA gap on M4 Pro across two model shapes
and six context lengths. Identifies the worst (model, ctx) cell and
includes a frame-capture playbook for diagnosing it. Sections 6 and 7
ship as labelled placeholders to be filled in after manual GPU frame
capture; that completes Phase 1 and feeds the Phase 2 brainstorm.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Verify final state

- [ ] **Step 1: Confirm three commits on `kv4-sdpa-profile`**

```bash
git log --oneline kv4-sdpa..HEAD
```

Expected: three commits in this order (newest first):
- `docs: add Phase 1 SDPA quantized profile report`
- `bench: add quantized vector SDPA sweep`
- `docs: add profile-first optimization design (Phase 1)`

- [ ] **Step 2: Confirm working tree is clean**

```bash
git status
```

Expected: `nothing to commit, working tree clean`. The CSV/stdout artifacts in `/tmp/` are intentionally not tracked.

---

## Self-Review

**Spec coverage check:**

- Bench script with shape matrix, three variants, decode shape, group_size=64/bits=4 → Task 1 Step 1.
- Wall-clock methodology, 10 warmup + 100 iters, median + p10/p90 → Task 1 Step 1 (`time_op`).
- CLI flags (`--out`, `--models`, `--ctx`, `--iters`, `--warmup`, `--device`) → Task 1 Step 1 (`main`).
- Stdout table format → Task 1 Step 1 (`print_table`).
- CSV schema (long form, with `git_sha` and `device`) → Task 1 Step 1 (`write_csv`).
- Correctness gate using tolerance from `test_quantized_sdpa_vector_matches_dequantized` → Task 1 Step 1 (`assert_close`, `ATOL=5e-3`, `RTOL=5e-3`); cross-checked against `python/tests/test_fast_sdpa.py:731-732` (the bits=4/fp16 path uses 5e-3, not 1e-2 — value confirmed against the source).
- Smoke test (ctx=1024, iters=5, warmup=2) → Task 1 Step 2.
- Sanity time bound (<5 ms) → Task 1 Step 2 expected output.
- Full sweep < 5 min → Task 2 Step 1 expected.
- Report sections 1–7 with §6/§7 as placeholders → Task 3 Step 2.
- Frame-capture playbook → Task 3 Step 2 §5.
- Three commits in order → Tasks 1 Step 4, 3 Step 4 (the spec was committed before this plan started, as `bf873cd5`).
- All on branch `kv4-sdpa-profile` → already checked out.

No spec sections are unrepresented.

**Placeholder scan:** No "TBD"/"TODO"/"implement later" steps. Every code step has the actual code. The only placeholders in the deliverables are §6/§7 of the *report*, which the spec explicitly approves as placeholders. Bracketed `[paste ...]` markers in Task 3 Step 2 are filled in by the executor from real run output and explicitly verified absent in Step 3.

**Type/identifier consistency:** `MODELS`, `ModelShape`, `DEFAULT_CTX`, `DTYPE`, `GROUP_SIZE`, `BITS`, `ATOL`, `RTOL`, `make_inputs`, `time_op`, `assert_close`, `git_sha`, `run_cell`, `print_table`, `write_csv`, `main` — names used consistently across the script. Variant labels (`fp16_sdpa`, `q4_sdpa`, `dq_then_sdpa`) match between `run_cell`, `print_table`, the CSV writer, and the report skeleton. CLI flag names match the spec.

No fixes needed.
