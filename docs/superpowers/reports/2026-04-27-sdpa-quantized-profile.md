# Phase 1 Report — Quantized Vector SDPA Profile

**Date:** 2026-04-27
**Spec:** [`../specs/2026-04-27-profile-first-optimization-design.md`](../specs/2026-04-27-profile-first-optimization-design.md)
**Plan:** [`../plans/2026-04-27-profile-first-optimization-phase1.md`](../plans/2026-04-27-profile-first-optimization-phase1.md)
**Status:** Phase 1 complete (data collected). Sections 6–7 to be filled in after manual GPU frame capture.

## 1. Setup

- **Device:** Apple M4 Pro.
- **NAX:** enabled per `bf760f92 Enable NAX selectively on M4 Pro`.
- **MLX commit:** `37b9fb09` (branch `kv4-sdpa-profile`, off `kv4-sdpa`).
- **macOS:** 26.3.1 (build 25D771280a).
- **Xcode:** 26.4.1 (build 17E202).
- **Python:** 3.13 via `uv`. MLX built editable from this branch.
- **Command:**

  ```bash
  uv run python benchmarks/python/sdpa_vector_quantized_bench.py --out /tmp/sdpa_phase1.csv
  ```

- **Iterations:** 100 measured + 10 warmup per cell. Median reported in the
  tables; p10/p90 in the CSV.

### Methodology caveats (read before interpreting numbers)

- Each iteration is timed as a single `mx.eval(out)` round-trip, which captures
  Python dispatch + Metal encoder + GPU execute together. Treat absolute
  milliseconds as having ~10% uncertainty from OS jitter and shared-system
  load. Treat *ratios* (`q4/fp16`, `q4/dq`) as reliable to ~5% — both
  numerator and denominator share the same jitter floor.
- `fp16_sdpa` is timed against the *original* fp16 K/V tensors, not the
  fp32→quantize→dequantize round-trip of the same data. The `dq_then_sdpa`
  column is the apples-to-apples "same packed tensors, different kernel"
  comparison; `q4/dq` is therefore the more honest sanity-check ratio.
- `dq_then_sdpa` includes the `mx.dequantize(...)` calls inside the timed
  region by design. It measures the dequant + fp16 SDPA path as a unit, which
  is what the kernel under test is supposed to beat.
- `q4_sdpa` and `dq_then_sdpa` are both verified against each other within
  `atol=rtol=5e-3` (per `python/tests/test_fast_sdpa.py:731-732`) before
  timing each cell. No correctness-gate failures occurred during this run.

## 2. Results

### llama3-8b (Hq=32, Hk=8, D=128)

```
   ctx   fp16_sdpa     q4_sdpa    dq_then_sdpa   q4/fp16     q4/dq
  1024      0.2185      0.2445          0.1977     1.12x     1.24x
  2048      0.1561      0.2191          0.2228     1.40x     0.98x
  4096      0.1965      0.3047          0.3184     1.55x     0.96x
  8192      0.3040      0.4791          0.5475     1.58x     0.88x
 16384      0.5033      0.9023          1.0394     1.79x     0.87x
 32768      0.9716      1.5965          1.8996     1.64x     0.84x
```

### qwen-gqa (Hq=28, Hk=4, D=128)

```
   ctx   fp16_sdpa     q4_sdpa    dq_then_sdpa   q4/fp16     q4/dq
  1024      0.1344      0.1680          0.1476     1.25x     1.14x
  2048      0.1176      0.1919          0.1792     1.63x     1.07x
  4096      0.1508      0.2660          0.2195     1.76x     1.21x
  8192      0.1989      0.4146          0.3256     2.08x     1.27x
 16384      0.3080      0.7927          0.6207     2.57x     1.28x
 32768      0.4977      1.3868          1.0324     2.79x     1.34x
```

CSV: `/tmp/sdpa_phase1.csv` (not committed — regenerate via the command in §1).

## 3. Where the gap lives

**Worst (model, ctx) cell by `q4/fp16` ratio:** `qwen-gqa` at `ctx=32768` →
**2.79×** slower than fp16 vector SDPA. That is also the cell where the
absolute gap is largest in milliseconds (1.39 ms vs 0.50 ms ≈ 0.89 ms
delta).

**Trend with ctx:**

- `llama3-8b`: 1.12× → 1.40× → 1.55× → 1.58× → 1.79× → 1.64×. Mostly
  growing, with a small dip at the longest ctx. Rough shape suggests the
  kernel becomes increasingly bandwidth-bound as ctx grows, then plateaus
  when fp16 also starts saturating bandwidth.
- `qwen-gqa`: 1.25× → 1.63× → 1.76× → 2.08× → 2.57× → 2.79×. Monotonically
  growing. The kernel does *not* close the gap as ctx scales — if anything
  it widens. This is the more concerning trend.

**Why qwen-gqa is worse than llama3-8b:** qwen-gqa has Hk=4 (vs Hk=8 on
llama3-8b), meaning each KV head is shared by 7 query heads (vs 4 on
llama). The fp16 vector kernel can amortize K/V loads across query heads
within a KV group, so its per-token cost grows slowly with the GQA ratio.
The quantized kernel's dequant work scales with K/V tile reads, so a
higher GQA ratio doesn't help it as much. The gap-vs-fp16 widening with
ctx confirms this: at long ctx, K/V bandwidth dominates total work, and
that is where the quantized path's dequant tax bites hardest.

## 4. Sanity check (q4 vs dequant→SDPA)

The premise of the new kernel is that running SDPA directly on packed 4-bit
K/V is faster than dequantizing first and then running fp16 SDPA. The
`q4/dq` column tests that premise.

- **llama3-8b**: q4_sdpa beats dq_then_sdpa (`q4/dq < 1.0×`) at ctx ≥ 2048
  (0.98× → 0.84×). At ctx=1024 it loses (1.24×). The kernel earns its keep
  on this shape from medium context onward.
- **qwen-gqa**: q4_sdpa **loses to dq_then_sdpa across the entire sweep**
  (1.07× → 1.34×). The new kernel never beats the dequant fallback on this
  shape. This is the correctness-of-purpose failure the spec called out:
  on qwen-gqa we would currently be better off calling
  `mx.dequantize(...)` followed by `mx.fast.scaled_dot_product_attention(...)`
  rather than `mx.fast.quantized_scaled_dot_product_attention(...)`.

This finding is what makes Phase 2 worth doing — and it argues that
"profile-first" was the correct framing: without this run we would not
have known the kernel underperforms its own fallback on a real model
shape.

## 5. Frame-capture playbook

Use this when Phase 2 starts (or whenever the worst cell from §3 needs
diagnosis).

1. **Pick the target cell** — the (model, ctx) with the highest `q4/fp16`
   ratio in §3. Currently: `qwen-gqa @ ctx=32768`. Secondary: `qwen-gqa @
   ctx=16384` (2.57×) since it runs faster, which makes the capture cycle
   tighter.
2. **Build MLX with debug shaders.** Verify the actual flag in
   `mlx/backend/metal/CMakeLists.txt` before running; do not commit a
   guessed flag name. The build env is otherwise the standard MLX dev build
   (`uv pip install -e .` with `CMAKE_BUILD_TYPE=Debug` and the metal-debug
   flag set as an env var or cmake arg).
3. **Reduce the workload to a single dispatch.**

   ```bash
   uv run python benchmarks/python/sdpa_vector_quantized_bench.py \
       --models qwen-gqa --ctx 32768 \
       --iters 1 --warmup 0
   ```

4. **Capture in Xcode.** Open Xcode → Debug → Capture GPU Frame, attach to
   the running Python process (use `--warmup 0` so the first dispatch is
   not consumed by warmup), trigger one frame.
5. **Inspect the `sdpa_vector_quantized` dispatch.** Read these counters/views
   and write the value into §6:
   - **Occupancy** — low (<25%) usually means register or
     threadgroup-memory pressure.
   - **ALU active vs memory wait** — ratio identifies bound class. >70%
     memory wait → bandwidth-bound. >70% ALU → compute-bound.
   - **L1/L2 cache hit rate on K/V loads** — low hit rate suggests the
     dequant unpack pattern is cache-unfriendly.
   - **Register pressure / spills** — any spills mean the kernel is
     ALU-bound for the wrong reason and a register-pressure fix is the
     first thing to try.

## 6. Findings

_To be filled in after frame capture. Bulleted observations only — no
conclusions._

## 7. Phase 2 candidates

_To be written after §6 is filled in. References §6._
