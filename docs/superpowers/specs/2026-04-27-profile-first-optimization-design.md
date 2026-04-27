# Profile-First Optimization of Quantized Vector SDPA — Phase 1 Design

**Date:** 2026-04-27
**Branch:** `kv4-sdpa-profile` (off `kv4-sdpa`)
**Scope:** Phase 1 only — measurement and reporting. No kernel changes.

## Goal

Close the performance gap between the new 4‑bit quantized vector SDPA Metal
kernel (`mlx/backend/metal/kernels/sdpa_vector_quantized.h`, landed in
3dc1f5a3) and the existing fp16 vector SDPA kernel.

"Profile-first" means: build the measurement loop and produce data before
proposing or implementing any kernel change. This document covers Phase 1
(data only). Phase 2 (kernel changes) will be brainstormed in a separate
spec once the Phase 1 report is in hand.

## Non-Goals (Phase 1)

- No kernel modifications.
- No CI integration of the benchmark.
- No plotting.
- No cross-device sweep — M4 Pro only.
- No prefill / multi-token decode shapes — vector kernel only.

## Workload

- **Hardware:** Apple M4 Pro (this branch's NAX configuration applies).
- **Mode:** Decode (B=1, seq_q=1).
- **Models:**
  - `llama3-8b`: Hq=32, Hk=8, D=128, V=128.
  - `qwen-gqa`: Hq=28, Hk=4, D=128, V=128.
- **Context lengths:** 1024, 2048, 4096, 8192, 16384, 32768.
- **Quantization config:** group_size=64, bits=4 (matches the fused kernel's
  routing path; see `python/tests/test_fast_sdpa.py:701`).

## Methodology

- Wall-clock via `time.perf_counter` with `mx.eval`/`mx.synchronize`.
- 10 warmup iters + 100 measured iters per (model, ctx, variant) cell.
- Inputs built once outside the timing loop; the op is the only thing timed.
- Per-iter deltas recorded individually so we can report **median** plus
  p10/p90 (median is robust to OS jitter; mean is not).
- `mx.random.seed(0)` per cell for reproducibility across reruns.
- For diagnostics: Xcode GPU Frame Capture on the worst-performing cell
  (manual, post-sweep). Methodology written into the report's playbook
  section.

### Variants compared per cell

| Variant         | What it measures                                                |
| --------------- | --------------------------------------------------------------- |
| `fp16_sdpa`     | `mx.fast.scaled_dot_product_attention` on fp16 K/V (the target) |
| `q4_sdpa`       | `mx.fast.quantized_scaled_dot_product_attention` on 4-bit K/V   |
| `dq_then_sdpa`  | `mx.dequantize(...)` → fp16 SDPA (sanity reference)             |

The `q4/fp16` ratio is the headline number ("the gap"). The `q4/dq_then_sdpa`
ratio is a sanity check: if the new kernel ever loses to the dequant fallback
on a real shape, that is a correctness-of-purpose failure.

## Deliverables

### 1. Bench script — `benchmarks/python/sdpa_vector_quantized_bench.py`

CLI:

```
--out PATH         optional; write CSV to PATH
--models LIST      default: llama3-8b,qwen-gqa
--ctx LIST         default: 1024,2048,4096,8192,16384,32768
--iters N          default: 100
--warmup N         default: 10
--device LABEL     default: M4Pro (recorded into CSV only)
```

Stdout (one table per model):

```
model=llama3-8b  Hq=32 Hk=8 D=128
ctx     fp16_sdpa  q4_sdpa  dq_then_sdpa  q4/fp16  q4/dq
1024      0.123      0.187     0.245        1.52x    0.76x
2048      ...
```

CSV (long form, one row per variant per cell):

```
model,Hq,Hk,D,V,ctx,variant,iters,median_ms,p10_ms,p90_ms,git_sha,device
llama3-8b,32,8,128,128,1024,fp16_sdpa,100,0.1234,0.1198,0.1290,3dc1f5a3,M4Pro
...
```

Long form (not pivot form) so future re-runs can be concatenated and diffed
without schema changes. `git_sha` from `git rev-parse --short HEAD`.

### 2. Report — `docs/superpowers/reports/2026-04-27-sdpa-quantized-profile.md`

Sections:

1. **Setup** — device, NAX state, MLX commit, OS / Xcode versions, the exact
   command run.
2. **Results table** — stdout table for both models + the q4/fp16 ratio.
3. **Where the gap lives** — short prose: which cells have the largest
   q4/fp16 ratio, and how the ratio scales with ctx (growing → bandwidth
   tax dominates; flat → fixed per-call overhead; shrinking → fp16
   saturates first).
4. **Sanity check** — q4_sdpa vs dq_then_sdpa ratio. Call out any cell where
   the new kernel loses to the fallback.
5. **Frame-capture playbook** — written so a future session can re-run
   without re-deriving:
   - Pick the worst (model, ctx) cell from §3.
   - Build with debug shaders enabled (verify the actual MLX/CMake flag at
     write time rather than committing a guessed flag name).
   - Run the script under Xcode → Debug → Capture GPU Frame, with
     `--iters 1 --warmup 0` to isolate a single dispatch.
   - Inspect the `sdpa_vector_quantized` dispatch.
   - Counters/views to read and what they mean:
     - **Occupancy** → low = register or threadgroup memory pressure.
     - **ALU active vs memory wait** → ratio identifies bound class.
     - **L1/L2 cache hit on K/V loads** → low = unpack pattern is
       cache-unfriendly.
     - **Register pressure / spills** → spills make the kernel ALU-bound
       for the wrong reason.
6. **Findings** — placeholder at PR time (`_To be filled in after frame
   capture._`); bulleted observations only, no conclusions.
7. **Phase 2 candidates** — placeholder at PR time; written after §6 is
   filled in. References §6.

## Test Plan (script-level)

- **Smoke:** `python benchmarks/python/sdpa_vector_quantized_bench.py
  --ctx 1024 --iters 5 --warmup 2` completes without error and prints a
  table.
- **Sanity:** at ctx=1024, all three variants finish in <5 ms each (catches
  "we accidentally measured something else").
- **Correctness gate:** before timing each cell, assert `q4_sdpa` output
  matches `dq_then_sdpa` output within the tolerance used in
  `test_quantized_sdpa_vector_matches_dequantized`
  (`python/tests/test_fast_sdpa.py:686`). Reuse the same tolerance values.
  If a cell fails, abort the whole run — the timing is meaningless if the
  kernel isn't computing the right thing on those shapes.
- **Full sweep:** default matrix runs end-to-end on M4 Pro in under ~5
  minutes.

## Commit Plan

On a new branch `kv4-sdpa-profile` off `kv4-sdpa`:

1. `docs: add profile-first optimization design (Phase 1)` — spec only.
2. `bench: add quantized vector SDPA sweep` — script only. Verify it runs on
   M4 Pro before commit 3.
3. `docs: add Phase 1 SDPA quantized profile report` — report with §1–§4
   filled in from a real run; §5 (playbook) included; §6/§7 placeholders.

## Open Items (resolved before Phase 2, not Phase 1)

- Exact `qwen-gqa` head config — confirmed Hq=28/Hk=4 per Section "Workload",
  but cross-check against a real Qwen-3 release before publishing the
  report; if a more representative GQA shape exists, swap it.
- Debug-shader build flag for the Xcode capture step — verify the actual
  CMake flag in `mlx/backend/metal/CMakeLists.txt` when writing the playbook
  rather than committing a guess.
