# Phase 3 Design — Measure the Quantized Vector SDPA Bottleneck Properly

**Date:** 2026-04-27
**Branch:** `kv4-sdpa-phase3-measure` (off `kv4-sdpa`)
**Depends on:**
- `docs/superpowers/reports/2026-04-27-sdpa-quantized-profile.md` (Phase 1)
- `docs/superpowers/reports/2026-04-27-sdpa-phase2-attempt.md` (Phase 2 negative result)

## Goal

Get **measured** data on what the quantized vector SDPA kernel is
actually bound by, so the next kernel change has a working hypothesis
behind it. Phase 1 §6 made a structural ALU-op-count argument that
was inferred from wall-clock ratios, not measured. Phase 2 tested two
levers against that hypothesis (§7.1 shared dequant, §7.2 fp16
dequant) — both didn't move the numbers. The bottleneck is somewhere
the structural reasoning didn't see.

Phase 3 produces no kernel change. Its only deliverable is data + an
analysis report that pinpoints the actual bottleneck, ranked by
counter evidence rather than op-count math.

## Non-Goals

- No kernel changes. Phase 3 is measurement only.
- No new MLX user-facing API.
- No multi-platform work — M4 Pro only.

## Two Independent Measurements

### M1 — Xcode Shader Profiler (interactive, by user)

Open the captured `.gputrace` in Xcode and read counters that the CLI
cannot reach. Phase 1 §6.1 established that `xctrace` exposes only
`RT Unit Active` on M4 Pro / macOS 26.3.1; the rest of the Apple GPU
counter set is gated to the Xcode UI. The captures already exist
from Phase 1's PR #4
(`benchmarks/python/sdpa_vector_quantized_capture.py`).

What to read, in this order, on the worst-cell q4 capture
(`/tmp/sdpa_q4_sdpa_qwen_gqa_32768.gputrace`):

1. **Limiter / Bottleneck** (Performance pane → Limiter column on
   the dispatch). Apple GPU Tools labels the dominant stall class.
   Single most useful reading.
2. **ALU active %** vs **Memory stall %**. Confirms the limiter.
   - >70% Memory stall → bandwidth or load-pattern bound.
   - >70% ALU active → compute bound. Then look at:
3. **F16 active % vs F32 active %**. If F32 active dominates and F16
   is near zero, the dequant arithmetic is genuinely fp32 — Phase 2
   §7.2 should have helped (and didn't, which would be confusing).
   If F16 active is high, Apple's compiler is already fp16-promoting
   and §7.2 was a no-op for that reason.
4. **L1 Buffer Cache Hit %**, **L2 Cache Hit %**. Low L2 hit on the
   K/V loads → uncoalesced or thrashing.
5. **Occupancy**. <25% means the kernel isn't filling the chip.
   Likely register-pressure-driven if low.
6. **Register allocation / spill bytes** (Pipeline Statistics on
   the kernel). Any nonzero spill means register pressure is the
   first problem to solve.

For comparison, repeat (1)–(6) on the fp16 capture
(`/tmp/sdpa_fp16_sdpa_qwen_gqa_32768.gputrace`). The deltas tell
us where the gap lives.

**Source-line profiling.** Click the "Profile" button on a captured
trace (stopwatch icon top-right). This re-runs the shader with full
counter collection (~30s) and unlocks **per-source-line ALU cost** in
the Shaders panel. With MLX built `-DMLX_METAL_DEBUG=ON`
(`mlx/CMakeLists.txt:39`), each `.metal` source line maps to its
cost. Open `sdpa_vector_quantized_2pass_1` and find the top 3 hot
lines. Are they:
- Dequant FMAs (`U(p & 0xF) * ks + kb`)?
- Packed loads (`((const device uint16_t*)q_keys)[simd_lid]`)?
- Softmax exp / max?
- AV updates (`o[j] = o[j] * factor + exp_score * vals[j]`)?

The hot-line ranking is the most actionable single piece of data
Phase 3 can produce.

### M2 — Dequant-only microbenchmark (autonomous, by harness)

A Python microbench (`benchmarks/python/sdpa_dequant_microbench.py`,
new file) that times `mx.dequantize` on the same shapes the SDPA
kernel processes per call, in isolation. Compares to:

- A pure fp16 read of the same byte volume (`mx.zeros_like(...) +
  source`) as a memory-bandwidth lower bound.
- A pure fp16 ALU op (`source * scale`) at the same data volume as
  an ALU-bandwidth reference.

If dequant runs ≈ at the fp16-read time, the dequant ALU is
essentially free (memory-load-bound). If dequant runs ≈ at the
fp16-multiply time, dequant ALU IS the cost — and Phase 2 §7.2's null
result is genuinely confusing (would point at compiler already
fp16-promoting). If dequant takes far longer than either, there's a
third bottleneck (the bit-extract pattern, or scale/bias lookup).

This is a **single-op** microbench that doesn't carry any of the
SDPA kernel's other work, so the bottleneck attribution is direct.

## Deliverables

1. **`benchmarks/python/sdpa_dequant_microbench.py`** — the
   microbench.
2. **`docs/superpowers/reports/2026-04-28-sdpa-phase3-measurement.md`**
   — the analysis report. Sections:
   1. Setup (device, MLX commit, build flags).
   2. M2 results (microbench numbers + interpretation).
   3. M1 results (counter values from Xcode, copy-pasted).
   4. Synthesis — what the actual bottleneck is, with citations to
      §2/§3 numbers.
   5. Phase 4 candidates — the (now informed) list of kernel
      changes to try, ranked by counter evidence.

## Workflow

User runs M1 (interactive Xcode), pastes counter readings into the
report. I run M2 autonomously and fill in those numbers. We then jointly
fill in the synthesis (§4 of the report).

Realistic time budget:
- M2 microbench script + run: 30 minutes (autonomous).
- M1 in Xcode: 30–60 minutes (user). The Profile button's source-line
  cost is the part that takes most of the time; counter-reading is
  fast once you know where to click.
- Synthesis: 15 minutes.

## Test Plan

- [ ] Microbench correctness: against a reference (call mx.dequantize
      with seeded inputs, recompute the dequant in numpy, compare
      within tolerance). The microbench is a benchmark, not a kernel
      change, so this gate is just to make sure we're timing what we
      think we are.
- [ ] Microbench numbers reproduce within 5% across consecutive runs.
- [ ] Captures open cleanly in Xcode (no "empty trace" failures —
      the multi-iter capture pattern from Phase 1's PR #4 already
      handles this).

## Commit Plan

On `kv4-sdpa-phase3-measure` (off `kv4-sdpa`):

1. `docs: add Phase 3 measurement design` — this spec.
2. `bench: add dequant-throughput microbenchmark` — the script.
3. `docs: add Phase 3 measurement report` — final analysis with both
   M1 and M2 sections filled in.

## Open Items

- Whether to also rebuild MLX with `-DMLX_METAL_DEBUG=ON` for the
  capture re-run. Yes — the Phase 1 PR #4 baseline used Debug for
  exactly this reason. If the captures from PR #4 are still on disk
  and were taken with debug shaders, they can be reused for the M1
  step. Re-generate if not.
- Whether to capture the `dq_then_sdpa` variant too. The fp16
  capture is the more direct comparison (both kernels are vector SDPA
  variants); `dq_then_sdpa` mixes a `mx.dequantize` op with a
  separate `fp16 SDPA` call so the per-dispatch attribution is more
  diffuse. Skip unless M1 leaves the bottleneck ambiguous.
