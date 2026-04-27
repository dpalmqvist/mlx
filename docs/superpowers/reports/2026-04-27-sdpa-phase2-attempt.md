# Phase 2 Report — Shared Dequant + fp16 Dequant: Both Did Not Move the Needle

**Date:** 2026-04-27
**Spec:** [`../specs/2026-04-27-phase2-shared-kv-dequant-design.md`](../specs/2026-04-27-phase2-shared-kv-dequant-design.md)
**Branch:** `kv4-sdpa-phase2`
**Status:** Negative result. Both Phase 1 §7.1 (share dequant across query
heads in a GQA group) and §7.2 (fp16-precision dequant) were tried and
**neither moved the qwen-gqa @ ctx=32768 q4/fp16 ratio measurably from
the Phase 1 baseline of 2.79×**. Kernel reverted to baseline. Phase 3
needs different profiling data — the bottleneck identified in Phase 1
§6 was either wrong or the proposed remedies don't actually attack it.

## 1. Setup

- **Device:** Apple M4 Pro (same as Phase 1).
- **MLX commit (baseline):** `bfeb8b4d` (kv4-sdpa, post-Phase 1 merge).
- **Bench:** `benchmarks/python/sdpa_vector_quantized_bench.py`
  unchanged from Phase 1.
- **Iterations:** 100 measured + 10 warmup per cell. Median reported
  (same as Phase 1).
- **Tests:** `python/tests/test_fast_sdpa.py` —
  `test_quantized_sdpa_vector_matches_dequantized` and
  `test_quantized_sdpa_vector_quality_vs_fp` ran clean across both
  attempts. The 164 pre-existing `test_sdpa_sliced` failures were
  unchanged.

## 2. Attempt A — Phase 1 §7.1 (share K/V dequant across query heads)

### What was implemented

Modified `sdpa_vector_quantized_2pass_1` so that one designated
"producer" simdgroup per threadgroup (`tidtg.y == 0 && tidtg.z == 0`)
dequants K/V into threadgroup-memory tiles `k_shared[D]` and
`v_shared[V]`. All `gqa_factor × q_seq_len` simdgroups in the
threadgroup then read from the shared tiles. Two `threadgroup_barrier`
calls per K-position iteration (one after K-dequant, one after
V-dequant). Producer dequants unconditionally on `use_key` so the
shared tiles are always valid for any consumer that needs them.

### Result — regression

| Cell | Phase 1 q4/fp16 | Attempt A q4/fp16 | Phase 1 q4/dq | Attempt A q4/dq |
|---|---:|---:|---:|---:|
| llama3-8b @ 32768 | 1.64× | **1.89×** | 0.84× | **0.97×** |
| qwen-gqa @ 32768  | 2.79× | **3.05×** | 1.34× | **1.39×** |

Worse on every cell. The barrier overhead exceeded the dequant ALU
savings.

### Why the spec's prediction was wrong

The spec assumed query-head simdgroups in a threadgroup were running
serially on a shared resource — i.e. that dequant ALU was duplicated
gqa_factor times AND that this duplication was on the wall-clock
critical path. Neither premise holds:

- **Simdgroups within a threadgroup run in parallel** on different
  Apple-GPU SIMD execution units. They issue their own dequant
  instructions concurrently. So per-iteration wall-clock latency is
  determined by the slowest *single simdgroup's* work, not aggregate
  threadgroup ALU.
- The shared-dequant design moves all dequant work onto the producer
  simdgroup (which now has the same per-iter work as before — it
  still does the dequant) while consumer simdgroups idle at the
  barrier. **The critical path is unchanged:** producer's per-iter
  cost was ≈ 30 ALU ops in the old kernel, still ≈ 30 ops in the new
  one. The barriers are pure addition.
- Aggregate ALU dropped (114 ops/iter vs 210), but if the GPU isn't
  ALU-saturated to the point of contention, that doesn't translate to
  wall-clock savings — and the regression suggests barrier overhead
  out-cost any contention reduction.

### Cost of two barriers per iter, back-of-envelope

At ctx=32768 with `blocks` set so that each block does ~1024 K
positions, that's ~32k K-iters per simdgroup. Two
`threadgroup_barrier` calls per iter = 64k barriers per simdgroup. At
even ~30 cycles per barrier wait, that's ≈ 2M cycles per simdgroup, or
~1.3 ms at the GPU's ~1.5 GHz. The kernel's full runtime is ~1.4 ms.
The barriers alone could plausibly cost as much as the kernel does
(though this is a loose upper bound; pipelined execution and barrier
fusion likely make the actual cost smaller).

## 3. Attempt B — Phase 1 §7.2 (fp16-precision dequant)

### What was implemented (two variants)

**B1 (surgical):** kept `typedef float U` for k[]/vals[]/o[] storage
and softmax accumulators, but rewrote the dequant FMAs to do
`T(nibble) * scale + bias` in T (the input dtype, fp16/bf16) and cast
the result to U=fp32:

```cpp
T ks = k_scales[qk_group_idx];
T kb = k_biases[qk_group_idx];
k[0] = static_cast<U>(T(p & 0xF) * ks + kb);
```

The hypothesis was that M4 Pro's 2× fp16 ALU throughput on FMAs would
nearly halve the dequant cost without affecting QK score / softmax /
AV-accumulator precision.

**B2 (aggressive):** changed `typedef float U` → `typedef T U`, so the
entire kernel including softmax max/sum_exp ladder runs in T.

### Result — neither moved the headline numbers

**B1 (surgical fp16 dequant FMA):**

| Cell | Phase 1 q4/fp16 | Attempt B1 q4/fp16 | Phase 1 q4/dq | Attempt B1 q4/dq |
|---|---:|---:|---:|---:|
| llama3-8b @ 32768 | 1.64× | 1.67× | 0.84× | 0.85× |
| qwen-gqa @ 32768  | 2.79× | 2.76× | 1.34× | 1.35× |

Within bench noise. q4-specific tests passed; B1 is correctness-clean.

**B2 (aggressive `typedef T U`):** failed to compile. `U max_score =
-INFINITY;` is fine for `T = float` and `T = half`, but for `T = bfloat`
the implicit `float → bfloat` conversion is rejected. Would need
explicit `U(-INFINITY)` casts and audit of every intermediate type
boundary. Not pursued because B1's null result already disproved the
hypothesis: fp16 dequant arithmetic alone wasn't worth the precision
risk for a measurable win.

### Why §7.2's prediction was wrong (or at least unmeasurable here)

Three plausible reasons; can't distinguish without GPU counters:

1. **Apple's Metal compiler may already promote fp16 ALU on this
   path automatically.** If `U(p & 0xF) * float_ks + float_kb` was
   already lowered to fp16 instructions on M4 Pro, B1's explicit
   typing changed nothing in the generated code.
2. **The dequant FMAs may not be the per-simdgroup critical path.**
   Even if they took 0 cycles, the per-iter cost is dominated by
   memory loads (the packed K/V load, scales, biases) and softmax
   exp/max — none of which B1 touched.
3. **The cast `static_cast<U>(T(...) * ks + kb)` may force a fp16 →
   fp32 conversion that costs as much as the FMA throughput delta
   saves.** Particularly likely if the back-end inserts the cast
   eagerly rather than fusing it.

§7.2 deserved more rigorous measurement (per-instruction cost via the
Xcode shader profiler) before committing engineering effort. We didn't
have that data and went on Phase 1 §6.3's structural reasoning, which
gave the right *direction* (dequant is the differentiator) but not the
right *quantification* (the fix is small in wall-clock terms).

## 4. Implication for Phase 1 §6's bound-class claim

Phase 1 §6 concluded the q4 kernel is **ALU-bound on dequant** based
on the per-K-position ALU op count (q4 = 30 ops, fp16 = 14 ops; ratio
2.14× ≈ measured 2.79×). After two negative attempts targeting that
specific class of work, the conclusion needs a hedge:

- The **op count** is genuinely 2.14× higher in q4. That part of §6
  is correct.
- "ALU-bound" was inferred from the op-count ratio matching the
  wall-clock ratio. But that match is also consistent with ANY
  bottleneck that's proportional to per-iter work — register
  pressure, instruction-fetch, threadgroup-memory pressure, etc.
  Without GPU counters, "ALU-bound" was a guess that fits the data,
  not a measurement.
- If the bottleneck were really ALU on the dequant FMAs, B1's surgical
  fp16 change should have moved numbers. It didn't. So either the
  bottleneck is somewhere else (something the dequant *also* exercises
  but isn't the FMA throughput), or the compiler is already fp16-ing.

The honest takeaway: **we don't know what the actual bottleneck is**.
The structural ALU-op accounting was useful for ruling things out
(bandwidth math) but not for predicting which lever moves the kernel.

## 5. What's actually needed for Phase 3

Both negative attempts share a root cause: we acted on a structural
hypothesis without measurement. The Phase 1 report acknowledged this
as a weakness (§6.1: CLI counter collection on M4 Pro is restricted)
but the spec for Phase 2 went ahead anyway. That was the wrong call.

**Pre-Phase-3 work, before any kernel change:**

1. **Get real per-instruction profiling data.** Options:
   - Use the existing `.gputrace` captures (see Phase 1's
     `benchmarks/python/sdpa_vector_quantized_capture.py`) and open
     them in Xcode interactively. Read counters that the CLI cannot
     reach: ALU active %, F16/F32 active %, Memory stall %, L1/L2
     cache hit rates, Register spill bytes, Occupancy.
   - Run the Xcode **Shader Profiler** (the "Profile" button on a
     captured trace) on `sdpa_vector_quantized_2pass_1`. This gives
     per-source-line cost, which would directly answer "is the
     dequant FMA the hot line, or is it the load, or the cast?"
2. **A/B against the fp16 reference.** Capture the same scene with
   the fp16 vector kernel and compare per-line cost between the two
   side-by-side. The lines that differ are where the gap lives. (We
   already produced a comparison trio in Phase 1 §5; what we
   *didn't* do was actually read the counters.)
3. **Microbench at the instruction level.** Build a stripped-down
   kernel that only does `n` rounds of the dequant inner loop, with
   no QK / softmax / AV / loads. Measure its ALU throughput in
   isolation. Compare to a similar microbench of the QK FMA loop. If
   dequant-FMA microbench shows ≪ 30 ops/cycle of cost, then dequant
   is *not* the bottleneck and Phase 1 §6 was wrong.

Only after one of these has produced a quantitative answer should we
attempt another kernel change.

## 6. Phase 3 candidates (revised, contingent on §5 measurement)

The Phase 1 §7 list doesn't survive Phase 2 unchanged. Updated:

- **§7.1 (share dequant)**: ruled out by §2 above. Barriers cost more
  than the savings. Don't retry without addressing barrier overhead
  first (e.g. unrolled loops that amortize barriers across multiple
  K-iters).
- **§7.2 (fp16 dequant)**: B1 shows ~zero impact; B2 untested. Worth
  re-trying B2 only if §5's measurements show fp16-vs-fp32 ALU is in
  fact the bottleneck on this kernel. Otherwise drop.
- **§7.3 (hoist scales/biases per-group)**: still small lever; no
  evidence yet whether it matters. Defer until §5 says yes or no.
- **NEW: investigate whether the kernel is memory-bound on the
  packed load itself.** The `((const device uint16_t*)q_keys)[simd_lid]`
  load is a 2-byte uncoalesced load per lane. fp16 SDPA does
  `keys[j]` for j in 0..qk_per_thread which is contiguous and
  vectorizable. The load pattern asymmetry might dominate. Profile
  with `Texture/Buffer Load` and `Load/Store stall` counters.
- **NEW: investigate register pressure.** q4 holds scales/biases
  (fp32) plus dequant intermediates per simdgroup. Higher register
  pressure → lower occupancy → less latency hiding across threadgroups.
  Read the Xcode register-allocation panel.

## 7. Conclusions

Two attempts targeting the bottleneck identified in Phase 1 §6 didn't
help. The cost was: ~2 hours of build + bench cycles, no shipped
code, no perf change. The information gained is also real:

- We know §7.1 (shared dequant via threadgroup memory + barriers) is
  not the right fix on this kernel.
- We know §7.2 in its surgical form is at best within noise, at worst
  zero, on this kernel.
- We know Phase 1 §6's ALU-bound conclusion was at best partial; the
  structural reasoning gave a direction but not a working lever.
- The next prerequisite is **measurement, not implementation**.

Kernel left at baseline (`bfeb8b4d` content). Phase 2 spec
(`docs/superpowers/specs/2026-04-27-phase2-shared-kv-dequant-design.md`)
is committed as a record of the original (wrong) hypothesis. This
report is the corrective.
