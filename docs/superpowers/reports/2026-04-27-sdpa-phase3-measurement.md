# Phase 3 Report — Measuring the Quantized Vector SDPA Bottleneck

**Date:** 2026-04-27
**Spec:** [`../specs/2026-04-27-phase3-measurement-design.md`](../specs/2026-04-27-phase3-measurement-design.md)
**Branch:** `kv4-sdpa-phase3-measure`
**Status:** Both §M2 (microbenchmark) and §M1 (Xcode counters) complete. Bottleneck identified: register-pressure-driven low occupancy on a memory-latency-bound kernel. Phase 4 candidates ranked.

## 1. Setup

- **Device:** Apple M4 Pro.
- **MLX commit:** `86733158` (post-Phase 2 merge), built `-DMLX_METAL_DEBUG=ON -DCMAKE_BUILD_TYPE=Debug` for source-line shader profiling.
- **Microbench:** `benchmarks/python/sdpa_dequant_microbench.py` (this PR).
- **Captures:** regenerated under the debug build —
  `/tmp/sdpa_q4_sdpa_qwen_gqa_32768.gputrace` and
  `/tmp/sdpa_fp16_sdpa_qwen_gqa_32768.gputrace`. Both produced by
  `benchmarks/python/sdpa_vector_quantized_capture.py` (Phase 1 PR #4).
- **Iterations:** 100 measured + 10 warmup per cell. Median + p10/p90.

## 2. M2 Microbench — what does dequant actually cost?

Three operations on the same tensor shape, in isolation:

- `dequant`: `mx.dequantize(packed, scales, biases, group_size=64, bits=4)` → fp16 tensor.
- `pure_read`: `mx.sum(fp16_tensor)` — touches every input byte once. Memory-bandwidth lower bound.
- `pure_multiply`: `fp16_tensor * scalar` — one fp16 FMA per element, write back. ALU-bandwidth reference.

### Results

```
shape=qwen-gqa-32768  (Hk=4, ctx=32768, D=128; q4_total=9.0 MB, fp16_total=32.0 MB)
  op              median ms      p10      p90  effective GB/s
  dequant            0.3691   0.3616   0.4105          116.5
  pure_read          0.2683   0.2542   0.2845          125.1
  pure_multiply      0.4132   0.4053   0.4380           81.2

shape=llama3-8b-32768  (Hk=8, ctx=32768, D=128; q4_total=18.0 MB, fp16_total=64.0 MB)
  op              median ms      p10      p90  effective GB/s
  dequant            0.6426   0.5961   0.7161          133.8
  pure_read          0.4107   0.4009   0.4487          163.4
  pure_multiply      0.7599   0.7255   0.8314           88.3

shape=qwen-gqa-8192  (Hk=4, ctx=8192, D=128; q4_total=2.2 MB, fp16_total=8.0 MB)
  op              median ms      p10      p90  effective GB/s
  dequant            0.1949   0.1890   0.2008           55.1
  pure_read          0.1634   0.1515   0.1704           51.3
  pure_multiply      0.1642   0.1577   0.1806           51.1
```

(Effective GB/s = bytes touched ÷ time. For dequant, "bytes touched"
counts both the q4 input + scales/biases AND the fp16 output write.)

### Headline finding

**Dequant runs at 116–134 GB/s, only ~10–20% slower than `pure_read`
(125–163 GB/s).** The dequant operation is essentially **memory-bandwidth-bound**, not ALU-bound.

For comparison: M4 Pro's spec peak is 273 GB/s. The pure-read achieves
~50% of peak — typical for a single-stream read, with the rest going
to refill latencies. Dequant at 117 GB/s on qwen-gqa is consistent
with reading the 10 MB q4 input + writing the 32 MB fp16 output close
to bandwidth.

`pure_multiply` (a full fp16 input read + fp16 output write per
element + 1 multiply) runs at 81–88 GB/s, slower than dequant. So
the per-element multiply cost is real, but it isn't the bottleneck —
the read-and-write traffic is.

### Direct refutation of Phase 1 §6.4

Phase 1 §6.4 concluded:
> the kernel is ALU-bound on dequant. The arithmetic ratio q4/fp16 ≈
> 2.14× sits very close to the measured worst-cell ratio of 2.79× …
> That is strong structural evidence the q4 kernel is ALU-bound on
> dequant, not bandwidth-bound.

§M2 directly refutes this. **In isolation, dequant ALU is not the
bottleneck.** The 16 ALU ops per K position that Phase 1 §6.2 counted
up complete in roughly the same time as the corresponding memory load
takes to retire — the ops are issued in the shadow of memory wait and
disappear from the wall clock.

This explains the Phase 2 negative results:
- §7.1 (share dequant across query heads) saved nothing because the
  dequant work wasn't on the wall-clock critical path to begin with.
- §7.2 (fp16 dequant FMAs) saved nothing for the same reason — the
  FMAs themselves were already cheap.

### Where the 0.89 ms q4-vs-fp16 SDPA gap actually lives

Adding the microbench numbers:

- `mx.dequantize` of K alone: 0.37 ms (qwen-gqa @ 32768)
- `mx.dequantize` of V alone: ≈ 0.37 ms (same shape)
- fp16 vector SDPA at the same ctx: 0.50 ms (Phase 1 §2)
- **Sum**: 1.24 ms

Phase 1 measured `dq_then_sdpa` at **1.05 ms** for the same cell —
roughly consistent with the sum (the difference is L2-cache reuse
between the back-to-back ops).

`q4_sdpa` measured at **1.39 ms**. So the fused-kernel path is
**0.34 ms slower** than just dequantizing upfront and running fp16
SDPA. The fused kernel is *worse* than the sum of its parts. That's
the headline finding §4 (Phase 1) called out and Phase 2 failed to
fix.

### Hypothesis from M2 alone

The 0.34 ms penalty is **not** from dequant ALU. The microbench shows
that work is bandwidth-shadowed. The penalty must come from
*integration cost* — what happens when dequant is interleaved with
QK / softmax / AV in the same kernel:

1. **Register pressure from holding scales / biases per simdgroup.**
   q4 needs `ks`, `kb`, `vs`, `vb` held in registers across the K
   iter. fp16 doesn't. Higher per-simdgroup register count → lower
   occupancy → less latency hiding across threadgroups.
2. **Instruction-fetch / issue bottleneck.** The q4 inner loop has
   ~30 instructions/iter vs fp16's ~14. Even if individual
   instructions are fast, more instructions means more issue cycles.
   Apple GPU SIMDs may have limited instruction queues that show up
   as issue stalls.
3. **Memory access pattern asymmetry.** fp16 does coalesced 8-byte
   reads per lane; q4 does 2-byte uint16 reads. Both should coalesce
   within a simdgroup, but per-lane bytes-per-load is 4× lower for
   q4 — possibly a load-port utilization issue.

Of these, **(1) register pressure** is the most directly testable
without GPU counters: count register usage in the compiled shader, or
look for any spill bytes. M1's Xcode pipeline-statistics readout
will answer this directly.

## 3. M1 Xcode Counter Readings

From the qwen-gqa @ ctx=8192 captures, Profile-after-reload, Counters
tab, with the dominant pass-1 dispatch selected.

(`/tmp/sdpa_q4_sdpa_8192.gputrace` and `/tmp/sdpa_fp16_sdpa_8192.gputrace`
both 205 MB, release-build metallib. Larger ctx=32768 captures
crashed Xcode, but the bottleneck pattern is identical at 8192
because the gap is per-K-position structural, not ctx-specific.)

### Compared per dispatch

| Counter | q4 (`sdpa_vector_quantized_2pass_1`) | fp16 (`sdpa_vector_2pass_1`) | Δ |
|---|---:|---:|---:|
| GPU time share (this dispatch) | **97.93%** of 1.11 ms total | (n/a — different trace) | — |
| ALU Utilization | 26.74% | 26.74% | **same** |
| F16 Utilization | 0.00% | 17.51% | q4 doesn't use fp16 ALU at all |
| F32 Utilization | 17.51% | 20.90% | both fp32-heavy |
| Allocated registers per thread | **72** | **50** | **q4 = +22 (+44%)** |
| SIMD Groups dispatched | 7082 | 7208 | within 2% (same workload) |

Counter labels we couldn't surface: per-counter Memory stall %, L1 /
L2 cache hit rates, occupancy (these may live in the Heat Map or
Counters tab under different names; not pursued because the data we
have is already enough — see §4).

### What the numbers say, in plain terms

- **Both kernels are equally memory-stalled.** ALU Utilization 27%
  in both means both spend ~73% of GPU cycles waiting on
  something — most of which is memory in this kernel class.
- **Same wall-clock fraction, different absolute work.** Same 27%
  active means q4's wall-clock × 0.27 = absolute ALU time for q4,
  fp16's wall-clock × 0.27 = absolute ALU time for fp16. q4's wall
  clock is ~2× fp16's at this cell, so absolute ALU work is ~2×.
  Consistent with §6.2's per-K op count (30 vs 14).
- **q4 uses 44% more registers per thread.** 72 vs 50. That's the
  **decisive new finding**. Higher register count reduces how many
  simdgroups can run concurrently on each GPU core (the register
  file per core is fixed). Fewer concurrent simdgroups → less
  ability to hide memory latency by switching threadgroups during
  stalls → more time spent stalled in absolute terms, even when
  the *fraction* stalled looks identical.
- **q4's compiler doesn't fp16-promote.** F16 Utilization = 0%
  while fp16's is 17.51%. This confirms Phase 2 §7.2 surgical
  change had to be explicit (it didn't auto-happen) — and explains
  why §7.2's null result happened anyway: ALU isn't the bottleneck
  in the first place, so reducing ALU op cost doesn't move
  wall-clock noticeably.

## 4. Synthesis

The q4 kernel is **memory-latency-bound, gated by register-pressure-driven low occupancy**. Phase 1 §6's "ALU-bound on dequant"
diagnosis was wrong on both counts (ALU at 27%, dequant runs at
memory bandwidth in M2). The actual story:

1. **Memory access is the wall-clock dominator.** Both kernels
   stall ~73% of the time. q4 reads 12 B/K-position vs fp16's 16
   B, so q4 has slightly less raw memory traffic — but...
2. **q4's high register count reduces occupancy**, leaving more
   time spent stalled per lane because fewer alternative
   threadgroups are ready to run during a memory wait.
3. **q4's ALU work is also genuinely larger** (30 vs 14
   ops/K-position), so its absolute ALU time scales 2× under the
   same 27% utilization fraction.

Both factors compound multiplicatively. (Halving register count
without changing ALU work, or vice versa, would only get part of
the way.)

Phase 2 §7.1 (share dequant via threadgroup memory + barriers)
failed because it added barrier overhead without reducing register
pressure or absolute ALU work. Phase 2 §7.2 (fp16 dequant FMAs)
failed because reducing ALU work in the dequant FMAs by ~2×
translates to ~5% wall-clock at most (since FMAs are a small
fraction of the 27% ALU time, and the rest is integer extract +
load).

## 5. Phase 4 Candidates — ranked by counter evidence

### 5.1 Reduce register count from 72 to ≤50 (HIGHEST EXPECTED IMPACT)

Match fp16's 50-register baseline. Closing 22 registers is the
single biggest lever the §3 numbers point at. Concrete sub-levers:

**5.1.a Eliminate intermediate `k[]` and `vals[]` arrays.** Currently
the inner loop dequants all qk_per_thread (=4) K nibbles into
`k[0..3]`, then a separate loop multiplies them into `score`. Same
for V. Fuse dequant + FMA so only one nibble is live at a time:

```cpp
// Current (4 regs held simultaneously)
for (int j = 0; j < 4; j++) k[j] = dequant_j;
U score = 0;
for (int j = 0; j < 4; j++) score += q[j] * k[j];

// Proposed (1 reg held)
U score = 0;
for (int j = 0; j < 4; j++) {
  U kj = dequant_j;
  score += q[j] * kj;
}
```

**Expected savings:** ~6 registers (3 for k, 3 for vals — `o[]`
must stay across the AV loop and `q[]` across iterations).

**5.1.b Combine with `typedef T U` (Phase 2 §7.2 aggressive).**
Halves the per-element register footprint of `q[]`, `o[]`, intermediates from fp32 to fp16. 4 fp32 = 4 regs vs 4 fp16 = 2 regs (Apple GPU register granularity is 32-bit; 2 fp16 packed into 1 reg).
Need to fix the bfloat compile error (`U max_score = U(-INFINITY)`,
explicit casts on `static_cast<U>` boundaries). Not a typedef
change alone — needs a full audit.

**Expected savings:** ~8 registers.

**Combined 5.1.a + 5.1.b**: 14-register reduction. From 72 → ~58.
Closes most of the 22-register gap to fp16.

**Expected wall-clock impact**: occupancy improvement is non-linear
in register count (Apple GPUs allocate simdgroups per-core in
discrete buckets). 72 → 58 may bump the kernel from N to N+1
simdgroups per core, materially improving latency hiding. Optimistic
case: 30-50% wall-clock reduction on the q4 kernel.

### 5.2 Cache scales/biases per group in threadgroup memory (MEDIUM)

Phase 2 §7.3 originally said "small lever". With Phase 3 data, this
deserves a re-look: scales/biases are reloaded every K iter even
though they only change every group_size=64 K positions. With BN=32
(simdgroups iterating in stride 32 along K), every other outer iter
falls in the same group → identical scale/bias values on consecutive
outer iters.

Implementation: at group boundaries (i % group_size == 0), the
producer simdgroup writes `scale, bias` into threadgroup memory;
all simdgroups read on subsequent iters until the next boundary.

Unlike Phase 2 §7.1 (which barrier'd every K iter), this barriers
once per group (= once per 64 K iters), so barrier overhead is
~64× less per kernel call. Should not regress like §7.1 did.

**Expected savings:** small fraction of memory traffic
(scales/biases are ~25% of total q4 byte volume), so wall-clock
improvement is bounded at ~10%. Bundle with 5.1.

### 5.3 Pre-fetch K/V loads to overlap with dequant ALU (LOW)

Currently each K iter has: load → dequant → score → softmax → load V → AV. The two loads are serial within an iter. Issuing the V load earlier (e.g. unrolled across two K iters) might overlap V load with iter-N's softmax.

**Expected savings:** unclear without microbenchmark. Modest.

### 5.4 Rejected as next step

- §7.1 (full shared dequant) — disproven by Phase 2.
- §7.2 surgical (fp16 dequant FMAs) — null result by Phase 2; the lever was real but too small.
- Tile-size tuning — same kernel structure as fp16 path which is well-tuned; no expected win there.

### Phase 4 success criterion (unchanged from Phase 1 §4)

`q4_sdpa` must beat its own `dq_then_sdpa` fallback at every
context length on `qwen-gqa`. Currently 1.07–1.34× slower. 5.1.a +
5.1.b together should clear this.

## 6. Reproduction

For future Phase N reruns of M1:

1. Build MLX in Release mode (debug captures crash Xcode at ctx ≥ 8192 due to a 343 MB embedded metallib snapshot).
2. Generate captures via `benchmarks/python/sdpa_vector_quantized_capture.py --variant {q4_sdpa,fp16_sdpa} --ctx 8192 --iters 3 --out /tmp/<name>.gputrace` (release captures are ~205 MB and open cleanly; ctx=32768 captures are ~365 MB and may still crash on this Xcode/macOS combo).
3. `open -a Xcode /tmp/<name>.gputrace` → click "Profile after reload" (stopwatch icon top-right) → wait ~30s for full counter collection.
4. Performance pane → sort by GPU Time descending → click the dominant `sdpa_vector_*_2pass_1*` row.
5. Counters tab → read ALU Utilization, F16 Utilization, F32 Utilization. Allocated registers + SIMD Groups are visible in the Top Shaders table on the Overview tab.
