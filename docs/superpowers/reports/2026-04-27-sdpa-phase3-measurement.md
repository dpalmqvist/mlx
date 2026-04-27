# Phase 3 Report — Measuring the Quantized Vector SDPA Bottleneck

**Date:** 2026-04-27
**Spec:** [`../specs/2026-04-27-phase3-measurement-design.md`](../specs/2026-04-27-phase3-measurement-design.md)
**Branch:** `kv4-sdpa-phase3-measure`
**Status:** §M2 (microbenchmark) complete and analyzed. §M1 (Xcode counters) pending — needs interactive Xcode work that the CLI cannot do on this hardware/OS combo (see Phase 1 §6.1).

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

## 3. M1 Xcode Counter Readings — TO BE FILLED IN

Pending interactive work in Xcode against the captures from §1.
Per-counter slots:

### sdpa_vector_quantized_2pass_1_float16_t_128_128_64_4 (q4 path)

| Counter | Value | Source |
|---|---|---|
| Limiter | _ | Performance pane → Limiter column |
| ALU active % | _ | Counters → Compute group |
| F16 active % | _ | Counters → Compute group |
| F32 active % | _ | Counters → Compute group |
| Memory stall % | _ | Counters → Memory group |
| L1 Buffer Cache Hit % | _ | Counters → Cache group |
| L2 Cache Hit % | _ | Counters → Cache group |
| Occupancy | _ | Pipeline overview |
| Register allocation | _ regs | Pipeline Statistics |
| Spill bytes | _ B | Pipeline Statistics |
| Top hot source line | _ | Shader Profiler (Profile button) |
| 2nd hot line | _ | Shader Profiler |
| 3rd hot line | _ | Shader Profiler |

### sdpa_vector_2pass_1_float16_128_128 (fp16 path, comparison)

| Counter | Value |
|---|---|
| Limiter | _ |
| ALU active % | _ |
| F16 active % | _ |
| Memory stall % | _ |
| L2 Cache Hit % | _ |
| Occupancy | _ |
| Register allocation | _ regs |
| Spill bytes | _ B |
| Top hot source line | _ |

The deltas between the two are the most actionable single piece of
data Phase 3 can produce.

## 4. Synthesis — TO BE WRITTEN AFTER §3

Pending §3. Likely structure:

- The §M2 result + §M1 register/occupancy delta tells us whether
  hypothesis (1) above is correct.
- If yes: Phase 4 is "reduce register pressure on q4 inner loop"
  (e.g., spill scales/biases to threadgroup memory but lazily, OR
  recompute group_idx per-iter to avoid per-lane register hold, OR
  fuse the K and V dequant into a single op so one set of state is
  reused).
- If no (registers are equal): the bottleneck is (2) instruction
  count, in which case the only lever is reducing instructions
  per iter (e.g., precompute scale*bias-table per group on launch).
- If memory stall is high: the bottleneck is (3) load pattern, in
  which case the lever is widening the per-lane packed load (e.g.,
  reading uint32 of nibbles even when only 2 are needed, then
  discarding).

## 5. Phase 4 Candidates — TO BE RANKED AFTER §3

Will reference §4. Empty until §3 is filled.

## 6. Workflow note

The Xcode UI work in §3 is gated on user time, not technical capability. Counters that the CLI cannot reach on M4 Pro / macOS 26.3 (Phase 1 §6.1) require Xcode-Debug-Capture-GPU-Frame manual readings.

To minimize the user's session: open `/tmp/sdpa_q4_sdpa_qwen_gqa_32768.gputrace`, click **Profile** (stopwatch icon top-right) to re-run with full counters, wait ~30s, then in the resulting view:

1. Performance pane (left nav, the "16 ms" entry) → sortable list of dispatches → click `sdpa_vector_quantized_2pass_1_*` → right inspector shows Limiter + counters.
2. Click **Shaders** view → select the kernel → see **per-source-line ALU cost**. Top 3 hot lines are the most actionable single readout.
3. Repeat steps 1–2 on `/tmp/sdpa_fp16_sdpa_qwen_gqa_32768.gputrace`.
