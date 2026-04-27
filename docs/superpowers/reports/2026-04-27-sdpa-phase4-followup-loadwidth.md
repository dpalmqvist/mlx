# Phase 4 Follow-up — Load-Width Microbench (§6.2) and Ceiling Acceptance (§6.4)

**Date:** 2026-04-27
**Branch:** `kv4-sdpa-phase4-regpressure`
**Predecessors:** Phase 4 attempt report (`2026-04-27-sdpa-phase4-attempt.md`), §6 pivots list.
**Status:** §6.2 (load-width diagnostic) complete. §6.3 (different axis) deferred. **§6.4 (accept the kernel ceiling) is the conclusion.**

## §6.2 — Lane-load-width microbench

### Method

`benchmarks/python/sdpa_load_width_microbench.py` (this PR). Three custom Metal kernels (via `mx.fast.metal_kernel`), each touching the same total byte volume but with different per-lane load widths:

- **uint16**: 2 bytes/lane/iter — matches the q4 kernel's `((const device uint16_t*)q_keys)[simd_lid]` pattern.
- **uint32**: 4 bytes/lane/iter — what the q4 kernel uses for D=256 (qk_per_thread=8) and what we'd widen the D=128 case to.
- **uint2** (`uint32x2`, 8 bytes): matches fp16 vector kernel's `keys[j]` for j in 0..qk_per_thread.

Each lane sums the loads into a uint accumulator and writes to a per-lane output, so the compiler can't dead-store-eliminate. 64 MB total bytes touched per kernel run, 100 measured iters + 10 warmup, M4 Pro.

### Result

```
kernel                     iters/lane  median ms      p10      p90  effective GB/s
uint16 (2 B/lane)                 512     0.4477   0.4249   0.5315          149.9
uint32 (4 B/lane)                 256     0.3912   0.3765   0.4230          171.5
uint2  (8 B/lane)                 128     0.3939   0.3780   0.4472          170.4
```

### Reading

- **uint16 is ~14% slower wall-clock** than uint32 on the same byte volume. Translates to 22 GB/s lost throughput.
- **uint32 = uint2** within bench noise. Beyond uint32, wider lane loads don't help.
- M4 Pro's effective load throughput in this pattern caps at ~170 GB/s, vs 273 GB/s spec peak.

The hypothesis from Phase 4 §5 ("smaller per-lane loads are throttled") is **partially confirmed**: there is a measurable load-width penalty, but it's modest (14%), not order-of-magnitude.

### What would Phase 5 buy if we widened the q4 loads?

Phase 3 §M1 measured the q4 kernel as ~73% memory-stalled. If load throughput improves by 14%, the wall-clock improvement is bounded by the share of time that's load-stall:

`wall_clock_improvement ≤ 0.14 × 0.73 ≈ 10%`

Realistic estimate, accounting for ALU overlap with loads in the integrated kernel: 5-8%. On qwen-gqa @ ctx=32768:

- Current q4_sdpa: 1.39 ms (q4/fp16 = 2.79×)
- After load widening (optimistic): ~1.25 ms (q4/fp16 ≈ 2.51×)
- Phase 1 success criterion: q4 < dq_then_sdpa = 1.05 ms (need ~24% reduction)

**A load-widening kernel rewrite would not clear the success criterion**. It would close ~20% of the gap to dq_then_sdpa but leave the rest. Combined with the structural cost of restructuring the kernel (each simdgroup processing 2 K positions per outer iter to use uint32 lanes on D=128), the engineering / risk ratio is poor.

## §6.4 — Accept the kernel ceiling

After **four kernel-rewrite attempts** (Phase 2 §7.1 share dequant, Phase 2 §7.2 fp16 dequant, Phase 4 §5.1.a fused dequant + FMA, Phase 4 §5.1.b typedef T U) and **one diagnostic** (this report's §6.2 load-width), the picture is:

| Lever | Phase | Result |
|---|---|---|
| Per-simdgroup ALU work | 2 §7.2 surgical, 4 §5.1.a, 4 §5.1.b | Null (3×) |
| Aggregate threadgroup ALU | 2 §7.1 | Regression |
| Lane load width | 4 §6.2 | Minor (~14% on synthetic, ~5-10% expected in integrated) |
| Memory bandwidth | M2 microbench | Already at ~50% of peak; close to ceiling |

The q4 vector decode kernel on M4 Pro is **structurally close to its ceiling within this code shape.** The remaining ~10% available from load-widening doesn't reach the Phase 1 success criterion (q4 < dq_then_sdpa on qwen-gqa).

The honest conclusion: **the kernel ceiling on qwen-style GQA shapes (Hq/Hk = 7) is the current performance.** The kernel is a strict win over the previous q4 decomposition (Phase 1 §A v2 SHIPPED notes "1.7× at medium context"), but it does *not* beat dequantize→fp16-SDPA at high GQA ratios. That's the inherent property of running fp16 SDPA on already-cached fp16 K/V vs running q4 SDPA with on-the-fly dequant per-simdgroup.

### Phase 1 status update (proposed)

The parent doc `~/devel/olmlx-model/M4_OPTIMIZATION_OPPORTUNITIES.md` Tier 1.5 §A "v2 SHIPPED" should note:

- Kernel-optimization workstream (Phases 2-4) attempted four kernel rewrites and one load-width diagnostic. All confirmed the kernel is at its ceiling for qwen-style GQA at long context.
- The user-facing perf gap on qwen-gqa long-ctx is explained by the kernel's inherent dequant-per-simdgroup overhead competing with fp16's L2-amortized loads, not by any source-level inefficiency we could find.
- The actual user-facing fix is the routing-level approach (§6.1 from the Phase 4 report): detect when q4_sdpa is going to lose to its own fallback and call mx.dequantize + mx.fast.scaled_dot_product_attention internally. This was deferred at the user's choice in favor of exploring §6.2-6.4 first; it remains the cheapest available actual improvement.

### Pivots remaining

- **§6.1 routing-level fix** — still on the table if/when the user wants the actual user-facing speedup.
- **§6.3 different axis** — NAX in SDPA prefill, speculative decoding, 2-bit quant. Independent workstreams; not "more of the same kernel" but pivots to genuinely different optimization targets.
- **Stop the q4 SDPA optimization workstream** — the kernel ceiling is documented; treat it as shipped at current performance and move to other parts of the M4 optimization parent document.

## Summary

§6.2 produced a real but small diagnostic finding (~14% load-width penalty on uint16 lane loads). It's not large enough to motivate a fifth kernel rewrite given the cumulative null/regression record. The honest move is §6.4: declare the kernel at its ceiling for qwen-style GQA shapes on M4 Pro and pivot to either §6.1 (routing-level) for a real user-facing improvement or §6.3 (different axis) for a different workstream entirely.
