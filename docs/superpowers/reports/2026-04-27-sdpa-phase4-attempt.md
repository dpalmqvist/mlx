# Phase 4 Report — Register Pressure Reduction: Null Result

**Date:** 2026-04-27
**Branch:** `kv4-sdpa-phase4-regpressure`
**Depends on:** `docs/superpowers/reports/2026-04-27-sdpa-phase3-measurement.md` Phase 4 candidates §5.1
**Status:** Negative result. Both §5.1.a (fused dequant + FMA, eliminating `k[]`/`vals[]` arrays) and §5.1.b (`typedef T U` for the surviving `q[]`/`o[]` arrays) tried, neither moved the qwen-gqa @ ctx=32768 q4/fp16 ratio measurably. Kernel reverted to baseline. **Four cumulative kernel-rewrite attempts** (Phase 2 §7.1, §7.2; Phase 4 §5.1.a, §5.1.b) — all targeting per-simdgroup work or register pressure — none have moved the bench numbers. The kernel appears to be at its structural ceiling in this code shape.

## 1. What was tried

### §5.1.a — Fused dequant + FMA

Eliminate the `thread U k[qk_per_thread]` and `thread U vals[v_per_thread]` arrays by fusing each nibble's dequant arithmetic directly into the QK score and AV accumulator FMAs:

```cpp
// Before:
for (j) k[j] = U(nibble_j) * ks + kb;
U score = 0;
for (j) score += q[j] * k[j];

// After:
U score = 0;
for (j) score += q[j] * (U(nibble_j) * ks + kb);  // transient nibble
```

Same fusion for V/AV. Applied to both single-pass `sdpa_vector_quantized` and 2-pass-1 `sdpa_vector_quantized_2pass_1` kernels. Phase 3 §5.1.a expected ~6 register reduction on the SSA assumption that `k[]`/`vals[]` were holding floats live across the dequant phase.

### §5.1.b — `typedef T U`

Layered on top of §5.1.a. Changed `typedef float U` → `typedef T U` so the surviving `q[]` and `o[]` thread arrays store in input dtype (fp16/bf16). Half-precision storage halves the per-element register footprint vs U=fp32.

Required compatibility fixes for bfloat16 instantiations:
- `U max_score = -INFINITY` → `U max_score = Limits<U>::finite_min` (matches the fp16 vector kernel's pattern at `sdpa_vector.h:91`)
- `(fmask[0] >= -INFINITY)` → `(fmask[0] >= Limits<T>::finite_min)`
- `sum_exp_score = 1` → `sum_exp_score = U(1)`

Both kernels (single-pass and 2pass_1) were updated.

## 2. Result — null on every cell

Phase 4 §5.1.a alone (qwen-gqa @ ctx=32768): 2.79× q4/fp16 (Phase 1 baseline: 2.79×) → unchanged.

§5.1.a + §5.1.b combined:

```
model=llama3-8b  Hq=32 Hk=8 D=128
   ctx   fp16_sdpa     q4_sdpa    dq_then_sdpa   q4/fp16     q4/dq
  1024      0.2211      0.2654          0.2100     1.20x     1.26x
  2048      0.1628      0.2277          0.2230     1.40x     1.02x
  4096      0.1994      0.3055          0.3200     1.53x     0.95x
  8192      0.3063      0.4799          0.5490     1.57x     0.87x
 16384      0.5088      0.8900          1.0174     1.75x     0.87x
 32768      0.9611      1.5916          1.8857     1.66x     0.84x

model=qwen-gqa  Hq=28 Hk=4 D=128
   ctx   fp16_sdpa     q4_sdpa    dq_then_sdpa   q4/fp16     q4/dq
  1024      0.1368      0.1689          0.1530     1.23x     1.10x
  2048      0.1307      0.1886          0.1794     1.44x     1.05x
  4096      0.1482      0.2652          0.2192     1.79x     1.21x
  8192      0.2013      0.4132          0.3245     2.05x     1.27x
 16384      0.2924      0.8010          0.6306     2.74x     1.27x
 32768      0.4975      1.3808          1.0299     2.78x     1.34x
```

Every cell within bench noise of Phase 1 baseline. q4_sdpa correctness clean (`test_quantized_sdpa_vector_matches_dequantized` passes, `test_quantized_sdpa_vector_quality_vs_fp` passes).

## 3. Why neither helped

Three plausible reasons; without re-capturing the actual post-§5.1 register count in Xcode we can't distinguish them definitively:

1. **The compiler was already SSA-optimizing.** When `k[j]` is read once into the QK FMA and then becomes dead, the compiler reuses the register without source-level help. The 72-vs-50 register count gap measured in Phase 3 §M1 reflects something more structural than dead-store elimination — possibly the integer extract intermediates, the int-to-float casts, the loaded packed bytes, or compiler scheduling decisions for the inner loop.

2. **Register count dropped but not to a new occupancy bucket.** Apple GPUs allocate simdgroups per core in discrete buckets of register file usage. If the q4 kernel was previously at e.g. 72 regs / N simdgroups per core, dropping to 65 regs might still allocate N simdgroups (no improvement). Only crossing the 50-reg boundary (or whatever the next bucket is) bumps occupancy.

3. **The Phase 3 register-pressure hypothesis was wrong.** Phase 3 §M1 measured 72 vs 50 regs and 27% ALU utilization in both kernels. We hypothesized that q4's higher register count → lower occupancy → more absolute time spent stalled. But that hypothesis was a structural inference, not a measurement of *actual* occupancy or *actual* memory-stall cycles in either kernel. If both kernels happen to run at the same occupancy despite the register count difference, the wall-clock 2× gap comes from something else entirely.

The most likely combination is (1) + (2): the compiler is already aggressively eliminating dead stores so source-level fusion doesn't help, and the actual register pressure reduction from §5.1.b is too small to bump occupancy. (3) can't be ruled out without an Xcode capture of the post-Phase-4 build.

## 4. Cumulative four-attempt summary

| Phase | Lever | Hypothesis | Result |
|---|---|---|---|
| 2 §7.1 | Share K/V dequant via threadgroup memory + barriers | Aggregate threadgroup ALU = bottleneck | **Regression** (qwen-gqa: 2.79× → 3.05×): barrier overhead exceeded any savings |
| 2 §7.2 surgical | Cast pattern `T(nibble) * ks + kb` for fp16 dequant FMA | fp16 ALU > fp32 ALU on Apple GPU | Null (within bench noise) |
| 4 §5.1.a | Fused dequant + FMA, eliminate k[]/vals[] | Reduce live register count | Null |
| 4 §5.1.b | `typedef T U` for q[]/o[] | Halve per-element register footprint | Null |

All four attempts share the assumption that the per-simdgroup ALU work or register footprint is the wall-clock bottleneck. The combined null/regression results are strong evidence that **those are not the levers on this kernel**. Phase 1 §6's "ALU-bound on dequant" and Phase 3 §5.1's "register pressure → occupancy" hypotheses both miss the actual cause.

## 5. What is the actual bottleneck?

Best-supported hypothesis after four attempts: the per-iter wall clock is dominated by **memory-load serialization**, not ALU or registers. Each lane in the simdgroup issues a 2-byte packed load per K position; this load can't be coalesced as efficiently as fp16's 8-byte coalesced load because the per-lane bytes are smaller. The latency of issuing 32 small loads vs 32 large loads may differ even when total bytes are the same, and that latency falls on the wall clock no matter how few ALU ops or registers we use.

A way to test this: kernel-level microbench that compares two stripped-down kernels — one doing 32k packed-uint16 loads per simdgroup, one doing 32k uint32-load + uint32-load (= the fp16 pattern) — same byte volume, different lane width. If the uint16 version is meaningfully slower wall-clock, the load pattern is the bottleneck.

## 6. Pivots, ranked

Given four attempts on the kernel haven't moved numbers, the next move shouldn't be a fifth attempt with the same assumption. Options:

### 6.1 Routing-level fix — use dq_then_sdpa for high-GQA shapes (CHEAPEST)

`dq_then_sdpa` already beats `q4_sdpa` on qwen-gqa at every measured ctx (1.07–1.34× faster). Phase 1 §4 marked this as the kernel's correctness-of-purpose failure. **The honest fix is to route around it**: in `mx.fast.quantized_scaled_dot_product_attention`, detect when `gqa_factor` is high (e.g. ≥ 5) and the ctx is long, and call `mx.dequantize` + `mx.fast.scaled_dot_product_attention` internally instead of dispatching the q4 kernel.

Expected impact: q4 SDPA on qwen-gqa improves from current 2.79× q4/fp16 to ~2.06× (the dq_then_sdpa ratio). Llama3-8b (GQA factor 4) keeps the q4 kernel since it's already net-beneficial there. Implementation lives in `mlx/backend/metal/scaled_dot_product_attention.cpp`, not in any kernel.

This is "give up optimizing the kernel; just don't run it on shapes where it loses". Feels like a defeat but is the right call given the data.

### 6.2 Memory-load microbench (DIAGNOSTIC, NOT FIX)

Build the §5 microbench above. If it confirms uint16 loads are slower than uint32 in isolation, the next kernel-level lever would be to widen the q4 kernel's loads (e.g. always load uint32 even when only 2 nibbles are needed; discard the unused half). That's a structural rewrite worth a separate Phase 5.

### 6.3 Different axis entirely

Phase 1 §7's other levers (NAX in SDPA prefill, speculative decoding, 2-bit quant) — none of which touch the q4 vector decode kernel. These are independent workstreams; not the right pivot if the goal is "make q4 vector decode faster", but the right pivot if the goal is "improve overall LLM decode throughput on M4 Pro".

### 6.4 Accept the ceiling, document, move on

The q4 vector decode kernel's ceiling on M4 Pro is the current performance. Phase 1 §A "v2 SHIPPED" already established the kernel is a strict win over the *previous* q4 decomposition (1.7× at medium context). It's not a strict win vs fp16 KV at qwen-style GQA shapes, and four kernel-rewrite attempts haven't changed that. Update Phase 1's status to "kernel ceiling reached; routing fallback is the active workstream" and move on.

## 7. Conclusion

Phase 4 produced no shipped kernel change and no perf improvement. Cost: ~3 hours of build + bench cycles across §5.1.a and §5.1.b. Information value:

- Strongest evidence to date that **per-simdgroup work / register pressure is not the lever** on this kernel, regardless of whether sharing across simdgroups (Phase 2 §7.1), reducing op precision (Phase 2 §7.2), or reducing live register count (Phase 4 §5.1.a/b).
- Implies the actual bottleneck is structural (memory load pattern, instruction scheduling, or kernel dispatch overhead) — none of which the four attempts could touch.
- The defensive next move is **routing-level**: detect when `q4_sdpa` is going to lose to its own fallback and route around it (§6.1). This ships an actual user-facing speedup on qwen-gqa shapes without touching the kernel further.

Recommendation: take the §6.1 routing-level fix as Phase 5 unless the user has a different priority.
