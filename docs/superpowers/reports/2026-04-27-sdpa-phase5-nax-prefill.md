# Phase 5 Report — §6.3 Different Axis: NAX in SDPA Prefill (already done)

**Date:** 2026-04-27
**Branch:** `kv4-sdpa-phase5-nax-prefill`
**Predecessor:** Phase 4 follow-up §6.3 deferred. The Phase 1 parent doc Tier 1.5 §B suggested NAX in SDPA prefill as "likely the single biggest win still on the table for long-prompt prefill on M4 Pro."
**Status:** §B is **already done** in the upstream MLX codebase. Prefill SDPA dispatches NAX whenever available (M4 Pro: yes, after Phase 1 §1's gate fix). Bench confirms ~6.7 TFLOPS at long context = ~89% of M4 Pro's NAX matmul peak. The parent doc's claim "the Metal SDPA kernel does not touch the matrix coprocessor today" is outdated.

## 1. What's actually shipping on M4 Pro

The prefill SDPA dispatch is at `mlx/backend/metal/scaled_dot_product_attention.cpp:166-190`:

```cpp
void sdpa_full_self_attention_metal(...) {
  if (metal::is_nax_available() && q.shape(3) != 80 &&
      (env::enable_tf32() || q.dtype() != float32)) {
    return sdpa_full_self_attention_nax(...);
  }
  ...
}
```

For typical fp16 inference (`q.dtype() == float16`), the second condition is true, so NAX is used whenever available. Phase 1 §1's `is_nax_available()` fix made NAX available on M4 Pro (`gen >= 16`), so prefill SDPA on M4 Pro **does** use the matrix coprocessor today.

The NAX SDPA kernel was added in MLX commit `54f1cc6e` (Add Neural Accelerator Support), well before the parent doc was written. The doc's claim that the SDPA kernel doesn't touch NAX has been stale since that landed.

## 2. Prefill SDPA performance — measurement

`benchmarks/python/sdpa_prefill_bench.py` (this PR). Prefill self-attention at qL = kL ∈ {512, 1024, 2048, 4096, 8192}, two model shapes (Llama-3 8B style, Qwen-style GQA), both fp16-native and dequantize-then-fp16 paths.

```
model=llama3-8b  Hq=32 Hk=8 D=128
      qL  fp16_prefill  q4_dq_prefill  q4_dq/fp16  fp16 TFLOPS
     512        0.8261         0.8556       1.04x        5.20
    1024        2.7949         2.8382       1.02x        6.15
    2048       10.5378        10.6252       1.01x        6.52
    4096       41.0598        41.1388       1.00x        6.69
    8192      164.1571       170.7444       1.04x        6.70

model=qwen-gqa  Hq=28 Hk=4 D=128
      qL  fp16_prefill  q4_dq_prefill  q4_dq/fp16  fp16 TFLOPS
     512        0.7831         0.7908       1.01x        4.80
    1024        2.5591         2.5787       1.01x        5.87
    2048        9.7536         9.7191       1.00x        6.16
    4096       38.6087        38.6371       1.00x        6.23
    8192      157.6660       157.1776       1.00x        6.10
```

Two findings:

1. **fp16 prefill caps at ~6.7 TFLOPS** at long context. M4 Pro's NAX matmul peak (Phase 1 §1) is ~7.7 TFLOPS — **prefill is at ~87% of peak**. Headroom exists but is bounded.
2. **`q4 → dequantize → fp16 SDPA` ≈ fp16-native** (1.00–1.04× ratio across all shapes). The dequantize cost is amortized into the prefill compute and effectively free at this scale. No need for a "quantized SDPA prefill" primitive.

## 3. Remaining headroom

After confirming NAX is operative, the remaining levers on the SDPA prefill path are tile-size tuning and similar micro-optimizations:

- **NAX SDPA tile sizes hardcoded** at `scaled_dot_product_attention.cpp:31-36`: `wm=4, wn=1, bq=64, bk=32`. No empirical sweep on M4 Pro recorded.
- Phase 1 §3 swept the quantized GEMM tile sizes and found defaults were already optimal (best correct alternative within −0.6% of default). The SDPA NAX tile defaults likely face a similar story but haven't been tested.
- Realistic upside if a tile sweep finds a winner: 5–15% on prefill TFLOPS, i.e. 6.7 → ~7.5 TFLOPS at qL=8192. A meaningful but small improvement.

The engineering cost would mirror Phase 1 §3: add an env-var tile override to `sdpa_full_self_attention_nax`, write a sweep harness with a correctness gate, run on M4 Pro. ~1-2 hours of work.

## 4. Reconciling with the parent doc

The Phase 1 parent doc `~/devel/olmlx-model/M4_OPTIMIZATION_OPPORTUNITIES.md` Tier 1.5 §B should be updated to reflect:

- The "NAX in SDPA prefill" lever is **already shipped** in upstream MLX (`steel_attention_nax`).
- M4 Pro's NAX gate fix (Phase 1 §1) made it operative on this device.
- Measured prefill TFLOPS at long context is ~6.7, ~87% of NAX matmul peak.
- Remaining headroom is in tile-size tuning, expected 5–15% upper bound.

The doc framed §B as the biggest remaining lever; in fact it's already captured. That changes the §6.3 conclusion: there isn't a clean "different axis" available *that's also a quick win*. The remaining Phase 1 §7 candidates (§C speculative decoding, §D 2-bit quant, §E fused decoder, §F AMX/ANE) are all multi-day workstreams.

## 5. Recommendation

After the cumulative four kernel-rewrite attempts in Phases 2–4 and this §6.3 finding, the optimization workstream's **highest-leverage remaining move** is back to **§6.1** (route around the q4 vector kernel on shapes where it loses to its own dequant fallback). That ships a real user-facing improvement (~26% on qwen-gqa @ ctx=32768, from q4/fp16 = 2.79× to ≈ 2.06× = the dq_then_sdpa ratio) with a small, contained code change in `scaled_dot_product_attention.cpp`.

Alternatively, if the user wants to pursue tile-size tuning on the NAX SDPA kernel as a §6.3 follow-up, that's a measurement-driven lever consistent with Phase 3's pattern. Realistic upside: 5–15% on prefill TFLOPS. Engineering: env-var override + sweep harness, ~1-2 hours.

§6.3 alternatives §C/§D/§E/§F are out of scope for this workstream as they're all multi-day pivots; recommend treating them as separate top-level workstreams if pursued.
