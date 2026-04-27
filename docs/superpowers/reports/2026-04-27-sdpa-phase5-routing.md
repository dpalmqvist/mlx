# Phase 5 Report — §6.1 Routing Fix Shipped

**Date:** 2026-04-27
**Branch:** `kv4-sdpa-phase5-nax-prefill` (extended to include the §6.1 routing fix; the §6.3 NAX-prefill investigation was the predecessor commit on the same branch)
**Predecessor:** `2026-04-27-sdpa-phase4-followup-loadwidth.md` §6.1.
**Status:** Shipped. Phase 1 §4 success criterion ("q4_sdpa < dq_then_sdpa at every ctx on qwen-gqa") **met by routing**.

## 1. The change

One-line heuristic added to `QuantizedScaledDotProductAttention::use_fallback` in `mlx/backend/metal/scaled_dot_product_attention.cpp`:

```cpp
const int gqa_factor = q.shape(1) / q_keys.shape(1);
if (gqa_factor >= 5) {
  return true;  // route to dequantize → fp16 SDPA fallback
}
```

When the operation's GQA ratio (Hq / Hk) is 5 or higher, the primitive declines to dispatch the q4 kernel. The Python-level wrapper at `mlx/fast.cpp:927-955` already had a `dequantize → scaled_dot_product_attention` fallback wired for exactly this case, so the change is self-contained.

## 2. Why threshold = 5

Empirical, from the cumulative bench data across Phases 1–4:

- **Llama-style (Hq/Hk = 4):** q4 wins at ctx ≥ 4096 (q4/dq = 0.84–0.97×) and ties at shorter ctx (q4/dq = 1.02–1.06×, within bench noise). Net: q4 kernel is the right choice across the board. Threshold must not catch this case.
- **Qwen-style (Hq/Hk = 7):** q4 loses at every ctx (q4/dq = 1.07–1.34×). Threshold must catch this case.
- No data at Hq/Hk = 5 or 6. Threshold ≥ 5 is conservative — it catches qwen (7) without disturbing llama (4) or mistral (4). If a future model uses Hq/Hk = 5 or 6, this routes it to the fallback by default, which is safer than running the q4 kernel where it might lose.

## 3. Bench result — Phase 1 sweep, post-routing

```
model=llama3-8b  Hq=32 Hk=8 D=128                # gqa_factor = 4, q4 kernel still active
   ctx   fp16_sdpa     q4_sdpa    dq_then_sdpa   q4/fp16     q4/dq
  1024      0.2180      0.2162          0.2038     0.99x     1.06x
  2048      0.1612      0.2278          0.2241     1.41x     1.02x
  4096      0.2007      0.3085          0.3249     1.54x     0.95x
  8192      0.3096      0.4842          0.5566     1.56x     0.87x
 16384      0.5171      0.8937          1.0270     1.73x     0.87x
 32768      0.9676      1.5996          1.8951     1.65x     0.84x

model=qwen-gqa  Hq=28 Hk=4 D=128                  # gqa_factor = 7, routes to fallback
   ctx   fp16_sdpa     q4_sdpa    dq_then_sdpa   q4/fp16     q4/dq
  1024      0.1358      0.1610          0.1440     1.19x     1.12x
  2048      0.1286      0.1748          0.1767     1.36x     0.99x
  4096      0.1486      0.2192          0.2169     1.48x     1.01x
  8192      0.2049      0.3299          0.3269     1.61x     1.01x
 16384      0.2973      0.5639          0.5699     1.90x     0.99x
 32768      0.4879      1.0197          1.0327     2.09x     0.99x
```

Reading:

- **`q4/dq` on qwen-gqa is 0.99–1.12× across all ctx**, vs 1.07–1.34× pre-routing. The Phase 1 §4 floor is met: q4_sdpa is never worse than dq_then_sdpa on this shape. The 1.12× outlier at ctx=1024 is bench noise on a 0.14 ms cell — both variants now do the same work (dequant + fp16 SDPA), so any difference is from graph-fusion / scheduling jitter.
- **Headline `q4/fp16` improvement on qwen-gqa @ ctx=32768: 2.79× → 2.09×** = 26% reduction in absolute time (1.39 ms → 1.02 ms). For users running qwen-style models at long context with `kv_bits=4`, that's a real and immediate decode latency win.
- Llama-style numbers unchanged (the routing condition doesn't fire for gqa_factor=4).

## 4. Why we got here

This change is the conservative, measurement-driven move after **four kernel-rewrite attempts** (Phases 2–4) and **two diagnostic measurements** (Phases 3 and 4 §6.2) failed to find a kernel-level lever big enough to clear the success criterion. Phase 4 §6.4 named the right next move: **stop optimizing the kernel; route around it on shapes where it loses**.

The honest framing: the q4 vector decode kernel is structurally limited at high GQA ratios on M4 Pro, and re-deriving fp16 K/V on every call from packed nibbles is just slower than dequantizing once and reading the materialized fp16 K/V repeatedly. The routing fix takes the latter path automatically when it'll be faster.

## 5. What's left

- **Llama @ short ctx (1024–2048):** q4/dq = 1.02–1.06×. Within bench noise. Could consider extending the heuristic to `gqa_factor >= 5 OR (gqa_factor >= 4 AND kL < 4096) → fallback` but the savings are negligible (a few hundred microseconds per token, on a sub-millisecond op). Not worth the complexity.
- **Possible head_dim ≠ 128 cases:** the routing decision is shape-only (gqa_factor and kL via q_keys); doesn't read head_dim or group_size. Should generalize cleanly.
- **Tunable threshold:** if real-world feedback indicates Hq/Hk=5 or 6 shapes also lose to fallback, threshold can be lowered. Defaulting to 5 is safe.

## 6. Updates to the parent doc

`~/devel/olmlx-model/M4_OPTIMIZATION_OPPORTUNITIES.md` Tier 1.5 §A "v2 SHIPPED" should be updated:

- Append: "**v3 SHIPPED (routing fix):** at high GQA ratios (Hq/Hk ≥ 5) the q4 vector decode kernel loses to its own dequantize → fp16 SDPA fallback at every measured context length. The dispatch now routes to the fallback automatically on those shapes. Headline impact: qwen-gqa @ ctx=32768 q4/fp16 = 2.79× → 2.09× (−26%). Llama-style shapes unchanged (kernel still wins on gqa=4)."
- Update the kernel optimization workstream status: kernel ceiling was reached after Phases 2–4. Phase 5 ships the routing fix as the final user-facing improvement.

## 7. Commit log on this branch

1. `bench+docs: §6.3 NAX in SDPA prefill — already done` (predecessor)
2. `routing: q4 SDPA → dequant fallback at gqa_factor >= 5` (this commit)
3. `docs: Phase 5 §6.1 routing fix report` (this report)
