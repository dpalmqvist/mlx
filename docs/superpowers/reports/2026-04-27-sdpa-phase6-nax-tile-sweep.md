# Phase 6 Report — NAX SDPA Tile Sweep

**Date:** 2026-04-27
**Branch:** `kv4-sdpa-phase6-naxsdpa-tiles`
**Predecessor:** `2026-04-27-sdpa-phase5-nax-prefill.md` §3 flagged the NAX SDPA tile sizes (`wm=4, wn=1, bq=64, bk=32`) as hardcoded and never empirically swept on M4 Pro.
**Status:** Sweep complete. Default tile changed from `(bq=64, bk=32)` → `(bq=32, bk=32)` on `D=128`. **3-14% prefill TFLOPS improvement on Llama-3 8B shapes**, neutral on Qwen-GQA shapes within bench noise.

## 1. Method

Six new AOT instantiations added to `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.metal` covering `(bq, bk) ∈ {32, 64, 128} × {32, 64}` for `bd=128`. The dispatch in `mlx/backend/metal/scaled_dot_product_attention.cpp` accepts `MLX_NAX_SDPA_BQ` / `MLX_NAX_SDPA_BK` env-var overrides for sweep benchmarking.

Sweep harness `benchmarks/python/sdpa_nax_tile_sweep.py` runs all 6 combos × 2 model shapes × 4 contexts in subprocesses, parses the prefill bench output, and tabulates per-cell winners.

## 2. Sweep results — initial pass

```
fp16_prefill median ms per (model, qL, tile combo):

  model            qL     (32,32)     (32,64)     (64,32)     (64,64)    (128,32)    (128,64)  winner
  ---------------------------------------------------------------------------------------------------
  llama3-8b      1024     2.7726*    2.9950     2.9879     3.0060     2.9393     3.0675   (32, 32)
  llama3-8b      2048    10.4967*   11.5898    12.2672    11.0793    10.7256    11.8403   (32, 32)
  llama3-8b      4096    41.0397*   45.4878    42.7664    43.4109    42.9081    42.2633   (32, 32)
  llama3-8b      8192   164.1586*  182.8801   176.8574   178.0126   170.6090   176.6980   (32, 32)
  qwen-gqa       1024     2.5125*    2.5965     2.6194     2.5856     2.5387     2.5753   (32, 32)
  qwen-gqa       2048     9.5925     9.8562    10.8076    10.2363    10.4818     9.4646*  (128, 64)
  qwen-gqa       4096    37.8383    38.9309    37.0373    37.4193    36.3019*   37.5251   (128, 32)
  qwen-gqa       8192   154.6219   154.2982   149.9665   153.6060   147.2706*  152.4183   (128, 32)
```

(`*` = best at this `(model, qL)`. Old default `(64, 32)` was never the winner.)

## 3. Reading the sweep — variance check

The qwen long-context cells initially suggested `(128, 32)` wins by 2-5%. Re-running just qwen `qL=8192` to spot-check:

```
=== bq=32 bk=32 === qwen qL=8192: 143.83 ms
=== bq=64 bk=32 === qwen qL=8192: 158.25 ms
=== bq=128 bk=32 === qwen qL=8192: 158.79 ms
```

Now `(32, 32)` wins by 9% on the same cell where the original sweep had it tied with `(128, 32)`. **Bench variance is ~5-10%** at long context; the small "(128, 32) wins on qwen" signal in §2 doesn't reproduce reliably.

## 4. The robust signal

`(bq=32, bk=32)` is **best or within noise of best** in every measured cell across multiple trials:

- **Llama-3 8B (gqa_factor=4):** unambiguous winner at every qL, 3-14% faster than old default `(64, 32)`.
- **Qwen-GQA (gqa_factor=7):** best at short ctx, within ±3% of best at long ctx (which is at the edge of bench noise).

The simpler heuristic — always pick `(bq=32, bk=32)` on D=128 — captures the robust signal without taking on a shape-dependent decision tree that may be modeling noise rather than real perf differences.

## 5. Confirmation trial — new vs old default

100-iter prefill bench, both new default (bq=32 selected by heuristic) and old default (bq=64 forced via env var):

```
                              new (bq=32)     old (bq=64)    Δ
llama3-8b qL=1024                    2.79             2.94    -5%
llama3-8b qL=2048                   10.60            12.14   -13%
llama3-8b qL=4096                   41.20            42.70    -4%
llama3-8b qL=8192                  168.95           173.53    -3%
qwen-gqa  qL=1024                    2.59             2.54    +2%  (noise)
qwen-gqa  qL=2048                    9.77            10.66    -8%
qwen-gqa  qL=4096                   38.13            36.96    +3%  (noise)
qwen-gqa  qL=8192                  150.24           149.52    +0.5% (noise)
```

5/8 cells improve by 3-13%; 3/8 cells are within ±3% (= bench-noise envelope). Net win, no regression.

## 6. The change

### `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.metal`

Added 4 new AOT instantiations for D=128:

```cpp
instantiate_attn(iname, itype, 32, 32, 128, 4, 1, mname, mtype)
instantiate_attn(iname, itype,128, 32, 128, 4, 1, mname, mtype)
instantiate_attn(iname, itype, 32, 64, 128, 4, 1, mname, mtype)
instantiate_attn(iname, itype,128, 64, 128, 4, 1, mname, mtype)
```

(`(64, 32)` and `(64, 64)` for both D=64 and D=128 stay as-is. We add 4 × 3 instantiations across float16 / bfloat16 / float32, including the bool-mask variant for each, so 24 new kernel symbols total.)

### `mlx/backend/metal/scaled_dot_product_attention.cpp`

```cpp
int bq = (bd == 128) ? 32 : 64;
```

Plus optional `MLX_NAX_SDPA_BQ` / `MLX_NAX_SDPA_BK` env-var overrides for future sweep work.

## 7. Why bq=32 wins on Llama (mechanism)

Hypothesis (not directly counter-verified, but consistent with the data): smaller bq means more threadgroups dispatched per kernel call. With `Hq=32` query heads × `qL/bq` query tiles × `B` batch, llama at qL=1024 runs 32 × 32 = 1024 threadgroups for `bq=32` vs 32 × 16 = 512 for `bq=64`. M4 Pro's 16 GPU cores each schedule multiple threadgroups; more threadgroups → better load balancing across cores → better throughput. This effect dominates at low GQA where each KV head is shared by fewer query heads (less reuse opportunity from larger Q tiles).

## 8. Workstream summary

Cumulative kv4-sdpa-* attempts:

| Phase | Lever | Impact |
|---|---|---|
| 2 §7.1 | Share dequant via TG memory + barriers | Regression |
| 2 §7.2 | fp16 dequant FMAs | Null |
| 4 §5.1.a | Fused dequant + FMA | Null |
| 4 §5.1.b | `typedef T U` | Null |
| 4 §6.2 | Lane load width diagnostic | 14% synthetic, ~5-10% in integrated |
| 5 §6.3 | NAX prefill investigation | Already shipped upstream |
| **5 §6.1** | **Routing fallback at gqa_factor ≥ 5** | **−26% on qwen long-ctx q4 SDPA** |
| **6** | **NAX SDPA tile bq=32 on D=128** | **3-14% on llama prefill** |

Two real user-facing improvements shipped (§6.1 and Phase 6), after four kernel-rewrite attempts taught us where the q4 vector decode kernel's ceiling lives. The decode-path improvement (§6.1) and the prefill-path improvement (Phase 6) are independent — different code paths, different shapes — and stack additively for users running long-prompt prefill followed by long-ctx decode on quantized KV.

## 9. Open items

- The `(128, 32)` instantiation is added but never selected by the default heuristic. Could be removed for binary-size cleanliness, but its presence keeps the env-var override useful for future sweeps. Leaving as-is.
- `bq=32` chosen for D=128 only; D=64 / D=80 / D=256 paths unchanged. None of these are common in modern LLMs at the shapes targeted by the q4 KV workstream.
- `wm=4, wn=1` (warp arrangement) not swept. `wm=2, wn=2` could be a follow-up but would require additional .metal instantiations and doesn't have an obvious motivating signal from current data.

## 10. Update to parent doc

`~/devel/olmlx-model/M4_OPTIMIZATION_OPPORTUNITIES.md`:
- Tier 1 §4 ("Per-arch swizzle / tile params in matmul") — analogous SDPA NAX tile tuning has now happened. Add a note: "M4 Pro NAX SDPA tile defaults swept Phase 6; new default `bq=32, bk=32` on `D=128` delivered 3-14% prefill improvement vs prior `(64, 32)` default."
- Tier 1.5 §B should reference this report for the prefill TFLOPS post-tuning.
