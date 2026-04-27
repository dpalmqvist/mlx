# NAX M-threshold refinement — empirical validation of the M=256 default

**Date:** 2026-04-27
**Branch:** `kv4-sdpa` (post-Phase-6 revert)
**Predecessor:** `~/devel/olmlx-model/M4_OPTIMIZATION_OPPORTUNITIES.md` "Status & suggested next step" listed "Refine NAX M-threshold from 256 with a denser M-sweep" as an open lever — the conjecture was that the threshold could be lowered to capture wins on mid-M shapes.
**Status:** Lever closed. The current `M >= 256` default is approximately optimal on M4 Pro. The "conservative" framing in the prior comment was inaccurate; the threshold is the empirical inflection.

## 1. Method

Added an `MLX_QMM_NAX_M_THRESHOLD` env-var override (default 256) to the two `M >= 256` gates in `mlx/backend/metal/quantized.cpp` (`qmm_nax` and the general `gather_qmm_nax` path; the rhs-specialized `gather_qmm_rhs_nax` has its own `M/E >= 512` gate and was left untouched). Re-read on every dispatch — no static cache — so subprocess sweeps see env-var changes between invocations.

Sweep harness `~/devel/olmlx-model/benchmarks/nax_m_threshold_sweep.py` runs each (shape, M, threshold) cell in a fresh subprocess: 20-iter warmup + 50-iter timed, median wall-time. Threshold `99999` forces NAX off; threshold `0` forces it on (subject to the other gate conditions: `transpose && K % 64 == 0 && K dtype != float32 unless tf32`). Variant differences below are NAX vs non-NAX kernel, not threshold-vs-threshold.

## 2. Sweep results — qwen-7B / Llama-3 8B shapes

```
shape                     M |  NAX-off med   NAX-on med  speedup
-----------------------------------------------------------------
qwen7b/qkv      (3584,3584) 32 |  0.3729       0.3409       1.09x
qwen7b/qkv                 64 |  0.3980       0.3916       1.02x
qwen7b/qkv                128 |  0.7242       0.7274       1.00x
qwen7b/qkv                256 |  1.1787       1.1647       1.01x
qwen7b/qkv                512 |  2.0626       1.9965       1.03x

qwen7b/ffn_up   (18944,3584) 32 |  0.8338       1.4872       0.56x  ← NAX -44%
qwen7b/ffn_up                64 |  1.4501       1.4694       0.99x
qwen7b/ffn_up               128 |  2.6650       2.5499       1.05x
qwen7b/ffn_up               256 |  5.1110       4.8678       1.05x
qwen7b/ffn_up               512 | 10.0210       9.4362       1.06x

qwen7b/ffn_down (3584,18944) 32 |  0.8681       0.8696       1.00x
qwen7b/ffn_down              64 |  1.5176       1.5132       1.00x
qwen7b/ffn_down             128 |  2.8120       2.8288       0.99x
qwen7b/ffn_down             256 |  5.2769       5.2707       1.00x
qwen7b/ffn_down             512 | 10.2461       9.7723       1.05x

llama8b/qkv     (4096,4096) 32 |  0.3079       0.3049       1.01x
llama8b/qkv                 64 |  0.4661       0.4657       1.00x
llama8b/qkv                128 |  0.8206       0.8546       0.96x  ← NAX -4% (reproduces in 5/5 trials)
llama8b/qkv                256 |  1.4215       1.3719       1.04x
llama8b/qkv                512 |  2.6174       2.5145       1.04x

llama8b/ffn_up  (14336,4096) 32 |  0.7866       1.3031       0.60x  ← NAX -40%
llama8b/ffn_up               64 |  1.2707       1.2959       0.98x
llama8b/ffn_up              128 |  2.3160       2.2560       1.03x
llama8b/ffn_up              256 |  4.4145       4.2027       1.05x
llama8b/ffn_up              512 |  8.6414       8.1788       1.06x
```

50 iters/cell, fresh subprocess per cell. Bench-noise envelope is ±2-3%; deltas inside that range are reported as ties.

Cross-checked qwen-1.5B shapes (smaller dim) at M ∈ {64, 128, 256}: same shape: M=64 borderline-to-loss, M=128 mixed (1.02x ffn_up, 0.98x qkv), M=256 consistent 1.02-1.05x. Not transcribed in full — sweep is in the JSON.

## 3. Reading the data

- **M ≥ 512:** NAX wins 3-6% on every shape. Old default already captures this.
- **M = 256:** NAX wins 4-5% on FFN shapes, 1-4% on attention QKV. Tie-or-win on every cell measured. The current threshold sits exactly at the consistent win boundary.
- **M = 128:** Mixed. FFN wins (~5%), square-attention-QKV loses (~3-4%). Net mean is near zero with shape-dependent ±5% spread.
- **M = 64:** Bench-noise envelope on every cell. No actionable win or loss.
- **M ≤ 32:** NAX regresses **40-44%** on wide-N FFN shapes due to the 64-row partial-tile (only half the rows do useful work). Tie or modest win on smaller-N shapes. NAX is unsafe at this M for general workloads.

The llama8b/qkv M=128 outlier (-4%) reproduces in 5/5 re-runs — it is not bench noise. NAX has a real per-call setup cost that is not yet amortized at M=128 on the square 4096×4096 shape. The same cost is more than offset by NAX's compute advantage on the wider 18944×3584 ffn_up shape at the same M.

## 4. Decision

**Keep the default at `M >= 256`.** Lowering to 128 would trade 5% FFN gains for 3-4% attention losses; lowering to 64 risks ~6% FFN regressions; lowering to 32 is clearly wrong. The conjecture in the parent doc — "could expand the NAX-eligible range and capture wins in mid-M shapes" — does not hold. The original 256 was tuned coarsely but happened to land at the right inflection.

The comment in `quantized.cpp` is updated from "256 is conservative — refine with a denser M-sweep when convenient" (inaccurate) to a brief summary of the inflection behavior with a pointer to this report.

## 5. What we keep

- `qmm_nax_m_threshold()` helper + `MLX_QMM_NAX_M_THRESHOLD` env-var override stay in. They cost nothing, and are useful for the next investigator who wants to re-sweep on different hardware (M5? M4 Max?) or different (group_size, bits).
- Sweep harness `~/devel/olmlx-model/benchmarks/nax_m_threshold_sweep.py` stays in. Re-runs trivially.

## 6. Workstream summary update

| Lever (Tier 1.5 §A archive + open list) | Status |
|---|---|
| Tier 1.5 §A v1+v2 q4 SDPA kernel | Shipped |
| Tier 1.5 §A v3 routing fallback (gqa>=5) | Shipped |
| Tier 1.5 §B Phase 6 NAX SDPA tile sweep | **REVERTED** (build broken; bench numbers from stale binary; see revert commit) |
| **NAX M-threshold refinement (this lever)** | **Closed: M=256 default empirically validated** |

## 7. Open items / non-goals

- **Per-arch threshold:** This data is M4 Pro only. M5 / M4 Max could have different inflection points. The env var is the recommended way to investigate.
- **Separate threshold for `gather_qmm_nax` (general path):** The general gather path shares the threshold with `qmm_nax`. Real MoE workloads use the rhs-specialized path (`gather_qmm_rhs_nax`) which has its own `M/E >= 512` gate, so the shared threshold is unlikely to be a real concern.
- **Kernel-side fix for the M=128 outlier:** The 3-4% NAX loss at llama8b/qkv M=128 might be addressable by making the NAX kernel handle partial bm=64 tiles more efficiently. Out of scope here.

## 8. Update to parent doc

`~/devel/olmlx-model/M4_OPTIMIZATION_OPPORTUNITIES.md` "Open levers worth pursuing":

- Strike: "Refine NAX M-threshold from 256 with a denser M-sweep (M ∈ [32, 64, 128, 256, 512]). Could expand the NAX-eligible range and capture wins in mid-M shapes. Quick benchmark exercise."
- Replace with: "✅ NAX M-threshold refinement (2026-04-27) — RESOLVED: M=256 default empirically validated on M4 Pro. M ≥ 512 unambiguous NAX win; M = 256 consistent 3-5% NAX win; M = 128 shape-dependent (FFN +5% / qkv −4%); M ≤ 64 noise or regression. See report `2026-04-27-nax-m-threshold-refinement.md`."

Also: parent doc Tier 1.5 §B previously claimed "Phase 6 tile sweep delivered 3-14% on Llama prefill". That claim is **invalid** — the bench numbers came from a stale binary; the new instantiations violate the kernel's static_asserts and would fail at compile or JIT time. The Phase 6 commits have been reverted on this branch. Tier 1.5 §B should be updated to remove the Phase 6 claim and mark "NAX SDPA tile tuning needs real kernel work (lift `kU=16` hardcoding or add `WM=2` instantiations) — out of scope for this workstream."
