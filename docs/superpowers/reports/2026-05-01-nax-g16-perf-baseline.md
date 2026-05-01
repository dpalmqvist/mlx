# NAX-on-g16 Performance Baseline — Report

**Date:** 2026-05-01
**Hardware:** Apple M4 Pro
**MLX commit:** 86c571daa8bbb86955275a8023672bb747acd5f1
**Spec / plan:** `docs/superpowers/specs/2026-05-01-nax-g16-perf-baseline-design.md`, `docs/superpowers/plans/2026-05-01-nax-g16-perf-baseline.md`

## Method

Single-build A/B via the `MLX_DISABLE_NAX` runtime gate (added in commit
0761db37). Two subprocesses, one with the gate set, run the same 13 cases.
Per case: 3 warmup iters, 10 timed iters, median wall time. Full harness:
`benchmarks/python/nax_g16_perf_bench.py`.

Preflight on this run: NAX-on arm reported `nax_available=True` `nax_flavor=g16`;
NAX-off arm reported `nax_available=False` `nax_flavor=none`. The gate flipped
correctly; speedup numbers reflect real kernel-path differences, not noise.

## Results

```
kernel          shape                                                nax_on    nax_off   speedup   TFLOPS
gemm_fused      M=2048 N=4096 K=4096                               13.371ms    9.301ms     0.70x     5.14
gemm_fused      M=2048 N=11008 K=4096                              37.569ms   25.007ms     0.67x     4.92
gemm_fused      M=512 N=4096 K=4096                                 3.480ms    2.465ms     0.71x     4.94
gemm_splitk     M=64 N=64 K=8192                                    0.334ms    0.120ms     0.36x     0.20
gemm_splitk     M=128 N=128 K=4096                                  0.220ms    0.125ms     0.57x     0.61
gemm_segmented  B=8 M=512 N=4096 K=4096                            26.789ms   18.388ms     0.69x     5.13
gemm_segmented  B=32 M=128 N=128 K=128                              0.180ms    0.144ms     0.80x     0.74
gather          tokens=2048 E=8 hidden=4096 expert_hidden=14336 top_k=2  1757.902ms  1757.262ms     1.00x     0.27
gather          tokens=512 E=8 hidden=4096 expert_hidden=4096 top_k=2  121.034ms  121.430ms     1.00x     0.28
sdpa_prefill    B=1 H=32 kL=2048 hd=128                            23.690ms   10.576ms     0.45x     2.90
sdpa_prefill    B=1 H=32 kL=8192 hd=128                           381.510ms  177.414ms     0.47x     2.88
sdpa_prefill    B=1 H=32 kL=512 hd=128                              1.701ms    0.877ms     0.52x     2.53
sdpa_prefill    B=1 H=32 kL=2048 hd=64                              6.765ms    5.649ms     0.84x     5.08
```

JSON archive: `docs/superpowers/reports/2026-05-01-nax-g16-perf-baseline.json`
(committed alongside this report).

## Ranking

By kernel_label group, highest-to-lowest median speedup (NAX-on / NAX-off,
>1 means NAX wins):

| Rank | Group           | Median speedup | Verdict         |
|------|-----------------|---------------|-----------------|
| 1    | gather          | 1.00x         | parity          |
| 2    | gemm_segmented  | 0.74x         | NAX hurts       |
| 3    | gemm_fused      | 0.70x         | NAX hurts       |
| 4    | sdpa_prefill    | 0.49x         | NAX hurts       |
| 5    | gemm_splitk     | 0.46x         | NAX hurts       |

Classification thresholds: NAX wins >= 1.05x, parity 0.95–1.05x, NAX hurts < 0.95x.

Detailed speedups per row, descending:

- gather (tokens=512, expert_hidden=4096):     1.00x  — parity
- gather (tokens=2048, expert_hidden=14336):   1.00x  — parity
- sdpa_prefill (kL=2048, hd=64):              0.84x  — NAX hurts
- gemm_segmented (B=32 small):                 0.80x  — NAX hurts
- gemm_fused (M=512):                          0.71x  — NAX hurts
- gemm_fused (M=2048, N=4096):                 0.70x  — NAX hurts
- gemm_segmented (B=8 large):                  0.69x  — NAX hurts
- gemm_fused (M=2048, N=11008):                0.67x  — NAX hurts
- gemm_splitk (M=128):                         0.57x  — NAX hurts
- sdpa_prefill (kL=512, hd=128):              0.52x  — NAX hurts
- sdpa_prefill (kL=8192, hd=128):             0.47x  — NAX hurts
- sdpa_prefill (kL=2048, hd=128):             0.45x  — NAX hurts
- gemm_splitk (M=64, N=64):                   0.36x  — NAX hurts (worst case)

## Observations

**Dominant signal — matmul (all three variants):** Every single matmul case is a
regression. NAX g16 is 25–65% slower than the non-NAX steel_gemm path across
gemm_fused, gemm_splitk, and gemm_segmented. The regression is worst in the
small-M splitk regime (M=64, N=64, K=8192 → 0.36x) and mildest in the batched
segmented case at small batch size (B=32, tiny tiles → 0.80x). The large-batch
segmented and all fused cases cluster tightly around 0.67–0.71x. This is a
consistent, shape-independent overhead — not a corner case.

**SDPA shape dependence:** SDPA shows clear shape sensitivity. The hd=64 variant
(kL=2048) is the least affected at 0.84x, while the hd=128 variants range
from 0.45–0.52x for longer sequences (kL=512, 2048, 8192). The kL=512 short
sequence (0.52x) is disproportionately slow relative to what one might expect
from a startup-cost explanation — the overhead appears fixed per iteration
regardless of kL. The hd=64 case benefits from fewer total FLOPs in the MMA
tiles relative to the threadgroup overhead, which explains its milder regression.

**Gather:** Both gather cases are essentially at parity (1.00x within measurement
noise). This is consistent with gather being memory-bound rather than
compute-bound — the NAX vs. non-NAX kernel path difference matters far less when
the bottleneck is DRAM bandwidth, not ALU throughput. The NAX g16 gather
implementation is not hurting and not helping at these shapes.

**Anomalous case:** The gemm_splitk M=64, N=64, K=8192 case at 0.36x is the
single most extreme regression. A 64x64 output tile is tiny relative to K=8192;
the NAX cooperative_tensor machinery (register layout transformations, scratch
staging) carries a per-threadgroup fixed cost that is not amortized by the
output tile size. The non-NAX splitk path launches many small threadgroups with
far lighter overhead, making this shape particularly unfavorable for NAX.

**Known g16 design costs visible in the data:**

1. *Threadgroup scratch staging* (every load_safe/load_rows/store_*): This is
   the primary suspect for the uniform ~0.67–0.71x matmul regression. Every
   load/store on g16 must stage through threadgroup memory because NAXFrag32
   lacks direct SIMD register cooperative_tensor load support. This adds 2
   round-trips per tile through shared memory compared to the non-NAX path that
   issues cooperative_tensor ops directly.

2. *SDPA wm=2 vs. standard wm=4*: The wm=2 instantiation (used for g16 SDPA
   because NAXFrag32 requires it) means half as many wavefronts are active on
   the SDPA kernel, reducing occupancy. This is consistent with the deep
   regression (0.45–0.52x) on the hd=128 SDPA shapes where the MMA pipeline
   dominates.

3. *Direct per-element addressing for the SDPA mask path*: Not exercised in this
   run (no bool mask). The float-mask SDPA path ran in all 4 SDPA cases.

## Recommendation for next phase

The worst kernel path is **gemm_splitk** (median ~0.46x, worst case 0.36x). The
suspected bottleneck is the per-tile threadgroup scratch round-trip in the NAXFrag32
load/store path: for a 64x64 output tile with K=8192, the scratch staging cost
is paid on every K-dimension iteration while the tile accumulation work is minimal,
inverting the usual compute/overhead ratio. The immediate investigation target should
be profiling the scratch staging latency in `steel_gemm_splitk_nax_g16` (and the
analogous `steel_gemm_fused_nax_g16` matmul kernel) with Metal GPU counters to
confirm whether shared-memory bandwidth or barrier synchronization is the dominant
cost — then evaluate whether a fused register-only path (bypassing scratch for
small tiles where cooperative_tensor store fits within the warpgroup) is feasible
on M4 Pro. SDPA at 0.45–0.52x for hd=128 is a close second-priority target, where
the wm=2 occupancy halving (vs the standard wm=4) is the suspected primary cost
rather than scratch staging.

## Caveats

- Wall-clock variance: 10-iter median, no fan/thermal control, no GPU clock pinning.
- Single-shape-per-row: no statistical sweep within a regime.
- NAX-off arm uses the legacy SIMD-only kernels — for SDPA on g16 that is
  `sdpa_full_self_attention_metal`; for matmul it is the non-NAX `steel_gemm_*`
  family. The comparison is "NAX g16 path" vs "what would run if NAX were
  unavailable on this device" — NOT "NAX g16 path" vs "NAX g17 path". The latter
  requires different hardware.
- The bool-mask SDPA path on g16 falls back to non-NAX (NAXFrag32 cooperative_tensor
  doesn't accept bool); not measured here.
- The bm=16 gather path on g16 also falls back to non-NAX (NAXFrag32 needs SM>=32);
  not measured here.
- 11008-N gemm_fused, 14336-expert_hidden gather, and 8192-kL SDPA cases each
  allocate large tensors (~150-900MB); if running on a base-tier M3 with 8GB
  unified memory consider re-running with smaller shapes.
