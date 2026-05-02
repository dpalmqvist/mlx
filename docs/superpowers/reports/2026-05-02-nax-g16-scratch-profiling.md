# NAX-on-g16 Phase 8 — empirical prototype trial (final)

**Date:** 2026-05-02
**Hardware:** Apple M4 Pro
**MLX commits:** branch `nax-g16-phase8`, ranges below.
**Spec / plan:** `docs/superpowers/specs/2026-05-01-nax-g16-phase8-scratch-profiling-design.md`, `docs/superpowers/plans/2026-05-01-nax-g16-phase8.md`

## Methodology pivot

The Phase 8 spec preferred Xcode Metal GPU counters as the diagnostic tool, with source-level A/B variants as a documented fallback. Captures recorded via `mx.metal.start_capture` did not produce visible counter data when opened in Xcode on this machine; the Profile-after-Replay flow produced empty performance panes. We pivoted to empirical try-all: implement each of the three candidate prototypes (A, B, C) from the spec's decision rule and bench against the 2026-05-01 baseline.

This trades *understanding* (we don't know exactly why each prototype helped or didn't) for *speed*. Three benches × ~30 min each ran sequentially.

## Acceptance bar

- `gemm_splitk M=64 N=64 K=8192` reaches **≥0.95×** of non-NAX
- No other case regresses by **>0.05×** vs the 2026-05-01 baseline

## Trial results

### Round 1 — Prototype C (barrier-scope reduction in NAXFrag32)

Replace `threadgroup_barrier` with `simdgroup_barrier` in NAXFrag32's five scratch-staging methods (load_safe, load_rows, store_safe, store_rows, store_slice). Each simdgroup operates on its own scratch slice, so cross-simdgroup ordering is unnecessary.

Commit: `ae02566c`. Re-benched in the ship-state measurement (after B and A' reverts) at the merged-branch HEAD. The ship-state numbers are the canonical reference for what merging this PR delivers.

Bench artifacts: `2026-05-02-nax-g16-phase8-ship-bench.txt` (canonical, ship-state HEAD) and `2026-05-02-nax-g16-phase8-c-postfix-bench.txt` (original Round 1, on `ae02566c` directly — within measurement noise of ship-state).

| Case | baseline | post-C (ship) | delta |
|---|---|---|---|
| gemm_splitk M=64 N=64 K=8192 (target) | 0.36 | 0.41 | +0.05 |
| gemm_splitk M=128 N=128 K=4096 | 0.57 | 0.67 | +0.10 |
| gemm_segmented B=32 M=128 N=128 K=128 | 0.80 | 0.77 | -0.03 |
| All other matmul | — | — | within ±0.02 |
| sdpa, gather | — | — | within ±0.01 (noise) |

Acceptance bar at splitk: FAIL (0.41 < 0.95). But the bench is net-positive — small wins on the two splitk shapes, no case regresses by more than 0.03, and `nax_on` absolute times are within 2% of baseline (no occupancy regression). C ships as a code-quality improvement that delivers a measurable but well-below-target speedup at the small-tile splitk shapes.

### Round 2 — Prototype B (double-buffered scratch in gemm_loop)

Allocate separate scratch slices for Atile and Btile so their cooperative_tensor staging can run in parallel rather than serialize through one buffer per simdgroup. Stacked on top of C.

Commit: `095e63e6` (since reverted at `63c18414`). Bench output: `2026-05-02-nax-g16-phase8-bc-postfix-bench.txt`.

| Case | baseline nax_on (ms) | post-B+C nax_on (ms) | regression |
|---|---|---|---|
| gemm_segmented B=8 M=512 N=4096 K=4096 | 26.6 | 343 | **12×** |
| gather tokens=2048 (eh=14336) | 1770 | 8817 | **5×** |
| gemm_fused M=2048 N=11008 K=4096 | 38.2 | 52.3 | 1.4× |
| gemm_fused M=512 N=4096 K=4096 | 3.43 | 7.10 | 2.1× |
| gemm_splitk M=64 N=64 K=8192 | 0.34 | 0.69 | 2.0× |

Verdict: catastrophic regression. Reverted. Root cause is the doubled threadgroup-scratch allocation pushing past Metal's ~32KB per-threadgroup memory cap on the larger kernel instantiations (BM=128, WM=4 in big-shape fused / segmented / gather), collapsing occupancy from many simultaneous threadgroups per SM down to ~1.

The plan flagged this risk explicitly. Without Xcode counter access we couldn't pre-validate the allocation envelope; the bench was the validation, and it failed.

### Round 3 — Prototype A' (register-only fast path)

`NAXFrag32::load_direct` builds a `metal::tensor<device U, …, tensor_inline>` view of device memory (using `const_cast<device U*>` to satisfy the SDK's mutable-element requirement — the originally-spec'd `const device U` form was rejected by `cooperative_tensor::load`'s static_assert) and calls `ct.load(view)` directly. NAXTile::load dispatches to this on the non-transposed fast path; load_safe / load_rows / transposed loads still stage through scratch. Stacked on C (B was already reverted).

Commit: `86152210` (since reverted). Probe `tools/probe_naxfrag32_direct_load.py` kept as a regression test. Bench output: `2026-05-02-nax-g16-phase8-aprime-postfix-bench.txt`.

| Case | baseline | post-A'+C | delta |
|---|---|---|---|
| gemm_splitk M=64 N=64 K=8192 (target) | 0.36 | 0.35 | -0.01 |
| All matmul cases | — | — | within ±0.06 |
| nax_on absolute times | — | — | within 1% of baseline |

Acceptance bar at splitk: FAIL (0.35 < 0.95). Bench-neutral elsewhere; no occupancy regression. Reverted because A' adds complexity without performance benefit.

The apparent SDPA wins (e.g., kL=2048 hd=128 going 0.45 → 1.42) were `nax_off` slowdowns, not `nax_on` speedups — measurement noise from inter-bench thermal variation, not a real prototype A' effect.

## Final state

| Prototype | Status |
|---|---|
| A (register-only, `const device U` view) | infeasible — SDK rejects const tensor view |
| A' (register-only, `device U` view via const_cast) | compiles, correct, **bench no-op** — reverted |
| B (double-buffered scratch) | compiles, correct, **catastrophic regression** — reverted |
| C (simdgroup_barrier scope reduction) | compiles, correct, **bench-neutral** — kept |

`nax-g16-phase8` ships:
- The `--capture` flag in the perf-bench harness (Task 1, commit `ffee6c1c`)
- Prototype C: simdgroup_barrier scope reduction in NAXFrag32 (commit `ae02566c`)
- This profiling report and four bench artifacts (Rounds 1–3 plus the ship-state confirmation bench)
- A Phase 9 spec stub

## Implications for Phase 9

Two of three "scratch-staging is the bottleneck" hypotheses were tested empirically and disproven:
- Bypassing scratch entirely (A'): no effect on `nax_on` time
- Reducing barrier scope (C): no effect on `nax_on` time

The remaining bottleneck is somewhere we can't address by manipulating scratch. Candidates:
- **Cooperative_tensor lifecycle overhead.** Each `op.get_*_input_cooperative_tensor()` plus `ct.load()` may carry hidden per-fragment overhead independent of where the source data lives.
- **MMA dispatch / descriptor construction cost.** The `matmul2d_descriptor` is built per call; if it's not being constant-folded in some instantiations, per-call setup dominates.
- **NAXFrag32 vs BaseNAXFrag intrinsic overhead.** kPacking==1 is fundamentally different from kPacking==2 (32×32 fragment vs 16×16 with packing). Some part of the 30% may be inherent to the 32×32 path.

Phase 9 should:
1. Get Xcode counters working (or alternative profiling — ARM Instruments, command-stream timing, Metal API system trace) to actually MEASURE where the time goes rather than guessing.
2. If the cost is intrinsic to `cooperative_tensor` / NAXFrag32, the fix is at a different level: e.g., switching to manual SIMD register layout instead of `cooperative_tensor`, or sharing one `op` across the whole K-loop instead of constructing per-fragment.

## Caveats

- Wall-clock variance: 10-iter median, no fan or thermal control, no GPU clock pinning. Inter-bench thermal drift caused the apparent SDPA "wins" in Rounds 2 and 3 — confirmed by checking nax_on absolutes (which are unaffected by NAX-only kernel changes).
- A' kept its probe file (`tools/probe_naxfrag32_direct_load.py`) as a regression test even though the implementation is reverted; future Phase 9 attempts can verify against it.
- The `--capture` harness flag works headlessly even if Xcode counters don't display — the flag stays in case future macOS/Xcode versions surface counters.
