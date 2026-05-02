# NAX-on-g16 Phase 8 — empirical prototype trial (Xcode pivot)

**Date:** 2026-05-02
**Hardware:** Apple M4 Pro
**MLX commit:** see `git log` on branch `nax-g16-phase8`
**Spec / plan:** `docs/superpowers/specs/2026-05-01-nax-g16-phase8-scratch-profiling-design.md`, `docs/superpowers/plans/2026-05-01-nax-g16-phase8.md`

## Methodology pivot

The Phase 8 spec preferred Xcode Metal GPU counters as the diagnostic
tool, with source-level A/B variants as a documented fallback if counters
were "infeasible or too coarse." Captures recorded via
`mx.metal.start_capture` did not produce visible counter data when opened
in Xcode on this machine; the Profile-after-Replay flow produced empty
performance panes. Rather than fight Xcode further, we pivoted to an
empirical try-all approach: implement each of the three candidate
prototypes from the spec's decision rule (A, B, C) and pick the winner by
the existing acceptance bar — no counter analysis required.

This trades *understanding* (we won't know exactly why one prototype
wins) for *speed* (each prototype is small enough to implement, build,
and bench in one cycle). If all three miss the bar, we revisit
diagnostic-only variants in a Phase 9 follow-up.

## Order of trial

Smallest change first, to fail fast:

1. **C — barrier-scope reduction.** Replace `threadgroup_barrier` with
   `simdgroup_barrier` in NAXFrag32's scratch-staging methods. Each
   simdgroup operates on its own scratch slice; cross-simdgroup ordering
   is unnecessary on the read-back. Surface area: ~5 lines in
   `nax_common.h`. No allocation impact.

2. **B — double-buffered scratch.** Give Atile and Btile separate scratch
   slices in `gemm_loop` so they can stage in parallel. Surface area:
   `gemm_nax.h` + `steel_gemm_fused_nax.h` + `steel_gemm_splitk_nax.h`,
   ~20 lines. Doubles per-simdgroup threadgroup memory footprint.

3. **A — register-only fast path.** Bypass scratch staging on the aligned
   non-transposed `(Role::*, transpose=false)` descriptors via a direct
   device-memory `cooperative_tensor.load(view)` call. Surface area: new
   `NAXFrag32::load_direct` in `nax_common.h` + dispatch in
   `NAXTile::load`. Has a feasibility risk — the SDK may not accept a
   device-memory tensor view for the cooperative tensor returned by
   `get_*_input_cooperative_tensor`.

Stop trying further prototypes the moment one clears the bar.

## Acceptance bar

- `gemm_splitk M=64 N=64 K=8192` reaches **≥0.95×** of non-NAX
- No other case in the 13-row table regresses by **>0.05×** from the
  pre-fix baseline (re-measured immediately before the prototype lands)

## Result

(Filled in after the trial.)
