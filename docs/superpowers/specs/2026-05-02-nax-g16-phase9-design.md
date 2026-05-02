# NAX-on-g16 Phase 9 — design spec (stub)

**Date:** 2026-05-02
**Status:** stub — to be expanded via brainstorming
**Predecessor:** Phase 8 (`docs/superpowers/reports/2026-05-02-nax-g16-scratch-profiling.md`)

## Problem restatement

After Phase 8 closed, the ~30% NAX matmul tax on g16 (Apple M3/M4) remains. Phase 8 disproved the spec's three scratch-staging hypotheses empirically:

- **A'** (register-only fast path, bypass scratch entirely): bench no-op
- **B** (double-buffered scratch, parallel A/B staging): catastrophic regression from threadgroup-memory cap overflow on larger instantiations
- **C** (simdgroup_barrier instead of threadgroup_barrier): bench-neutral; shipped as a code-quality improvement

`nax_on` absolute times changed by <1% across all three trials → the bottleneck is not in scratch staging or barrier scope.

## Candidate next directions

1. **Get profiling working.** Xcode Metal counters were unusable on this machine in Phase 8 — find an alternative or fix the Xcode setup. Options: ARM Instruments, command-stream timing, Metal API system trace, source-level diagnostic A/B variants (deliberately produce wrong outputs to isolate cost contributors).

2. **Cooperative_tensor lifecycle.** Each `op.get_*_input_cooperative_tensor()` + `ct.load()` carries some per-fragment overhead independent of where source data lives. Investigate whether reusing one `op` across the K-loop (vs constructing per fragment) helps. The matmul2d_descriptor is shape-only and could be hoisted out.

3. **NAXFrag32 alternative path.** kPacking==1 (32×32 fragment) may have intrinsic overhead vs kPacking==2 (16×16 with packing) on g16. Consider a manual SIMD register layout path that bypasses `cooperative_tensor` entirely for the inner loop.

4. **SDPA hd=128 wm=2 occupancy.** Out of Phase 8 scope, but the 0.45–0.52× SDPA regression has a known root cause (wm=2 instead of wm=4) that's separate from matmul. Could be its own phase if we want.

## Acceptance bar

Same as Phase 8: `gemm_splitk M=64 N=64 K=8192` reaches ≥0.95× of non-NAX, no other case regresses by >0.05×. If a different cost contributor is identified that puts the bar out of reach, document and revise.

## Out of scope

- Anything not on g16 hardware (g17 / BaseNAXFrag is unaffected)
- Bool-mask SDPA, bm=16 gather (still fall back to non-NAX)

(This is a stub; brainstorming will expand it once a profiling path is chosen.)
