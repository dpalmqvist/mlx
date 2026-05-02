# NAX-on-g16 Phase 10 — design spec (stub)

**Date:** 2026-05-02
**Status:** stub — to be expanded via brainstorming
**Predecessor:** Phase 9 (`docs/superpowers/reports/2026-05-02-nax-g16-phase9-attribution.md`)

## Problem restatement

Phase 9 attributed the NAX matmul tax on g16 (Apple M3/M4) via three diagnostic kernel variants (V1 no-mma, V2 zero-load, V3 hoisted-op). On the trustworthy gemm_fused M=2048 N=4096 K=4096 shape:

- **cost(mma) ≈ 4.14ms** = 38% of V0 nax_on (10.999ms) — **dominant**
- **cost(ct.load) ≈ 0.84ms** = 8% of V0 nax_on
- **cost(per-iter setup) ≈ 0.08ms** = ~1% of V0 nax_on (compiler already CSE'd it; V3 hoisting was a no-op for perf, but is a valid refactor that passes correctness probes)

Phase 9 also revised the framing of the matmul "tax": V0 nax_on is 0.85× of V0 nax_off on gemm_fused (10.999 vs 9.317ms). The 30% tax language from earlier phases conflated NAX-vs-non-NAX speedup with intrinsic NAX overhead. The actual gap is 1.68ms, of which a chunk is just NAX paying for an mma op the non-NAX kernel also pays for via `simdgroup_matrix_multiply`. We don't yet know how much of the gap is "NAX mma slower than non-NAX mma" vs "extra NAX-only work outside mma."

The Phase 9 attribution decision rule named mma as the dominant contributor → Phase 10's job. The fix sits at a different level than scratch staging or coop-tensor lifecycle: it's about the kPacking==1 path itself.

## Candidate next directions

1. **BaseNAXFrag-style alternative path on g16.** The kPacking==2 path (16×16 fragment with packing) used by BaseNAXFrag is the existing mature NAX implementation on g17. On g16 we ruled out kPacking==2 because of an SDK-level issue (matmul2d at non-32 tile sizes), but the underlying "use multiple smaller mmas instead of one 32×32" idea may be portable. Investigate whether a manual SIMD-register layout that bypasses `cooperative_tensor` for the inner loop is feasible on g16, or whether a hybrid path (cooperative_tensor for setup, hand-coded mma scheduling for the K-loop) helps.

2. **Compare NAX mma vs non-NAX mma directly.** Phase 9 measured cost(mma) within the NAX kernel only. To know whether NAX mma is slower than the equivalent non-NAX `simdgroup_matrix_multiply`, build a back-to-back microbenchmark that runs each on the same fragment shape with everything else equal. If NAX mma is competitive with non-NAX mma, the matmul tax is *not* in mma after all — it's elsewhere (kernel scheduling, occupancy, dispatch overhead). If NAX mma is slower, that confirms Phase 9's attribution and bounds the achievable improvement.

3. **Profile NAX mma instruction count / cycles directly.** Phase 9 demoted Xcode counters and Apple Instruments as profiling avenues. If either becomes available, run the kernel under those tools and look at the mma instruction's actual stall / cycle behavior. This may reveal whether the cost is intrinsic to the NAX hardware op or downstream of register pressure / occupancy.

4. **gemm_splitk-specific optimization.** Phase 9 noted that splitk's nax_on=1.145ms / nax_off=0.140ms gap (0.12× speedup) is dominated by per-dispatch overhead, not K-loop body cost. A separate phase could target splitk specifically — e.g., inlining the kernel, reducing setup, or even falling back to non-NAX on small shapes where NAX dispatch overhead dwarfs the work.

## Acceptance bar

To be set once a direction is chosen. Phase 9's bar (≥0.95× of non-NAX on gemm_splitk M=64 N=64 K=8192) is likely unreachable for splitk specifically — the 0.12× current state is dominated by dispatch overhead that no kernel-body change will fix. A more realistic bar for Phase 10 is "improve gemm_fused M=2048 N=4096 K=4096 from 0.85× toward 0.95×" *or* "establish a hard upper bound on what's achievable on g16's NAX hardware via the comparison microbenchmark in candidate #2."

## Out of scope

- SDPA wm=2 occupancy regression on hd=128 — separate phase if pursued.
- Anything not on g16 hardware (g17 / BaseNAXFrag is unaffected).
- Bool-mask SDPA, bm=16 gather (still fall back to non-NAX).

## Open questions for brainstorming

- How much of the 0.85× gap on gemm_fused is recoverable on g16 hardware, given NAX mma's intrinsic cost?
- Is the "manual SIMD register layout" path realistic without SDK support, or is it a multi-month effort?
- Is splitk worth optimizing, or is the right answer to fall back to non-NAX on small shapes (and what's the tile-size threshold)?

(This is a stub; brainstorming will scope Phase 10 once the user picks a direction.)
