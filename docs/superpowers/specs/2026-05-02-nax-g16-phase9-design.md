# NAX-on-g16 Phase 9 — design spec

**Date:** 2026-05-02
**Status:** approved (brainstormed 2026-05-02)
**Predecessor:** Phase 8 (`docs/superpowers/reports/2026-05-02-nax-g16-scratch-profiling.md`)

## Problem restatement

Phase 8 closed with the ~30% NAX matmul tax on g16 (Apple M3/M4) still unexplained. Three scratch-staging hypotheses (A' register-only fast path, B double-buffered scratch, C simdgroup-scope barriers) were tested empirically; only C shipped, as a bench-neutral code-quality improvement. Phase 8 named three remaining suspects:

- **Cooperative_tensor lifecycle overhead** — per-fragment `op.get_*_input_cooperative_tensor()` + `ct.load()` may carry hidden setup cost.
- **MMA dispatch / descriptor construction cost** — per-call setup may not constant-fold.
- **NAXFrag32 (kPacking==1) intrinsic overhead** — fundamentally different from BaseNAXFrag's kPacking==2 path.

Phase 8 also established that without working profiler counters we cannot pre-validate hypotheses; the bench is the validation, and untargeted fixes either no-op or regress catastrophically. Phase 9 must measure where the time goes before, or alongside, attempting any fix.

## Approach

**Empirical attribution via diagnostic kernel variants** — the documented fallback in Phase 8's spec that was never executed. Build three variants of the existing NAX matmul (V1–V3), each deliberately deleting one cost contributor. Run them against the current ship-state baseline (V0) on two diagnostic shapes. Subtract to attribute cost.

**Confirmation via Apple Instruments / Metal System Trace (A2)** — a separate Apple tool from Xcode's Metal counters. Run on V0 only after V1–V3 lands, to get an independent occupancy / bandwidth / timeline read. Tiebreaker if V1–V3 attribution is ambiguous; sanity check otherwise.

This pairing was chosen over alternatives because V1–V3 has known feasibility (just C++/Metal source edits and bench harness work, no Apple tooling) while A2 is a low-cost parallel investigation that complements rather than blocks. Xcode Metal counters (Phase 8's failed first choice) are explicitly *not* on the path for Phase 9; if they start working they're a bonus.

## Components

### Diagnostic variants

Three new instantiations of the NAX matmul inner-loop kernel(s), gated by build-time selection. The exact gating mechanism (preprocessor macro `NAX_DIAG_VARIANT={1,2,3}` vs. a kernel template parameter vs. separate kernel symbols) is deferred to the implementation plan, which will read the current `gemm_loop_nax` parameterization first. All variants take an extra `device float* dce_sink` kernel argument; baseline V0 ignores it.

- **V1 — no-mma.** Keep `op` construction and `ct.load(...)`; replace each `mma(...)` invocation with a per-thread volatile write of one fragment lane to `dce_sink`. The result is a kernel that performs all the staging and load work but no matrix multiplication. `cost(mma) := V0 − V1`.

- **V2 — zero-load.** Keep `op` construction and `mma(...)`; skip `ct.load(...)` so fragments stay default-initialized (zero or garbage). The result is incorrect but cheap: it isolates the cost of pulling data from scratch into NAX registers. `cost(ct.load) := V1 − V2`.

- **V3 — hoisted-op.** Construct `mpp::tensor_ops::matmul2d op{desc}` plus the input and destination cooperative tensors *once* above the K-loop and reuse across iterations. If the descriptor is genuinely shape-only (likely), the result is correct. If the descriptor encodes per-call positional state, the result will be wrong; that's still informative. `cost(per-iter setup) := V0 − V3`.

DCE protection: each variant must produce side effects the compiler cannot elide. Per-thread sink writes are the baseline mechanism. Verification (see Risks) is a gate before we trust any variant's bench numbers.

### Bench harness

Extend `benchmarks/nax_g16_perf_bench.py` (or add a sibling `nax_g16_diag_bench.py`) with a `--diag` mode. The mode dispatches V0–V3 against the diagnostic shape set, prints an attribution table (per-shape rows × per-variant columns plus computed `cost(mma)`, `cost(ct.load)`, `cost(per-iter setup)` columns), and writes the raw output to `docs/superpowers/reports/`.

### Diagnostic shapes

- `gemm_splitk M=64 N=64 K=8192` — the worst-case Phase 8 target (0.36×).
- `gemm_fused M=2048 N=4096 K=4096` — ship-typical larger shape (~0.78×).

Two shapes is the minimum to confirm the attribution generalizes. Adding a second shape is cheap once the variants compile.

## Decision rule

After running V0–V3 on both shapes:

| Dominant contributor | Phase 9 action |
|---|---|
| **per-iter setup** (largest of `V0−V3`) | Ship the hoisted-op pattern in the real `gemm_loop_nax`. Target the acceptance bar this phase. |
| **ct.load** (largest of `V1−V2`) | Likely scratch bandwidth or layout. Document attribution; design phase 10 layout rethink. No fix this phase — scope risk too high. |
| **mma** (largest of `V0−V1`) | kPacking==1 intrinsic on g16. Fix is the BaseNAXFrag-style alternative path. Document; design phase 10. |
| **ambiguous** (top two within 0.05× on both shapes) | Run A2 Instruments to break the tie; re-decide. |

The decision rule is intentionally tight: only the per-iter-setup path triggers a same-phase fix, because hoisting `op` is a small, localized change (likely a few dozen lines in `gemm_loop_nax`). The other two named fixes are large enough to warrant their own design phase.

## Acceptance bar

- **Diagnosis (must):** attribution table populated for both diagnostic shapes; dominant cost contributor identified per shape; ambiguity (if any) resolved via A2 or documented as such.
- **Fix (conditional):** if the dominant contributor is per-iter setup, ship the hoisted-op fix and reach **≥0.95×** of non-NAX on `gemm_splitk M=64 N=64 K=8192`, with no other case from the standard perf-bench set regressing by **>0.05×** vs. the post-Phase-8 baseline. If the contributor is ct.load or mma, Phase 9 ships the report and a Phase 10 spec stub; no kernel fix this phase.

## A2 Instruments pass

Run after V1–V3 attribution lands. Target: V0 baseline only, both diagnostic shapes. Expected outputs: GPU timeline, kernel occupancy, memory bandwidth utilization. Use as tiebreaker when V1–V3 attribution is ambiguous; otherwise as an independent sanity check on the named bottleneck.

If Instruments turns out to be as broken on this machine as Xcode counters were in Phase 8: V1–V3 is self-contained, attribution still lands, and we document the second tooling failure as evidence for a future investment in alternative profiling infrastructure.

## Out of scope

- SDPA wm=2 occupancy regression on hd=128 — separate phase if pursued.
- BaseNAXFrag-style alternative NAX path on g16 — would be Phase 10 if mma cost dominates.
- Anything not on g16 hardware (g17 / BaseNAXFrag is unaffected).
- Bool-mask SDPA, bm=16 gather (already documented as falling back to non-NAX).
- Fixing Xcode Metal counters on this machine — explicitly de-prioritized; A2 (Instruments) is the chosen alternative.

## Risks

- **DCE elision.** Per-thread sink writes may not be enough; the compiler could still hoist or merge work in unexpected ways. Mitigation: inspect generated Metal IR (or AIR / Metal compiler intermediate) for each variant before benching, and verify each variant *contains* / *omits* the operations we expect. This is a hard gate before any V1–V3 number is trusted.
- **V3 ≡ V0.** Apple's compiler may already constexpr-fold or hoist `op` construction implicitly. If so, V3 cost is zero — useful negative result, eliminates the lifecycle suspect, focuses attention on ct.load and mma.
- **A2 also broken.** Instruments may suffer the same fate as Xcode counters on this machine. Phase 9 is engineered so V1–V3 stands on its own; A2 failure degrades to "we have an attribution but no independent confirmation," which is acceptable.
- **Variant divergence vs. real kernel.** If gating uses a separate kernel symbol or template, the variant may end up materially different from V0 in code paths beyond the one operation we meant to delete. Mitigation: prefer minimal in-place gating (preprocessor macro switching a single statement) over forking the kernel; gating choice is settled by the implementation plan after reading the current code.
- **Thermal variance.** Phase 8 saw apparent SDPA "wins" that were actually `nax_off` slowdowns from inter-bench thermal drift. Phase 9 attribution depends on `nax_on` deltas being real; the bench harness must collect enough iterations to swamp inter-run drift. The existing perf-bench median-of-N is likely sufficient but should be sanity-checked against a same-trial reproducibility run.

## Open questions

None blocking. Gating mechanism for variants and exact bench-harness shape (extend existing vs. sibling) are implementation-plan decisions, not design decisions.
