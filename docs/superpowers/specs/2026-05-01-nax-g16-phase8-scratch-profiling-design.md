# NAX-on-g16 Phase 8 — matmul scratch profiling + fix

**Date:** 2026-05-01
**Status:** design
**Hardware:** Apple M3/M4 family (g16 NAX flavor)
**Predecessors:** Phase 7 (`docs/superpowers/specs/2026-05-01-nax-g16-phase7-design.md`), perf baseline (`docs/superpowers/reports/2026-05-01-nax-g16-perf-baseline.md`)

## Goal

Reduce the uniform ~30% NAX matmul tax on g16 for `gemm_fused` and
`gemm_splitk`. The perf baseline (commit `86c571da`, M4 Pro) measured every
matmul case as a regression, ranging 0.36×–0.80× of non-NAX. Worst case is
`splitk M=64 N=64 K=8192` at 0.36×.

Profile to identify the dominant cost contributor, then prototype the
smallest fix that reaches parity (≥0.95× of non-NAX at the worst case, no
regressions elsewhere).

## Phase shape

Single PR. Sequential gates: profiling → fix selection → prototype → bench.
If the prototype misses the acceptance bar, the PR ships the profiling
report only and the fix moves to Phase 9.

## Profiling

### Tool

Xcode Metal GPU capture via `MTLCaptureManager`, triggered from
`benchmarks/python/nax_g16_perf_bench.py` (extended with a `--capture` flag
that wraps a single iteration in a capture scope). Output is a `.gputrace`
file per case, reviewed in Xcode for per-kernel counters.

### Cases

Small set, shapes drawn from the existing baseline so deltas align with the
report on `main`:

- `splitk M=64 N=64 K=8192` — worst case, primary diagnostic
- `splitk M=128 N=128 K=4096` — confirms the small-tile pattern
- `fused M=512 N=4096 K=4096` — typical mid-size
- `fused M=2048 N=11008 K=4096` — confirms shape-independence claim

For each case capture both arms (`MLX_DISABLE_NAX=1` set and unset). The
analysis works in terms of counter *deltas* between arms, not absolute
numbers. The non-NAX arm is the steel_gemm baseline.

### Counters of interest

- ALU active (% of cycles)
- Threadgroup memory bandwidth (GB/s, % of peak)
- Memory stall cycles (DRAM + threadgroup)
- Barrier stall cycles
- Occupancy (active threads / max per SM)
- Threadgroup memory pressure (occupancy ceiling from per-TG scratch size)

## Decision rule (fix selection)

After profiling, classify the dominant cost and prototype exactly one of:

| Dominant signal | Prototype |
|---|---|
| Threadgroup memory bandwidth saturated (>70% peak), ALU starved (<40%) | **A. Register-only fast path** for `(Role, transpose=false, aligned)` descriptors — bypass scratch staging on the fast path. Keep scratch for safe/rows/transpose. Requires SDK feasibility check first. |
| Bandwidth moderate (<70%), barrier stall cycles dominate | **B. Double-buffered scratch** — give Atile and Btile separate scratch slices so they stage in parallel. |
| Barrier stalls + per-simd-only scratch access pattern visible | **C. Barrier-scope reduction** — replace `threadgroup_barrier` with `simdgroup_barrier` in `NAXFrag32::load_rows`/`store_rows`/`store_safe`/`store_slice` where each simdgroup's scratch slice is genuinely simdgroup-private. |
| None of the above | Halt prototype. Ship profiling report. Defer fix to Phase 9. |

If both B's and C's signals fire (barrier stalls dominate *and* the access
pattern is per-simd-only), prefer C first — it's a one-line per-call-site
change with no allocation impact. If C lands a measurable but partial
improvement, B may still be done in a follow-up phase.

If the data points to a fix not in this list, revise the spec — do not
silently pivot.

## Prototype changes (one of)

### A. Register-only fast path

Lands in `mlx/backend/metal/kernels/steel/nax_common.h`. Add
`NAXFrag32::load_direct<Role, transpose>` that constructs a
`cooperative_tensor` directly from device memory (no scratch staging).
Dispatch in `NAXTile::load` to call it on the aligned, non-transposed path.
Existing scratch path remains for `load_safe`, `load_rows`, and the
transposed cases.

**Feasibility check (mandatory before A is selected):** add
`tools/probe_naxfrag32_direct_load.py` that attempts a register-only load
for the `(Role::Left, false)` and `(Role::Right, false)` descriptors and
verifies bit-exact equality against the scratch path. If the probe fails
(MPP/MTL doesn't expose this capability for NAXFrag32 descriptors on g16),
drop A from the decision rule and re-evaluate B/C from the same counters.

### B. Double-buffered scratch

Lands in `mlx/backend/metal/kernels/steel/gemm/gemm_nax.h` (the inner
`gemm_loop`) and the kernel-scope scratch sizing in
`steel_gemm_fused_nax.h` and `steel_gemm_splitk_nax.h`.

- Allocate two scratch pointers per simdgroup: `sg_scratch_a` and
  `sg_scratch_b`.
- Pass `sg_scratch_a` to `Atile.load(...)` and `sg_scratch_b` to
  `Btile.load(...)`.
- The output-tile `Dtile.store`/`store_safe` epilogue can reuse
  `sg_scratch_a` since it's the only consumer at that point.
- Double `kScratchSize` only when `NAXFrag_::kPacking == 1`. The
  BaseNAXFrag (kPacking==2) path is unchanged.

**Occupancy guard:** validate via Xcode capture that the doubled
allocation does not push the kernel past Metal's per-threadgroup memory cap
(~32KB on g16) or drop occupancy. If it does, scope this to a single shape
class (e.g., only kernels with WM*WN ≤ 4) and gate via a kernel template
parameter.

### C. Barrier-scope reduction

Lands inside `NAXFrag32::load_rows`/`store_rows`/`store_safe`/`store_slice`
in `nax_common.h`. No caller changes.

- Audit each barrier in those methods. Per-simdgroup scratch slices are
  guaranteed by callers (`sg_scratch = scratch + sg_id * kFragRows *
  kFragCols`), so cross-simdgroup ordering is unnecessary on the read-back.
- Replace `threadgroup_barrier(mem_flags::mem_threadgroup)` with
  `simdgroup_barrier(mem_flags::mem_threadgroup)` where the data flow is
  fully per-simd.
- Leave any cross-simd barriers untouched — those exist for the kernel-level
  K-iter ordering, which is still threadgroup-scoped.

## Validation

### Correctness

All existing NAX probes pass:

- `tools/probe_nax_frag32.py`
- `tools/probe_naxfrag32_transpose.py`
- `tools/probe_naxfrag32_store_slice.py`
- `tools/probe_naxfrag32_sdpa.py` (Phase 7 SDPA — verifies the shared
  nax_common.h hierarchy still works for SDPA)
- `tools/repro_kge_bk_bug.py`
- `tools/repro_gather_g16.py`

For prototype A specifically, also `tools/probe_naxfrag32_direct_load.py`
must pass.

### Performance

Rerun `benchmarks/python/nax_g16_perf_bench.py` full table on M4 Pro.
Acceptance bar:

- `splitk M=64 N=64 K=8192` reaches **≥0.95× of non-NAX** (parity)
- No other case in the 13-row table regresses by >0.05× from the
  pre-fix baseline. The pre-fix baseline is the row of speedup numbers in
  `docs/superpowers/reports/2026-05-01-nax-g16-perf-baseline.md` (NAX-on
  arm timings re-measured on the same machine immediately before the
  prototype lands, since absolute wall times will drift between runs).

### Cross-arch sanity

The fix must not regress kPacking==2 (BaseNAXFrag, g17 hardware). All
changes in `nax_common.h` and `gemm_nax.h` must be guarded by `if constexpr
(NAXFrag_::kPacking == 1)` or live behind dispatch paths only reachable
from `_g16` kernel instantiations. No g17 code path may take a different
branch as a side effect.

## Risks

1. **A's feasibility is unknown until probed.** Phase 1 design notes said
   Metal SDK lacks bounded tensor views; whether unbounded direct
   device-memory cooperative_tensor loads work for NAXFrag32 descriptors on
   g16 is open. Mitigation: the feasibility probe runs before A is
   committed to. If it fails, fall back to B (if counters pointed there) or
   halt.

2. **B may break occupancy.** Per-threadgroup memory budget on g16 is
   ~32KB. A `WM=WN=2` `gemm_splitk_nax` kernel currently uses
   `4 × 32 × 32 × 4 = 16KB` scratch in fp32. Doubling lands at 32KB which
   may push past the cap once MPP's own internal scratch is added.
   Mitigation: validate occupancy via Xcode capture immediately after
   prototype.

3. **C may be a no-op.** If Metal already coalesces threadgroup-scoped
   barriers down to simdgroup scope when the data flow allows, the change
   does nothing. Worth ~30 minutes to try; abandon quickly if profiling
   shows no delta.

4. **Acceptance bar may be unreachable.** If the counters identify a
   bottleneck that isn't addressable by A/B/C (e.g., per-MMA descriptor
   setup cost intrinsic to NAXFrag32), the PR ships the profiling report
   only and the fix design becomes Phase 9. This is the explicit fallback,
   not a failure mode.

## Out of scope

- SDPA hd=128 `wm=2` occupancy regression — separate root cause, separate
  phase.
- `gemm_segmented` — same `gemm_loop` underneath, but its per-batch-segment
  epilogue adds scope; if the chosen fix lands in `gemm_loop` it
  *automatically* benefits segmented, but segmented validation is a
  follow-up phase.
- gather — already at parity (1.00×) per the baseline; no work needed.
- bool-mask SDPA and bm=16 gather — both still fall back to non-NAX on g16
  and are unaffected by this work.

## Deliverables

- `docs/superpowers/reports/2026-05-XX-nax-g16-scratch-profiling.md` —
  counter results per case, dominant-cost classification, fix-selection
  decision.
- One of {A, B, C} prototyped and merged if it clears the bar; otherwise
  the PR is profiling-only and a Phase 9 spec is written.
- Refreshed perf baseline numbers in a follow-on report (if a fix lands).
- `tools/probe_naxfrag32_direct_load.py` (only if prototype A is taken).

## Files touched

Profiling-only path:
- `benchmarks/python/nax_g16_perf_bench.py` — add `--capture` flag

Prototype A:
- `mlx/backend/metal/kernels/steel/nax_common.h`
- `tools/probe_naxfrag32_direct_load.py` (new)

Prototype B:
- `mlx/backend/metal/kernels/steel/gemm/gemm_nax.h`
- `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.h`
- `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk_nax.h`

Prototype C:
- `mlx/backend/metal/kernels/steel/nax_common.h`
