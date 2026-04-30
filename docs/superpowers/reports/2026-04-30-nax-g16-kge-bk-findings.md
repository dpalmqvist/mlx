# NAX g16 — K≥bk correctness bug: investigation, root cause, and fix

**Date:** 2026-04-30
**Branch:** `nax-g16-phase2`
**PR:** [dpalmqvist/mlx#14](https://github.com/dpalmqvist/mlx/pull/14)
**Status:** Resolved. Root cause identified, splitk_nax + segmented_nax ported to NAXFrag32, gather_mm_rhs_nax gated on g16. All sweep cases pass; net regression delta on `python/tests/test_blas.py` is +2 (12 → 10 baseline failures), all remaining failures pre-existing fp16 batched-matmul bugs.

## 1. Symptom

Phase 2 work on NAX-on-g16 reached the point where smoke tests in the `(M, N, K) = (128, 64, 64)` ballpark passed but `python/tests/test_blas.py` still failed on shapes with larger K. The Phase 2 handoff memory recorded a representative failure table:

```
| Shape              | gemm_k_iters | remainder | result    |
|--------------------|--------------|-----------|-----------|
| 128×128×256        | 1            | 0         | OK        |
| 128×128×257        | 1            | 1         | FAIL (94) |
| 128×128×384        | 1            | 128       | FAIL      |
| 128×128×512        | 2            | 0         | FAIL (146)|
| 32×32×256          | 1            | 0, M<bm   | FAIL (70) |
| 256×256×512        | 2            | 0         | OK        |
```

The handoff memory framed this as a `gemm_loop` correctness bug — "passes only when ALL simdgroups in a threadgroup take the kAlignedM=true && kAlignedN=true branch AND only 1 outer K iter AND no remainder" — and pointed forward to candidates inside `gemm_loop` and the fused-kernel preamble (dispatch_bool, NAXTile wrapping, return-by-value of `Dtile`, `STEEL_PRAGMA_NO_UNROLL` interactions).

**That framing was wrong.** Every candidate inside `gemm_loop` was eliminated by direct disproof (Section 3). The real bug was upstream of `gemm_loop`: dispatch.

## 2. Method — building a reproducer

`tools/repro_kge_bk_bug.py` exercises `mx.matmul` for fp32 inputs across an M×N grid {64×128, 128×128, 128×256, 256×256} and a K sweep {128, 256, 257, 384, 512, 640, 768, 1024}, comparing against a numpy reference with random integers in [-3, 3] (so fp32 reductions are exact and the verdict is binary).

Initial run (M=N=128) exposed a more informative pattern than the handoff table:

```
  K=128  iters=0  rem=128  OK
  K=256  iters=1  rem=0    OK
  K=257  iters=1  rem=1    FAIL  (err=409)
  K=384  iters=1  rem=128  FAIL  (err=460)
  K=512  iters=2  rem=0    FAIL  (err=479)
  K=640  iters=2  rem=128  FAIL  (err=569)
  K=768  iters=3  rem=0    FAIL  (err=606)
  K=1024 iters=4  rem=0    FAIL  (err=737)
```

Two new datapoints emerged that the handoff table didn't capture:

- **K=128 (`gemm_k_iters_aligned == 0`, only the `!kAlignedK` remainder branch runs) PASSES.** So the remainder branch by itself is fine.
- **K=256 (single outer iter, no remainder) PASSES.** So a single full BK pass is fine.
- Failure starts the moment we need a second piece of work in any direction — second outer iter (K=512), or outer iter + remainder (K=257/384), or more.

Then the M×N sweep showed that the failure threshold *moves with threadgroup count*:

```
| TG count   | shape    | K=257 | K=512 | K=640 |
|------------|----------|-------|-------|-------|
| 1          | 64×128   | FAIL  | FAIL  | FAIL  |
| 2          | 128×128  | FAIL  | FAIL  | FAIL  |
| 4          | 128×256  | OK    | OK    | FAIL  |
| 8          | 256×256  | OK    | OK    | FAIL  |
```

This was the second clue that the failure wasn't really about K — it was about which kernel got dispatched. The `K > 2*max(M,N)` and `K >= 3*max(M,N)` thresholds in `mlx/backend/metal/matmul.cpp` exactly trace this pattern (see Section 4).

## 3. What was eliminated (each by direct test, not by reading)

Each candidate from the handoff was tried in turn against the same K-sweep harness; all left the failure pattern *unchanged*:

| Hypothesis | Test | Outcome |
|---|---|---|
| Return-by-value of NAXTile (RVO failure / cooperative-tensor copy) corrupts Dtile | Changed `gemm_loop` to take `Dtile` by reference; updated all 4 callers | Identical failures |
| `dispatch_bool` lambda nesting interferes with specialization | Bypassed `dispatch_bool` in fused_nax; called `gemm_loop` with `kAlignedM=N=K=true` directly | Identical failures (K=128 changed because we forced wrong specialization, but K=256/512 unchanged) |
| `NAXTile<…, NAXFrag32>::load` + `tile_matmad_nax` template wrapping vs raw `NAXFrag32::load_rows` + `mma` | Added `MLX_NAX_DEBUG_RAW` branch inside `gemm_loop` that bypasses NAXTile/tile_matmad_nax for `kPacking==1 && TM=TN=TK=1`, mirroring `tools/probe_outer_inner_k.py` exactly | Identical failures |
| `NAXTile` constructor leaves frag in indeterminate state visible to subsequent loads | Hoisted `a_frag` / `b_frag` declarations out of inner loop to match the probe verbatim | Identical failures |
| `NAXTile<AccumType,…>` declared but uncleared in fused kernel preamble | (Implicitly tested by the by-reference change, since `Dtile.clear()` runs first inside the new gemm_loop) | Identical failures |

Meanwhile the existing probes — `probe_nax_frag32`, `probe_multik_accumulate`, `probe_multisg_multik`, `probe_outer_inner_k` (8 simdgroups × 2 outer K iters × BK=256, structurally identical to the failing production case) — all **passed** at every K value. Same `load_rows`. Same scratch. Same `mma`. Same outer/inner loop. Same pointer advance. Same NAXFrag32 layout. Yet production failed.

That mismatch only made sense if production wasn't actually running the same kernel binary as the probes.

## 4. Root cause — dispatch, not gemm_loop

`mlx/backend/metal/matmul.cpp:949` routes `mx.matmul` for `use_nax && batch_size_out == 1` to `steel_gemm_splitk_axpby_nax` whenever:

```cpp
K >= 3 * std::max(M, N) ||
(std::max(M, N) <= 1024 && K > 2 * std::max(M, N))
```

Computing this for the failure table:

| Shape           | max(M,N) | K   | Dispatch         | Verdict |
|-----------------|----------|-----|------------------|---------|
| 128×128×256     | 128      | 256 | regular fused    | OK      |
| 128×128×257     | 128      | 257 | **splitk_nax**   | FAIL    |
| 128×128×384     | 128      | 384 | **splitk_nax**   | FAIL    |
| 128×128×512     | 128      | 512 | **splitk_nax**   | FAIL    |
| 256×256×512     | 256      | 512 | regular fused    | OK      |
| 256×256×640     | 256      | 640 | **splitk_nax**   | FAIL    |
| 64×128×257      | 128      | 257 | **splitk_nax**   | FAIL    |
| 128×256×640     | 256      | 640 | **splitk_nax**   | FAIL    |

The correlation is exact. **Every failure routes to `steel_gemm_splitk_axpby_nax`. Every pass routes to `steel_matmul_regular_axpby_nax` (the fused path).**

`steel_gemm_splitk_nax.h:100` (pre-PR) declared the accumulator as:

```cpp
constexpr short TM = SM / 16;
constexpr short TN = SN / 16;
NAXTile<AccumType, TM, TN> Dtile;
```

`NAXTile<…>` defaults `NAXFrag_ = BaseNAXFrag` (kPacking==2), and the `/16` in `TM`/`TN` matches BaseNAXFrag's 16×16 frag. BaseNAXFrag's MMA goes through the `(16, 32, 16)` matmul2d descriptor — **the exact descriptor Phase 1 documented as broken on g16, the entire reason `NAXFrag32` exists.**

`steel_gemm_segmented_nax.h:82` and `steel_gemm_gather_nax.h:88` had the same defect. `steel_gemm_fused_nax.h` was correct because Phase 2 had wired its `NAXFrag_` template parameter and `_g16` instantiation. The other three kernels never received that work — Task 2.7 in the Phase 2 plan was supposed to gate them off until Phase 3, but the gate was never landed.

Confirmation: a 4-line gate (`metal::nax_arch_flavor() != metal::NAXArchFlavor::kG16`) on the splitk dispatch alone made all 32 K-sweep cases pass.

## 5. Fix

PR #14, commit `cff2afbb`:

- **`steel_gemm_splitk_nax`** — full NAXFrag32 port mirroring `steel_gemm_fused_nax`. Add `class NAXFrag_ = BaseNAXFrag` template parameter; derive `TM = SM / NAXFrag_::kFragRows`, `TN = SN / NAXFrag_::kFragCols` (with `static_assert` divisibility); allocate `threadgroup AccumType scratch_buf[WM*WN*kFragRows*kFragCols]` for `kPacking==1` (size 1 placeholder for `kPacking==2`); compute `sg_scratch = scratch_buf + simd_group_id * (kFragRows*kFragCols)`; thread `(threadgroup T*)sg_scratch` through `gemm_loop` and `Dtile.store{,_safe}` (instead of `nullptr`). New `instantiate_..._g16` macros emit `_g16`-suffixed kernels for fp16/bf16/fp32. `matmul.cpp` appends `_g16` to the kname when `nax_arch_flavor() == kG16`.
- **`steel_gemm_segmented_nax`** — same pattern.
- **`steel_gemm_gather_nax`** — gated off on g16 instead of ported. Two reasons it can't be done now:
  - `NAXTile::store_slice` has no `kPacking==1` implementation (the spec defers it to Phase 5).
  - Gather instantiates `bm=16` and `bm=32` shapes (`SM = bm/wm = 16`), incompatible with NAXFrag32's 32×32 frag.
  - Falls through to non-NAX `gather_mm_rhs`, which works correctly on g16.

## 6. Verification

- **`tools/repro_kge_bk_bug.py`** — 32/32 cases pass (was 24/32 pre-fix).
- **All 5 NAX probes** (`probe_nax_frag32`, `probe_multisg_partial_k`, `probe_multik_accumulate`, `probe_multisg_multik`, `probe_outer_inner_k`) — pass.
- **`python/tests/test_blas.py`** — 10 failures, down from 12 baseline. Two regained: `test_gemv_gemm_same_precision` and `(1, 64, 4096) @ (1, 4096, 64) transpose='nn'`. Remaining 10 are all fp16 batched-matmul shapes (e.g. `(3, …, …)`, `(16, 768, 128)`, `test_matrix_vector_attn`) that were already failing on the pre-PR Phase 2 head; they are unrelated to this change.

## 7. Lessons

**Probes that exhaust the suspected layer don't disprove a bug — they only narrow it.** All five existing probes targeted the kernel primitives; none crossed the dispatch boundary. The kernel-layer probes were correct and well-designed — but the bug lived above them. A "probe" that exercises `mx.matmul` with the actual production shapes is the only thing that *would* have caught this earlier. `tools/repro_kge_bk_bug.py` now plays that role.

**When a failure pattern is suspiciously sensitive to a parameter that "shouldn't matter" (here: threadgroup count), it's worth checking whether that parameter actually changes which kernel binary runs.** The threadgroup-count sensitivity in the M×N sweep was the single most diagnostic datapoint — it's hard to construct a within-kernel race that masks under more parallelism, but trivially easy to explain by noting that more threadgroups happens to imply different `max(M,N)` and therefore a different dispatch decision.

**The Phase 2 design was self-consistent; the implementation skipped the gate.** The Phase 2 spec correctly listed splitk/gather/segmented as deferred-to-Phase-3 with an explicit "gate them off on g16" requirement (Task 2.7). That gate never landed, so on g16 these kernels silently routed to the broken (16,16,16) descriptor instead of either being fixed or being disabled. The handoff memory's "Where to look next" section then sent investigation in the wrong direction by anchoring on the kernel-body candidates.

## 8. References

- PR: https://github.com/dpalmqvist/mlx/pull/14
- Spec (Phase 2 design): `docs/superpowers/specs/2026-04-28-nax-g16-phase2-design.md`
- Plan (Phase 2 tasks): `docs/superpowers/plans/2026-04-28-nax-g16-phase2.md` — Task 2.7 (gate non-fused NAX paths) was the deferred work that this PR substitutes a partial port for.
- Probes: `tools/probe_nax_frag32.py`, `tools/probe_multisg_partial_k.py`, `tools/probe_multik_accumulate.py`, `tools/probe_multisg_multik.py`, `tools/probe_outer_inner_k.py` (existing); `tools/repro_kge_bk_bug.py` (new).
- Files touched: `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk_nax.{h,metal}`, `…/steel_gemm_segmented_nax.{h,metal}`, `mlx/backend/metal/matmul.cpp`.
