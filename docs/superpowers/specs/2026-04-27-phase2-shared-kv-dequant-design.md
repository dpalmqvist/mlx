# Phase 2 Design — Share K/V Dequant Across Query Heads in a GQA Group

**Date:** 2026-04-27
**Branch:** `kv4-sdpa-phase2` (off `kv4-sdpa`)
**Depends on:** `docs/superpowers/reports/2026-04-27-sdpa-quantized-profile.md` (Phase 1 §6 / §7)
**Scope:** Phase 1 §7.1 only. §7.2 (fp16-precision dequant) and §7.3 (hoist scales) are deferred to Phase 3.

## Goal

Close the 2.79× q4-vs-fp16 gap on `qwen-gqa @ ctx=32768` (Phase 1 §3) by
sharing K/V dequant across the query heads in a GQA group. Phase 1 §6
established that the kernel is **ALU-bound on dequant**, and §6.4 showed
that the per-query-head dequant work is the structural reason q4 scales
worse with GQA ratio than fp16 (which gets implicit GQA amortization
through L2 reuse of K/V loads).

The success criterion (Phase 1 §4): **`q4_sdpa` must beat
`dq_then_sdpa` at every measured context length on `qwen-gqa`.** This
is the floor — the kernel must be faster than its own
`mx.dequantize(...) → fp16 SDPA` fallback. Currently the fallback wins
at every ctx on qwen-gqa (1.07–1.34×).

## Non-Goals (Phase 2)

- No fp16-precision dequant (Phase 3, Phase 1 §7.2).
- No scale/bias hoisting (Phase 3, Phase 1 §7.3).
- No new bench script — re-run the Phase 1 sweep harness as-is.
- No prefill / multi-token decode shapes — vector kernel only.
- No single-pass kernel changes for ctx < 1024. The 2-pass kernel
  routes for ctx ≥ 1024 (Phase 1 §A v2 SHIPPED note); it's where the
  measured gap lives. The single-pass kernel can absorb the same
  refactor in a follow-up if needed but has no measured regression.

## Background — what's already shared and what isn't

Read together with `mlx/backend/metal/kernels/sdpa_vector_quantized.h`,
specifically `sdpa_vector_quantized_2pass_1` (lines 283–497). The grid
layout is:

- `tid.x = kv_head_idx` (range: `num_kv_heads`)
- `tid.y = batch_idx`
- `tid.z = block_idx`
- `tidtg.x` ∈ [0, BD=32) → simd_lid within one simdgroup
- `tidtg.y` ∈ [0, gqa_factor) → which query head *within* the GQA
  group this simdgroup serves
- `tidtg.z` ∈ [0, q_seq_len) → q position within the prompt block

Per threadgroup: `BD × gqa_factor × q_seq_len` threads, organized as
`gqa_factor × q_seq_len` simdgroups of 32 lanes each. All simdgroups
in the threadgroup share `kv_head_idx`, `batch_idx`, `block_idx` —
i.e. they iterate over the **same** K/V positions. They differ only in
the query they're computing scores against.

The current inner loop (lines 392–488) has each simdgroup
independently:

1. Load K nibbles for its lane (line 411 — same address as every
   other simdgroup in the threadgroup, since `simd_lid` matches and
   pointer base shares `kv_head_idx`).
2. Dequant K nibbles to floats using shared scale/bias (line 404).
3. Compute QK score against its query (lines 429–433).
4. Update softmax accumulators.
5. Repeat for V.

Steps 1 and 2 produce **bit-identical results** across all simdgroups
in the threadgroup. The device-memory load (step 1) gets L1-cached so
its bytes-moved cost amortizes implicitly. But step 2 — the dequant
ALU — runs `gqa_factor` times per K position because each simdgroup
re-derives the float K for its own QK.

For qwen-gqa: `gqa_factor=7`, so the dequant ALU runs 7× per K
position. Phase 1 §6.2 measured 16 dequant-related ALU ops per K
position (8 for K + 8 for V); 7× redundancy = 112 ALU ops per K
position spent on duplicate work alone, vs ~14 ops of real work.

## Design — share dequant via threadgroup memory

Hoist the dequant out of the per-simdgroup inner loop. One
"producer" simdgroup dequants K (and V) into threadgroup memory; all
simdgroups in the threadgroup read from there.

Concretely, per outer-loop iteration:

```
producer simdgroup (simd_gid == 0 within the threadgroup):
  load K nibbles, dequant to floats, write to k_shared[BD]
  load V nibbles, dequant to floats, write to v_shared[BD]
threadgroup_barrier(mem_threadgroup)
all simdgroups (including the producer):
  read k_shared into thread-local k[qk_per_thread]
  compute QK score against own query, simd_sum
  update softmax max/sum_exp
  read v_shared into thread-local vals[v_per_thread]
  update AV accumulator
threadgroup_barrier(mem_threadgroup)   # before producer writes next iter
```

Threadgroup memory layout (added):

- `threadgroup U k_shared[BD]` — `BD=32` floats = 128 bytes
- `threadgroup U v_shared[BD]` — 128 bytes

Plus the existing `threadgroup U outputs[BN * BD]` (already 4 KB — the
new tiles are noise on top).

### Why one producer simdgroup, not split work

Two natural alternatives:

1. **One producer simdgroup** (chosen). Simple, correct, 1× dequant
   cost amortized across all consumers. Producer has 32 lanes which
   cover the 32-lane K tile perfectly — no per-lane scatter.
2. **Split dequant across simdgroups** — e.g. simdgroup 0 dequants
   K, simdgroup 1 dequants V, others wait. More parallelism in
   theory, but adds a barrier between K-share and V-share, doubling
   barrier traffic. Revisit only if profile shows the producer
   simdgroup is the bottleneck.

Option 1 keeps the producer's lane mapping isomorphic to the existing
kernel (each lane handles `qk_per_thread` nibbles for its own
`simd_lid` slice of head_dim), which makes the diff small.

### Subtlety: producer-simdgroup load amplification

The producer simdgroup does the same per-lane work it currently does
(load 1 packed uint16 + 4 dequant FMAs per K position, for D=128
qk_per_thread=4). Non-producers skip that and instead read 4 floats
from threadgroup memory + the existing QK/softmax/AV work. So:

| Simdgroup | Per-K ALU ops (current) | Per-K ALU ops (new) |
|---|---:|---:|
| Producer | 30 | 30 (unchanged) |
| Non-producer | 30 | **14** (no dequant) |

Threadgroup-wide per-K: was `gqa_factor × 30` ops; becomes `30 +
(gqa_factor − 1) × 14`.

For qwen-gqa (gqa_factor=7): 30 + 6 × 14 = 114 ops, vs 7 × 30 = 210
today. **45% reduction in threadgroup-wide ALU per K position.**

Per-query-head (the latency the user sees, since simdgroups within a
threadgroup execute in parallel on different SIMD units): the
critical path is bounded by the slowest simdgroup, which is the
producer at 30 ops per K. So per-query-head latency drops from 30
ops → 30 ops on the producer's path **but the non-producers no longer
contend for ALU bandwidth with the producer's dequant work**, because
they're idle through the dequant phase (or doing the threadgroup load
+ QK). On a SIMD-bound kernel this should map to ~roughly the
producer's wall-clock cost, i.e. q4 ratio drops from 2.14× toward
~1.0× the fp16 ALU work + barrier overhead.

This is the structural lever that fp16 already enjoys implicitly via
L2-cached K/V loads. We're making the q4 path explicit about it.

### Pass-2 aggregator — unchanged

`sdpa_vector_2pass_2` is type-agnostic and operates on the per-block
partial outputs / sums / maxs that pass-1 writes. Its layout
contract with pass-1 is unchanged (pass-1 still writes one partial
per `(q_batch_head_idx, q_seq_idx, block_idx)` tuple at the same
strides). Verified at `sdpa_vector_quantized.h:366,375-376`. No
pass-2 changes.

### Single-pass kernel

The single-pass kernel (`sdpa_vector_quantized` at lines 16–275)
serves ctx < 1024. It uses a different threadgroup layout
(`q_batch_head_idx = tid.x`, no GQA grouping per threadgroup), so
the same fix doesn't drop in. Phase 2 leaves it as-is — the measured
gap on llama3-8b @ ctx=1024 is 1.12× and on qwen-gqa @ ctx=1024 is
1.25×; tolerable until proven otherwise. If a follow-up wants to
share dequant in single-pass too, it'd need a separate threadgroup
restructure (group by `kv_head_idx` like the 2-pass already does).

## Workload (same as Phase 1, for re-bench)

- **Hardware:** Apple M4 Pro.
- **Mode:** Decode (B=1, seq_q=1).
- **Models:** llama3-8b (Hq=32, Hk=8, D=128), qwen-gqa (Hq=28, Hk=4, D=128).
- **Context lengths:** 1024, 2048, 4096, 8192, 16384, 32768.
- **Quant config:** group_size=64, bits=4.

## Methodology — verification + measurement

### Correctness

- **Existing tests must pass** without tolerance changes:
  `python/tests/test_fast_sdpa.py` — specifically
  `test_quantized_sdpa_vector_matches_dequantized` (atol=rtol=5e-3 from
  `test_fast_sdpa.py:731-732`) and the broader fp16-quality bound
  test. Adding a tg-memory barrier and reordering loads must not
  change the math.
- **Bench correctness gate** (already in
  `benchmarks/python/sdpa_vector_quantized_bench.py`) compares
  `q4_sdpa` against `dq_then_sdpa` per cell at the same tolerance.
  Re-running the Phase 1 sweep is itself a correctness check.

### Performance

Re-run Phase 1 bench unchanged:

```bash
uv run python benchmarks/python/sdpa_vector_quantized_bench.py \
    --out /tmp/sdpa_phase2.csv
```

The Phase 2 win is judged on **two ratios** versus Phase 1's
recorded numbers (Phase 1 report §2):

- **`q4/dq` on qwen-gqa**: must drop below 1.0× at every ctx
  (currently 1.07–1.34×). This is the success criterion.
- **`q4/fp16` on qwen-gqa @ ctx=32768**: should drop from 2.79× to
  ~1.2× or less based on the §6 ALU-ratio math (Phase 2 should
  remove ~6/7ths of the dequant overhead on qwen-gqa).

If `q4/dq` does not drop below 1.0× at every ctx on qwen-gqa, Phase 2
has not landed §7.1's intended structural fix — even if other ratios
improve. Investigate before merging.

## Deliverables

### 1. Kernel changes — `mlx/backend/metal/kernels/sdpa_vector_quantized.h`

Modify `sdpa_vector_quantized_2pass_1` only. Add `k_shared` and
`v_shared` threadgroup arrays, refactor the inner loop to:

1. Producer simdgroup (`simd_gid == 0` within threadgroup) dequants
   K to `k_shared`.
2. Threadgroup barrier.
3. All simdgroups read `k_shared` into thread-local `k[]`, compute
   QK + softmax.
4. Producer dequants V to `v_shared`.
5. Threadgroup barrier.
6. All simdgroups read `v_shared` into thread-local `vals[]`, update
   AV.
7. Threadgroup barrier (so the producer can rewrite `k_shared` next
   iter without racing the non-producers' read).

Note: Metal threadgroup barriers are cheap (single-cycle on Apple
GPUs) compared to ALU savings, but more than zero — keep barrier
count to the minimum necessary.

### 2. No new tests

The existing `test_fast_sdpa.py` and the bench's correctness gate
cover the change. Adding kernel-internal tests for a refactor that
preserves behavior is overkill. If a test failure surfaces during
verification, the right fix is the kernel, not a new test.

### 3. Phase 2 report — `docs/superpowers/reports/2026-04-28-sdpa-phase2-shared-dequant.md`

Sections, mirroring the Phase 1 report's shape:

1. **Setup** — device, MLX commit on this branch, OS / Xcode versions.
2. **Results table** — re-run Phase 1 bench output, both models × all
   ctx, three variants. Table layout identical to Phase 1 §2 so
   diffing is mechanical.
3. **Comparison vs Phase 1** — q4 ratios before/after for both models.
4. **Sanity check** — q4 vs dq_then_sdpa, with the success-criterion
   verdict (yes/no per cell).
5. **Discussion** — does the measured improvement match the §6
   prediction (~6/7ths reduction in dequant overhead on qwen-gqa)? If
   not, what plausibly explains the residual? (Likely candidates:
   barrier overhead, non-producer simdgroup idle time, register
   pressure from `k_shared` reads.)
6. **Phase 3 candidates** — references Phase 1 §7.2 (fp16 dequant) and
   §7.3 (hoist scales) plus anything the Phase 2 measurement turned
   up.

## Test Plan (kernel-level)

- [ ] **Pre-change:** run the existing `test_fast_sdpa.py` SDPA tests
      to establish a clean baseline. Record any pre-existing failures
      so they aren't blamed on Phase 2.
- [ ] **Post-change:** same test run; expect zero new failures and
      zero tolerance changes.
- [ ] **Bench correctness gate:** re-run the Phase 1 sweep; confirm no
      `correctness gate failed` errors at any cell.
- [ ] **Performance:** the success criterion above. Both qwen-gqa
      `q4/dq < 1.0` everywhere and qwen-gqa `q4/fp16 ≤ ~1.5` at
      ctx=32768.

## Commit Plan

On `kv4-sdpa-phase2` (already created off `kv4-sdpa`):

1. `docs: add Phase 2 design (shared K/V dequant)` — this spec.
2. `docs: add Phase 2 implementation plan`
3. `kernel: share K/V dequant across query heads in 2pass_1` — the
   actual kernel diff. Keep this one commit; the change is one
   logical unit.
4. `docs: add Phase 2 measurement report` — Phase 1-bench re-run
   numbers + comparison.

## Open Items (resolve before merging Phase 2)

- Single-pass kernel: do we leave it untouched and accept the small
  q4-vs-fp16 gap at ctx ≤ 512, or backport the same fix? Decide after
  seeing the 2-pass numbers; if 2-pass closes to <1.2× we may not
  need to touch single-pass at all.
- Producer-simdgroup work imbalance: the producer does ~2× the work
  of consumers per K-iter. Is this a problem on M4 Pro's SIMD
  scheduler? Likely no (consumers wait on the barrier anyway), but
  worth measuring producer vs consumer ALU-active time if the Phase
  2 numbers come in below the §6 prediction.
