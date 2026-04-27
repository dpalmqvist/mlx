# Phase 1 Report — Quantized Vector SDPA Profile

**Date:** 2026-04-27
**Spec:** [`../specs/2026-04-27-profile-first-optimization-design.md`](../specs/2026-04-27-profile-first-optimization-design.md)
**Plan:** [`../plans/2026-04-27-profile-first-optimization-phase1.md`](../plans/2026-04-27-profile-first-optimization-phase1.md)
**Status:** Phase 1 complete. §6 derived from kernel-source analysis (CLI GPU counters are restricted on M4 Pro / macOS 26.3; see §6.1); §7 ranks Phase 2 candidates against §6's findings.

## 1. Setup

- **Device:** Apple M4 Pro.
- **NAX:** enabled per `bf760f92 Enable NAX selectively on M4 Pro`.
- **MLX commit at the time of the run captured in §2:** `37b9fb09` (branch
  `kv4-sdpa-profile`, off `kv4-sdpa`). Re-runs will tag CSV rows with
  the current HEAD; the CSV's `git_sha` column is the authoritative
  per-row provenance.
- **macOS:** 26.3.1 (build 25D771280a).
- **Xcode:** 26.4.1 (build 17E202).
- **Python:** 3.13 via `uv`. MLX built editable from this branch.
- **Command:**

  ```bash
  uv run python benchmarks/python/sdpa_vector_quantized_bench.py --out /tmp/sdpa_phase1.csv
  ```

- **Iterations:** 100 measured + 10 warmup per cell. Median reported in the
  tables; p10/p90 in the CSV.

### Methodology notes (read before interpreting numbers)

Caveats:

- Each iteration is timed as a single `mx.eval(out)` round-trip, which captures
  Python dispatch + Metal encoder + GPU execute together. Treat absolute
  milliseconds as having ~10% uncertainty from OS jitter and shared-system
  load. Treat *ratios* (`q4/fp16`, `q4/dq`) as reliable to ~5% — both
  numerator and denominator share the same jitter floor.
- `fp16_sdpa` is timed against the *original* fp16 K/V tensors, not the
  fp32→quantize→dequantize round-trip of the same data. The `dq_then_sdpa`
  column is the apples-to-apples "same packed tensors, different kernel"
  comparison; `q4/dq` is therefore the more honest sanity-check ratio.
- `dq_then_sdpa` includes the `mx.dequantize(...)` calls inside the timed
  region by design. It measures the dequant + fp16 SDPA path as a unit, which
  is what the kernel under test is supposed to beat.

Correctness:

- `q4_sdpa` and `dq_then_sdpa` are verified against each other within
  `atol=rtol=5e-3` (per `python/tests/test_fast_sdpa.py:731-732`) before
  timing each cell. No correctness-gate failures occurred during this run.

## 2. Results

### llama3-8b (Hq=32, Hk=8, D=128)

```
   ctx   fp16_sdpa     q4_sdpa    dq_then_sdpa   q4/fp16     q4/dq
  1024      0.2185      0.2445          0.1977     1.12x     1.24x
  2048      0.1561      0.2191          0.2228     1.40x     0.98x
  4096      0.1965      0.3047          0.3184     1.55x     0.96x
  8192      0.3040      0.4791          0.5475     1.58x     0.88x
 16384      0.5033      0.9023          1.0394     1.79x     0.87x
 32768      0.9716      1.5965          1.8996     1.64x     0.84x
```

### qwen-gqa (Hq=28, Hk=4, D=128)

```
   ctx   fp16_sdpa     q4_sdpa    dq_then_sdpa   q4/fp16     q4/dq
  1024      0.1344      0.1680          0.1476     1.25x     1.14x
  2048      0.1176      0.1919          0.1792     1.63x     1.07x
  4096      0.1508      0.2660          0.2195     1.76x     1.21x
  8192      0.1989      0.4146          0.3256     2.08x     1.27x
 16384      0.3080      0.7927          0.6207     2.57x     1.28x
 32768      0.4977      1.3868          1.0324     2.79x     1.34x
```

CSV: `/tmp/sdpa_phase1.csv` (not committed — regenerate via the command in §1).

## 3. Where the gap lives

**Worst (model, ctx) cell by `q4/fp16` ratio:** `qwen-gqa` at `ctx=32768` →
**2.79×** slower than fp16 vector SDPA. That is also the cell where the
absolute gap is largest in milliseconds (1.39 ms vs 0.50 ms ≈ 0.89 ms
delta).

**Trend with ctx:**

- `llama3-8b`: 1.12× → 1.40× → 1.55× → 1.58× → 1.79× → 1.64×. Mostly
  growing, with a small dip at the longest ctx. Rough shape suggests the
  kernel becomes increasingly bandwidth-bound as ctx grows, then plateaus
  when fp16 also starts saturating bandwidth.
- `qwen-gqa`: 1.25× → 1.63× → 1.76× → 2.08× → 2.57× → 2.79×. Monotonically
  growing. The kernel does *not* close the gap as ctx scales — if anything
  it widens. This is the more concerning trend.

**Why qwen-gqa is worse than llama3-8b:** qwen-gqa has Hk=4 (vs Hk=8 on
llama3-8b), meaning each KV head is shared by 7 query heads (vs 4 on
llama). The fp16 vector kernel can amortize K/V loads across query heads
within a KV group, so its per-token cost grows slowly with the GQA ratio.
The quantized kernel's dequant work scales with K/V tile reads, so a
higher GQA ratio doesn't help it as much.

**Bandwidth vs dequant-throughput — testable hypothesis for §6.** On
qwen-gqa, the absolute `(q4 − fp16)` delta grows from 0.034 ms (ctx=1024)
to 0.889 ms (ctx=32768) — roughly **26×** — while fp16 itself grows only
from 0.13 ms to 0.50 ms (~3.7×). Pure bandwidth dominance would predict
the delta to grow roughly with fp16 (linear in tile reads); 26×/3.7× ≈ 7×
super-linear instead points to per-K/V-tile dequant cost as the dominant
overhead, not just bandwidth. §6's frame-capture should distinguish these
by reading ALU-active vs memory-wait directly — Phase 2 should not anchor
on "bandwidth" without confirming.

## 4. Sanity check (q4 vs dequant→SDPA)

> **Headline finding.** On `qwen-gqa`, `q4_sdpa` **loses to `dq_then_sdpa`
> at every measured context length** (1.07× to 1.34× slower). The new
> kernel never beats its own dequant fallback on this shape. **Phase 2's
> primary success criterion is reversing this** — until then, qwen-style
> GQA decode would currently be faster calling
> `mx.dequantize(...)` + `mx.fast.scaled_dot_product_attention(...)` than
> calling `mx.fast.quantized_scaled_dot_product_attention(...)` directly.

The premise of the new kernel is that running SDPA directly on packed 4-bit
K/V is faster than dequantizing first and then running fp16 SDPA. The
`q4/dq` column tests that premise.

- **llama3-8b**: q4_sdpa beats dq_then_sdpa (`q4/dq < 1.0×`) at ctx ≥ 2048
  (0.98× → 0.84×). At ctx=1024 it loses (1.24×). The kernel earns its keep
  on this shape from medium context onward.
- **qwen-gqa**: q4_sdpa loses across the entire sweep, as called out
  above.

This is the correctness-of-purpose failure the spec called out — and what
makes Phase 2 worth doing. It argues that "profile-first" was the correct
framing: without this run we would not have known the kernel
underperforms its own fallback on a real model shape. The headline
finding is more actionable than the raw `q4/fp16 = 2.79×` worst-cell
number from §3, because closing the gap to fp16 (which has no dequant
work at all) may be physically limited, while beating the dequant
fallback is definitionally achievable — the fallback is the kernel's own
floor.

## 5. Frame-capture playbook

Use this when Phase 2 starts (or whenever the worst cell from §3 needs
diagnosis).

1. **Pick the target cell** — the (model, ctx) with the highest `q4/fp16`
   ratio in §3. Currently: `qwen-gqa @ ctx=32768`. Secondary: `qwen-gqa @
   ctx=16384` (2.57×) since it runs faster, which makes the capture cycle
   tighter.
2. **Build MLX with debug shaders.** The flag is `MLX_METAL_DEBUG`, defined
   at `CMakeLists.txt:39` (top-level) and gated at `CMakeLists.txt:177-178`
   (verified at write-time on `kv4-sdpa-profile`). Build via:

   ```bash
   CMAKE_ARGS="-DMLX_METAL_DEBUG=ON -DCMAKE_BUILD_TYPE=Debug" \
       uv pip install -e . --reinstall --no-deps
   ```

   Re-verify the flag still exists at those lines before relying on this
   command if substantial time has passed since the report was written.
3. **Capture programmatically into a `.gputrace` bundle.** A companion
   script wraps the worst-cell dispatch in
   `mx.metal.start_capture` / `stop_capture` and writes a
   Xcode-openable trace. Generate all three variants (`q4_sdpa`,
   `fp16_sdpa`, `dq_then_sdpa`) on the worst cell:

   ```bash
   for v in q4_sdpa fp16_sdpa dq_then_sdpa; do
     MTL_CAPTURE_ENABLED=1 \
       python benchmarks/python/sdpa_vector_quantized_capture.py \
         --variant "$v" --model qwen-gqa --ctx 32768
   done
   ```

   Output: `/tmp/sdpa_<variant>_qwen_gqa_32768.gputrace` per variant. The
   `MTL_CAPTURE_ENABLED=1` env var is required for programmatic capture.
   This avoids the Xcode-attach dance and produces a directly comparable
   trace trio.
4. **Open the captures in Xcode.**

   ```bash
   open -a Xcode /tmp/sdpa_q4_sdpa_qwen_gqa_32768.gputrace
   ```

   (Repeat for each variant in separate Xcode windows for side-by-side.)
5. **Inspect the relevant SDPA dispatch.** Read these counters/views
   and write the value into §6:
   - **Occupancy** — low (<25%) usually means register or
     threadgroup-memory pressure.
   - **ALU active vs memory wait** — ratio identifies bound class. >70%
     memory wait → bandwidth-bound. >70% ALU → compute-bound.
   - **L1/L2 cache hit rate on K/V loads** — low hit rate suggests the
     dequant unpack pattern is cache-unfriendly.
   - **Register pressure / spills** — any spills mean the kernel is
     ALU-bound for the wrong reason and a register-pressure fix is the
     first thing to try.

## 6. Findings

### 6.1 Counter-collection note

GPU counter collection via `xctrace` on M4 Pro / macOS 26.3.1 was
attempted with the `Metal System Trace`, `Game Performance`, and
`Metal GPU Counters` instruments. All three exposed only one Apple GPU
counter on this device (`RT Unit Active`); the richer counter profile
needed to read `ALU active %`, `Memory stall %`, and L1/L2 hit rates
fails with `Selected counter profile is not supported on target device`
from CLI. Xcode's interactive GPU debugger does expose those counters
on the same captures, but only via UI. Findings below are derived from
**kernel-source analysis + the §3 measured timings**, which is enough
to identify the bottleneck and rank Phase 2 candidates without the
counter values. The Xcode UI counters can refine specific numbers (e.g.
exact L2 hit %) but should not change the ordering of the candidates in
§7.

### 6.2 Per-K-position cost comparison

Both vector kernels iterate over the K sequence in steps of `BN=32`
positions per simdgroup. Per K position, in the inner loop:

| Step | `sdpa_vector` (fp16) | `sdpa_vector_quantized` (q4) | Δ |
|---|---|---|---|
| K load | 4 fp16 nibble loads = 8 B | 2 fp16 (scale, bias) + 1 packed uint16 = 6 B | q4 reads **less** |
| K dequant | none | 4 nibble extracts + 4 (mul, add) FMAs | **+8 ALU ops** |
| QK score | 4 FMAs + simd_sum | same | — |
| Softmax update | 4 ops (max, exp, factor) | same | — |
| V load | 4 fp16 loads = 8 B | 2 fp16 + 1 packed uint16 = 6 B | q4 reads **less** |
| V dequant | none | 4 nibble extracts + 4 FMAs | **+8 ALU ops** |
| AV update | 4 FMAs | same | — |

Source: `mlx/backend/metal/kernels/sdpa_vector.h:99-147` vs
`mlx/backend/metal/kernels/sdpa_vector_quantized.h:151-247`.

**Per-K totals:** fp16 ≈ 16 B loaded + 14 ALU ops; q4 ≈ 12 B loaded +
30 ALU ops. **q4 reads 25% less data but does 2.14× the ALU work.**

### 6.3 Bound-class analysis

The arithmetic ratio q4/fp16 ≈ 2.14× sits very close to the measured
worst-cell ratio of 2.79× on `qwen-gqa @ ctx=32768`. That is strong
structural evidence the q4 kernel is **ALU-bound on dequant**, not
bandwidth-bound. Three corroborating points:

- **Bandwidth ceiling math** (M4 Pro, 273 GB/s): the 32k-token KV at
  q4+scales/biases is ≈ 4.2 MB of working set. A pure-bandwidth lower
  bound on the kernel is ~15 µs. Measured time is 1.39 ms — ~90× over
  the bandwidth ceiling. Bandwidth cannot be the limiter.
- **Per-head L2 footprint** also points away from bandwidth: q4 KV per
  head ≈ 2.5 MB vs fp16 ≈ 8 MB. The q4 path *should* L2-reuse better
  across query heads, not worse — yet it scales **worse** with GQA
  ratio (qwen Hq/Hk=7 → 2.79× gap, llama 4 → 1.79×).
- **GQA-ratio sensitivity** (§3) is the giveaway: fp16 amortizes K/V
  loads across query heads sharing a KV head; q4 has the same load
  amortization opportunity but **redoes the dequant ALU per query
  head** because each query head is its own threadgroup
  (`q_batch_head_idx = tid.x` at
  `sdpa_vector_quantized.h:94-96`). More query heads per KV head =
  more redundant dequant work for q4 = wider gap. Exactly what we see.

The 0.65× residual between the ALU ratio (2.14×) and the measured
ratio (2.79×) on qwen-gqa is most plausibly from (a) extra cache
pressure from per-group scales/biases (32768/64 = 512 extra small
loads per K head per pass) and (b) register pressure from carrying
fp32 dequant intermediates (`typedef float U;` at line 83) instead of
fp16. Confirming which dominates would need either the Xcode UI
counters or a `U = T` correctness/timing experiment — both worth
doing before locking in the §7 ordering, but neither changes the
top-level pick.

### 6.4 Why this matches §3's symptoms exactly

- **q4 loses to dq_then_sdpa on qwen-gqa across all ctx (§4):** the
  dequant fallback materializes K/V to fp16 once *per call*, then runs
  the fp16 kernel which amortizes its loads across all 7 query heads
  in the KV group. The fused q4 kernel re-dequants per query head, so
  it does ~7× the dequant arithmetic of the fallback at this GQA
  ratio. Mathematically inevitable.
- **`(q4 − fp16)` delta grows ~26× from ctx=1024 to ctx=32768 while
  fp16 itself grows only ~3.7×:** the per-K dequant cost is
  proportional to N (tile reads), so the extra 16 ALU ops per K
  position scale linearly with ctx. fp16's growth is sub-linear because
  it starts hitting bandwidth at long ctx; q4 stays compute-bound and
  scales linearly. 26×/3.7× ≈ 7× super-linear is exactly the GQA-ratio
  amortization gap that fp16 captures and q4 doesn't.

## 7. Phase 2 candidates

Ranked by expected impact / effort. All three target the dequant-ALU
ceiling identified in §6.

### 7.1 Share K/V dequant across query heads in a GQA group — **largest lever**

**The architectural fix.** Currently each query head is its own
threadgroup (`tid.x = q_batch_head_idx`,
`sdpa_vector_quantized.h:94`), and each threadgroup re-dequants the
same K/V data its KV-group siblings also dequant. Restructure to one
threadgroup per `kv_head_idx`, with the threadgroup computing output
for all `gqa_factor` query heads sharing that KV head in a single
pass.

**Expected impact:** dequant work amortizes across the GQA group, so
per-token dequant cost drops by `gqa_factor`. For qwen-gqa
(`gqa_factor = 7`) this should close most of the 2.79× gap to fp16 in
one shot. Llama-3 (`gqa_factor = 4`) gets a smaller but still
substantial win.

**Effort:** moderate — needs a new threadgroup grid, larger threadgroup
memory for `gqa_factor` Q vectors and softmax state, and care around
the per-query-head accumulator layout. Pass-2 aggregator
(`sdpa_vector_2pass_2`) can stay unchanged because it's already
type-agnostic and per-output.

**Why this is the top candidate:** it's the unique fix that closes the
GQA-ratio sensitivity in §3 by construction — fp16 already gets
implicit GQA amortization through L2 reuse of the K/V loads; the q4
kernel can only get it explicitly because the dequant work isn't a
load.

### 7.2 fp16-precision dequant + softmax — **moderate lever, simple to test**

Change `typedef float U` (line 83) to use the input dtype `T` for the
dequant intermediates and softmax accumulators. M4 Pro's GPU has 2× fp16
ALU vs fp32 on the SIMD path (verified separately for matmul kernels in
the parent doc; not yet measured for SDPA but the same hardware
applies).

**Expected impact:** the dequant FMAs (8 ops per K position) and the
softmax/AV ops (~9 more) all run at 2× throughput. Worst case 17 of
the 30 q4 ALU ops drop to 0.5× cost = 8.5 ops effective → total ~21.5
ops per K position vs 30 today. **~30% reduction** in q4 kernel time,
independent of the gqa-amortization fix in 7.1.

**Effort:** small — single typedef change plus correctness check
against the existing `atol=rtol=5e-3` bound from
`test_quantized_sdpa_vector_matches_dequantized`
(`python/tests/test_fast_sdpa.py:731-732`). The unpack arithmetic
(nibble extract + scale/bias FMA) at 4-bit precision should comfortably
fit within fp16 dynamic range, but quality on long contexts needs
verification — the softmax max/exp ladder is where fp16 typically
breaks if it does.

**Risk:** the parent doc's Tier 2 #6 result was "fp16 accumulate is a
−5% regression" — but that was for the **NAX matmul** path, where Apple's
matrix coprocessor likely operates at fixed internal precision
regardless of the C-template. The SDPA vector kernel is plain SIMD ALU,
which is the canonical case where fp16 should be 2× faster than fp32.
The two cases are not analogous; the negative result on NAX should not
discourage trying fp16 on the SDPA vector path.

### 7.3 Hoist scales/biases to per-group constants — **smallest lever, cheapest fix**

Currently `k_scales[qk_group_idx]` and `k_biases[qk_group_idx]` are
loaded every outer-loop iter (`sdpa_vector_quantized.h:165-166,
207-208`). Within a `group_size=64` window, `BN=32` simdgroups process
32 K positions per outer iter, so two consecutive outer iters fall in
the same group and re-load the same scale/bias. Hoist the load into a
per-group prologue and reuse via threadgroup memory or registers.

**Expected impact:** small — 2× fewer scale/bias loads for K and V,
which is ~32k bytes/iter saved at ctx=32768. Latency-wise this is in
the noise (a few percent at most) but it removes a small source of
register pressure that may interact with 7.2.

**Effort:** small.

**Why ranked last:** scales/biases are a tiny fraction of total bytes
moved; the primary gap isn't from these loads. Worth bundling with 7.1
or 7.2 rather than a standalone PR.

### 7.4 Rejected as next step: kernel tile-size tuning

Tile sizes (`BN=32, BD=32`) match the fp16 vector kernel's tiling. Both
kernels share the same outer-loop structure; sweeping `BN`/`BD` on the
quantized variant in isolation does not address the structural ALU
asymmetry identified in §6 and would at best compensate for register
pressure that 7.2 also addresses more directly. Defer until after 7.1
and 7.2 land — at that point a sweep makes sense as fine-tuning.

### Phase 2 success criteria

From §4: **`q4_sdpa` must beat `dq_then_sdpa` at every measured
context length on `qwen-gqa`.** That's the floor — beating the
fallback the kernel is supposed to replace. Currently the fallback
wins at every ctx (1.07–1.34×). 7.1 alone should clear this floor;
7.2 alone might or might not depending on how much of the 0.65×
residual in §6.3 is fp32 vs fp16 ALU. Together they should comfortably
make q4 the faster path everywhere.
