# NAX g16 Phase 5 — gather_nax port (design)

## Goal

Port `steel_gemm_gather_nax` to dispatch through `NAXFrag32` on g16, restoring NAX-accelerated `gather_mm_rhs` for Apple gen-16 GPUs. Phase 1 introduced `NAXFrag32` (32×32 fragment, kPacking==1) to work around the broken (16,16,16) `matmul2d` descriptor on g16. Phase 2 ported `steel_gemm_fused_nax`, `steel_gemm_splitk_nax`, and `steel_gemm_segmented_nax`. `steel_gemm_gather_nax` remained gated off because it has two extra requirements: `NAXFrag32::store_slice` (still a `static_assert` stub) and `bm=16` instantiations that conflict with NAXFrag32's 32×32 minimum.

## Decision: option A (drop bm=16 on g16)

`steel_gemm_gather_nax` instantiates three `bm` values (`steel_gemm_gather_nax.metal:31-33`):

```
bm=16, BN=128, BK=128, WM=1, WN=4   // SM = 16
bm=32, BN=128, BK=128, WM=1, WN=4   // SM = 32
bm=64, BN=128, BK=128, WM=2, WN=4   // SM = 32
```

The bm rule (`matmul.cpp:2032-2042`) selects bm by tokens-per-expert (`M/E`):

| `M/E` | bm |
|-------|----|
| > 48  | 64 |
| 25-48 | 32 |
| ≤ 24  | 16 |

NAXFrag32 has `kFragRows = kFragCols = 32`. With `BM=16, WM=1`, `SM=16`, so `TM = SM / kFragRows = 16/32 = 0` — impossible. Two paths to enable bm=16 on g16: build a sub-32 staging wrapper (~3-5 days, complex) or drop bm=16 on g16 and let it fall through to the existing non-NAX `gather_mm_rhs` (the same path that runs today). We pick the latter:

- The status quo on g16 is already "all gather runs non-NAX". Dropping bm=16 on g16 is strictly an improvement: bm=32 and bm=64 gain NAX, bm=16 stays at parity.
- bm=16 corresponds to the smallest matmul shape (16×128×K) where the non-NAX Steel GEMM is already efficient — the marginal speedup of adding NAX is smallest precisely where the engineering cost is highest.
- A sub-32 wrapper can be added later (Phase 6) if MoE-decode benchmarks on g16 show meaningful daylight between bm=32+ NAX and the bm=16 non-NAX fallback.

## Components

### 1. `NAXFrag32::store_slice` — implement (replace stub)

**Location**: `mlx/backend/metal/kernels/steel/gemm/nax.h:943`.

Currently:

```cpp
template <typename T, typename... Args>
METAL_FUNC static void store_slice(
    const thread dtype_frag_t<T>&,
    Args...) {
  static_assert(sizeof(T) < 0,
      "NAXFrag32::store_slice not yet implemented; Phase 5 SDPA needs this");
}
```

Replace with the same staged-store pattern as `store_safe`/`store_rows` (lines 845-888):

1. `store(src, scratch, 32)` — write the 32×32 fragment to threadgroup scratch.
2. `threadgroup_barrier(mem_threadgroup)`.
3. Per-thread loop over `kElemsPerFrag` (8 elements per lane on g16): each lane copies its share from scratch to device memory, gated by the rectangular bounds `(r ∈ [start_x, stop_x), c ∈ [start_y, stop_y))`.

The signature must take a `threadgroup T* scratch` parameter (matching `store_safe`/`store_rows`). The element addressing uses `flat = lane * kElemsPerFrag + i`, `r = flat / 32`, `c = flat % 32`, exactly mirroring the existing two methods.

### 2. `NAXTile::store_slice` — thread scratch through

**Location**: `nax.h:1347-1371`.

Today's signature:

```cpp
METAL_FUNC void store_slice(device U* dst, int ld, short2 start, short2 stop) const;
```

Add `threadgroup AccumType* scratch` parameter and pass it down to `NAXFrag_t::store_slice` for the kPacking==1 case. Resolve the TODO at line 1355-1357 ("Phase 5 must redesign this dispatch"). For BaseNAXFrag, the scratch parameter is unused (kPacking==2 doesn't need staging) — same shape as Phase 2's `store`/`store_rows`/`store_safe`.

### 3. `steel_gemm_gather_nax.h` — parameterize on NAXFrag, allocate scratch

**Location**: `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather_nax.h`.

Mirror the Phase 2 splitk/segmented port:

- Add `class NAXFrag_ = BaseNAXFrag` template parameter.
- Compute `TM = SM / NAXFrag_::kFragRows`, `TN = SN / NAXFrag_::kFragCols` (currently hardcoded `/16`).
- When `NAXFrag_::kPacking == 1`, allocate per-simdgroup `threadgroup AccumType sg_scratch[32 * 32]` and pass it as the last argument to `gemm_loop` and to `Ctile.store` / `Ctile.store_slice`. (Same pattern as `steel_gemm_splitk_nax.h:104-109` post-Phase-2.)

The `Ctile.store_slice(C, ld, start, stop, sg_scratch)` calls at lines 120-131 simply gain the trailing `sg_scratch` argument.

### 4. `steel_gemm_gather_nax.metal` — add g16 instantiations

**Location**: `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather_nax.metal:31-33`.

Add `instantiate_..._g16` macros (using `NAXFrag32`) for **bm=32 and bm=64 only** — not bm=16. Pattern matches Phase 2's `steel_gemm_splitk_nax.metal` and `steel_gemm_segmented_nax.metal`. Existing `BaseNAXFrag` instantiations stay (used on non-g16).

### 5. `matmul.cpp` dispatcher — gate by bm, not by arch flavor

**Location**: `mlx/backend/metal/matmul.cpp:2424-2434`.

Today:

```cpp
if (metal::is_nax_available() &&
    metal::nax_arch_flavor() != metal::NAXArchFlavor::kG16 &&
    (env::enable_tf32() || a.dtype() != float32)) {
  return gather_mm_rhs_nax(a, b, rhs_indices, out, d, s);
}
gather_mm_rhs(a, b, rhs_indices, out, d, s);
```

After:

```cpp
if (metal::is_nax_available() &&
    (env::enable_tf32() || a.dtype() != float32)) {
  const bool g16 =
      metal::nax_arch_flavor() == metal::NAXArchFlavor::kG16;
  // On g16 the gather kernel uses NAXFrag32, which requires bm >= 32.
  // The bm rule inside gather_mm_rhs_nax picks bm=16 when M/E <= 24
  // (see matmul.cpp:2032-2042). In that case, fall through to the non-NAX
  // path. bm=32 and bm=64 are NAX-accelerated on g16.
  const int K = a.shape(-1);
  const int M = a.size() / K;
  const int E = b.shape(0);
  const bool g16_skip = g16 && (M / E <= 24);
  if (!g16_skip) {
    return gather_mm_rhs_nax(a, b, rhs_indices, out, d, s);
  }
}
gather_mm_rhs(a, b, rhs_indices, out, d, s);
```

The duplicated `M/E <= 24` rule is intentional and small. Add a cross-reference comment at the dispatcher and at the bm-selection inside `gather_mm_rhs_nax`.

## Data flow

```
gather_mm_rhs_nax kernel
  ├── allocates threadgroup AccumType sg_scratch[32 * 32]   (kPacking==1 only)
  └── inner loop over RHS gather indices
        ├── gemm_loop(... sg_scratch)                       [unchanged from Phase 2]
        └── if (offset_next - offset == SM):
              Ctile.store(C, ld, sg_scratch)
            else:
              Ctile.store_slice(C, ld, start, stop, sg_scratch)
                └── NAXTile::store_slice(..., scratch)
                    └── NAXFrag32::store_slice(..., scratch)
                        ├── store(src, scratch, 32)         [stage to threadgroup]
                        ├── threadgroup_barrier(mem_threadgroup)
                        └── per-thread bounded copy from scratch to device
```

## Tests

### New probe: `tools/probe_naxfrag32_store_slice.py`

Compares `NAXFrag32::store_slice` output against `BaseNAXFrag::store_slice` on identical fragments across rectangular slice boundaries:

- Full slice (start=(0,0), stop=(32,32)) — should match `store`.
- Row-only slice (e.g. start=(0,8), stop=(32,24)) — first/last rows clipped.
- Column-only slice (e.g. start=(4,0), stop=(28,32)) — first/last cols clipped.
- Both axes clipped (e.g. start=(4,8), stop=(28,24)).
- Empty slice (start == stop) — no writes.
- Slice with non-zero `off_x`/`off_y` (multi-frag tile case).

Each case writes to a device buffer pre-filled with a sentinel; verify untouched bytes still hold the sentinel and touched bytes hold the expected frag values.

### Regression sweep

Extend `tools/repro_kge_bk_bug.py` style — drive `mx.gather_mm` with shapes that hit each bm path on g16:

| M/E | bm chosen | g16 path |
|-----|-----------|----------|
| 200 | 64 | NAX (NAXFrag32) |
| 32  | 32 | NAX (NAXFrag32) |
| 8   | 16 | non-NAX fallback |

For each shape, compare against numpy reference at fp32 (existing `test_gather_mm_*` rtol).

### Existing tests

`python/tests/test_blas.py` — `test_gather_mm_*` and `test_gather_mm_sorted*` must pass on g16 with the new dispatch.

## Risks and mitigations

1. **store_slice off-by-one in bound order**. NAXTile passes `(start, stop)` as `short2(col, row)` swapped — easy to mis-thread between `start_x/stop_x` (rows) and `start_y/stop_y` (cols) at the NAXFrag32 layer. Mitigate with the probe (which has asymmetric row/col slices that catch a swapped axis).
2. **Dispatcher rule duplication drift**. The `M/E <= 24` boundary is now in two places (dispatcher and `gather_mm_rhs_nax`). Mitigate with cross-reference comments at both sites.
3. **Scratch lifetime across the gather index loop**. Gather maintains `Ctile` accumulator across multiple index iterations. Scratch is reused across iterations (read after barrier in `store_slice`), so it must be declared at simdgroup scope for the kernel lifetime — not inside the index loop. Same pattern as `steel_gemm_splitk_nax.h` post-Phase-2.

## Out of scope

- **Sub-32 staging wrapper for bm=16 on g16** — Phase 6 if MoE-decode benchmarks justify it.
- **Future SDPA store_slice usage** (the original TODO author's note at `nax.h:935`) — same impl will serve, no extra work needed.

## References

- Phase 2 design: `docs/superpowers/specs/2026-04-28-nax-g16-phase2-design.md`.
- Phase 2 K>=bk findings (root cause of why splitk/segmented/gather all needed porting): `docs/superpowers/reports/2026-04-30-nax-g16-kge-bk-findings.md`.
- Phase 1 NAXFrag32 introduction: `mlx/backend/metal/kernels/steel/gemm/nax.h:540-1000` (the NAXFrag32 struct).
