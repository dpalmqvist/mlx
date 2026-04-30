# NAX g16 — Phase 2 design (steel_gemm_fused integration)

**Status:** brainstormed 2026-04-28, refined in light of Phase 1 discoveries.
**Supersedes the Phase 2 portion of:** `docs/superpowers/specs/2026-04-27-nax-g16-fix-plan.md` and the Phase 2 section of `docs/superpowers/plans/2026-04-27-nax-g16-fix.md` (the original spec was written before Phase 1 surfaced the scratch-threading constraint).
**Hardware target:** Apple M4 Pro (`applegpu_g16s`).
**Branch / worktree:** `nax-g16-phase2` at `~/.config/superpowers/worktrees/mlx/nax-g16-phase2`.

## Goal

Re-enable NAX (`is_nax_available() == true`) on g16s for **dense fused matmul only**, by routing `steel_gemm_fused_nax` through the new `NAXFrag32` (32×32 frag, `(32, 32, 32)` descriptor) added in Phase 1. Other NAX-using paths (splitk, gather, segmented gemm, qmm, SDPA) remain gated off on g16 until Phase 3+.

Success criterion: `python/tests/test_blas.py` and `MLX_METAL_NO_NAX=1 python/tests/test_blas.py` both pass on M4 Pro, with no regression on the g17+ fused path (it continues to use `BaseNAXFrag` via the default template arg).

## Why a fresh design instead of executing the existing plan

Phase 1 left five `// TODO(nax-g16-fix)` markers in `NAXTile`'s safe/rows/slice methods because Metal forbids local `threadgroup T buf[N]` declarations inside non-kernel device functions. The original plan's Task 1.5 sketched a "scratch staging" idea but never resolved how the buffer flows from kernel scope down to per-frag staging. This document fixes that.

A pre-design probe (`tools/probe_mpp_bounded_view.py`) also ruled out an alternative path: MPP's `tensor_inline` does **not** honor extents as bounds for `cooperative_tensor.load` — sentinels in the buffer past the declared extent leak into the loaded cooperative tensor (max|err| 50–100 vs the truncated reference). MPP's `tensor_handle` would bounds-check, but it cannot be constructed from a raw pointer inside a kernel (compiler error: "candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 2 were provided") and rejects threadgroup memory. Therefore in-kernel safe I/O on a raw pointer **must** stage or hard-code per-element bounds.

We picked staging at the NAXTile level (Approach B in the brainstorm) because it keeps NAXFrag32's per-thread layout invisible outside `nax.h` and reuses the contiguous-threadgroup `load`/`store` already validated in Phase 1.

## Architecture

The contract:

- **NAXFrag32** owns its layout, per-thread element ordering, `mma`, and (per Phase 1, commit `47379949`) the `load_safe` / `load_rows` / `store_safe` / `store_rows` family. The safe/rows methods take a `threadgroup T* scratch` parameter and stage device↔scratch internally using a row-per-lane fill pattern, then call the contiguous threadgroup `load`/`store` for the cooperative-tensor round-trip. **(Phase 2 adds device-pointer overloads of the contiguous `load`/`store` for the aligned path; the safe/rows surface is unchanged from Phase 1.)**
- **NAXTile** dispatches on `kPacking`. For `kPacking == 1` (NAXFrag32) callers, every method takes a `threadgroup T* scratch = nullptr` parameter and forwards it to the corresponding NAXFrag32 method — including the previously implicit-TODO aligned `load`/`store` whose signatures don't currently match NAXFrag32 either.
- **The kernel** owns scratch sizing, allocation, and per-simdgroup base-pointer computation. It also owns the template parameterization on the frag class.
- **The host dispatcher** owns kernel-name selection — append `_g16` to the kname when `nax_arch_flavor() == kG16`.

**Reconciliation with the implementation plan:** Section 2 of this spec (NAXTile dispatch) was originally drafted assuming staging happened at the NAXTile layer. Phase 1 already located the staging inside NAXFrag32's safe methods, so the corresponding plan task forwards scratch through NAXTile to NAXFrag32 rather than open-coding the lane-cooperative copy at the NAXTile layer. The kernel allocator, the call sites, and the validation gates are all unchanged.

Five files change, in dependency order:

1. `mlx/backend/metal/kernels/steel/gemm/nax.h` — NAXFrag32 device overloads + NAXTile dispatch on `kPacking`.
2. `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.h` — kernel parameterized on `NAXFrag_`; scratch alloc; `gemm_epilogue` kPacking==1 staging.
3. `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.metal` — `_g16` instantiations (subset of tile tuples that satisfy 32-divisibility).
4. `mlx/backend/metal/device.{h,cpp}` — `NAXArchFlavor` enum, `nax_arch_flavor()` helper, relax `is_nax_available()` gate.
5. `mlx/backend/metal/matmul.cpp` (+ `kernels.h`, `jit_kernels.cpp`, `nojit_kernels.cpp` if the kname is built internally) — `_g16` suffix wiring; non-fused paths gated `flavor != kG16`.

No other files change. No host-side API surface changes.

## Component design

### 1. NAXFrag32 device-pointer overloads

Two new methods, each a one-line address-space variant of an existing threadgroup overload.

```cpp
template <typename T, typename U>
METAL_FUNC static void load(
    thread dtype_frag_t<T>& dst,
    const device U* src,
    const short ld);

template <typename T, typename U>
METAL_FUNC static void store(
    const thread dtype_frag_t<T>& src,
    device U* dst,
    const short ld);
```

Body: identical to the existing `threadgroup` overloads — construct a `tensor_inline` view over the raw pointer with extents `(32, ld)`, get a left-input cooperative tensor from a `(32, 32, 32)` descriptor, `cT.load(view)` (or `cT.store(view)` for store), copy `cT[i]` into / out of the per-thread frag.

Address-space mismatch: the existing `tensor<threadgroup const U, ...>` becomes `tensor<device const U, ...>` (and similarly for store). Phase 1's bounded-view probe T1 already proved the `(32, 32, 32)` descriptor with full extents over a device pointer gives 0 max|err|, so this is a known-safe addition.

Validation: extend `tools/probe_nax_frag32.py` with `test_mma_via_dv_load_store` — same body as the existing `test_mma_via_tg_load_store` but using device pointers. Pass criterion: max|err| < 1e-3.

### 2. NAXTile dispatch on `kPacking` — the five TODO sites

Each of the five methods (`load`, `load_safe`, `load_rows`, `store`, `store_safe`, `store_rows`, plus `store_slice` which is deferred) gains an `if constexpr (NAXFrag_t::kPacking == 1)` branch and a trailing `threadgroup U* scratch = nullptr` parameter.

**Lane mapping for staging:** simdgroup is 32 lanes, frag is 32 rows × 32 cols. Each lane is responsible for one row — lane `r` copies row `r` of the 32-element-wide frag tile. This uses every lane, requires no cross-lane shuffling, and produces stride-1 accesses along the column dimension which is coalescing-friendly for the device side.

**Barrier discipline:** before each store-into-scratch, after each load-from-scratch — every per-frag iteration costs two `threadgroup_barrier(mem_flags::mem_threadgroup)`. Acceptable for Phase 2; perf tuning is out of scope.

**store_safe (illustrative — store_rows drops the col bound, load_safe and load_rows mirror in reverse with zero-fill):**

```cpp
template <typename U>
METAL_FUNC void store_safe(
    device U* dst, const int ld, const short2 dst_tile_dims,
    threadgroup U* scratch = nullptr) const {
  if constexpr (NAXFrag_t::kPacking == 1) {
    const ushort lane = simd_lane_id();
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        constexpr short m_off = idx_row.value * NAXFrag_t::kFragRows;
        constexpr short n_off = idx_col.value * NAXFrag_t::kFragCols;

        // 1. Drop the 32x32 frag into scratch (contiguous, ld=32).
        NAXFrag_t::store(frag_at<idx_row.value, idx_col.value>(),
                         scratch, /*ld=*/short(32));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Lane r writes row r of scratch -> device with bounds.
        const short row_lim = max(short(0), short(dst_tile_dims.y - m_off));
        const short col_lim = max(short(0), short(dst_tile_dims.x - n_off));
        if (lane < row_lim) {
          STEEL_PRAGMA_UNROLL
          for (short c = 0; c < 32; ++c) {
            if (c < col_lim) {
              dst[(m_off + lane) * ld + n_off + c] = scratch[lane * 32 + c];
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      });
    });
  } else {
    // existing kPacking==2 body, unchanged.
  }
}
```

**load_safe / load_rows:** mirror the above. Lane `r` writes row `r` of scratch from device with `(r < row_lim && c < col_lim) ? src[...] : T(0)` for zero-fill on out-of-bounds, then `NAXFrag_t::load(scratch, 32)`.

**store_rows / load_rows:** drop the column bound (`col_lim` is always 32).

**Aligned `load` and `store`:** also need an `if constexpr` branch since NAXFrag32 has no offset/stride args. The kPacking==1 branch computes a per-frag base pointer (`dst + m_off * ld + n_off`) and calls `NAXFrag_t::store(device_ptr, ld)` directly — **no scratch, no barrier, no staging**. Fast path for fully-aligned tiles.

**store_slice:** under kPacking==1, becomes
```cpp
static_assert(false,
    "NAXTile::store_slice not implemented for NAXFrag32 (kPacking==1); "
    "Phase 5 SDPA migration must add it.");
```
Compile-time fail if any caller surfaces in Phase 2; Phase 5 (SDPA) lifts it.

**BaseNAXFrag callers unaffected:** every existing call site passes the existing positional args; the new trailing `threadgroup U* scratch` param defaults to `nullptr`, which the kPacking==2 branch never reads.

### 3. Kernel scratch sizing (`steel_gemm_fused_nax.h`)

Three changes inside the `gemm` kernel:

**(a) Add a frag-class template parameter** with `BaseNAXFrag` as the default so g17+ instantiations stay bit-identical:

```cpp
template <typename T, int BM, int BN, int BK, int WM, int WN,
          bool transpose_a, bool transpose_b,
          typename AccumType = float,
          class NAXFrag_ = mlx::steel::BaseNAXFrag>     // NEW
[[kernel, max_total_threads_per_threadgroup(WM*WN*32)]]
void gemm(...);
```

**(b) Compute TM/TN from the frag class:**

```cpp
constexpr short TM = SM / NAXFrag_::kFragRows;
constexpr short TN = SN / NAXFrag_::kFragCols;
```

For BaseNAXFrag (kFragRows=kFragCols=16), this is the existing formula. For NAXFrag32 (32, 32), the kernel now requires `SM, SN >= 32 && SM % 32 == 0 && SN % 32 == 0`. Tile tuples violating this are not instantiated as `_g16` (Section 4 below).

**(c) Conditionally allocate scratch:**

```cpp
if constexpr (NAXFrag_::kPacking == 1) {
  threadgroup AccumType scratch_buf[WM * WN * 32 * 32];
  threadgroup AccumType* scratch = scratch_buf + simd_group_id * 1024;
  // ... pass `scratch` to Dtile.{load,store}_{safe,rows} and to gemm_epilogue.
} else {
  // existing g17+ body, no scratch alloc — zero threadgroup-memory cost.
}
```

The `if constexpr` keeps the g17+ path's threadgroup-memory budget unchanged. Sizing on the g16 path: `WM * WN * 1024 * sizeof(AccumType)`. For typical `WM=WN=2` and `AccumType=float`, that's 16 KB — within Apple GPU's 32 KB threadgroup budget but tracked as a concrete cost. If a g16 instantiation hits the budget ceiling, we'd shrink the (WM, WN) for that kernel; not anticipated for Phase 2 tuples.

**(d) gemm_epilogue staging:** under `if constexpr (NAXFrag_::kPacking == 1)`, the epilogue receives `scratch` as a new param, stages C frag-by-frag into it (lane-r-copies-row-r with bounds), calls `NAXFrag_::load(scratch, 32)` for the addmm registers, and the final write goes through `Dtile.store_safe(D, ldd, dims, scratch)` (which itself reuses the same scratch buffer). One `threadgroup` allocation feeds both load-C and store-D.

**Type compatibility of the shared scratch:** the buffer is sized in `AccumType` (typically `float`). When loading C of input type `T` (e.g. `half`), the staging code casts the scratch pointer to `(threadgroup T*)` for the lane-cooperative copy and the `NAXFrag_::load(scratch, 32)` call. This is byte-safe as long as `sizeof(AccumType) >= sizeof(T)`, which holds for the supported dtype combinations (half/bfloat→float, half/bfloat→half, float→float). The kernel asserts this with a `static_assert(sizeof(AccumType) >= sizeof(T))` inside the kPacking==1 branch.

### 4. `_g16` instantiations (`steel_gemm_fused_nax.metal`)

For each existing tile tuple `(T, BM, BN, BK, WM, WN, transpose_a, transpose_b, AccumType)`, add a sibling that explicitly passes `NAXFrag32` as the `NAXFrag_` template arg and appends `_g16` to the exported kernel name.

**Tile-divisibility filter (must be applied during Task 2.4):** skip any tuple where `(BM/WM) % 32 != 0` or `(BN/WN) % 32 != 0`. The exact set of skipped tuples comes from a one-shot audit during the implementation task; this design just specifies the rule.

**Compile-time cost:** doubles the number of fused-nax kernel instantiations. Acceptable.

**Symbol verification:** `nm build/.../libmlx.dylib | grep _g16` must list the new symbols after build (Task 2.4 step).

### 5. Dispatcher and host-side wiring

**`device.h`:**

```cpp
enum class NAXArchFlavor { kNone, kG16, kG17Plus };
bool is_nax_available();
NAXArchFlavor nax_arch_flavor();
```

**`device.cpp`:**

- Relax the gate from `gen >= (arch == 'p' ? 18 : 17)` to `gen >= 16` in `is_nax_available()`, with a comment linking to this spec.
- Implement `nax_arch_flavor()` as a lazy-init static returning `kNone` when NAX is disabled, `kG16` when `gen == 16`, `kG17Plus` when `gen > 16`.

**`matmul.cpp`:**

- In `steel_matmul_regular_axpby_nax`, when `nax_arch_flavor() == kG16`, append `_g16` to the kname string before passing it to `get_steel_gemm_fused_nax_kernel`. **Build the suffix at the kname-string level** — do not push the flavor parameter down into the kernel-builder. Smallest change.
- **Tile-fallback strategy:** for Phase 2, restrict the g16 path to a single known-good tile tuple (the largest one in the existing instantiation list that satisfies 32-divisibility). The dispatcher picks that tuple unconditionally on g16. Concrete tuple chosen during Task 2.5 once the audit completes. Phase 3+ broaden.

**Non-fused gating** (Task 2.7 — gates partial Phase 2 from breaking other paths on g16):

For each NAX use-site outside `steel_matmul_regular_axpby_nax`, AND `nax_arch_flavor() != NAXArchFlavor::kG16` into the existing `is_nax_available()` check. Concrete sites:

- `matmul.cpp` — splitk, gather, segmented gemm.
- `quantized.cpp` — qmm_nax, fp_quantized_nax.
- `scaled_dot_product_attention.cpp` — steel_attention_nax.

Each gets a one-line comment linking to this spec: "Phase 3+ will land NAXFrag32 plumbing for this path."

### 6. Validation gates

Run in this order. Halt and surface to the human partner on any FAIL.

1. `python tools/probe_nax_descriptor.py` — baseline descriptor sweep, no change. Sanity.
2. `python tools/probe_nax_frag32.py` — Phase 1 tests + new `test_mma_via_dv_load_store` (Task 2.0). Validates NAXFrag32 device-pointer overloads.
3. **Aligned smoke test** (after Task 2.5) — small `A @ B` of 32-divisible shapes on g16. First end-to-end `_g16` correctness check.
4. `python python/tests/test_blas.py` — covers unaligned shapes, exercises the lane-cooperative scratch staging from Section 2.
5. `MLX_METAL_NO_NAX=1 python python/tests/test_blas.py` — non-regression on the non-NAX path.
6. After Task 2.7: `python python/tests/test_quantized.py` and `python python/tests/test_fast_sdpa.py` — confirm the gates we added force a fallback to non-NAX on g16 for these paths (else the partial Phase 2 produces wrong output for them).

`tools/probe_mpp_bounded_view.py` is committed alongside this spec for the record. It is not a gate — its result is permanent.

## What is intentionally not in Phase 2

- `NAXFrag32::store_slice` (Phase 5 SDPA needs it).
- `_g16` instantiations for tile tuples failing 32-divisibility (broadened in a later phase if perf demands).
- Re-enabling NAX for splitk / gather / segmented / qmm / SDPA on g16 (Phases 3, 4, 5).
- Removing the `flavor != kG16` gate added in Section 5 (Phase 6).
- Performance measurement vs g17+ baseline (Phase 6).

## Open risks

1. **Threadgroup-memory pressure** on the g16 path. `WM*WN*1024*sizeof(AccumType)` on top of the existing steel-gemm staging. If a real instantiation hits the 32 KB ceiling, Task 2.4 shrinks (WM, WN) for `_g16`. Symptom would be a Metal compile error citing threadgroup-memory size; mitigation is documented above.
2. **Per-element-ordering correctness for `cT.load` over a contiguous 32×32 threadgroup buffer.** Phase 1's `test_mma_via_tg_load_store` validated this register→tg→register, but the staging pattern in Section 2 is the first time we'll write **selectively-masked** data into the 32×32 buffer and round-trip through `cT.load`. If `cT.load` reorders / packs elements in a way that depends on bytes outside `(row_lim, col_lim)` of the 32×32 region, zero-fill must cover that region too. Mitigation: lane-r-zeros-its-row covers all 1024 elements, so any element `cT.load` reads is either valid data or zero. This should be safe by construction; if T2 of the bounded-view probe somehow indicates otherwise we revisit.
3. **Mixed-precision dtype combinations** (e.g., half A → float C accum). Phase 1's `NAXFrag32::load` uses `get_left_input_cooperative_tensor<T, T, T>()` — same-type for all three slots. If a real test_blas case instantiates with mismatched left/right/dest types, the cooperative tensor's per-thread layout may not match what NAXFrag32 expects. Mitigation: surfaces during Task 2.4 build or Task 2.6 test_blas — if that error appears, NAXFrag32::load needs explicit `<AType, BType, CType>` template args threaded through.

## Mapping to plan tasks

This design will be implemented as Phase 2 tasks 2.0–2.7 in the implementation plan that supersedes the Phase 2 section of `2026-04-27-nax-g16-fix.md`:

- 2.0 — NAXFrag32 device-pointer overloads + probe_nax_frag32 extension.
- 2.1 — NAXArchFlavor enum, nax_arch_flavor(), relax is_nax_available gate.
- 2.2 — NAXTile dispatch on kPacking==1 (the five TODO sites + aligned `load`/`store`).
- 2.3 — steel_gemm_fused_nax.h kernel parameterization + scratch alloc + gemm_epilogue staging.
- 2.4 — _g16 instantiations in steel_gemm_fused_nax.metal (with tile-divisibility audit).
- 2.5 — matmul.cpp dispatcher kname suffix + smoke test.
- 2.6 — test_blas.py validation (NAX + MLX_METAL_NO_NAX=1).
- 2.7 — gate non-fused NAX paths off on g16 + non-regression spot-checks.

Phase 2 final validation runs the matrix from §6 and lands the result.
