# NAX g16 Phase 7 — SDPA NAX correctness on g16 (design)

## Goal

Make `mx.fast.scaled_dot_product_attention` produce correct output on Apple gen-16 GPUs (M3/M4 Pro family) by porting the SDPA NAX kernel to use `NAXFrag32` (32×32 fragment, (32,32,32) descriptor) instead of `BaseNAXFrag` (16×16, (16,32,16) descriptor). Achieved by deduplicating the NAX class hierarchy between the gemm and attn trees first, then enabling NAXFrag32 dispatch in SDPA the same way Phase 1+2 did for matmul.

## Symptom

`python/tests/test_fast_sdpa.py::test_sdpa` fails on g16 hardware:

```
test_sdpa (B=1, qsl=20, ksl=20, head_dim=64, n_q_heads=3, n_kv_heads=3,
           mask=None, transpose=False, dtype=mlx.core.float16)
AssertionError: 0.031585693359375 not less than or equal to 0.0003
```

Max diff is ~100× the expected fp16 tolerance — wrong output, not numerical drift.

Diagnostic confirmed the bug: temporarily adding `metal::nax_arch_flavor() != metal::NAXArchFlavor::kG16` to the dispatcher at `scaled_dot_product_attention.cpp:177` makes all 18 SDPA tests pass. So the bug is the NAX path itself on g16, not anything else.

## Root cause

`mlx/backend/metal/kernels/steel/attn/nax.h::BaseNAXFrag` uses `(16, 32, 16)` `matmul2d_descriptor` (lines 401, 473). `device.cpp:840-849` explicitly notes that descriptor is broken on g16 — the very reason Phase 1 introduced `NAXFrag32` for the gemm tree.

Phase 1+2+5+6 ported the gemm tree (`steel/gemm/nax.h`) but `attn/nax.h` is a separate file (forked from gemm/nax.h at some point in upstream history and never re-synced). The SDPA NAX path never saw the porting and continues to use the broken descriptor on g16.

## Why this wasn't caught

- The Phase 6 audit said "SDPA verified clean" — but it only checked transpose handling on `NAXFrag32::load`. SDPA doesn't use `NAXFrag32` at all; the audit verified the wrong thing.
- The bug was upstream MLX (`#2772 Add Neural Accelerator Support`, `#2811 Centralize NAX condition`) and predates our kv4-sdpa work. Upstream presumably tested only on gen 17+ where (16,32,16) works.
- `test_sdpa` was failing on `main` going back at least to Phase 5 validation; the Phase 5 implementer marked it "pre-existing" without investigating.

## Design

### Architecture

Move the canonical NAX class hierarchy (`Role`, `BaseNAXFrag`, `NAXFrag32`, `NAXTile`, `tile_matmad_nax`) into a new shared header `mlx/backend/metal/kernels/steel/nax_common.h`. Both `gemm/nax.h` and `attn/nax.h` become thin re-export headers that include `nax_common.h` plus any tree-specific extensions (none today).

After dedup, Phase 7 mirrors Phase 1+2 for matmul:

1. Parameterize the SDPA NAX kernel on `class NAXFrag_ = mlx::steel::BaseNAXFrag`.
2. Update every `NAXTile::load*` call site in the SDPA kernel to pass explicit `<Role::Left/Right, transpose_b>` template args (Phase 6 hardening removed defaults; quantized was updated, SDPA's separate copy was missed).
3. Allocate per-simdgroup threadgroup scratch when `kPacking==1`, thread through any `store_slice` / `store_safe` / `store_rows` calls.
4. Add `_g16` Metal kernel instantiations using `NAXFrag32` (for SDPA tile shapes that satisfy SM ≥ 32).
5. Append `_g16` to the dispatched kernel name when `nax_arch_flavor() == kG16`.

The (Role, transpose) parameterization from Phase 6 is built in: NAXFrag32 already correctly threads transpose flags into the load descriptor. SDPA inherits correct transpose handling automatically.

### Components

#### 1. `mlx/backend/metal/kernels/steel/nax_common.h` (new)

Single source of truth for the NAX class hierarchy. Contents (copied verbatim from current `gemm/nax.h`):

- `enum class Role { Left, Right };`
- `struct BaseNAXFrag { ... };` (with kPacking==2)
- `struct NAXFrag32 { ... };` (with kPacking==1, store_slice impl from Phase 5, role-aware load from Phase 6)
- `template<...> struct NAXTile { ... };` (with constexpr-if branches on kPacking, no-default Role/transpose params)
- `template<...> tile_matmad_nax(...);`

Header guard `#pragma once`. Namespace `mlx::steel`. Includes `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` and `<metal_stdlib>`.

#### 2. `mlx/backend/metal/kernels/steel/gemm/nax.h` and `mlx/backend/metal/kernels/steel/attn/nax.h`

After dedup:

```cpp
// File: mlx/backend/metal/kernels/steel/{gemm,attn}/nax.h
#pragma once
#include "mlx/backend/metal/kernels/steel/nax_common.h"
```

If any tree-specific extension exists (none today; verified during Task 1's audit), it goes in the tree-local file after the include.

#### 3. `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h`

Add `class NAXFrag_ = mlx::steel::BaseNAXFrag` template parameter at the end of the kernel template list. Mirrors `steel_gemm_fused_nax.h`.

For every `NAXTile::load*` call, pass `<Role::Left, transpose_a>` for Q-loads, `<Role::Right, transpose_b>` for K/V-loads. Audit during implementation: identify each call site (Q@K^T has transpose_b=true; softmax@V has transpose_b=false; both transpose_a=false typically).

When `NAXFrag_::kPacking == 1`, allocate `threadgroup AccumType scratch[WM*WN*32*32]` per simdgroup at kernel scope (mirrors `steel_gemm_fused_nax.h:107-120`). Pass to `gemm_loop` (if SDPA uses one) and to `Ctile.store_slice` / `store_safe` / `store_rows` calls.

#### 4. `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.metal`

Add `instantiate_kernel(..., "_g16", ..., float, mlx::steel::NAXFrag32)` macros for every existing dtype × shape combination (mirrors `steel_gemm_fused_nax.metal`). Drop any instantiation with SM < 32 from the `_g16` set — those would fail `static_assert(SM % NAXFrag_::kFragRows == 0)` since kFragRows=32 for NAXFrag32.

#### 5. `mlx/backend/metal/scaled_dot_product_attention.cpp`

Inside `sdpa_full_self_attention_nax`, after the kname is built, append `_g16` when `metal::nax_arch_flavor() == metal::NAXArchFlavor::kG16`. Mirror the pattern from `matmul.cpp:2080` (Phase 5 gather dispatch).

If after auditing tile shapes we find any SDPA instantiation has SM<32, that case falls through to the non-NAX SDPA path early — same intra-kernel fallback pattern as Phase 5's bm=16.

#### 6. Audit `mlx/backend/metal/kernels/steel/attn/{attn.h,mma.h,loader.h,transforms.h,params.h}`

Read each file. Surface any direct `NAXTile`/`BaseNAXFrag`/NAX-related usage. Bring divergent code up to current parity. If a file is unchanged from upstream and doesn't reference NAX, leave it alone.

### Tests

#### `tools/probe_naxfrag32_sdpa.py` (new)

Modeled on `tools/probe_naxfrag32_transpose.py`. Exercises the actual SDPA call shapes:

- `transpose_b=true` Q@K^T matmul: load Q as Role::Left,false; K as Role::Right,true; mma with bool_constant<true> for transpose_b.
- `transpose_b=false` softmax@V matmul: load Q as Role::Left,false; V as Role::Right,false; mma with bool_constant<false>.
- Both NAXFrag32 and BaseNAXFrag instantiations (the two should give identical results for inputs that fit; if they differ, that's a separate bug).

Each case uses small integer inputs in [-3, 3] so fp32 reductions are exact.

Currently RED for the NAXFrag32 transpose_b=true case (the SDPA bug). Goes GREEN after Phase 7.

#### `python/tests/test_fast_sdpa.py`

All 18 tests must pass on g16 post-Phase-7. The currently-failing `test_sdpa` case must pass.

#### Regression

- `python/tests/test_blas.py` (matmul + gather_mm subsets, plus full): all pass post-dedup. The dedup must not regress the gemm tree.
- `python/tests/test_quantized.py`: all pass. Quantized's `<Role::Left, false>` annotations from Phase 6 must keep working through the new shared header.
- All Phase 1/2/5/6 NAX probes: pass.

### Out of scope

- **Performance**. Correctness-only.
- **Attention vector kernels** (`test_sdpa_vector*`). They already pass; not affected by this fix.
- **Sub-32 SDPA tile shapes**. If any instantiation has SM<32, fall through to non-NAX or BaseNAXFrag variant — defer a proper sub-32 wrapper to a possible Phase 8.
- **Probe/test inline-HEADER deduplication**. The probes still each carry their own inlined NAXFrag32 source; combining those is a follow-up.
- **kv4-sdpa work**. Confirmed innocent.

### Risks and mitigations

1. **Dedup breaks gemm tree compile.** The shared header must export every public symbol gemm tree consumers used. *Mitigation*: rebuild + run `test_blas.py -k matmul -k gather_mm` BEFORE touching the SDPA kernel. If gemm passes, dedup is clean.

2. **`attn/` tree has tree-specific NAX behavior we miss.** The diff vs `gemm/nax.h` was 899 lines but mostly attn/ "missing" gemm/'s additions. *Mitigation*: file-by-file audit (Component 6); preserve any divergence as deliberate.

3. **SDPA kernel uses SM<32 tile shapes.** Could force a partial gate. *Mitigation*: audit instantiations in Task 1 of the plan; design the `_g16` instantiation list around what fits. Document the fallback path explicitly.

4. **Probe coverage gap.** The `attn/` tree might have call patterns the gemm-tree probes don't exercise. *Mitigation*: the new SDPA probe (Component-test 1) replicates the SDPA kernel's exact load+mma sequence.

## References

- Phase 1 design (NAXFrag32 introduction): `docs/superpowers/specs/2026-04-27-nax-g16-fix-plan.md`
- Phase 2 design: `docs/superpowers/specs/2026-04-28-nax-g16-phase2-design.md`
- Phase 5 design: `docs/superpowers/specs/2026-04-30-nax-g16-phase5-design.md`
- Phase 6 design: `docs/superpowers/specs/2026-04-30-nax-g16-phase6-design.md`
- Phase 6 audit (the one that missed SDPA): `docs/superpowers/reports/2026-05-01-nax-g16-transpose-audit.md`
- The "broken descriptor" comment: `mlx/backend/metal/device.cpp:840-849`
- The current SDPA dispatch: `mlx/backend/metal/scaled_dot_product_attention.cpp:177`
