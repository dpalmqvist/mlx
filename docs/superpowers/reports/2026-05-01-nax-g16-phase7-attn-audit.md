# Phase 7 attn/ tree audit

Captures NAX usage in `mlx/backend/metal/kernels/steel/attn/` for the
Phase 7 SDPA-on-g16 port. Used by Task 2 (dedup) and Task 4 (kernel
parameterization).

## attn/nax.h vs gemm/nax.h diff

- **Diff size:** 899 lines total (`attn/nax.h` is 887 lines; `gemm/nax.h` is 1535 lines)
- **Tree-specific code in attn/ to preserve post-dedup:**

  The `+` lines in the diff (code that exists in `attn/nax.h` but NOT in
  `gemm/nax.h`) are the old, role-unaware `NAXTile` method signatures:

  | Method | attn/nax.h signature | gemm/nax.h signature |
  |---|---|---|
  | `load` | `load(src, ld)` — no role/transpose | `load<role,transpose>(src, ld, scratch)` |
  | `store` | `store(dst, ld)` — no scratch | `store(dst, ld, scratch)` |
  | `load_rows` | `load_rows(src, ld, n_rows)` — no role/scratch | `load_rows<role,transpose>(src, ld, n_rows, scratch)` |
  | `load_safe` | `load_safe(src, ld, src_tile_dims)` — no role/scratch | `load_safe<role,transpose>(src, ld, src_tile_dims, scratch)` |
  | `store_rows` | `store_rows(dst, ld, n_rows)` — no scratch | `store_rows(dst, ld, n_rows, scratch)` |
  | `store_safe` | `store_safe(dst, ld, dst_tile_dims)` — no scratch | `store_safe(dst, ld, dst_tile_dims, scratch)` |
  | `store_slice` | `store_slice(dst, ld, start, stop)` — no scratch | `store_slice(dst, ld, start, stop, scratch)` |

  Also: `tile_matmad_nax` in attn has only the `kPacking==2` branch (no
  `kPacking==1` / NAXFrag32 branch).

  These are NOT semantically new functionality unique to attn — they are
  simplified versions of gemm's methods, valid only for BaseNAXFrag
  (kPacking==2). They represent the call signatures the existing SDPA kernel
  uses (verified: `steel_attention_nax.h` calls `Qtile.load(...)`,
  `Ktile.load(...)`, `Vtile.load_rows(...)`, `Vtile.load(...)`,
  `Otile.store_rows(...)`, `Otile.store(...)` without role or scratch args).

- **Verdict:** `attn/nax.h` is a **strict subset** of `gemm/nax.h` in terms of
  NAXFrag capability (only BaseNAXFrag, no NAXFrag32, no Role enum, no kPacking).
  The "attn-only" signatures are simpler method overloads that call BaseNAXFrag
  directly. **Dedup is safe as a pure re-export of gemm/nax.h**, but the existing
  SDPA kernel (`steel_attention_nax.h`) calls the simpler role-unaware signatures
  that exist in `attn/nax.h` but not in `gemm/nax.h`. Task 4 must either:
  (a) add role-unaware overloads back to the shared header, or
  (b) update all call sites in `steel_attention_nax.h` to pass explicit
  `role`/`transpose` args (required for NAXFrag32 anyway — so (b) is the right
  path for the g16 port).

## SDPA kernel call sites (steel_attention_nax.h)

All NAXTile declarations and their load/store usage in
`mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h`:

| Line | Tile | Type | Dims (TileRows×TileCols) | Use | Role | Transpose | Notes |
|---|---|---|---|---|---|---|---|
| 143 | `otile_t` = `NAXTile<AccumType, TQ, TD>` | AccumType (float) | TQ×TD = 1×(BD/kU) | Output accumulator O | — | — | Declared once, accumulated over all KB blocks |
| 201 | `stile_t` = `NAXTile<AccumType, TQ, TK>` | AccumType (float) | TQ×TK = 1×(BK/kU) | Score S = Q@K^T | — | — | Per-KB, cleared each iteration |
| 212 | `Qtile` = `NAXTile<T, 1, 1>` | T (input dtype) | 1×1 | Q tile for one (iq, id) step | Left | false | Loaded row-by-row |
| 213 | `Ktile` = `NAXTile<T, 2, 1>` | T (input dtype) | 2×1 | K tile for one (ik, id) step; TileRows=2 because MMA reads 2 K frags | Right | true | K^T transpose happens in the mma call (transpose_b=true) |
| 315 | `mtile_t` = `NAXTile<melem_t, TQ, TK>` | melem_t (MaskType or bool) | TQ×TK | Attention mask M | Left | false | Loaded from device mask buffer; added to S |
| 429 | `Vtile` = `NAXTile<T, 1, 2>` | T (input dtype) | 1×2 | V tile for one (ik, id) step; TileCols=2 because MMA reads 2 V frags | Right | false | V is not transposed |

Load/store call sites:

| Line | Tile | Call | Condition | Role | Transpose | Notes |
|---|---|---|---|---|---|---|
| 219–222 | `Qtile` | `Qtile.load_rows(Q+off, Q_stride, lim_rows_q - iq*kU)` | `!align_Q && is_last_q` | Left | false | Partial Q tile (last block) |
| 224 | `Qtile` | `Qtile.load(Q+off, Q_stride)` | otherwise | Left | false | Full Q tile |
| 228–231 | `Ktile` | `Ktile.load_rows(K+off, K_stride, lim_rows_k - ik*kU)` | `!align_K && is_last_k` | Right | true | Partial K tile (last block); K role=Right, transpose=true |
| 233 | `Ktile` | `Ktile.load(K+off, K_stride)` | otherwise | Right | true | Full K tile |
| 326–332 | `mfrag` (via `mtile_t::NAXFrag_t::load`) | `mtile_t::NAXFrag_t::load(mfrag, mask, M_stride, Int<1>{}, row_pos, col_pos)` | in-bounds mask | Left | false | Direct NAXFrag_t::load (not NAXTile wrapper) |
| 355–363 | `mfrag` (via `mtile_t::NAXFrag_t::load_safe`) | `mtile_t::NAXFrag_t::load_safe(mfrag, mask, M_stride, Int<1>{}, qL, kL, row_pos, col_pos)` | out-of-bounds mask | Left | false | Direct NAXFrag_t::load_safe (not NAXTile wrapper) |
| 433–437 | `Vtile` | `Vtile.load_rows(V+off, V_stride, lim_rows_k - ik*kU)` | `!align_K && is_last_k` | Right | false | Partial V tile (last block); V role=Right, transpose=false |
| 439 | `Vtile` | `Vtile.load(V+off, V_stride)` | otherwise | Right | false | Full V tile |
| 478 | `Otile` | `Otile.store_rows(O, O_stride, lim_rows_q)` | `!align_Q && is_last_q` | — | — | Partial output store |
| 480 | `Otile` | `Otile.store(O, O_stride)` | otherwise | — | — | Full output store |

**Total NAXTile load/store sites: 10** (2 for Q, 2 for K, 2 for mask via NAXFrag_t direct, 2 for V, 2 for O).

## SDPA tile_matmad_nax calls

The SDPA kernel does NOT call `tile_matmad_nax`. It calls
`NAXFrag_t::mma(...)` directly with two fused-operand overloads:

| Line | Fused frags | (transpose_a, transpose_b) | C=, A=, B= | Purpose |
|---|---|---|---|---|
| 236–243 | 2 K frags | (false, true) | `Stile.frag_at(iq,ik)`, `Stile.frag_at(iq,ik+1)` ← `Qtile.frag_at(0,0)` @ `Ktile.frag_at(0,0)` / `Ktile.frag_at(1,0)` | Q @ K^T for S |
| 442–449 | 2 V frags | (false, false) | `Otile.frag_at(iq,id)`, `Otile.frag_at(iq,id+1)` ← `Stile.frag_at(iq,ik)` @ `Vtile.frag_at(0,0)` / `Vtile.frag_at(0,1)` | P @ V for O |

Both calls use the `TN==1 && TM%2==0` fused-pair path in `tile_matmad_nax`
(BaseNAXFrag, kPacking==2), which expects a paired-frag signature. The
NAXFrag32 port (Task 4) must route these through NAXFrag32's `mma()` using
kPacking==1 single-frag dispatch.

## SDPA tile shapes

Existing instantiations from `steel_attention_nax.metal` (all `wm=4, wn=1`):

| bq | bk | bd | wm | wn | SQ = bq/(wm*kU) | threads |
|---|---|---|---|---|---|---|
| 64 | 32 | 128 | 4 | 1 | 16 | 128 |
| 64 | 32 |  64 | 4 | 1 | 16 | 128 |
| 64 | 64 | 128 | 4 | 1 | 16 | 128 |
| 64 | 64 |  64 | 4 | 1 | 16 | 128 |

SQ=16 for ALL existing variants. Since BaseNAXFrag uses a (16,32,16) descriptor
(kFragRows=16), SQ=16 exactly fills one frag row (TQ=1). This is the broken
configuration on g16.

Phase 7 _g16 variants (wm=2 → SQ=32, fitting NAXFrag32's 32x32 frag):

| bq | bk | bd | wm | wn | SQ = bq/(wm*kU) | threads |
|---|---|---|---|---|---|---|
| 64 | 32 | 128 | 2 | 1 | 32 | 64 |
| 64 | 32 |  64 | 2 | 1 | 32 | 64 |
| 64 | 64 | 128 | 2 | 1 | 32 | 64 |
| 64 | 64 |  64 | 2 | 1 | 32 | 64 |

Threadgroup size: 128 threads (regular, wm=4) → 64 threads (g16, wm=2). Halves
simdgroup parallelism per threadgroup.

Note: `kU=16` is hardcoded on line 128 of `steel_attention_nax.h`. For NAXFrag32
the frag granularity is 32, so `kU` must be 32 in the g16 kernel variant — or the
loop structure (TQ, TK, TD) must be recalculated with NAXFrag32's kFragRows=32.
The `static_assert(TQ == 1, ...)` check must still pass. With wm=2, bq=64,
kNWarps=2, kU=32: TQ = 64/(2*32) = 1. Consistent.

## Other attn/ files using NAX

| File | NAX usage | Action needed for Phase 7 |
|---|---|---|
| `attn/kernels/steel_attention_nax.h` | 6 NAXTile declarations; 10 load/store sites; 2 direct `NAXFrag_t::mma` calls | Port to NAXFrag32 (Task 4): add `<Role, transpose>` to all load sites, update mma dispatch, change kU=32 |
| `attn/kernels/steel_attention_nax.metal` | Instantiation table only (macros); no NAX code | Task 5: add `_g16` instantiation macro set with wm=2 |
| `attn/mma.h` | Has `tile_matmad` (non-NAX, simdgroup_matrix based) for the standard attention kernel; NOT NAX | No action |
| `attn/attn.h` | Standard GEMM kernel using `tile_matmad` (non-NAX); includes `mma.h`, `loader.h`, `params.h` | No action |
| `attn/loader.h` | BlockLoader for standard tiled attention (threadgroup staging, no NAX) | No action |
| `attn/transforms.h` | Activation transforms (non-NAX) | No action |
| `attn/params.h` | `AttnParams` / `AttnMaskParams` structs (no NAX) | No action |
| `attn/nax.h` | BaseNAXFrag + NAXTile with old signatures; missing Role, kPacking, NAXFrag32 | Task 2: replace with thin re-export of `gemm/nax.h` (or a shared `nax_common.h`) |

## Phase 7 sequencing implication

The audit confirms Phase 7 can proceed as planned with no scope blockers.
`attn/nax.h` is a strict subset of `gemm/nax.h` with no attn-unique functionality,
so Task 2 (dedup) can safely replace it with a re-export. The SDPA kernel uses
the old role-unaware NAXTile method signatures (no `scratch`, no `Role`), which
means Task 4 must update all 10 load/store call sites to add explicit `<Role,
transpose>` template args — required for NAXFrag32 correctness on g16 and
straightforward given the Role assignments documented in the table above. The
`kU=16` hardcoded constant must become `kU=32` (or NAXFrag_t::kFragRows) in the
g16 kernel variant. The `mma()` calls use a fused-pair signature that is
BaseNAXFrag-specific; the NAXFrag32 path uses single-frag dispatch and must be
plumbed through `tile_matmad_nax` or a new direct mma loop.
