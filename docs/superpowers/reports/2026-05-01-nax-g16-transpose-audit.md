# NAX-on-g16 transpose audit (Phase 6)

Audit of every dispatch site that uses `NAXFrag32` with `transpose_a`
or `transpose_b` flags, post the Phase 6 layer-1 fix
(commit 072e6ac9 + hardening 6d5710ba).

## Scope

Every site where mlx might dispatch a g16-specific NAX kernel and where
the kernel uses `NAXFrag32::load` (directly or via `NAXTile::load`).

The Phase 6 layer-1 fix parameterized `NAXFrag32::load` on `(Role,
transpose)` template parameters and updated `NAXTile::load` / `NAXTile::load_safe`
to forward them via `load_rows<role, transpose>`. The central call site is
`gemm_nax.h` (`gemm_loop`), which now passes `<Role::Left, transpose_a>`
and `<Role::Right, transpose_b>` explicitly. All four g16 GEMM kernel
families share that single `gemm_loop` call.

## matmul.cpp

### steel_gemm_fused_nax_g16

- **Dispatch**: `matmul.cpp:176` (`steel_matmul_regular_axpby_nax`) —
  kernel name built at line 220, `_g16` suffix appended at line 228.
- **Transpose configs reachable**: all four — `nn`, `nt`, `tn`, `tt` —
  instantiated at `steel_gemm_fused_nax.metal:29-33`.
- **Load path**: `gemm_nax.h:66,70,75,79,116,118` — passes
  `<Role::Left, transpose_a>` for A and `<Role::Right, transpose_b>` for
  B in every branch (aligned, unaligned, partial-K).
- **Status**: fixed by Phase 6 core change (gemm_nax.h)
- **Evidence**: `test_blas.py -k matmul` — 9/9 PASS (matmul, batched,
  dtypes, shapes, unaligned, gather_matmul, grad, block_masked, empty)

### steel_gemm_splitk_nax_g16

- **Dispatch**: `matmul.cpp:661` (`steel_gemm_splitk_axpby_nax`) —
  kernel name built at line 720, `_g16` suffix appended at line 729.
- **Transpose configs reachable**: all four — `nn`, `nt`, `tn`, `tt` —
  instantiated at `steel_gemm_splitk_nax.metal:29-33`.
- **Load path**: same `gemm_loop` in `gemm_nax.h` as fused — Role+transpose
  forwarded correctly.
- **Status**: fixed by Phase 6 core change (gemm_nax.h)
- **Evidence**: `test_blas.py -k matmul` — 9/9 PASS (splitk is exercised
  by large-K shapes in test_matmul_shapes / test_matmul_unaligned)

### steel_gemm_segmented_nax_g16

- **Dispatch**: `matmul.cpp:2460` (`segmented_mm`) — `use_nax` gate at
  line 2512, kernel name built at line 2526+, `_g16` suffix at line 2561.
  Calls `get_steel_gemm_segmented_nax_kernel` at line 2577.
- **Transpose configs reachable**: all four — `nn`, `nt`, `tn`, `tt` —
  instantiated at `steel_gemm_segmented_nax.metal:29-33`.
- **Load path**: same `gemm_loop` in `gemm_nax.h` — Role+transpose
  forwarded correctly.
- **Status**: fixed by Phase 6 core change (gemm_nax.h)
- **Evidence**: `test_blas.py -k matmul` — 9/9 PASS

### steel_gather_mm_rhs_nax_g16

- **Dispatch**: `matmul.cpp:1996` (`gather_mm_rhs_nax`) — kernel name
  built at line 2062, `_g16` suffix at line 2081. Calls
  `get_steel_gemm_gather_nax_kernel` at line 2105 with `transpose_a=false`
  (hardcoded).
- **Transpose configs reachable**: `nn` and `nt` only — `trans_a` is
  always `false` in both the dispatcher (line 2111) and the instantiation
  macros (`steel_gemm_gather_nax.metal:46-48`). `bm=16` cases fall back to
  non-NAX path (line 2047-2051).
- **Load path**: same `gemm_loop` in `gemm_nax.h` — `<Role::Left,
  transpose_a=false>` and `<Role::Right, transpose_b>` forwarded correctly.
- **Status**: fixed by Phase 6 core change (gemm_nax.h)
- **Evidence**: `test_blas.py -k gather_mm` — 2/2 PASS
  (gather_mm_sorted, gather_mm_sorted_vjp)

## quantized.cpp / quantized_nax.h / fp_quantized_nax.h

### qmm_nax / gather_qmm_nax (affine and fp modes)

- **Dispatch**: `quantized.cpp:718` (`qmm_nax`) and `quantized.cpp:913`
  (`gather_qmm_nax`) — gated on `is_nax_available()`, `transpose`, K%64==0,
  and M threshold. No `nax_arch_flavor()` g16 check; no `_g16` kernel suffix.
- **Kernel headers**: `quantized_nax.h` and `fp_quantized_nax.h` use
  `NAXTile<T, TM, TK>` **without** an explicit `NAXFrag_` type argument —
  they get the default `BaseNAXFrag` (kPacking==2), not `NAXFrag32`
  (kPacking==1).
- **Load path**: `quantized_nax.h:1038,1040,1171,1601,1603,1639` and
  `fp_quantized_nax.h:298,300,430,936,938,976` call
  `Atile.template load<Role::Left, false>` — these are the calls hardened
  in Phase 6 Task 2a commit 6d5710ba. Because `NAXFrag_t::kPacking==2`
  for `BaseNAXFrag`, the `if constexpr (kPacking == 1)` branch in
  `NAXTile::load` is never taken; the Role+transpose params are accepted
  syntactically but ignored at runtime (kPacking==2 path uses per-element
  layout-invariant loads). No `NAXFrag32` hazard exists.
- **Status**: verified clean — quantized kernels do not instantiate
  NAXFrag32; the explicit `<Role::Left, false>` params from Task 2a are
  correct defensive annotations.
- **Evidence**: `test_quantized.py` — 27/27 PASS

### gather_qmm_rhs_nax (affine and fp modes)

- **Dispatch**: `quantized.cpp:1259` (`gather_qmm_rhs_nax`) — same
  `is_nax_available()` gate, no arch-flavor check, no `_g16` suffix.
- **Kernel headers**: same as above — `NAXTile` defaults to `BaseNAXFrag`;
  no NAXFrag32 involvement.
- **Status**: verified clean — same reasoning as qmm_nax above.
- **Evidence**: `test_quantized.py` — 27/27 PASS (includes gather_qmm
  variants)

## scaled_dot_product_attention.cpp / kernels/steel/attn/

### sdpa_full_self_attention_nax

- **Dispatch**: `scaled_dot_product_attention.cpp:177` — gated on
  `is_nax_available()` and `q.dtype()`. No `nax_arch_flavor()` g16 check;
  no `_g16` kernel suffix. Kernel name built as `steel_attention_*` (line 60).
- **Kernel headers**: `kernels/steel/attn/nax.h` is a self-contained copy
  of the NAXTile infrastructure that defines its own `BaseNAXFrag` (line 27)
  and `NAXTile` (line 536). The attention kernel instantiations at
  `steel_attention_nax.h:143,201,212,213,315,429` use `NAXTile<T, TQ, TD>`,
  `NAXTile<T, 1, 1>`, etc. — all without an explicit `NAXFrag_` argument,
  so all get `BaseNAXFrag` (kPacking==2).
- **NAXFrag32**: absent — `attn/nax.h` does not define `NAXFrag32`, and
  neither `steel_attention_nax.h` nor `steel_attention_nax.metal` reference
  `NAXFrag32`, `kPacking==1`, `_g16`, or `NAXArchFlavor`.
- **Status**: verified clean — SDPA path never reaches NAXFrag32; no
  transpose/layout hazard.
- **Evidence**: `test_fast_sdpa.py` — `test_quantized_sdpa_*` 2/2 PASS;
  `test_sdpa` has a pre-existing failure (max diff 0.0316 > 0.0003 atol)
  that also reproduces on `main` before any Phase 6 change and is therefore
  unrelated to this audit. The SDPA NAX dispatch itself is not implicated.

## Summary

| Site | File:line | Trans configs | Status | Evidence |
|---|---|---|---|---|
| steel_gemm_fused_nax_g16 | matmul.cpp:176 / gemm_nax.h:66 | nn,nt,tn,tt | Fixed by Phase 6 (gemm_nax.h) | test_blas matmul 9/9 PASS |
| steel_gemm_splitk_nax_g16 | matmul.cpp:661 / gemm_nax.h:66 | nn,nt,tn,tt | Fixed by Phase 6 (gemm_nax.h) | test_blas matmul 9/9 PASS |
| steel_gemm_segmented_nax_g16 | matmul.cpp:2460 / gemm_nax.h:66 | nn,nt,tn,tt | Fixed by Phase 6 (gemm_nax.h) | test_blas matmul 9/9 PASS |
| steel_gather_mm_rhs_nax_g16 | matmul.cpp:1996 / gemm_nax.h:66 | nn,nt only | Fixed by Phase 6 (gemm_nax.h) | test_blas gather_mm 2/2 PASS |
| qmm_nax / gather_qmm_nax | quantized.cpp:718,913 / quantized_nax.h | transpose only (A-side) | Verified clean (BaseNAXFrag) | test_quantized 27/27 PASS |
| gather_qmm_rhs_nax | quantized.cpp:1259 / quantized_nax.h | nn,nt | Verified clean (BaseNAXFrag) | test_quantized 27/27 PASS |
| sdpa_full_self_attention_nax | scaled_dot_product_attention.cpp:177 / attn/nax.h | N/A (no transpose param) | Verified clean (BaseNAXFrag, no g16 path) | test_fast_sdpa quantized 2/2 PASS |

**Total: 7 sites**
- Verified clean: 3
- Fixed by Phase 6 (gemm_nax.h core change): 4
- Needs separate spec: 0

## Notes / observations

1. **Single fix point, four beneficiaries.** All four g16 GEMM kernel
   families (fused, splitk, segmented, gather) share the same `gemm_loop`
   in `gemm_nax.h`. The Phase 6 fix to that one file — parameterizing
   `NAXTile::load` on `(Role, transpose)` and forwarding them through
   `load_rows<role, transpose>` — automatically corrects all four.

2. **Quantized kernels are isolated from the g16 NAXFrag32 hazard.**
   They use `NAXTile` with the default `BaseNAXFrag` (kPacking==2) and are
   not gated on `nax_arch_flavor()`. The `<Role::Left, false>` annotations
   added by Task 2a are syntactically required (no default params) and
   semantically inert for the kPacking==2 branch.

3. **SDPA has its own NAX infrastructure copy.** `kernels/steel/attn/nax.h`
   is a fork of the gemm `nax.h` that retains only `BaseNAXFrag`; it does
   not import or reference `NAXFrag32`. Any future g16-specific SDPA work
   would require adding `NAXFrag32` to that copy and gating the dispatch
   on `nax_arch_flavor()` — that is a separate future spec.

4. **Pre-existing SDPA test failure is unrelated.** `test_fast_sdpa.py
   test_sdpa` fails with the same error on `main` (before any Phase 6
   commit), confirming it is a pre-existing issue not caused by this work.

5. **gather_mm_rhs is A-non-transposed by design.** The gather path hard-
   codes `trans_a=false` both in the dispatcher (matmul.cpp:2111) and in the
   kernel instantiation macros. This is correct for the gather-RHS use case
   (only B is optionally transposed).
