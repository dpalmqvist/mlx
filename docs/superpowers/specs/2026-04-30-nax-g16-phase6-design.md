# NAX g16 Phase 6 — transposed matmul correctness fix (design)

## Goal

Fix a correctness regression on Apple gen-16 GPUs where `steel_gemm_fused_nax_g16` (and every kernel that uses NAXFrag32 with non-default transpose flags) produces wrong results. Restore correct output for all matmul, quantized matmul, and SDPA paths that involve `transpose_a=true` or `transpose_b=true` on g16.

## Symptom

`mx.matmul` with any transpose configuration produces output with relative error >1.0 (i.e., output is fundamentally wrong, not numerical drift) on g16 hardware. Reproducer:

```python
import mlx.core as mx, numpy as np
mx.random.seed(0)
N = 128
a = mx.random.normal((N, N), dtype=mx.float32)
b = mx.random.normal((N, N), dtype=mx.float32)
print(np.max(np.abs(np.array(a @ b)   - (np.array(a) @ np.array(b)))))    # 0.0   (works)
print(np.max(np.abs(np.array(a @ b.T) - (np.array(a) @ np.array(b).T))))  # ~63   (broken)
```

The error scales with N. All three transpose configurations (`a @ b.T`, `a.T @ b`, `a @ a.T`) fail.

## Bisect

| commit | `a @ b.T` err | NAX state on g16 |
|---|---|---|
| `af78575c` (Phase 1 merge) | 1.9e-05 ✓ | gated off (gen<17) |
| `52d6ce8a` (Phase 2: dispatch fused_nax_g16) | 6.3e+01 ✗ | enabled |
| `17d903fe` (current main, post-Phase 5) | 6.3e+01 ✗ | enabled |

Phase 1 introduced NAXFrag32 + `steel_gemm_fused_nax_g16`; Phase 2 wired the dispatch and re-enabled `is_nax_available()` for gen 16. The bug arrived with Phase 2's dispatch wiring — exercising Phase 1's untested-with-transpose code.

## Why no test caught this

Every existing NAX probe (`tools/probe_nax_frag32.py`, `probe_naxfrag32_store_slice.py`, the multisg/multik probes) hardcodes `transpose_left=false, transpose_right=false`. `tools/probe_nax_frag32.py::test_mma_register_only`, `test_mma_via_tg_load_store`, and `test_mma_via_dv_load_store` — the three end-to-end mma probes — all use `transpose_a=false, transpose_b=false`.

`python/tests/test_blas.py -k matmul` does cover transpose, and reported 10 fp16 transpose failures. The Phase 5 validation report misclassified these as "pre-existing on main" because they reproduce on main; in fact they reproduce on main *because* of this bug, which has been in main since Phase 2 merged.

## Affected dispatch sites

Every kernel that uses NAXFrag32 with transpose flags. Phase 1+2+5 left this set:

| dispatch site | file | non-transposed | transposed |
|---|---|---|---|
| fused_nax_g16 | matmul.cpp | works | broken |
| splitk_nax_g16 | matmul.cpp | works | broken |
| segmented_nax_g16 | matmul.cpp | works | broken |
| gather_nax_g16 (nt) | matmul.cpp | works | broken |
| quantized_matmul (kG16-gated paths) | quantized.cpp | likely broken if reachable | likely broken |
| SDPA prefill (kG16-gated paths) | scaled_dot_product_attention.cpp | likely broken if reachable | likely broken |

The audit (component 5) verifies each.

## Diagnosis hypothesis

`NAXFrag32::load` (nax.h:667+) calls `cooperative_tensor::load(view)` with a hardcoded `transpose_left=false, transpose_right=false` descriptor (nax.h:671-676). `NAXFrag32::mma` (nax.h:617-657) uses runtime `transpose_a/transpose_b` flags in *its* descriptor. If MPP's per-thread element layout for `cooperative_tensor` is determined by the descriptor's `(role, transpose)`, then the load and mma populate/expect different element positions in `dtype_frag_t`, producing scrambled values when the mma flags don't match the load's hardcoded flags.

This is a **leading hypothesis, not confirmed**. Three plausible layers:

- **Layer 1 (load)**: most likely. NAXFrag32::load uses wrong descriptor for the role+transpose actually used by the consumer.
- **Layer 2 (mma)**: NAXFrag32::mma routes elements wrong inside the per-thread → cooperative_tensor copy.
- **Layer 3 (descriptor)**: the (32,32,32) descriptor on g16 has different transpose semantics than the (16,32,16) descriptor BaseNAXFrag uses on non-g16.

A focused diagnostic probe (component 1 below) pins down the layer before any fix is applied.

## Components

### 1. Diagnostic probe — `tools/probe_naxfrag32_transpose.py`

Pure isolation test, ~150-200 lines. Steps:

- Allocate two known fp32 32×32 matrices A, B with small integer values (range [-3, 3]) so reductions are exact.
- Inside a Metal kernel, load both via `NAXFrag32::load`. Call `NAXFrag32::mma` 4 times — once per `(transpose_a, transpose_b) ∈ {F,T}²`.
- For each, the kernel computes one matmul into a separate output tile.
- Python-side: compare each output against the matching numpy reference (`A @ B`, `A @ B.T`, `A.T @ B`, `A.T @ B.T`).
- Expected on current main: `(F,F)` passes, all three transposed cases fail.

Then a **second sub-probe** that isolates layer 1 vs 2/3:

- For the `(F, T)` case (transpose_b=true), construct a NEW load path that uses a `transpose_right=true` descriptor when populating `ct_b`.
- If results become correct, the bug is layer 1.
- If still broken, the bug is layer 2 or 3 — the load was already correct.

The probe is committed first as a RED test. After the fix, it goes GREEN.

### 2. Layer-1 fix (most likely path) — parameterize `NAXFrag32::load` on `(role, transpose)`

If the diagnostic confirms layer 1:

```cpp
namespace mlx::steel {
enum class Role { Left, Right };
}

struct NAXFrag32 {
  template <Role role, bool transpose, typename T, typename U>
  METAL_FUNC static void load(
      thread dtype_frag_t<T>& dst,
      const threadgroup U* src,
      short ld) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, 32,
        /*transpose_left=*/  (role == Role::Left)  ? transpose : false,
        /*transpose_right=*/ (role == Role::Right) ? transpose : false,
        /*relaxed_precision=*/true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct = (role == Role::Left)
        ? op.template get_left_input_cooperative_tensor<T,T,T>()
        : op.template get_right_input_cooperative_tensor<T,T,T>();
    metal::dextents<int32_t, 2> ext(32, ld);
    metal::tensor<threadgroup U, metal::dextents<int32_t, 2>, metal::tensor_inline>
        view(const_cast<threadgroup U*>(src), ext);
    ct.load(view);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      dst[i] = static_cast<T>(ct[i]);
    }
  }
  // ... same parameterization for load_safe, load_rows, device overloads
};
```

The signature exposes role+transpose at the call site. Existing `load_safe`/`load_rows`/device overloads get the same parameterization (4 functions × 4 variants = 16 instantiations, but most are template-deduplicated).

`NAXFrag32::store` and `store_slice` are NOT affected — they only handle the output (Cn0/Dtile), which is always the destination cooperative_tensor (one role, transpose semantics tied to the mma descriptor used at store time).

### 3. NAXTile plumbing — thread `(role, transpose)` from gemm_loop

`NAXTile<T, ..., NAXFrag_>::load` (nax.h:1146+) gains template params `<Role role, bool transpose>`. For `kPacking==1`, forwards them to `NAXFrag_t::load`. For `kPacking==2` (BaseNAXFrag), the params are ignored — BaseNAXFrag's per-element dr/dc loads are layout-invariant.

`gemm_nax.h::gemm_loop` already has `transpose_a, transpose_b` template params. Update the load calls:

```cpp
Atile.template load<Role::Left,  transpose_a>(A + A_offset, lda, scratch);
Btile.template load<Role::Right, transpose_b>(B + B_offset, ldb, scratch);
```

Same for the K-aligned-tail path (lines 95-130 of gemm_nax.h).

`load_safe` likewise. `load_rows` is only used internally inside `NAXTile::load` (kPacking==1 staging) — it inherits the role+transpose parameterization from there.

### 4. Layer-2/3 contingencies (predesigned, applied only if probe rules out layer 1)

**If layer 2** (mma element routing): Inside `NAXFrag32::mma`, the `ct_a[i] = A[i]; ct_b[i] = B[i]` copies need to permute based on transpose. Concrete shape depends on what the probe reveals about per-thread element ordering. Task description in the implementation plan: "investigate via probe; fix the routing."

**If layer 3** (descriptor itself): The (32,32,32) descriptor on g16 has unexpected transpose semantics. Mitigation paths:
- (a) Use multiple smaller mma calls assembled to handle transpose externally (significant scope, may warrant separate spec).
- (b) Gate transposed paths on g16 to fall back to non-NAX (one-line change to `is_nax_available()` for the transposed case).

If the probe identifies layer 3, Phase 6 ships option (b) as a minimal fix and a separate spec is opened for option (a).

### 5. Audit pass — `docs/superpowers/reports/2026-04-30-nax-g16-transpose-audit.md`

Walk every g16 dispatch site that could use NAXFrag32 with transpose. For each:
- Locate the dispatch (kname suffix `_g16` or `kG16` arch flavor check).
- Identify whether transpose_a / transpose_b can be true on that path.
- Confirm the load+mma layout assumption holds after the fix from component 2.

Sites to audit:
1. `mlx/backend/metal/matmul.cpp` — fused_nax_g16, splitk_nax_g16, segmented_nax_g16, gather_nax_g16 dispatchers.
2. `mlx/backend/metal/quantized.cpp` and the kernel headers `mlx/backend/metal/kernels/quantized_nax.h`, `mlx/backend/metal/kernels/fp_quantized_nax.h`.
3. `mlx/backend/metal/scaled_dot_product_attention.cpp` and `mlx/backend/metal/kernels/steel/attn/`.

Each section: site identifier, transpose paths covered, verdict (verified / fixed by core change / needs separate spec). The audit is committed as part of the Phase 6 PR.

## Tests

- `tools/probe_naxfrag32_transpose.py` — RED on current main, GREEN after fix. 4 cases (FF/FT/TF/TT) plus a sub-probe that isolates the layer.
- `python/tests/test_blas.py -k matmul -v` — must pass cleanly post-fix on g16. The 10 fp16-transpose failures from Phase 5 validation must disappear.
- `python/tests/test_blas.py -k gather_mm` — keep passing (gather is the most recent NAX path; verify no regression).
- All Phase 1/2/5 NAX probes — keep passing.

## Out of scope

- **Performance**. This is a correctness-only fix.
- **Restoring bm=16 NAX gather** on g16 (Phase 5 design choice; possible Phase 7).
- **Reworking the `dtype_frag_t` intermediate** (Approach C from brainstorming).
- **Layer-3 deep fix beyond the gate fallback** — if the probe identifies layer 3, Phase 6 ships only the gate; deep fix needs its own spec.

## Risks and mitigations

1. **MPP per-thread layout assumption**. The leading hypothesis assumes MPP's `cooperative_tensor` layout depends on `(role, transpose)`. If this turns out false, the layer-1 fix is a no-op. Mitigated by component 1's diagnostic probe — two sub-probes pin down the layer before any fix is applied.

2. **Hidden BaseNAXFrag transpose path also broken**. BaseNAXFrag works for transpose on non-g16 hardware (otherwise prod matmul would have been broken for years). But if Phase 6's parameterization implicitly affects BaseNAXFrag, it could break the previously-working path. Mitigated by: BaseNAXFrag's `load` ignores the new template params (its per-element dr/dc loads are layout-invariant). Verified by running `test_blas.py -k matmul` on non-g16 hardware (or via CI if available) post-fix.

3. **Audit scope creep**. The audit might reveal a separate latent bug in quantized or SDPA paths. Mitigation: the audit deliverable is a triage doc; fixes for newly-discovered bugs go to separate specs/PRs. Phase 6 ships the matmul fix + audit doc.

## References

- Phase 1 spec: `docs/superpowers/specs/2026-04-27-nax-g16-fix-plan.md` (NAXFrag32 introduction)
- Phase 2 design: `docs/superpowers/specs/2026-04-28-nax-g16-phase2-design.md`
- Phase 5 design: `docs/superpowers/specs/2026-04-30-nax-g16-phase5-design.md`
- Existing NAX probes: `tools/probe_nax_frag32.py`, `probe_naxfrag32_store_slice.py`
