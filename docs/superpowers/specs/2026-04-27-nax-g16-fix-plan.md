# NAX correctness on g16s (M4 Pro): plan for a real fix

## Status

`is_nax_available()` is gated to `gen >= (arch == 'p' ? 18 : 17)` again, so
NAX is disabled on M4 Pro. This restores correctness for dense matmul,
quantized matmul, and SDPA at the cost of the qmm "wins" reported in
`bf760f92` — those wins were measured on a kernel that produces wrong
output (see `tools/check_qmm_nax_correctness.py`).

This document records what we know and what a real re-enable would take.

## Evidence

`tools/probe_nax_descriptor.py` sweeps `mpp::tensor_ops::matmul2d` on the
current GPU using the layout-agnostic API (`tensor_inline` +
`cT.load(handle)` / `cT.store(handle)`). Inputs are integers in [-3, 3]
so fp32 reductions are exact and any non-zero error is a real divergence.

On M4 Pro (applegpu_g16s), single-simdgroup execution:

| descriptor (M, N, K, relaxed) | max\|err\| |
|---|---|
| (16, 32, 16, _) — what mlx uses | 67  |
| (32, 16, 16, _)                 | 45  |
| (16, 16, 32, _)                 | 74  |
| (32, 32, 16, _)                 | 83  |
| (16, 32, 32, _)                 | 70  |
| (32, 16, 32, _)                 | 74  |
| **(32, 32, 32, _)**             | **0** |

`(M=N=K=16)` is rejected at compile time by an MPP `static_assert`
("at least one of M, N, K must be 32 if both inputs are cooperative
tensors"). Larger descriptors (`(32, 64, 32)`, `(64, 32, 32)`) require
`execution_simdgroups<N>` rather than `execution_simdgroup` and were not
tested.

Conclusion: on g16s, `matmul2d<execution_simdgroup>` is correct only for
the symmetric `(32, 32, 32)` descriptor. The current NAX kernels use
`(16, 32, 16)` and are therefore wrong everywhere on g16s.

## Layout for the working descriptor

`tools/probe_nax_descriptor.py` (the layout-dump variant in earlier
versions of the script) decoded the per-thread element coordinates for
`(32, 32, 32)`:

- Per-thread capacity: 32 elements (= 32×32 / 32 lanes).
- Lane base `(a, b)`:
  - `a = ((lane & 1) << 1) | ((lane & 8) >> 1)` ∈ {0, 2, 4, 6}
  - `b = ((lane & 2) >> 1) | ((lane & 4) >> 1) | ((lane & 16) >> 2)` ∈ {0..7}
- Element `i` maps to `(a + dr[i % 8], b + dc[i / 8])` with
  - `dr = {0, 1, 8, 9, 16, 17, 24, 25}`
  - `dc = {0, 8, 16, 24}`

This gives the per-thread coverage of an 8 × 4 = 32 element shape inside
a 32 × 32 tile.

## What "real fix" requires

1. **New frag class** alongside `BaseNAXFrag` — call it `NAXFrag32`:
   - `kFragRows = kFragCols = 32`, `kElemsPerFrag = 32`, `kElemRows = 8`,
     `kElemCols = 4`.
   - `get_coord` rewritten to match the layout above (or, preferably,
     I/O routed through `tensor_inline` + `cT.load/store` so we don't
     hard-code any layout — this also removes the same risk on future
     GPU generations).
2. **Descriptor switch** to
   `matmul2d_descriptor(32, 32, 32, transpose_a, transpose_b, false)`
   for the g16-specific path.
3. **Threadgroup staging** — keep the existing tg-memory pipeline for
   K-loop reuse and bandwidth, but stage into `tensor_inline` views and
   load via `ctA.load(tg_tensor)` instead of element-indexed copies.
4. **Re-instantiate** the four NAX gemm kernels (`steel_gemm_fused_nax`,
   `steel_gemm_splitk_nax`, `steel_gemm_gather_nax`,
   `steel_gemm_segmented_nax`) with the 32×32 frag, plus matching qmm
   (`quantized_nax`, `fp_quantized_nax`) and SDPA (`steel_attention_nax`)
   variants.
5. **Runtime dispatcher** in `matmul.cpp`, `quantized.cpp`,
   `scaled_dot_product_attention.cpp`: pick the g16 vs g17+ kernel name
   from `metal::is_nax_available()` plus an architecture flavor flag.
6. **Validation**: run `tools/probe_nax_descriptor.py`,
   `tools/check_qmm_nax_correctness.py`, and the full
   `python/tests/test_quantized.py` / linalg / dense-matmul suites on
   g16s before re-enabling.

## Scope

~10 files, multi-day refactor. Owns its own branch and review.
The g17+ path that currently works must not regress.

## Files

- `mlx/backend/metal/device.cpp` — `is_nax_available()` gate (this revert).
- `tools/probe_nax_descriptor.py` — descriptor sweep + layout dump.
- `tools/check_qmm_nax_correctness.py` — qmm correctness probe.

## Why this isn't a one-line patch

The MMA descriptor is a `constexpr` template parameter, so different
descriptors compile to different kernels. Switching to `(32, 32, 32)`
without also switching the frag size means each MMA covers a 32×32 tile
but `BaseNAXFrag` (16×16) only consumes 8 elements per thread — three
quarters of the MMA output is silently dropped. We confirmed this
empirically: pairing the working descriptor with mlx's existing
indexed-access layout still gives wrong results (max\|err\| = 101 in
`tools/probe_nax_descriptor.py` history). The correctness comes from the
**combination** of (32, 32, 32) descriptor + matching frag + tile sizes
+ I/O via `cT.load/store`.
