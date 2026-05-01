# NAX-on-g16 Performance Baseline — Design Spec

**Status:** Approved 2026-05-01.
**Goal:** Produce a measured, kernel-by-kernel ranking of where g16 NAX is helping, hurting, or at parity vs. the non-NAX fallback, so the next performance phase can target the highest-leverage path.
**Non-goal:** Changing any kernel code. This work is measurement-only.

## Background

NAX matmul, gather, and SDPA all run correctly on Apple gen-16 GPUs (M3/M4) after Phases 1, 2, 5, 6, 7 (PRs #13–#17, all merged to main). The g16 path uses `NAXFrag32` (32×32 fragment, kPacking==1) instead of the standard `BaseNAXFrag` (16×16, kPacking==2), with three g16-specific costs that are likely to show up in benchmarks:

1. **Threadgroup scratch staging** for `load_safe`, `load_rows`, `store_safe`, `store_rows`, `store_slice` — every bounded I/O traverses a 32×32 threadgroup buffer with the associated barrier cost. Non-g16 NAX paths use bounded cooperative_tensor views directly.
2. **`wm=2` for SDPA prefill** — `attention_nax_g16` runs with 64 threads/threadgroup instead of the standard 128, because NAXFrag32 requires SQ≥32 and `bq=64, wm=4` would give SQ=16. Halves the threadgroup width.
3. **Direct per-element addressing** for mid-kernel reads of the destination tile (e.g., the SDPA mask add path uses `get_coord()+dr_dc()` rather than a cooperative_tensor load).

No measurements exist today to confirm these costs are real or rank them. This spec defines the harness that produces those numbers.

## Approach

**Single build, runtime A/B via env var.** The simplest, cleanest mechanism. Rejected alternatives: two separate builds (slow iteration), per-shape-feature dispatch tricks (shape-distorted comparison), unmeasured knob-sweeping (doesn't answer "is NAX helping?").

### Component 1 — Runtime NAX gate

Add an env var read on first call to `is_nax_available()` in `mlx/backend/metal/device.cpp`. The check happens inside the existing `_check_nax` lambda that's already cached behind `static bool is_nax_available_`, so the perf cost of the gate is exactly one `getenv` per process.

```cpp
bool is_nax_available() {
#ifdef MLX_METAL_NO_NAX
  return false;
#else
  auto _check_nax = []() {
    if (const char* env = std::getenv("MLX_DISABLE_NAX")) {
      if (env[0] != '\0' && env[0] != '0') return false;
    }
    bool can_use_nax = false;
    if (__builtin_available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      can_use_nax = true;
    }
    auto& d = metal::device(mlx::core::Device::gpu);
    auto gen = d.get_architecture_gen();
    can_use_nax &= gen >= 16;
    return can_use_nax;
  };
  static bool is_nax_available_ = _check_nax();
  return is_nax_available_;
#endif
}
```

**Caching note:** because the result is cached at first call, flipping `MLX_DISABLE_NAX` mid-process does not work — the bench harness must drive A/B via separate subprocesses.

**`nax_arch_flavor()` is downstream of `is_nax_available()`** and already returns `kNone` when NAX is unavailable, so it inherits the gate automatically with no second change.

### Component 2 — Bench harness

A single Python script `benchmarks/python/nax_g16_perf_bench.py`:

- Single source of truth for the case list (op, kernel_label, shape, work-flop-or-byte estimator).
- Spawns two subprocesses with `/Users/daniel/mlx/.venv/bin/python`, one with `MLX_DISABLE_NAX=1` set in the env, one without. Each child runs every case, prints one JSON record per case to stdout, exits.
- Parent reads both JSON streams, joins on `(kernel_label, shape)`, prints a comparison table to stdout, and writes the joined records to `nax_g16_perf_<timestamp>.json` next to the script.
- One numerical-equivalence sanity check at startup (single small fp16 matmul, NAX-on vs NAX-off, `np.allclose(rtol=1e-2)`). If it fails, abort before any timing — that means the gate didn't flip.

**Per-case timing:** 3 warmup iters, 10 timed iters, take the median. Between iters, `mx.eval()` the result; after the timed window, `mx.synchronize()` to fence GPU completion. Reuse `benchmarks/python/time_utils.py` if its API fits the subprocess shape; otherwise inline a small timer.

## Shape grid

Thirteen cases, fp16 inputs throughout. Picked to (i) hit g16 NAX dispatch (avoiding `bm=16` gather and bool-mask SDPA, both of which fall through to non-NAX on g16), (ii) cover at least one Llama-style prefill shape per op, and (iii) include small/large boundary cases.

| kernel_label | shape | notes |
| --- | --- | --- |
| gemm_fused | M=2048, N=4096, K=4096 | Llama-7B qkv proj on 2k prefill |
| gemm_fused | M=2048, N=11008, K=4096 | Llama-7B MLP up-proj |
| gemm_fused | M=512, N=4096, K=4096 | shorter prefill |
| gemm_splitk | M=64, N=64, K=8192 | synthetic splitk trigger |
| gemm_splitk | M=128, N=128, K=4096 | milder splitk |
| gemm_segmented | B=8, M=512, N=4096, K=4096 | batched matmul, MHA-style |
| gemm_segmented | B=32, M=128, N=128, K=128 | small batched |
| gather | tokens=2048, E=8, hidden=4096, expert_hidden=14336, top_k=2 | Mixtral-style prefill (bm≥32) |
| gather | tokens=512, E=8, hidden=4096, expert_hidden=4096, top_k=2 | smaller MoE |
| sdpa_prefill | B=1, H=32, kL=2048, hd=128 | Llama-7B prefill |
| sdpa_prefill | B=1, H=32, kL=8192, hd=128 | long-seq case |
| sdpa_prefill | B=1, H=32, kL=512, hd=128 | short-seq boundary |
| sdpa_prefill | B=1, H=32, kL=2048, hd=64 | smaller head dim |

Excluded on purpose:
- `bm=16` gather (falls back to non-NAX `gather_mm_rhs` on g16 — A/B is meaningless).
- Bool-mask SDPA (NAXFrag32 cooperative_tensor doesn't accept bool — falls back to `sdpa_full_self_attention_metal` on g16).
- Transposed matmul cases. Phase 6 fixed correctness; transpose perf is a follow-up if (a) shows we need it.

## Output format

Printed table:

```
op              shape                                       nax_on    nax_off   speedup  TFLOPS
gemm_fused      M=2048 N=4096 K=4096                        4.21ms    5.83ms    1.39x    16.3
gemm_fused      M=2048 N=11008 K=4096                       ...
gemm_splitk     M=64 N=64 K=8192                            ...
gemm_segmented  B=8 M=512 N=4096 K=4096                     ...
gather          tokens=2048 E=8 H=4096 EH=14336 k=2         ...
sdpa_prefill    B=1 H=32 kL=2048 hd=128                     ...
```

JSON dump:
```json
[
  {
    "kernel_label": "gemm_fused",
    "shape": {"M": 2048, "N": 4096, "K": 4096},
    "nax_on_ms": 4.21,
    "nax_off_ms": 5.83,
    "speedup": 1.39,
    "nax_on_tflops": 16.3
  },
  ...
]
```

`speedup > 1` means NAX is faster (the desired direction).

TFLOPS is computed from shape:
- matmul/gather: `2 * M * N * K / time` (×B for batched, ×top_k for gather).
- SDPA: `4 * B * H * Q * kL * hd / time` (Q@K^T + softmax-cheap + S@V; treat softmax as zero work for ranking).

For shapes that are obviously bandwidth-bound (none in this grid, but if added later) report GB/s alongside TFLOPS.

## File layout

| File | Status | Purpose |
| --- | --- | --- |
| `mlx/backend/metal/device.cpp` | edit | ~5-line `MLX_DISABLE_NAX` gate inside `_check_nax` |
| `benchmarks/python/nax_g16_perf_bench.py` | new | bench harness (subprocess A/B + table + JSON) |
| `docs/superpowers/specs/2026-05-01-nax-g16-perf-baseline-design.md` | this file | the design |
| `docs/superpowers/plans/2026-05-01-nax-g16-perf-baseline.md` | created next | implementation plan |
| `docs/superpowers/reports/2026-05-01-nax-g16-perf-baseline.md` | follow-up after run | the actual numbers and ranking |

## Risks and mitigations

- **Static caching of `is_nax_available_`** could mask a broken gate. Mitigation: numerical-equivalence sanity check at harness startup — if NAX-on and NAX-off produce different results, the gate works; if they don't, abort.
- **Subprocess startup variance.** Mitigation: each subprocess runs *all* cases, not one per process; per-case timing is wall-clock around the kernel itself, not around the subprocess.
- **`mx.synchronize` quirks.** Mitigation: use the same timing dance other MLX benches use (per `time_utils.py`).
- **Shape coverage gaps.** Acknowledged: 13 cases is wide-thin by design. The deliverable is a *ranking* of paths to attack, not exhaustive characterization. Follow-up phases pick the loser and sweep deep.

## Success criteria

- The runtime gate is committed and the sanity check passes (NAX-on and NAX-off produce different timings *and* the same numerical result).
- A single bench run produces the table and JSON for all 13 cases on g16 hardware.
- The follow-up perf-baseline report ranks the 5 kernel paths by NAX-on/NAX-off speedup with one of {NAX wins, NAX hurts, parity} per kernel.
- Decision in hand for the next phase: which kernel path to attack first.

## Out of scope

- Any kernel-code change.
- Comparison against non-MLX baselines (MPS, Accelerate, CPU).
- Two-build A/B (rejected for this work; runtime gate replaces it).
- CI/regression integration.
- Transposed matmul, bool-mask SDPA, `bm=16` gather (the dispatch-fallback paths — different question for a different day).
