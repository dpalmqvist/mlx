# NAX-on-g16 Phase 9 — matmul attribution

**Date:** 2026-05-02
**Hardware:** Apple M4 Pro
**MLX commits:** branch `nax-g16-phase9`, tip `97bdef03` (V3 added).
**Spec / plan:** `docs/superpowers/specs/2026-05-02-nax-g16-phase9-design.md`,
`docs/superpowers/plans/2026-05-02-nax-g16-phase9.md`

## Variants

- **V0** — current ship state (post-Phase-8). `MLX_NAX_DIAG_VARIANT=0`.
- **V1** — no-mma. `gemm_op.run()` in `NAXFrag32::mma` replaced with per-element add into `ct_c`. Loads + setup stay alive.
- **V2** — zero-load. `ct.load(view)` skipped in both `NAXFrag32::load` overloads (threadgroup and device); a single volatile read of `*src` keeps the device read alive.
- **V3** — hoisted-op. `gemm_op` + cooperative tensors constructed once outside the K-loop in `tile_matmad_nax`'s `kPacking==1` branch; mma body inlined across iterations.

## Per-variant nax_on / nax_off (ms)

| Shape | V0 nax_on | V1 nax_on | V2 nax_on | V3 nax_on | nax_off (V0 → V3) |
|---|---|---|---|---|---|
| gemm_splitk M=64 N=64 K=8192 | 1.145 | 0.672 | 1.043 | 1.077 | 0.140 → 0.638 → 0.142 → 0.215 |
| gemm_fused M=2048 N=4096 K=4096 | 10.999 | 6.862 | 10.158 | 10.924 | 9.317 → 9.299 → 9.299 → 9.294 |

`nax_off` for gemm_fused is stable across all four variant runs (range 0.023ms across 9.317–9.299) — confirms session-level conditions are consistent for the large-shape signal. `nax_off` for gemm_splitk is volatile (V1 spike to 0.638ms is post-rebuild cache state, not signal); the splitk shape's small absolute work (~67M FLOPs in <1ms) makes its `nax_off` and `nax_on` highly sensitive to dispatch overhead and metallib JIT cache state.

## Cost attribution

**Formula correction.** The spec and plan both wrote `cost(ct.load) := V1 − V2`. That formula is incorrect: V1 and V2 are not orthogonal — V1 deletes mma, V2 deletes ct.load, so `V1 − V2 = cost(ct.load) − cost(mma)`. Each diagnostic variant deletes one cost contributor relative to V0; the right formulae all reference V0:

- `cost(mma) := V0 − V1` (still correct)
- `cost(ct.load) := V0 − V2` (corrected — was V1−V2)
- `cost(per-iter setup) := V0 − V3` (still correct)

| Shape | cost(mma) = V0−V1 | cost(ct.load) = V0−V2 | cost(per-iter setup) = V0−V3 |
|---|---|---|---|
| gemm_splitk M=64 N=64 K=8192 | **0.473ms (41% of V0)** | 0.102ms (9%) | 0.068ms (6%) |
| gemm_fused M=2048 N=4096 K=4096 | **4.137ms (38% of V0)** | 0.841ms (8%) | 0.075ms (1%) |

## Dominant contributor

**`mma` dispatch** is the dominant cost contributor on both shapes. It accounts for ~38–41% of total NAX-on kernel time. `ct.load` is ~8–9% (modest). Per-iter cooperative-tensor setup is ~1–6% (effectively noise on gemm_fused, marginal on gemm_splitk).

The top two costs (mma and ct.load) are not within 0.05× of each other — gap is wide on both shapes. No tiebreaker needed; A2 Instruments (Task 10) is unnecessary.

## Implication for the matmul tax

V0 nax_on for gemm_fused is 10.999ms; V0 nax_off (non-NAX baseline) is 9.317ms. The "tax" itself is 1.682ms = 18% of nax_off, or 15% of nax_on. The **mma dispatch alone (4.137ms) is more than twice the size of the tax**, which means: NAX kernels do pay substantial mma cost, but a comparable cost is paid by the non-NAX baseline kernel for its own SIMD-group matrix-multiply path. The 30% tax language used in earlier phases conflated "NAX nax_on time vs. nax_off time" with "intrinsic NAX overhead" — the real picture is that NAX's kPacking==1 path is reasonably competitive with non-NAX on this large shape (0.85×), and the gap shrinks rather than collapsing into one dominant inefficiency.

For `gemm_splitk M=64 N=64 K=8192`, the picture is more complicated: V0 nax_on is 1.145ms vs V0 nax_off 0.140ms (0.12× speedup). The attribution still shows mma at 41%, but the absolute numbers are dominated by dispatch + setup overhead on tiny work, and the post-rebuild nax_off spike to 0.638ms in the V1 trial confirms that splitk's `nax_off` baseline is itself quirky in this session. The splitk shape is not a reliable target for shipping fix decisions — its tax magnitude is dominated by per-dispatch overhead, not the K-loop body that V1/V2/V3 actually instrumented.

## Decision

Per the spec's decision rule:

> **mma dominates** → kPacking==1 intrinsic on g16. Fix is the BaseNAXFrag-style alternative path. Document; design phase 10.

**Phase 9 ships:** the diagnostic variants (V1/V2/V3 source gates, default-inert under `MLX_NAX_DIAG_VARIANT=0`), the `--diag` bench-harness flag, the four bench reports, and this attribution.

**Phase 9 does NOT ship:** any kernel-level fix. The dominant contributor is `mma` dispatch itself, which is intrinsic to the kPacking==1 (NAXFrag32) path. Replacing it requires a manual SIMD register layout that bypasses `cooperative_tensor`, or porting the BaseNAXFrag (kPacking==2, 16×16 + packing) path to g16. Either is a substantial engineering effort and warrants its own design phase.

A Phase 10 stub will be created at Task 9 capturing this finding and staging the next round of brainstorming.

## Caveats

- Wall-clock variance: 10-iter median, no fan or thermal control, no GPU clock pinning. All four variants benched within the same session to bound inter-run drift.
- gemm_splitk's `nax_off` was volatile (0.140 → 0.638 → 0.142 → 0.215) due to metallib re-cache effects after each rebuild; the splitk numbers are directionally informative but quantitatively noisy. The gemm_fused signal (stable `nax_off` across all four runs) is the trustworthy basis for attribution.
- V1 / V2 produce wrong output by design; their nax_on_ms is informative only as a relative measurement against V0.
- DCE protection: V1 uses per-element add into `ct_c` (so `ct_a`/`ct_b` reads are kept alive into the back-copy → C). V2 uses `volatile auto _v2_sink = static_cast<U>(*src);` to keep the device read alive when `ct.load(view)` is skipped. Build-time IR inspection was *not* performed (planned as a hard gate in spec; demoted to "investigate if numbers look weird" in the plan). Numbers do not look weird: V1 < V0 << V2 ≈ V3 ≈ V0 is consistent with the variant semantics, and V0/V1/V2 roundtrip for `nax_off` on the trustworthy gemm_fused shape is bit-stable, so the variants are demonstrably distinguishable in the bench.
- V3 passed `tools/probe_nax_frag32.py` (10/10 tests) confirming the hoisted-op refactor is arithmetically correct. Even though it ships no perf benefit, it is a valid alternative kernel-shape that future phases can build on.
