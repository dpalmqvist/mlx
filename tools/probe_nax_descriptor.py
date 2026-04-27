"""
Sweep MPP matmul2d descriptors on the current GPU using the truly
layout-agnostic API (tensor_inline + cooperative_tensor.load/store).
This bypasses any layout assumption — if a descriptor still produces
the wrong answer, the bug is in MPP / hardware, not in mlx's frag.

Constraint from MPP: at least one of M, N, K must be 32 when both
inputs are cooperative tensors.
"""
import sys
import mlx.core as mx
import numpy as np

HEADER = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""


def make_source(M, N, K, relaxed):
    rp = "true" if relaxed else "false"
    return f"""
constexpr int M={M}, N={N}, K={K};
constexpr auto desc = matmul2d_descriptor(
    M, N, K,
    /*transpose_left=*/false,
    /*transpose_right=*/false,
    /*relaxed_precision=*/{rp},
    matmul2d_descriptor::mode::multiply);

matmul2d<desc, execution_simdgroup> op;
auto ctA = op.get_left_input_cooperative_tensor<float, float, float>();
auto ctB = op.get_right_input_cooperative_tensor<float, float, float>();
auto ctC = op.get_destination_cooperative_tensor<decltype(ctA), decltype(ctB), float>();

dextents<int32_t, 2> extA(M, K);
dextents<int32_t, 2> extB(K, N);
dextents<int32_t, 2> extC(M, N);
tensor<device float, dextents<int32_t, 2>, tensor_inline> tA((device float*)A, extA);
tensor<device float, dextents<int32_t, 2>, tensor_inline> tB((device float*)B, extB);
tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, extC);

ctA.load(tA);
ctB.load(tB);
for (uint16_t i=0; i<ctC.get_capacity(); ++i) ctC[i] = 0.0f;
op.run(ctA, ctB, ctC);
ctC.store(tC);
"""


def make_source_tg_staged(M, N, K, relaxed):
    rp = "true" if relaxed else "false"
    return f"""
constexpr int M={M}, N={N}, K={K};
constexpr auto desc = matmul2d_descriptor(
    M, N, K,
    /*transpose_left=*/false,
    /*transpose_right=*/false,
    /*relaxed_precision=*/{rp},
    matmul2d_descriptor::mode::multiply);

threadgroup float tgA[M*K];
threadgroup float tgB[K*N];

uint tid = thread_position_in_threadgroup.x;
for (uint i = tid; i < M*K; i += 32) tgA[i] = ((device float*)A)[i];
for (uint i = tid; i < K*N; i += 32) tgB[i] = ((device float*)B)[i];
threadgroup_barrier(mem_flags::mem_threadgroup);

matmul2d<desc, execution_simdgroup> op;
auto ctA = op.get_left_input_cooperative_tensor<float, float, float>();
auto ctB = op.get_right_input_cooperative_tensor<float, float, float>();
auto ctC = op.get_destination_cooperative_tensor<decltype(ctA), decltype(ctB), float>();

dextents<int32_t, 2> extA(M, K);
dextents<int32_t, 2> extB(K, N);
dextents<int32_t, 2> extC(M, N);
tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tA(tgA, extA);
tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tB(tgB, extB);
tensor<device      float, dextents<int32_t, 2>, tensor_inline> tC(C, extC);

ctA.load(tA);
ctB.load(tB);
for (uint16_t i=0; i<ctC.get_capacity(); ++i) ctC[i] = 0.0f;
op.run(ctA, ctB, ctC);
ctC.store(tC);
"""


def run_one(M, N, K, rp):
    src = make_source(M, N, K, rp)
    name = f"sweep_M{M}N{N}K{K}_rp{int(rp)}"
    k = mx.fast.metal_kernel(name=name, input_names=["A", "B"],
                             output_names=["C"], source=src, header=HEADER)
    A_np = np.random.RandomState(0).randint(-3, 4, (M, K)).astype(np.float32)
    B_np = np.random.RandomState(1).randint(-3, 4, (K, N)).astype(np.float32)
    A = mx.array(A_np); B = mx.array(B_np)
    out = k(inputs=[A, B], grid=(32, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[(M, N)], output_dtypes=[mx.float32])
    mx.eval(out[0])
    ref = A_np @ B_np
    return float(np.max(np.abs(np.asarray(out[0]) - ref)))


def run_one_tg(M, N, K, rp):
    src = make_source_tg_staged(M, N, K, rp)
    name = f"tg_sweep_M{M}N{N}K{K}_rp{int(rp)}"
    k = mx.fast.metal_kernel(name=name, input_names=["A", "B"],
                             output_names=["C"], source=src, header=HEADER)
    A_np = np.random.RandomState(0).randint(-3, 4, (M, K)).astype(np.float32)
    B_np = np.random.RandomState(1).randint(-3, 4, (K, N)).astype(np.float32)
    A = mx.array(A_np); B = mx.array(B_np)
    out = k(inputs=[A, B], grid=(32, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[(M, N)], output_dtypes=[mx.float32])
    mx.eval(out[0])
    ref = A_np @ B_np
    return float(np.max(np.abs(np.asarray(out[0]) - ref)))


def main():
    configs = [
        (16, 32, 16, False),
        (16, 32, 16, True),
        (32, 16, 16, False),
        (32, 16, 16, True),
        (16, 16, 32, False),
        (16, 16, 32, True),
        (32, 32, 16, False),
        (32, 32, 16, True),
        (32, 32, 32, False),
        (32, 32, 32, True),
        (16, 32, 32, False),
        (32, 16, 32, False),
        (32, 64, 32, False),
        (64, 32, 32, False),
    ]
    print(f"{'(M, N, K, rp)':<22}  {'max|err|':>10}  verdict")
    print("-" * 50)
    for c in configs:
        try:
            err = run_one(*c)
            verdict = "OK" if err < 1e-3 else "WRONG"
            print(f"{str(c):<22}  {err:>10.4g}  {verdict}")
        except Exception as e:
            line = str(e).splitlines()[0][:80]
            print(f"{str(c):<22}  {'ERR':>10}  {line}")

    print()
    print(f"{'(M, N, K, rp) [tg-staged]':<28}  {'max|err|':>10}  verdict")
    print("-" * 56)
    for c in [(32, 32, 32, False), (32, 32, 32, True), (16, 32, 16, False)]:
        try:
            err = run_one_tg(*c)
            verdict = "OK" if err < 1e-3 else "WRONG"
            print(f"{str(c):<28}  {err:>10.4g}  {verdict}")
        except Exception as e:
            line = str(e).splitlines()[0][:80]
            print(f"{str(c):<28}  {'ERR':>10}  {line}")


if __name__ == "__main__":
    sys.exit(main())
