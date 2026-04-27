# Copyright © 2026 Apple Inc.
"""Phase 4 §6.2 diagnostic microbench: lane-load-width throughput on M4 Pro.

Compares three custom Metal kernels that each touch the same total byte volume
but with different per-lane load widths, to test whether the q4 vector SDPA
kernel's 2-byte-per-lane uint16 load is structurally throttled vs wider loads.

Hypothesis (from Phase 4 §5): four cumulative kernel-rewrite attempts targeting
per-simdgroup ALU and register pressure haven't moved the bench. The remaining
candidate bottleneck is memory-load *issue rate* — Apple GPU may have load-port
limits that throttle 2-byte loads more than 4- or 8-byte loads per byte
delivered.

Each kernel iterates a per-lane sum over a flat 64 MB buffer, with iteration
count scaled inversely to load width so total bytes touched is constant. The
kernel writes the per-lane accumulator to an output array so the compiler
can't dead-store-eliminate the loads.

If kernel-B (uint32 / 4 B) is much faster than kernel-A (uint16 / 2 B) on the
same byte volume, load-width is a real lever and Phase 5 should widen the q4
kernel's lane loads. If they're within noise, load-width isn't the issue and
the kernel ceiling per Phase 4 §6.4 stands.

See docs/superpowers/reports/2026-04-27-sdpa-phase4-attempt.md §5–§6.
"""

import argparse
import statistics
import time

import mlx.core as mx


# Total byte volume per kernel run, in bytes.
TOTAL_BYTES = 64 * 1024 * 1024  # 64 MB

# Threadgroup config: 32 lanes per simdgroup, one simdgroup per threadgroup,
# many threadgroups across the GPU. Mirrors the SDPA vector kernel's tile
# topology.
LANES_PER_SIMDGROUP = 32
SIMDGROUPS_PER_TG = 1


# Each kernel: each lane reads ITERS values from `buf` at strided offsets,
# accumulates a uint sum, writes to `out` keyed on lane_id.
# `buf` is a flat uint32 buffer (we cast inside for narrower loads).

KERNEL_BODY_UINT16 = """
    uint tg_id = threadgroup_position_in_grid.x;
    uint lane_id = thread_index_in_simdgroup;
    uint global_lane = tg_id * 32 + lane_id;
    const device uint16_t* buf16 = (const device uint16_t*)buf;
    uint sum = 0;
    for (uint i = 0; i < ITERS; i++) {
        sum += (uint)buf16[global_lane + i * GLOBAL_STRIDE];
    }
    out[global_lane] = sum;
"""

KERNEL_BODY_UINT32 = """
    uint tg_id = threadgroup_position_in_grid.x;
    uint lane_id = thread_index_in_simdgroup;
    uint global_lane = tg_id * 32 + lane_id;
    const device uint32_t* buf32 = (const device uint32_t*)buf;
    uint sum = 0;
    for (uint i = 0; i < ITERS; i++) {
        sum += buf32[global_lane + i * GLOBAL_STRIDE];
    }
    out[global_lane] = sum;
"""

KERNEL_BODY_UINT2X32 = """
    uint tg_id = threadgroup_position_in_grid.x;
    uint lane_id = thread_index_in_simdgroup;
    uint global_lane = tg_id * 32 + lane_id;
    const device uint2* buf2 = (const device uint2*)buf;
    uint sum = 0;
    for (uint i = 0; i < ITERS; i++) {
        uint2 v = buf2[global_lane + i * GLOBAL_STRIDE];
        sum += v.x + v.y;
    }
    out[global_lane] = sum;
"""


def make_kernel(name: str, body: str):
    return mx.fast.metal_kernel(
        name=name,
        input_names=["buf"],
        output_names=["out"],
        source=body,
    )


def time_kernel(kernel, buf, lane_total: int, iters: int, global_stride: int,
                warmup: int, samples: int):
    body_wrapped = (
        kernel.__wrapped__ if hasattr(kernel, "__wrapped__") else kernel
    )

    grid = (lane_total, 1, 1)
    threadgroup = (LANES_PER_SIMDGROUP, 1, 1)

    def run():
        return kernel(
            inputs=[buf],
            template=[("ITERS", iters), ("GLOBAL_STRIDE", global_stride)],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(lane_total,)],
            output_dtypes=[mx.uint32],
        )[0]

    for _ in range(warmup):
        out = run()
        mx.eval(out)
    times = []
    for _ in range(samples):
        tic = time.perf_counter()
        out = run()
        mx.eval(out)
        times.append((time.perf_counter() - tic) * 1e3)
    times.sort()
    median = statistics.median(times)
    p10 = times[max(0, int(0.10 * samples) - 1)]
    p90 = times[min(samples - 1, int(0.90 * samples) - 1)]
    return median, p10, p90


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simdgroups", type=int, default=2048,
                   help="Number of simdgroups dispatched (= total lanes / 32).")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    lane_total = args.simdgroups * LANES_PER_SIMDGROUP
    # Buffer big enough that all lanes' iters fit, with stride = lane_total.
    buf_bytes = TOTAL_BYTES
    buf_uint32_count = buf_bytes // 4
    buf = mx.zeros(buf_uint32_count, dtype=mx.uint32) + mx.array(7, dtype=mx.uint32)
    mx.eval(buf)

    # Per-kernel: bytes per lane per iter, iter count chosen so each lane
    # touches ~equal bytes. Cap at buffer size / lanes / bytes_per_iter
    # so we never read out of bounds.
    bytes_per_lane_target = TOTAL_BYTES // lane_total

    runs = [
        ("uint16 (2 B/lane)", KERNEL_BODY_UINT16,
         bytes_per_lane_target // 2, 2),
        ("uint32 (4 B/lane)", KERNEL_BODY_UINT32,
         bytes_per_lane_target // 4, 4),
        ("uint2  (8 B/lane)", KERNEL_BODY_UINT2X32,
         bytes_per_lane_target // 8, 8),
    ]

    print("Lane-load-width microbench. Phase 4 §6.2.")
    print(f"  total bytes touched: {TOTAL_BYTES // (1024*1024)} MB")
    print(f"  simdgroups: {args.simdgroups}")
    print(f"  lanes total: {lane_total}")
    print(f"  iters/sample: {args.samples} (warmup {args.warmup})")
    print()
    print(f"  {'kernel':<25}  {'iters/lane':>10}  {'median ms':>9}  "
          f"{'p10':>7}  {'p90':>7}  {'effective GB/s':>14}")

    for name, body, iters, bytes_per_iter in runs:
        kernel = make_kernel(name.split()[0], body)
        # global_stride = lane_total in uint16/uint32/uint2 elements
        # but buf is uint32; for uint16 stride is lane_total*2, for uint32 it
        # equals lane_total, for uint2 stride is lane_total/2.
        # We pass the right stride for each kernel.
        if "uint16" in name:
            global_stride = lane_total
        elif "uint32" in name:
            global_stride = lane_total
        else:  # uint2
            global_stride = lane_total
        median, p10, p90 = time_kernel(
            kernel, buf, lane_total, iters, global_stride,
            args.warmup, args.samples,
        )
        bytes_touched = lane_total * iters * bytes_per_iter
        gb_s = (bytes_touched / 1e9) / (median / 1e3)
        print(f"  {name:<25}  {iters:>10}  {median:>9.4f}  "
              f"{p10:>7.4f}  {p90:>7.4f}  {gb_s:>13.1f}")


if __name__ == "__main__":
    main()
