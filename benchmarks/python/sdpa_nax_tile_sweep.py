# Copyright © 2026 Apple Inc.
"""Phase 6 sweep harness: NAX SDPA tile sizes for D=128 prefill.

Sweeps (bq, bk) ∈ {32, 64, 128} × {32, 64} over the prefill SDPA
benchmark using the MLX_NAX_SDPA_BQ / MLX_NAX_SDPA_BK env-var
overrides plumbed in mlx/backend/metal/scaled_dot_product_attention.cpp.

Each combo runs in a subprocess so the env override is fresh per
combo. Reports median fp16 prefill ms per (model, qL, combo) and
identifies the winner per cell.
"""

import argparse
import csv
import io
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFILL_BENCH = REPO_ROOT / "benchmarks" / "python" / "sdpa_prefill_bench.py"
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")


# (bq, bk) combos AOT-instantiated in steel_attention_nax.metal.
TILE_COMBOS = [
    (32, 32),
    (32, 64),
    (64, 32),   # current default
    (64, 64),
    (128, 32),
    (128, 64),
]


TABLE_RE = re.compile(
    r"model=(\S+)\s+Hq=\d+\s+Hk=\d+\s+D=\d+\s*\n"
    r"\s+qL\s+fp16_prefill[^\n]*\n"
    r"((?:\s+\d+[^\n]*\n)+)",
    re.MULTILINE,
)


def parse_table(stdout: str):
    """Extract (model, qL) -> fp16_prefill_median rows."""
    rows = {}
    for m in TABLE_RE.finditer(stdout):
        model = m.group(1)
        for line in m.group(2).strip().split("\n"):
            parts = line.split()
            if not parts:
                continue
            qL = int(parts[0])
            fp16_ms = float(parts[1])
            q4_dq_ms = float(parts[2])
            rows[(model, qL)] = {"fp16_prefill": fp16_ms, "q4_dq_prefill": q4_dq_ms}
    return rows


def run_combo(bq: int, bk: int, qLs: list[int], iters: int, warmup: int,
              models: str):
    env = os.environ.copy()
    env["MLX_NAX_SDPA_BQ"] = str(bq)
    env["MLX_NAX_SDPA_BK"] = str(bk)
    cmd = [
        PYTHON, str(PREFILL_BENCH),
        "--qL", ",".join(str(q) for q in qLs),
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--models", models,
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        print(f"  SUBPROCESS FAILED bq={bq} bk={bk}:", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(1)
    return parse_table(proc.stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qL", default="1024,2048,4096,8192")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--models", default="llama3-8b,qwen-gqa")
    p.add_argument("--out", help="optional CSV output path")
    args = p.parse_args()

    qLs = [int(c) for c in args.qL.split(",") if c.strip()]

    print(f"NAX SDPA tile sweep on M4 Pro. {len(TILE_COMBOS)} combos × "
          f"{len(args.models.split(','))} models × {len(qLs)} contexts.")
    print(f"iters={args.iters} warmup={args.warmup}")
    print()

    all_results = {}  # (bq, bk) -> {(model, qL): {fp16_prefill, q4_dq_prefill}}
    for bq, bk in TILE_COMBOS:
        print(f"  running (bq={bq}, bk={bk}) ...", flush=True)
        all_results[(bq, bk)] = run_combo(bq, bk, qLs, args.iters, args.warmup,
                                          args.models)

    # Build comparison table.
    models = args.models.split(",")
    print()
    print("fp16_prefill median ms per (model, qL, tile combo):")
    print()
    header = f"  {'model':<12} {'qL':>6}  " + "  ".join(
        f"{f'({bq},{bk})':>10}" for bq, bk in TILE_COMBOS) + "  winner"
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows_for_csv = []
    for model in models:
        for qL in qLs:
            cells = []
            for tile in TILE_COMBOS:
                fp16 = all_results[tile].get((model, qL), {}).get("fp16_prefill")
                cells.append((tile, fp16))
            valid = [(t, v) for t, v in cells if v is not None]
            if not valid:
                continue
            best_tile, best_ms = min(valid, key=lambda x: x[1])
            line = f"  {model:<12} {qL:>6}  "
            for tile, ms in cells:
                if ms is None:
                    line += f"{'n/a':>10}  "
                else:
                    marker = "*" if tile == best_tile else " "
                    line += f"{ms:>9.4f}{marker} "
            line += f" {best_tile}"
            print(line)
            for tile, ms in cells:
                if ms is not None:
                    rows_for_csv.append((model, qL, tile[0], tile[1], ms))

    print()
    print("  * = best at this (model, qL).")
    print(f"  default = (64, 32). Winners that aren't (64, 32) suggest a")
    print(f"  default-change candidate.")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "qL", "bq", "bk", "fp16_prefill_ms"])
            for row in rows_for_csv:
                w.writerow(row)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
