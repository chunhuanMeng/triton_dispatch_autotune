#!/usr/bin/env python3
"""Fixed-config standalone parity benchmark for the three Triton templates.

This is phase 1 of the worker/Inductor parity experiment.  It deliberately
benchmarks the exact kernels defined in bench_worker.py, using the same input
layout, grids, configs, correctness reference, and amortized event timing.
Inductor generated-kernel measurements are added only after this phase passes.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass

import torch
import triton

from bench_worker import (
    NUM_SMS,
    kernel_bmg_decode,
    kernel_bmg_persistent,
    kernel_triton_mm,
)


@dataclass(frozen=True)
class Case:
    template: str
    shape: tuple[int, int, int]
    config: tuple[int, int, int, int, int]


CASES = [
    # Standard configs must come from Inductor's int8_mm_configs, not from
    # the BMG-only Oracle lists.
    Case("triton_mm", (1, 2048, 1408), (64, 64, 32, 2, 4)),
    Case("triton_mm", (4, 2048, 768), (128, 128, 32, 2, 8)),
    Case("triton_mm", (32, 4096, 4096), (128, 128, 32, 2, 8)),
    Case("triton_mm", (32, 4096, 7168), (256, 128, 128, 3, 8)),
    # BMG persistent and tiled2d use their template-specific Oracle lists.
    Case("bmg_persistent", (1, 2048, 1408), (8, 512, 32, 2, 8)),
    Case("bmg_persistent", (4, 2048, 768), (8, 512, 32, 2, 16)),
    Case("bmg_persistent", (32, 4096, 4096), (32, 256, 32, 2, 8)),
    Case("bmg_persistent", (32, 4096, 7168), (32, 256, 32, 2, 8)),
    Case("bmg_decode", (1, 2048, 1408), (8, 512, 32, 2, 8)),
    Case("bmg_decode", (4, 2048, 768), (8, 512, 32, 2, 16)),
    Case("bmg_decode", (4, 2048, 1408), (32, 256, 32, 2, 8)),
    Case("bmg_decode", (32, 4096, 4096), (32, 256, 32, 2, 8)),
    Case("bmg_decode", (32, 4096, 7168), (32, 256, 32, 2, 8)),
    # Persistent-native configs from ORACLE_PERFORMANCE.md.
    Case("bmg_persistent", (512, 5120, 5120), (128, 512, 64, 2, 32)),
    Case("bmg_persistent", (1024, 5120, 5120), (128, 512, 64, 2, 32)),
    Case("bmg_persistent", (2048, 4096, 4096), (256, 128, 64, 3, 16)),
]


def make_call(A, B, C, case: Case):
    M, N, K = case.shape
    bm, bn, bk, ns, nw = case.config
    strides = (
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )

    if case.template == "triton_mm":
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        def call():
            return kernel_triton_mm[grid](
                A, B, C,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                GROUP_M=8, EVEN_K=(K % bk == 0), USE_FAST_ACCUM=False,
                M=M, N=N, K=K,
                stride_am=strides[0], stride_ak=strides[1],
                stride_bk=strides[2], stride_bn=strides[3],
                stride_cm=strides[4], stride_cn=strides[5],
                num_stages=ns, num_warps=nw,
            )

    elif case.template == "bmg_persistent":
        grid = (min(NUM_SMS, triton.cdiv(M, bm) * triton.cdiv(N, bn)),)

        def call():
            return kernel_bmg_persistent[grid](
                A, B, C,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                GROUP_M=8, NUM_SMS=NUM_SMS,
                M=M, N=N, K=K,
                stride_am=strides[0], stride_ak=strides[1],
                stride_bk=strides[2], stride_bn=strides[3],
                stride_cm=strides[4], stride_cn=strides[5],
                num_stages=ns, num_warps=nw,
            )

    elif case.template == "bmg_decode":
        grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))

        def call():
            return kernel_bmg_decode[grid](
                A, B, C,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                M=M, N=N, K=K,
                stride_am=strides[0], stride_ak=strides[1],
                stride_bk=strides[2], stride_bn=strides[3],
                stride_cm=strides[4], stride_cn=strides[5],
                num_stages=ns, num_warps=nw,
            )

    else:
        raise ValueError(case.template)

    return call, grid


def benchmark(case: Case, warmup: int, iterations: int, trials: int) -> dict:
    M, N, K = case.shape
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="xpu")
    B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device="xpu")
    reference = torch._int_mm(A, B)
    C = torch.zeros((M, N), dtype=torch.int32, device="xpu")
    call, grid = make_call(A, B, C, case)

    for _ in range(warmup):
        call()
    torch.xpu.synchronize()
    wrong = torch.count_nonzero(C != reference).item()
    max_abs_diff = (C - reference).abs().max().item()

    times_us = []
    for _ in range(trials):
        C.zero_()
        for _ in range(warmup):
            call()
        torch.xpu.synchronize()
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        torch.xpu.synchronize()
        times_us.append(start.elapsed_time(end) / iterations * 1000.0)

    time_us = statistics.median(times_us)
    tops = 2.0 * M * N * K / (time_us * 1e-6) / 1e12
    bw = (M * K + K * N + 4 * M * N) / (time_us * 1e-6) / 1e9
    return {
        "template": case.template,
        "M": M, "N": N, "K": K,
        "BM": case.config[0], "BN": case.config[1], "BK": case.config[2],
        "num_stages": case.config[3], "num_warps": case.config[4],
        "grid": str(grid),
        "correct": int(wrong == 0),
        "wrong_elements": wrong,
        "max_abs_diff": max_abs_diff,
        "trial_times_us": ";".join(f"{x:.4f}" for x in times_us),
        "time_us": time_us,
        "tops": tops,
        "tops_eff_pct": tops / 234.0 * 100.0,
        "bw_gbps": bw,
        "bw_eff_pct": bw / 456.0 * 100.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--template", choices=("triton_mm", "bmg_persistent", "bmg_decode"))
    parser.add_argument("--csv", default="standalone_fixed_config_results.csv")
    args = parser.parse_args()

    cases = [c for c in CASES if args.template is None or c.template == args.template]
    rows = []
    for case in cases:
        print(f"RUN {case.template} shape={case.shape} config={case.config}", flush=True)
        row = benchmark(case, args.warmup, args.iterations, args.trials)
        rows.append(row)
        print(
            f"  correct={row['correct']} wrong={row['wrong_elements']} "
            f"grid={row['grid']} time_us={row['time_us']:.3f} "
            f"TOPS={row['tops']:.2f} BW={row['bw_gbps']:.1f}",
            flush=True,
        )

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {args.csv}")


if __name__ == "__main__":
    main()
