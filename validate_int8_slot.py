"""Calibrate the INT8 ALU2 slot constant on Xe2.

BF16 is already established (see XE2_PERF_METHODOLOGY.md 1.5-1.7):
    1 ALU2 slot = 256 MAC = 512 FLOP   ->  peak = 20 * 2048 * f

For INT8 each 32-bit register slot packs 4 elements instead of 2, so one
dpas.8x8 should eat 32 K-steps instead of 16:
    1 ALU2 slot = 512 MAC = 1024 OPS   ->  peak = 20 * 4096 * f = 196.6 TOPS @2.4GHz

This script only *runs* the GEMMs; the check is done by parsing the unitrace
CSV afterwards:
    M*N*K / XVE_INST_EXECUTED_ALU2_ALL  ==  512   (exactly, if the model holds)

Run:
    unitrace -q -g ComputeBasic python validate_int8_slot.py 12 > /tmp/uni_int8.log
"""
import sys

import torch

SHAPES = [
    (8192, 8192, 8192),
    (4096, 4096, 4096),
    (2048, 7168, 28672),
    (2048, 5120, 5120),
]

if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    for M, N, K in SHAPES:
        a = torch.randint(-8, 8, (M, K), device="xpu", dtype=torch.int8)
        b = torch.randint(-8, 8, (K, N), device="xpu", dtype=torch.int8)
        for _ in range(5):
            c = torch._int_mm(a, b)
        torch.xpu.synchronize()
        for _ in range(iters):
            c = torch._int_mm(a, b)
        torch.xpu.synchronize()
        del a, b, c
        torch.xpu.empty_cache()
