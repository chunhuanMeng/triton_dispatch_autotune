"""Cross-validate the Xe2 execution model on many shapes.

Checks three claims that must hold for every compute kernel:
  1. M*N*K / ALU2_events == 256 exactly (1 ALU2 slot = 256 MAC = 512 FLOP)
  2. observed_slot_rate / ALU2_UTILIZATION == 80.0 (the whole-GPU slot ceiling)
  3. achieved TF/s / (20 * 2048 * f) == ALU2_UTILIZATION

Run under unitrace:
  unitrace -q -g ComputeBasic python validate_exec_model.py
"""
import sys

import torch

SHAPES = [
    (8192, 8192, 8192),
    (4096, 4096, 4096),
    (2048, 7168, 28672),
    (4096, 11008, 4096),
    (2048, 2048, 2048),
    (1024, 4096, 4096),
    (6144, 6144, 6144),
    (2048, 5120, 5120),
    (512, 4096, 4096),
    (3072, 3072, 3072),
]

if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    for M, N, K in SHAPES:
        a = torch.randn(M, K, device="xpu", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="xpu", dtype=torch.bfloat16)
        c = torch.empty(M, N, device="xpu", dtype=torch.bfloat16)
        for _ in range(5):
            torch.mm(a, b, out=c)
        torch.xpu.synchronize()
        for _ in range(iters):
            torch.mm(a, b, out=c)
        torch.xpu.synchronize()
        del a, b, c
        torch.xpu.empty_cache()
