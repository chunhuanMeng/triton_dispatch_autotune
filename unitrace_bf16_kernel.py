"""
Simple standalone kernel-only script for unitrace device-timing analysis.
Usage: python unitrace_bf16_kernel.py <mode> [M N K BM BN BK NS NW]
    mode: 'triton' or 'onednn'
Defaults are the original shape/config; all dimensions and Triton config fields
can be overridden for an isolated device-timing run.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton
from bench_worker import kernel_bmg_decode, _make_inputs
from search_space import GemmConfig

MODE = sys.argv[1]
values = [int(value) for value in sys.argv[2:]]
defaults = [128, 2048, 768, 32, 256, 32, 2, 8]
M, N, K, bm, bn, bk, ns, nw = (values + defaults[len(values):])
NUM_ITERS = 20

A, B, C, _ = _make_inputs(M, N, K, "bf16")
cfg = GemmConfig(bm, bn, bk, ns, nw)

if MODE == "triton":
    bm, bn, bk = cfg.BLOCK_M, cfg.BLOCK_N, cfg.BLOCK_K
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))

    def fn():
        kernel_bmg_decode[grid](
            A, B, C,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            ACC_TYPE=triton.language.float32,
            M=M, N=N, K=K,
            stride_am=A.stride(0), stride_ak=A.stride(1),
            stride_bk=B.stride(0), stride_bn=B.stride(1),
            stride_cm=C.stride(0), stride_cn=C.stride(1),
            num_stages=cfg.num_stages, num_warps=cfg.num_warps,
        )
else:  # onednn
    def fn():
        torch.mm(A, B, out=C)

# Warmup (also ensures Triton JIT compilation happens outside the profiled region)
for _ in range(10):
    fn()
torch.xpu.synchronize()

# Profiled region
for _ in range(NUM_ITERS):
    fn()
torch.xpu.synchronize()

print(f"Done: {MODE} ({M},{N},{K}) x {NUM_ITERS}")
