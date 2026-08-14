"""Where does the un-captured scheduling headroom on (2048,2048,2048) go?

oneDNN pays a 1.09375x wave-quantization penalty on this shape.  A Triton tile
with T=1024 pays only 1.0156x, so ~7.7% ought to be available -- yet the measured
win is only 2.5%.  This script asks the counters where the other ~5% went.

Hypothesis: Triton's K-loop still carries the descriptor-rebuild overhead
(instr/dpas, ALU1 slots, extra L1 traffic) documented in experiments 5 and 9,
and that eats most of the scheduling advantage.

Run:
    unitrace -q -g ComputeBasic python probe_residual_gap.py > /tmp/uni_gap.log 2>&1
"""
import torch

from probe_wave_quantization import _gemm

M = N = K = 2048
ITERS = 40

CONFIGS = [
    ("triton_64x64_nw4",    64,  64, 32, 2,  4, 8),
    ("triton_128x128_nw16", 128, 128, 32, 2, 16, 8),
    ("triton_256x128_nw16", 256, 128, 32, 2, 16, 8),
]


def main():
    a = torch.randn(M, K, device="xpu", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="xpu", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="xpu", dtype=torch.bfloat16)

    # oneDNN first so it is easy to find in the trace
    for _ in range(5):
        torch.mm(a, b, out=c)
    torch.xpu.synchronize()
    for _ in range(ITERS):
        torch.mm(a, b, out=c)
    torch.xpu.synchronize()

    for name, BM, BN, BK, ns, nw, gm in CONFIGS:
        T = (M // BM) * (N // BN)

        def fn():
            return _gemm[(T,)](a, b, c, M, N, K,
                               a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                               c.stride(0), c.stride(1),
                               BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=gm,
                               num_stages=ns, num_warps=nw)
        for _ in range(5):
            fn()
        torch.xpu.synchronize()
        for _ in range(ITERS):
            fn()
        torch.xpu.synchronize()
        print(f"ran {name}: grid={T} local={nw*16}")


if __name__ == "__main__":
    main()
