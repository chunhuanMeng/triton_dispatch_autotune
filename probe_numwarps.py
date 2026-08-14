"""Why does num_warps 8 -> 16 give +11% at identical tile / T / scheduling cost?

(2048,2048,2048), tile 128x128x32, num_stages=2, GROUP_M=8:
    nw=8   ->  236.39 us
    nw=16  ->  214.31 us
T=256 and the wave-quantisation cost (1.0156) are identical, so the difference
must come from inside the workgroup.

Leading hypothesis: register pressure.
    accumulator floats per lane = BM*BN / (nw*16)
        nw=8  -> 128  (likely forces large-GRF mode -> 4 HW threads/XVE)
        nw=16 ->  64  (fits the default 128-GRF budget -> 8 HW threads/XVE)
If nw=8 halves the threads per XVE, occupancy halves and the latency hiding
collapses -- which would show up as n_regs / n_spills / GRF mode, not as
instruction count.

This script prints Triton's own compile-time metadata for a ladder of configs.
"""
import torch

from probe_wave_quantization import _gemm

M = N = K = 2048
BM, BN, BK, NSTAGE, GM = 128, 128, 32, 2, 8

CONFIGS = [
    # (BM, BN, BK, num_stages, num_warps)
    (128, 128, 32, 2, 4),
    (128, 128, 32, 2, 8),
    (128, 128, 32, 2, 16),
    (128, 128, 32, 2, 32),
    (256, 128, 32, 2, 16),
    (256, 128, 32, 2, 32),
    (64, 64, 32, 2, 4),
    (64, 64, 32, 2, 8),
]


def main():
    a = torch.randn(M, K, device="xpu", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="xpu", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="xpu", dtype=torch.bfloat16)

    print(f"{'tile':>14} {'nw':>3} {'D':>4} {'acc float/lane':>15} "
          f"{'n_regs':>7} {'n_spills':>9} {'shared':>7} {'threads/WG':>11}")
    print("-" * 86)
    for bm, bn, bk, ns, nw in CONFIGS:
        T = (M // bm) * (N // bn)
        k = _gemm[(T,)](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=GM,
                        num_stages=ns, num_warps=nw)
        torch.xpu.synchronize()
        D = bm * bn * bk // (nw * 2048)
        acc = bm * bn // (nw * 16)
        n_regs = getattr(k, "n_regs", "?")
        n_spills = getattr(k, "n_spills", "?")
        shared = getattr(k, "shared", getattr(k, "metadata", None)
                         and getattr(k.metadata, "shared", "?"))
        print(f"{f'{bm}x{bn}x{bk}':>14} {nw:3d} {D:4d} {acc:15d} "
              f"{str(n_regs):>7} {str(n_spills):>9} {str(shared):>7} {nw*16:11d}")


if __name__ == "__main__":
    main()
