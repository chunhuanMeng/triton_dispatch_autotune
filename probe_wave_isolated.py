"""Isolate wave quantization as a SINGLE variable.

Why a tile sweep cannot attribute anything
------------------------------------------
    T = M*N / (BM*BN)          D = BM*BN*BK / (nw*2048)
Changing the tile moves T and D together, so a tile sweep confounds
"scheduling granularity" with "instruction amortisation" and occupancy.
The first attempt (probe_wave_quantization.py) did exactly that and therefore
could not attribute its own result.

Clean design
------------
Hold the kernel config COMPLETELY fixed (tile, BK, num_warps, num_stages) and
vary N so that T = (M/BM)*(N/BN) lands at different remainders mod 20.
D, nw, BK and the K-loop body are then identical; comparing achieved TF/s
(work-normalised) isolates the scheduling loss.

    predicted cost = 20*ceil(T/20) / T

Constraints that must hold or the experiment is meaningless:
  * N % BN == 0        -> zero tile-edge padding, otherwise two effects mix
  * result verified    -> a wrong grid silently produces garbage
  * compare adjacent N -> keeps the A+B working set (vs 18 MB L2) similar

Run:
    python probe_wave_isolated.py
"""
import math
import statistics

import torch

from probe_wave_quantization import _gemm, bench

M, K = 2048, 2048
BM, BN, BK, NSTAGE, NW, GM = 128, 128, 32, 2, 16, 8
NUM_XE = 20
REP = 300
TRIALS = 5

# N = 128*j  ->  T = 16*j.  j sweeps the remainder mod 20.
J_LIST = [14, 15, 16, 17, 18, 19, 20, 21]


def run():
    a = torch.randn(M, K, device="xpu", dtype=torch.bfloat16)
    cases = []
    for j in J_LIST:
        N = BN * j
        T = (M // BM) * (N // BN)
        waves = math.ceil(T / NUM_XE)
        tail = T - (waves - 1) * NUM_XE
        cost = NUM_XE * waves / T
        b = torch.randn(K, N, device="xpu", dtype=torch.bfloat16)
        c = torch.empty(M, N, device="xpu", dtype=torch.bfloat16)

        def fn(a=a, b=b, c=c, N=N, T=T):
            return _gemm[(T,)](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
                num_stages=NSTAGE, num_warps=NW)

        fn()
        torch.xpu.synchronize()
        err = (c.float() - torch.mm(a, b).float()).abs().max().item()
        assert err < 1.0, f"N={N}: wrong result, max err={err}"
        ws = (M * K + K * N + M * N) * 2 / 2**20
        cases.append(dict(j=j, N=N, T=T, waves=waves, tail=tail, cost=cost,
                          flop=2 * M * N * K, ws=ws, fn=fn, t=[]))

    for c in cases:
        for _ in range(20):
            c["fn"]()
    torch.xpu.synchronize()
    for _ in range(TRIALS):
        for c in cases:
            c["t"].append(bench(c["fn"], warmup=5, rep=REP))

    for c in cases:
        c["med"] = statistics.median(c["t"])
        c["tf"] = c["flop"] / c["med"] * 1e-6

    print(f"固定 config: {BM}x{BN}x{BK} ns={NSTAGE} nw={NW} "
          f"D={BM*BN*BK//(NW*2048)} GROUP_M={GM}   M={M} K={K}，只改 N\n")
    print(f"{'N':>6} {'T':>5} {'波':>4} {'末波':>7} {'预测代价':>9} "
          f"{'us':>9} {'TF/s':>8} {'工作集MB':>9} {'std':>6}")
    print("-" * 76)
    for c in cases:
        tail = f"{c['tail']}/20"
        print(f"{c['N']:6d} {c['T']:5d} {c['waves']:4d} {tail:>7} "
              f"{c['cost']:9.4f} {c['med']:9.2f} {c['tf']:8.2f} {c['ws']:9.1f} "
              f"{statistics.stdev(c['t']):6.2f}")

    print("\n相邻 N 配对（工作集最接近，唯一差别是末波占用）：")
    print(f"{'对比':>20} {'预测提升':>9} {'实测提升':>9} {'差':>8}")
    print("-" * 50)
    for i in range(len(cases) - 1):
        lo, hi = cases[i], cases[i + 1]
        pred = lo["cost"] / hi["cost"]
        obs = hi["tf"] / lo["tf"]
        label = f"N={lo['N']}->{hi['N']}"
        print(f"{label:>20} {pred:9.4f} {obs:9.4f} {obs - pred:+8.4f}")


if __name__ == "__main__":
    run()
