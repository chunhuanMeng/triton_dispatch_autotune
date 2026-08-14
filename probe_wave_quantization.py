"""Test the wave-quantization hypothesis from XE2_PERF_METHODOLOGY.md exp 8b.

Claim under test
----------------
On (2048,2048,2048) oneDNN uses T=128 tiles on 20 persistent workgroups
-> ceil(128/20)=7 waves, last wave only 8/20 occupied -> 1.094x scheduling cost.
A Triton tile giving T=256 (128x128) should land at 13 waves / 16-of-20 tail
-> 1.016x, i.e. ~7.7% less scheduling loss.

Two things must be checked separately, because they are NOT the same:
  1. ALU2_events == M*N*K/256 exactly?  (Triton should never compute padding
     tiles, so this is expected to be 1.00 for every tile that divides evenly.
     It says nothing about speed.)
  2. Does wall time actually track ceil(T/20)/(T/20)?  This is the real question.
     If the hardware keeps >1 workgroup resident per Xe core, the quantization
     granularity is not 20 and the model does not apply.

Run plain for timings:
    python probe_wave_quantization.py
Run under unitrace for ALU2_events / occupancy:
    unitrace -q -g ComputeBasic python probe_wave_quantization.py 2>&1 | tee /tmp/uni_wave.log
"""
import math
import sys

import torch
import triton
import triton.language as tl

M, N, K = 2048, 2048, 2048
NUM_XE_CORES = 20


@triton.jit
def _gemm(A, B, C, M, N, K,
          sam, sak, sbk, sbn, scm, scn,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
          GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(sam, sak),
                              offsets=(pid_m * BLOCK_M, 0),
                              block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(sbk, sbn),
                              offsets=(0, pid_n * BLOCK_N),
                              block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_ptr, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BLOCK_K))
        b_ptr = tl.advance(b_ptr, (BLOCK_K, 0))
    c_ptr = tl.make_block_ptr(base=C, shape=(M, N), strides=(scm, scn),
                              offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
                              block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    tl.store(c_ptr, acc.to(tl.bfloat16), boundary_check=(0, 1))


# (BM, BN, BK, num_stages, num_warps, GROUP_M)
CONFIGS = [
    (256, 256, 32, 2, 32, 8),
    (256, 128, 32, 2, 16, 8),   # what oneDNN uses -> T=128
    (128, 256, 32, 2, 16, 8),
    (128, 128, 32, 2,  8, 8),   # the proposal -> T=256
    (128, 128, 32, 3,  8, 8),
    (128, 128, 64, 2,  8, 8),
    (128, 128, 32, 2, 16, 8),
    (128,  64, 32, 2,  4, 8),   # T=512
    ( 64, 128, 32, 2,  4, 8),
    ( 64,  64, 32, 2,  4, 8),   # T=1024
]

REP = 100


def bench(fn, warmup=20, rep=REP):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.xpu.synchronize()
    return s.elapsed_time(e) / rep * 1e3   # us


def main():
    torch.manual_seed(0)
    a = torch.randn(M, K, device="xpu", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="xpu", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="xpu", dtype=torch.bfloat16)
    ref = torch.mm(a, b)

    flop = 2 * M * N * K
    t_dnn = bench(lambda: torch.mm(a, b, out=c))
    print(f"shape = ({M},{N},{K}) bf16   算法下界 ALU2 = {M*N*K//256:,}\n")
    print(f"{'tile':>14} {'ns':>3} {'nw':>3} {'T':>5} {'波':>4} {'末波':>6} "
          f"{'调度代价':>9} {'us':>9} {'TF/s':>7} {'vs oneDNN':>10} {'max err':>9}")
    print("-" * 100)
    print(f"{'oneDNN':>14} {'-':>3} {'-':>3} {'128':>5} {'7':>4} {'8/20':>6} "
          f"{1.09375:9.3f} {t_dnn:9.2f} {flop/t_dnn*1e-6:7.2f} {1.0:10.3f} {'-':>9}")

    rows = []
    for BM, BN, BK, ns, nw, gm in CONFIGS:
        T = math.ceil(M / BM) * math.ceil(N / BN)
        waves = math.ceil(T / NUM_XE_CORES)
        tail = T - (waves - 1) * NUM_XE_CORES
        cost = NUM_XE_CORES * waves / T
        grid = (T,)
        try:
            fn = lambda: _gemm[grid](a, b, c, M, N, K,
                                     a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                     c.stride(0), c.stride(1),
                                     BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=gm,
                                     num_stages=ns, num_warps=nw)
            fn(); torch.xpu.synchronize()
            err = (c.float() - ref.float()).abs().max().item()
            t = bench(fn)
        except Exception as ex:
            print(f"{f'{BM}x{BN}x{BK}':>14} {ns:3d} {nw:3d} {T:5d} {waves:4d} "
                  f"{f'{tail}/20':>6} {cost:9.3f}   FAIL: {str(ex)[:40]}")
            continue
        rows.append((BM, BN, BK, ns, nw, T, cost, t))
        print(f"{f'{BM}x{BN}x{BK}':>14} {ns:3d} {nw:3d} {T:5d} {waves:4d} "
              f"{f'{tail}/20':>6} {cost:9.3f} {t:9.2f} {flop/t*1e-6:7.2f} "
              f"{t_dnn/t:10.3f} {err:9.3f}")

    if rows:
        best = min(rows, key=lambda r: r[-1])
        print(f"\n最快 Triton: {best[0]}x{best[1]}x{best[2]} ns={best[3]} nw={best[4]} "
              f"T={best[5]} 调度代价={best[6]:.3f} -> {best[7]:.2f} us "
              f"({t_dnn/best[7]:.3f}x oneDNN)")
        print("\n调度代价 vs 实测时间的相关性（若模型成立，两列应同向）：")
        for r in sorted(rows, key=lambda r: r[6]):
            print(f"  代价 {r[6]:.3f}  ->  {r[7]:8.2f} us   ({r[0]}x{r[1]}, T={r[5]})")


if __name__ == "__main__":
    main()
