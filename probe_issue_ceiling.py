"""Find the real instruction issue ceiling on Xe2.

The methodology doc assumes 160 instr/clk (160 XVE x 1/clk) but that was never
tested: across 373 GEMM samples ISSUED/clk never went above 80.25.  If the true
ceiling is 80 rather than 160, then kernels sitting at ~77/clk are issue-bound
after all.

This kernel is almost pure scalar ALU work: one small load, a long chain of
independent arithmetic, one store.  If the machine can issue more than 80
instructions per clock, this is where it will show.

Run under:
    unitrace -q -g ComputeBasic python probe_issue_ceiling.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def scalar_storm(out_ptr, n_elements, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    # Eight independent accumulators so the ALU pipeline is never dependency
    # stalled -- we want to measure issue throughput, not latency.
    a0 = offs.to(tl.int32)
    a1 = a0 + 1
    a2 = a0 + 2
    a3 = a0 + 3
    a4 = a0 + 5
    a5 = a0 + 7
    a6 = a0 + 11
    a7 = a0 + 13
    for _ in range(ITERS):
        a0 = a0 * 3 + 1
        a1 = a1 * 5 + 2
        a2 = a2 * 7 + 3
        a3 = a3 * 11 + 5
        a4 = a4 ^ 0x5A5A
        a5 = a5 ^ 0x3C3C
        a6 = a6 + a0
        a7 = a7 + a1
    acc = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
    tl.store(out_ptr + offs, acc, mask=mask)


def main() -> None:
    n = 1 << 22
    out = torch.empty(n, device="xpu", dtype=torch.int32)
    BLOCK = 1024
    ITERS = 512
    grid = (triton.cdiv(n, BLOCK),)
    for _ in range(3):
        scalar_storm[grid](out, n, ITERS=ITERS, BLOCK=BLOCK, num_warps=8)
    torch.xpu.synchronize()
    for _ in range(20):
        scalar_storm[grid](out, n, ITERS=ITERS, BLOCK=BLOCK, num_warps=8)
    torch.xpu.synchronize()
    print("done", out[:4].tolist())


if __name__ == "__main__":
    main()
