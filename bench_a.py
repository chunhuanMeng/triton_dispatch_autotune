#!/usr/bin/env python3
"""
Simple benchmark to test triton_mm kernel performance.
Compares bench_worker triton_mm vs Inductor triton_mm.
"""
import torch
import triton
import triton.language as tl

# Config and shape for testing
M, N, K = 512, 4096, 4096
bm, bn, bk, ns, nw = 256, 128, 64, 3, 16


# ═══ Template 1: bench_worker triton_mm (with Inductor optimizations) ═══
@triton.jit
def kernel_triton_mm(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K).to(tl.int32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k_idx * BLOCK_K
        a_mask = offs_k[None, :] < (K - k_off)
        b_mask = offs_k[:, None] < (K - k_off)
        a_k_idx_vals = offs_k[None, :] + k_off
        b_k_idx_vals = offs_k[:, None] + k_off

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        a = tl.load(A + idx_m * stride_am + idx_n * stride_ak, mask=a_mask, other=0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        b = tl.load(B + idx_m * stride_bk + idx_n * stride_bn, mask=b_mask, other=0)

        acc = tl.dot(a, b, acc, out_dtype=tl.int32)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    tl.store(C + idx_m * stride_cm + idx_n * stride_cn, acc, mask=mask)


def main():
    print(f"Testing triton_mm kernel")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"Config: BM={bm}, BN={bn}, BK={bk}, NS={ns}, NW={nw}")
    print("-" * 50)

    # Create tensors
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='xpu')
    B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='xpu')
    C = torch.zeros((M, N), dtype=torch.int32, device='xpu')

    # Benchmark bench_worker triton_mm
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    fn = lambda: kernel_triton_mm[grid](A, B, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_stages=ns, num_warps=nw)

    # Warmup
    print("Warming up...")
    for _ in range(5):
        fn()
    torch.xpu.synchronize()

    # Benchmark
    print("Benchmarking...")
    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    torch.xpu.synchronize()
    s.record()
    for _ in range(200):
        fn()
    e.record()
    torch.xpu.synchronize()
    t1 = s.elapsed_time(e) / 200 * 1000
    print(f"bench_worker triton_mm: {t1:.2f} us")

    # Now test Inductor via torch.compile
    print("-" * 50)
    print("Testing Inductor triton_mm via torch.compile...")

    def mm(A, B):
        return torch.ops.aten.mm(A, B)

    # Warmup for compile
    compiled_mm = torch.compile(mm, backend='inductor')
    out = compiled_mm(A, B)
    torch.xpu.synchronize()

    s.record()
    for _ in range(200):
        out = compiled_mm(A, B)
    e.record()
    torch.xpu.synchronize()
    t2 = s.elapsed_time(e) / 200 * 1000
    print(f"Inductor triton_mm: {t2:.2f} us")

    print("-" * 50)
    print(f"Ratio (Inductor/bench_worker): {t2/t1:.2f}x")
    if t2/t1 < 1.1:
        print("PASS: Performance is close!")
    else:
        print(f"WARN: Inductor is {t2/t1:.2f}x slower than bench_worker")


if __name__ == "__main__":
    main()