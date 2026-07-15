#!/usr/bin/env python3
"""
Benchmark with Inductor-style kernel using Inductor's 11 int8_mm_configs.
Tests max tuning with optimized kernel.
"""
import torch
import triton
import triton.language as tl

# Inductor-style kernel (with full optimizations from triton_mm.py.jinja)
@triton.jit
def kernel_inductor(A, B, C, M, N, K,
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
    # Pointer alignment optimization (from triton_mm.py.jinja)
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

    # Rematerialize rm and rn
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    tl.store(C + idx_m * stride_cm + idx_n * stride_cn, acc, mask=mask)


# Inductor's 11 int8_mm_configs
INT8_MM_CONFIGS = [
    (64, 64, 32, 2, 4),      # config_1
    (64, 128, 32, 3, 4),     # config_2
    (128, 64, 32, 3, 4),     # config_3
    (64, 128, 32, 4, 8),     # config_4
    (128, 64, 32, 4, 8),     # config_5
    (64, 32, 32, 5, 8),      # config_6
    (32, 64, 32, 5, 8),      # config_7
    (128, 128, 32, 2, 8),    # config_8
    (64, 64, 64, 3, 8),      # config_9
    (128, 256, 128, 3, 8),   # config_10
    (256, 128, 128, 3, 8),   # config_11
]

# Test shapes
TEST_SHAPES = [
    (512, 4096, 2048),   # Typical prefill
    (1, 2048, 1408),     # Small M
    (2048, 4096, 14336), # Large K
    (512, 4096, 4096),   # Square-ish
]


def bench_config(A, B, C, M, N, K, config, num_iters=200):
    """Benchmark a config."""
    bm, bn, bk, ns, nw = config
    
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    fn = lambda: kernel_inductor[grid](A, B, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_stages=ns, num_warps=nw)
    
    # Warmup
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    
    # Benchmark
    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    torch.xpu.synchronize()
    s.record()
    for _ in range(num_iters):
        fn()
    e.record()
    torch.xpu.synchronize()
    return s.elapsed_time(e) / num_iters * 1000  # us


def main():
    print("=== Max Tune with Inductor 11 int8_mm_configs ===\n")
    
    for M, N, K in TEST_SHAPES:
        print(f"Shape ({M}, {N}, {K}):")
        
        A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='xpu')
        B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='xpu')
        C = torch.zeros((M, N), dtype=torch.int32, device='xpu')
        
        # Find best config
        best_time = float('inf')
        best_config = None
        
        for config in INT8_MM_CONFIGS:
            time_us = bench_config(A, B, C, M, N, K, config)
            if time_us < best_time:
                best_time = time_us
                best_config = config
            print(f"  Config {config}: {time_us:.2f} us")
        
        print(f"  BEST: Config {best_config}: {best_time:.2f} us")
        
        # Also benchmark oneDNN
        s = torch.xpu.Event(enable_timing=True)
        e = torch.xpu.Event(enable_timing=True)
        torch.xpu.synchronize()
        s.record()
        for _ in range(200):
            torch._int_mm(A, B)
        e.record()
        torch.xpu.synchronize()
        onednn_time = s.elapsed_time(e) / 200 * 1000
        print(f"  oneDNN: {onednn_time:.2f} us")
        print(f"  Speedup vs oneDNN: {onednn_time/best_time:.2f}x")
        print()


if __name__ == "__main__":
    main()