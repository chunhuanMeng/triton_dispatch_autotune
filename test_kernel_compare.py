#!/usr/bin/env python3
"""
Compare Inductor-style kernel vs Bench-style kernel.
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


# Bench-style kernel (with explicit mask)
@triton.jit  
def kernel_bench(A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k_idx * BLOCK_K
        a_mask = (rm[:, None] < M) & (offs_k[None, :] + k_off < K)
        b_mask = (offs_k[:, None] + k_off < K) & (rn[None, :] < N)
        a = tl.load(A + rm[:, None] * stride_am + (offs_k[None, :] + k_off) * stride_ak,
                    mask=a_mask, other=0)
        b = tl.load(B + (offs_k[:, None] + k_off) * stride_bk + rn[None, :] * stride_bn,
                    mask=b_mask, other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)

    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc, mask=mask)


def test_kernel_equivalence():
    """Test that both kernels produce identical results."""
    print("=== Kernel Equivalence Test ===\n")
    
    test_shapes = [
        (512, 4096, 2048),   # Typical prefill
        (1, 2048, 1408),     # Small M
        (2048, 4096, 14336), # Large K
    ]
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    GROUP_M = 8
    num_iters = 100
    
    for M, N, K in test_shapes:
        print(f"Shape ({M}, {N}, {K}):")
        
        A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='xpu')
        B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='xpu')
        C1 = torch.zeros((M, N), dtype=torch.int32, device='xpu')
        C2 = torch.zeros((M, N), dtype=torch.int32, device='xpu')
        
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        # Warmup
        for _ in range(5):
            kernel_inductor[grid](A, B, C1, M, N, K,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1), C1.stride(0), C1.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)
            kernel_bench[grid](A, B, C2, M, N, K,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1), C2.stride(0), C2.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)
        torch.xpu.synchronize()
        
        # Benchmark kernel 1
        s1 = torch.xpu.Event(enable_timing=True)
        e1 = torch.xpu.Event(enable_timing=True)
        torch.xpu.synchronize()
        s1.record()
        for _ in range(num_iters):
            kernel_inductor[grid](A, B, C1, M, N, K,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1), C1.stride(0), C1.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)
        e1.record()
        torch.xpu.synchronize()
        time1 = s1.elapsed_time(e1) / num_iters * 1000  # us
        
        # Benchmark kernel 2
        s2 = torch.xpu.Event(enable_timing=True)
        e2 = torch.xpu.Event(enable_timing=True)
        torch.xpu.synchronize()
        s2.record()
        for _ in range(num_iters):
            kernel_bench[grid](A, B, C2, M, N, K,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1), C2.stride(0), C2.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)
        e2.record()
        torch.xpu.synchronize()
        time2 = s2.elapsed_time(e2) / num_iters * 1000  # us
        
        # Compare results
        diff = (C1 - C2).abs().max().item()
        
        print(f"  Inductor-style: {time1:.4f} us")
        print(f"  Bench-style:    {time2:.4f} us")
        print(f"  Ratio: {time1/time2:.4f}x")
        print(f"  Result diff: {diff}")
        print(f"  Results match: {diff == 0}")
        print()


def test_against_onednn():
    """Test performance against oneDNN."""
    print("\n=== Performance vs oneDNN ===\n")
    
    M, N, K = 512, 4096, 2048
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    GROUP_M = 8
    num_iters = 200
    
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='xpu')
    B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='xpu')
    C = torch.zeros((M, N), dtype=torch.int32, device='xpu')
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    # Warmup
    for _ in range(10):
        kernel_bench[grid](A, B, C, M, N, K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)
        torch._int_mm(A, B)
    torch.xpu.synchronize()
    
    # Benchmark triton
    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    torch.xpu.synchronize()
    s.record()
    for _ in range(num_iters):
        kernel_bench[grid](A, B, C, M, N, K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)
    e.record()
    torch.xpu.synchronize()
    triton_time = s.elapsed_time(e) / num_iters * 1000  # us
    
    # Benchmark oneDNN
    s.record()
    for _ in range(num_iters):
        torch._int_mm(A, B)
    e.record()
    torch.xpu.synchronize()
    onednn_time = s.elapsed_time(e) / num_iters * 1000  # us
    
    print(f"Shape ({M}, {N}, {K}):")
    print(f"  Triton (64x64x64): {triton_time:.4f} us")
    print(f"  oneDNN: {onednn_time:.4f} us")
    print(f"  Speedup: {onednn_time/triton_time:.4f}x")
    
    # Calculate TOPS
    tops_triton = 2 * M * N * K / (triton_time * 1e-6) / 1e12
    tops_onednn = 2 * M * N * K / (onednn_time * 1e-6) / 1e12
    print(f"  Triton TOPS: {tops_triton:.2f} ({tops_triton/234*100:.1f}% of peak)")
    print(f"  oneDNN TOPS: {tops_onednn:.2f} ({tops_onednn/234*100:.1f}% of peak)")


if __name__ == "__main__":
    test_kernel_equivalence()
    test_against_onednn()