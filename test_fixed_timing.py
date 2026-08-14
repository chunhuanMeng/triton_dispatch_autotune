#!/usr/bin/env python3
"""Test fixed timing with proper warmup."""
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_triton_mm(A, B, C, M, N, K,
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


def test_timing():
    """Test timing with proper warmup."""
    print("=== Test Fixed Timing ===\n")
    
    # Test shape
    M, N, K = 512, 4096, 2048
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    GROUP_M = 8

    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='xpu')
    B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='xpu')
    C = torch.zeros((M, N), dtype=torch.int32, device='xpu')

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Warmup
    for _ in range(5):
        kernel_triton_mm[grid](A, B, C, M, N, K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)
    torch.xpu.synchronize()

    # Benchmark triton
    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    torch.xpu.synchronize()
    s.record()
    for _ in range(200):
        kernel_triton_mm[grid](A, B, C, M, N, K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)
    e.record()
    torch.xpu.synchronize()
    triton_time = s.elapsed_time(e) / 200 * 1000
    print(f'Triton (64x64x64): {triton_time:.2f} us')

    # oneDNN with FIXED warmup (assign result to _)
    for _ in range(50):
        _ = torch._int_mm(A, B)
    torch.xpu.synchronize()

    s.record()
    for _ in range(200):
        _ = torch._int_mm(A, B)
    e.record()
    torch.xpu.synchronize()
    onednn_time = s.elapsed_time(e) / 200 * 1000
    print(f'oneDNN (fixed warmup): {onednn_time:.2f} us')

    print(f'\nSpeedup: {onednn_time/triton_time:.4f}x')
    
    # Calculate TOPS
    tops_triton = 2 * M * N * K / (triton_time * 1e-6) / 1e12
    tops_onednn = 2 * M * N * K / (onednn_time * 1e-6) / 1e12
    print(f'Triton TOPS: {tops_triton:.2f} ({tops_triton/196.6*100:.1f}% of peak)')
    print(f'oneDNN TOPS: {tops_onednn:.2f} ({tops_onednn/196.6*100:.1f}% of peak)')


if __name__ == "__main__":
    test_timing()