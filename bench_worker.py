"""
Triton kernel benchmark worker.
For each 5-dim config, tests all 3 templates and returns the best time.
Kernels match the Inductor Jinja templates exactly.
"""
import torch
import triton
import triton.language as tl
from search_space import TEMPLATES, GemmConfig, is_valid_for_template


# ═══ Template 1: triton_mm (pointer indexing, 1D grid + GROUP_M swizzle) ═══
@triton.jit
def kernel_triton_mm(
    A, B, C, M, N, K,
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


# ═══ Template 2: bmg_persistent (block_ptr, persistent 1D grid) ═══
@triton.jit
def kernel_bmg_persistent(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n
    width = GROUP_M * grid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size

        rm = pid_m * BLOCK_M
        rn = pid_n * BLOCK_N

        a_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(rm, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
        b_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(0, rn), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptr, boundary_check=(0, 1), padding_option="zero")
            b = tl.load(b_ptr, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            a_ptr = tl.advance(a_ptr, (0, BLOCK_K))
            b_ptr = tl.advance(b_ptr, (BLOCK_K, 0))

        # Store
        idx_m = rm + tl.arange(0, BLOCK_M)
        idx_n = rn + tl.arange(0, BLOCK_N)
        mask = (idx_m[:, None] < M) & (idx_n[None, :] < N)
        tl.store(C + idx_m[:, None] * stride_cm + idx_n[None, :] * stride_cn, acc, mask=mask)


# ═══ Template 3: bmg_decode (block_ptr, 2D grid, non-persistent) ═══
@triton.jit
def kernel_bmg_decode(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N

    a_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(rm, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, rn), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_ptr, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptr = tl.advance(a_ptr, (0, BLOCK_K))
        b_ptr = tl.advance(b_ptr, (BLOCK_K, 0))

    c_ptr = tl.make_block_ptr(base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(rm, rn), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    tl.store(c_ptr, acc, boundary_check=(0, 1))


# ═══ Benchmark Functions ═══

NUM_SMS = 20  # Arc Pro B60 has 20 Xe-cores

def _bench_template(A, B, C, M, N, K, config, template, num_iters=200):
    """Benchmark one (config, template) pair. Returns time_us or None."""
    bm, bn, bk = config.BLOCK_M, config.BLOCK_N, config.BLOCK_K
    ns, nw = config.num_stages, config.num_warps

    try:
        if template == "triton_mm":
            grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
            fn = lambda: kernel_triton_mm[grid](A, B, C, M, N, K,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
                num_stages=ns, num_warps=nw)

        elif template == "bmg_persistent":
            grid = (NUM_SMS,)
            fn = lambda: kernel_bmg_persistent[grid](A, B, C, M, N, K,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, NUM_SMS=NUM_SMS,
                num_stages=ns, num_warps=nw)

        elif template == "bmg_decode":
            grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
            fn = lambda: kernel_bmg_decode[grid](A, B, C, M, N, K,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                num_stages=ns, num_warps=nw)
        else:
            return None

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

    except Exception:
        return None


def bench_config_all_templates(M, N, K, config, num_iters=200):
    """
    Benchmark a 5-dim config on all 3 templates, return best (time_us, winning_template).
    """
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='xpu')
    B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='xpu')
    C = torch.zeros((M, N), dtype=torch.int32, device='xpu')

    best_time = None
    best_template = None

    for template in TEMPLATES:
        if not is_valid_for_template(M, N, K, config, template):
            continue
        time_us = _bench_template(A, B, C, M, N, K, config, template, num_iters)
        if time_us is not None:
            if best_time is None or time_us < best_time:
                best_time = time_us
                best_template = template

    return best_time, best_template


def bench_onednn(M, N, K, num_iters=200):
    """Benchmark oneDNN _int_mm baseline."""
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='xpu')
    B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='xpu')

    for _ in range(50):
        torch._int_mm(A, B)
    torch.xpu.synchronize()

    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    torch.xpu.synchronize()
    s.record()
    for _ in range(num_iters):
        torch._int_mm(A, B)
    e.record()
    torch.xpu.synchronize()
    return s.elapsed_time(e) / num_iters * 1000
