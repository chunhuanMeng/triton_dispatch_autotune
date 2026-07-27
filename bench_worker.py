"""
Triton kernel benchmark worker.
For each 5-dim config, tests all 3 templates and returns the best time.
Kernels match the Inductor Jinja templates exactly.
"""
import os
import traceback

import torch
import triton
import triton.language as tl
from search_space import (
    TEMPLATES,
    GemmConfig,
    is_valid_for_template,
    template_config_keys,
)
from bench_inductor_worker import (
    bench_config_all_templates as bench_inductor_config_all_templates,
    bench_one_template as bench_inductor_one_template,
    bench_inductor_onednn,
)


# ═══ Template 1: triton_mm (pointer indexing, 1D grid + GROUP_M swizzle) ═══
@triton.jit
def kernel_triton_mm(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
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
    # Pointer alignment optimization (from Inductor triton_mm.py.jinja)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K).to(tl.int32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        if not EVEN_K:
            a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
            b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        if EVEN_K:
            a = tl.load(A + idx_m * stride_am + idx_n * stride_ak)
        else:
            a = tl.load(A + idx_m * stride_am + idx_n * stride_ak,
                        mask=a_mask, other=0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        if EVEN_K:
            b = tl.load(B + idx_m * stride_bk + idx_n * stride_bn)
        else:
            b = tl.load(B + idx_m * stride_bk + idx_n * stride_bn,
                        mask=b_mask, other=0)

        if USE_FAST_ACCUM:
            acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        else:
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # Rematerialize rm and rn
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    tl.store(C + idx_m * stride_cm + idx_n * stride_cn, acc, mask=mask)


# ═══ Template 2: bmg_persistent (block_ptr, persistent 1D grid) ═══
@triton.jit
def kernel_bmg_persistent(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    start_pid = tl.program_id(0).to(tl.int32)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n
    width = GROUP_M * grid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_M
        rn = pid_n * BLOCK_N

        a_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(rm, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
        b_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(0, rn), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptr, boundary_check=(0, 1), padding_option="zero")
            b = tl.load(b_ptr, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(a, b, acc, out_dtype=ACC_TYPE)
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
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int32)
    pid_n = tl.program_id(1).to(tl.int32)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N

    a_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(rm, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, rn), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_ptr, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b, acc, out_dtype=ACC_TYPE)
        a_ptr = tl.advance(a_ptr, (0, BLOCK_K))
        b_ptr = tl.advance(b_ptr, (BLOCK_K, 0))

    idx_m = rm + tl.arange(0, BLOCK_M)
    idx_n = rn + tl.arange(0, BLOCK_N)
    mask = (idx_m[:, None] < M) & (idx_n[None, :] < N)
    tl.store(C + idx_m[:, None] * stride_cm + idx_n[None, :] * stride_cn, acc, mask=mask)


# ═══ Experimental Template: bmg_decode_mloop ═══
# Same 2D-grid decode idea, but the grid is 1D (only over N). Each program
# instance loops internally over all M-tiles, loading each B K-tile from
# DRAM exactly once per iteration and reusing it in registers across every
# M-tile before advancing. This removes the "B read once per pid_m" traffic
# multiplication present in kernel_bmg_decode when M > BLOCK_M.
#
# Hardcoded to exactly 4 M-tiles (matches M=128, BLOCK_M=32 for this
# experiment). Triton's AST frontend does not support building loop-carried
# lists/tuples of tensors via mutation (append / index assignment), so the
# 4 accumulators and 4 A block pointers are kept as separate named values.
@triton.jit
def kernel_bmg_decode_mloop4(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid_n = tl.program_id(0).to(tl.int32)
    rn = pid_n * BLOCK_N

    b_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, rn), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))

    a_ptr0 = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(0 * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    a_ptr1 = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(1 * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    a_ptr2 = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(2 * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    a_ptr3 = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(3 * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))

    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        # Load this K-tile of B exactly once; reuse across every M-tile below.
        b = tl.load(b_ptr, boundary_check=(0, 1), padding_option="zero")

        a0 = tl.load(a_ptr0, boundary_check=(0, 1), padding_option="zero")
        acc0 = tl.dot(a0, b, acc0, out_dtype=ACC_TYPE)
        a_ptr0 = tl.advance(a_ptr0, (0, BLOCK_K))

        a1 = tl.load(a_ptr1, boundary_check=(0, 1), padding_option="zero")
        acc1 = tl.dot(a1, b, acc1, out_dtype=ACC_TYPE)
        a_ptr1 = tl.advance(a_ptr1, (0, BLOCK_K))

        a2 = tl.load(a_ptr2, boundary_check=(0, 1), padding_option="zero")
        acc2 = tl.dot(a2, b, acc2, out_dtype=ACC_TYPE)
        a_ptr2 = tl.advance(a_ptr2, (0, BLOCK_K))

        a3 = tl.load(a_ptr3, boundary_check=(0, 1), padding_option="zero")
        acc3 = tl.dot(a3, b, acc3, out_dtype=ACC_TYPE)
        a_ptr3 = tl.advance(a_ptr3, (0, BLOCK_K))

        b_ptr = tl.advance(b_ptr, (BLOCK_K, 0))

    idx_n = rn + tl.arange(0, BLOCK_N)

    idx_m0 = 0 * BLOCK_M + tl.arange(0, BLOCK_M)
    mask0 = (idx_m0[:, None] < M) & (idx_n[None, :] < N)
    tl.store(C + idx_m0[:, None] * stride_cm + idx_n[None, :] * stride_cn, acc0, mask=mask0)

    idx_m1 = 1 * BLOCK_M + tl.arange(0, BLOCK_M)
    mask1 = (idx_m1[:, None] < M) & (idx_n[None, :] < N)
    tl.store(C + idx_m1[:, None] * stride_cm + idx_n[None, :] * stride_cn, acc1, mask=mask1)

    idx_m2 = 2 * BLOCK_M + tl.arange(0, BLOCK_M)
    mask2 = (idx_m2[:, None] < M) & (idx_n[None, :] < N)
    tl.store(C + idx_m2[:, None] * stride_cm + idx_n[None, :] * stride_cn, acc2, mask=mask2)

    idx_m3 = 3 * BLOCK_M + tl.arange(0, BLOCK_M)
    mask3 = (idx_m3[:, None] < M) & (idx_n[None, :] < N)
    tl.store(C + idx_m3[:, None] * stride_cm + idx_n[None, :] * stride_cn, acc3, mask=mask3)


def bench_mloop_variant(A, B, C, M, N, K, config, num_iters=200, dtype="bf16"):
    """Benchmark the experimental bmg_decode_mloop4 kernel.

    Hardcoded to exactly 4 M-tiles (M must equal 4 * BLOCK_M).
    """
    bm, bn, bk = config.BLOCK_M, config.BLOCK_N, config.BLOCK_K
    ns, nw = config.num_stages, config.num_warps
    if M != 4 * bm:
        return None
    acc_type = tl.float32

    try:
        grid = (triton.cdiv(N, bn),)
        fn = lambda: kernel_bmg_decode_mloop4[grid](A, B, C,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            ACC_TYPE=acc_type,
            M=M, N=N, K=K,
            stride_am=A.stride(0), stride_ak=A.stride(1),
            stride_bk=B.stride(0), stride_bn=B.stride(1),
            stride_cm=C.stride(0), stride_cn=C.stride(1),
            num_stages=ns, num_warps=nw)

        for _ in range(5):
            fn()
        torch.xpu.synchronize()

        s = torch.xpu.Event(enable_timing=True)
        e = torch.xpu.Event(enable_timing=True)
        torch.xpu.synchronize()
        s.record()
        for _ in range(num_iters):
            fn()
        e.record()
        torch.xpu.synchronize()
        return s.elapsed_time(e) / num_iters * 1000

    except Exception:
        if os.environ.get("XE2_BENCH_DEBUG_ERRORS") == "1":
            print(
                f"BENCH ERROR template=bmg_decode_mloop4 shape=({M},{N},{K}) "
                f"config={config}",
                flush=True,
            )
            traceback.print_exc()
        return None


# ═══ Benchmark Functions ═══

NUM_SMS = 20  # Arc Pro B60 has 20 Xe-cores


def _dtype_policy(dtype):
    if dtype in ("int8", torch.int8):
        return torch.int8, torch.int32, "int8"
    if dtype in ("bf16", "bfloat16", torch.bfloat16):
        return torch.bfloat16, torch.bfloat16, "bf16"
    if dtype in ("fp16", "float16", torch.float16):
        return torch.float16, torch.float16, "fp16"
    raise ValueError(f"unsupported dtype: {dtype}")


def _make_inputs(M, N, K, dtype="int8"):
    input_dtype, output_dtype, dtype_name = _dtype_policy(dtype)
    if input_dtype == torch.int8:
        A = torch.randint(-128, 127, (M, K), dtype=input_dtype, device="xpu")
        B = torch.randint(-128, 127, (K, N), dtype=input_dtype, device="xpu")
    else:
        A = torch.randn((M, K), dtype=input_dtype, device="xpu")
        B = torch.randn((K, N), dtype=input_dtype, device="xpu")
    C = torch.empty((M, N), dtype=output_dtype, device="xpu")
    return A, B, C, dtype_name

def _bench_template(A, B, C, M, N, K, config, template, num_iters=200,
                    dtype="int8"):
    """Benchmark one (config, template) pair. Returns time_us or None."""
    bm, bn, bk = config.BLOCK_M, config.BLOCK_N, config.BLOCK_K
    ns, nw = config.num_stages, config.num_warps
    _, _, dtype_name = _dtype_policy(dtype)
    acc_type = tl.int32 if dtype_name == "int8" else tl.float32

    try:
        if template == "triton_mm":
            grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
            fn = lambda: kernel_triton_mm[grid](A, B, C,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
                EVEN_K=(K % bk == 0), USE_FAST_ACCUM=False,
                ACC_TYPE=acc_type, ALLOW_TF32=False,
                M=M, N=N, K=K,
                stride_am=A.stride(0), stride_ak=A.stride(1),
                stride_bk=B.stride(0), stride_bn=B.stride(1),
                stride_cm=C.stride(0), stride_cn=C.stride(1),
                num_stages=ns, num_warps=nw)

        elif template == "bmg_persistent":
            grid = (min(NUM_SMS, triton.cdiv(M, bm) * triton.cdiv(N, bn)),)
            fn = lambda: kernel_bmg_persistent[grid](A, B, C,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, NUM_SMS=NUM_SMS,
                ACC_TYPE=acc_type,
                M=M, N=N, K=K,
                stride_am=A.stride(0), stride_ak=A.stride(1),
                stride_bk=B.stride(0), stride_bn=B.stride(1),
                stride_cm=C.stride(0), stride_cn=C.stride(1),
                num_stages=ns, num_warps=nw)

        elif template == "bmg_decode":
            grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
            fn = lambda: kernel_bmg_decode[grid](A, B, C,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                ACC_TYPE=acc_type,
                M=M, N=N, K=K,
                stride_am=A.stride(0), stride_ak=A.stride(1),
                stride_bk=B.stride(0), stride_bn=B.stride(1),
                stride_cm=C.stride(0), stride_cn=C.stride(1),
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
        if os.environ.get("XE2_BENCH_DEBUG_ERRORS") == "1":
            print(
                f"BENCH ERROR template={template} shape=({M},{N},{K}) "
                f"config={config}",
                flush=True,
            )
            traceback.print_exc()
        return None


def bench_config_all_templates(M, N, K, config, num_iters=200, dtype="int8"):
    """
    Benchmark a 5-dim config on all 3 templates, return best (time_us, winning_template).
    """
    if dtype == "bf16" and os.environ.get("XE2_BF16_BENCH_BACKEND", "inductor") == "inductor":
        return bench_inductor_config_all_templates(
            M, N, K, config, num_iters=num_iters, dtype=dtype
        )

    A, B, C, _ = _make_inputs(M, N, K, dtype)

    best_time = None
    best_template = None

    for template in TEMPLATES:
        if not is_valid_for_template(M, N, K, config, template):
            continue
        time_us = _bench_template(
            A, B, C, M, N, K, config, template, num_iters, dtype=dtype
        )
        if time_us is not None:
            if best_time is None or time_us < best_time:
                best_time = time_us
                best_template = template

    return best_time, best_template


def bench_one_template(M, N, K, config, template, num_iters=200, dtype="int8"):
    """Benchmark one explicit six-dimensional (template, config) choice."""
    if dtype == "bf16" and os.environ.get("XE2_BF16_BENCH_BACKEND", "inductor") == "inductor":
        return bench_inductor_one_template(
            M, N, K, config, template, num_iters=num_iters, dtype=dtype
        )

    if not is_valid_for_template(M, N, K, config, template):
        return None
    A, B, C, _ = _make_inputs(M, N, K, dtype)
    return _bench_template(
        A, B, C, M, N, K, config, template, num_iters, dtype=dtype
    )


def bench_onednn(M, N, K, num_iters=200, dtype="int8"):
    """Benchmark the oneDNN/ATen baseline using the matching timer backend."""
    if dtype == "bf16" and os.environ.get("XE2_BF16_BENCH_BACKEND", "inductor") == "inductor":
        return bench_inductor_onednn(M, N, K, num_iters=num_iters, dtype=dtype)

    A, B, C, _ = _make_inputs(M, N, K, dtype)
    dtype_name = _dtype_policy(dtype)[2]

    # Triton benchmarks write into a preallocated C buffer. Use the same
    # output-buffer model for floating-point oneDNN/mm benchmarks. _int_mm
    # has no compatible out= variant, so INT8 keeps the existing path.
    if dtype_name == "int8":
        fn = lambda: torch._int_mm(A, B)
    else:
        fn = lambda: torch.mm(A, B, out=C)

    for _ in range(50):
        fn()
    torch.xpu.synchronize()

    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    torch.xpu.synchronize()
    s.record()
    for _ in range(num_iters):
        fn()
    e.record()
    torch.xpu.synchronize()
    return s.elapsed_time(e) / num_iters * 1000
