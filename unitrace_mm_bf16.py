"""Run an Inductor-compiled BF16 torch.mm for UnitTrace device timing.

Examples:
  python unitrace_mm_bf16.py 4 1536 2048
  python unitrace_mm_bf16.py 4 1536 2048 --iterations 100 --warmup 20

The GEMM is A[M,K] @ B[K,N] -> C[M,N].  The call is compiled by Inductor with
max-autotune.  Set XE2_PARITY_TEMPLATE=bmg_decode before launching if the
candidate list should contain only ATen/oneDNN plus the BMG tiled2d template.
Run this process under UnitTrace with --device-timing and inspect the selected
compiled GEMM kernel row in the device timing summary.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--template",
        choices=("all", "bmg_decode", "mm"),
        default=None,
        help="restrict Inductor candidates; default preserves the environment",
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be >= 0 and iterations must be > 0")

    if args.template is not None:
        if args.template == "all":
            os.environ.pop("XE2_PARITY_TEMPLATE", None)
        else:
            os.environ["XE2_PARITY_TEMPLATE"] = args.template

    import torch

    from torch._inductor import config as inductor_config

    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm = True
    inductor_config.fx_graph_cache = False
    inductor_config.autotune_local_cache = False

    # Create inputs before the measured loop.  The output buffer is reused so
    # allocation and output-layout overhead are not part of the GEMM call.
    a = torch.randn((args.M, args.K), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((args.K, args.N), dtype=torch.bfloat16, device="xpu")
    c = torch.empty((args.M, args.N), dtype=torch.bfloat16, device="xpu")

    @torch.compile(mode="max-autotune-no-cudagraphs")
    def run_mm(x, y):
        return torch.mm(x, y)

    with torch.inference_mode():
        for _ in range(args.warmup):
            c = run_mm(a, b)
        torch.xpu.synchronize()

        for _ in range(args.iterations):
            c = run_mm(a, b)
        torch.xpu.synchronize()

    print(
        f"Done: Inductor torch.mm BF16 shape=({args.M},{args.N},{args.K}) "
        f"warmup={args.warmup} iterations={args.iterations} "
        f"template={os.environ.get('XE2_PARITY_TEMPLATE', 'all')}"
    )


if __name__ == "__main__":
    main()
