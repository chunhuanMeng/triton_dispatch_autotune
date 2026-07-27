#!/usr/bin/env python3
"""Full BF16 config tuning with the real Inductor XPU templates.

For one shape and one template, injects the Cartesian config space into the
actual Inductor heuristic, then lets AlgorithmSelector benchmark every
candidate through TritonBenchmarkRequest/do_bench.  No external Triton
template is implemented here.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, "/home/sdp/meng/pytorch")

BLOCK_M = [4, 8, 16, 32, 64, 128, 256]
BLOCK_N = [32, 64, 128, 256, 512]
BLOCK_K = [32, 64, 128, 256]
NUM_STAGES = [1, 2, 3, 4]
NUM_WARPS = [4, 8, 16, 32]
TEMPLATES = ("triton_mm", "bmg_persistent", "bmg_tiled2d")


def all_configs(M: int, N: int, K: int, raw: bool):
    from torch._inductor.heuristics.template.triton import GemmConfig

    values = itertools.product(BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_WARPS)
    result = []
    for bm, bn, bk, ns, nw in values:
        # The raw mode sends every listed combination to Inductor.  Default
        # mode removes combinations that are structurally invalid for this
        # shape, while retaining the complete Cartesian search space otherwise.
        if not raw:
            if bk > K or bn > N or bm > max(M * 8, 64):
                continue
            if bm * bn < nw * 256:
                continue
            if (bm * bn) // (nw * 16) > 128:
                continue
        result.append(GemmConfig(bm, bn, bk, ns, nw))
    return result


def is_target(choice, template: str) -> bool:
    name = choice.name.lower()
    if template == "triton_mm":
        return name.startswith("triton_mm_") and "bmg_" not in name
    return f"bmg_{template.removeprefix('bmg_')}" in name


def configure_heuristic(template: str, configs) -> None:
    from torch._inductor.heuristics.template import get_template_heuristic
    from torch._inductor.kernel.mm import (
        bmg_persistent_mm_template,
        bmg_tiled2d_mm_template,
        mm_template,
    )

    template_obj = {
        "triton_mm": mm_template,
        "bmg_persistent": bmg_persistent_mm_template,
        "bmg_tiled2d": bmg_tiled2d_mm_template,
    }[template]
    heuristic = get_template_heuristic(template_obj.uid, "xpu", "mm")
    heuristic.mm_configs = list(configs)
    heuristic.exhaustive_configs = list(configs)
    if hasattr(heuristic, "should_scale_configs"):
        heuristic.should_scale_configs = False


def run(M: int, N: int, K: int, template: str, raw: bool):
    import logging
    import torch
    import torch._inductor.config as inductor_config
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    logging.getLogger("torch._inductor.autotune_process").setLevel(logging.WARNING)
    logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.WARNING)
    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm = True
    inductor_config.fx_graph_cache = False
    inductor_config.autotune_local_cache = False

    configs = all_configs(M, N, K, raw)
    configure_heuristic(template, configs)

    captured = []
    original_log = AlgorithmSelectorCache.log_results
    original_call = AlgorithmSelectorCache.__call__

    @staticmethod
    def patched_log(name, input_nodes, timings, elapse, precompile_elapse,
                    prescreening_elapse=None, hint_override=None,
                    is_collective=False):
        captured.append({choice.name: timing for choice, timing in timings.items()})
        return original_log(name, input_nodes, timings, elapse, precompile_elapse,
                            prescreening_elapse, hint_override, is_collective)

    def patched_call(self, name, choices, input_nodes, layout, *args, **kwargs):
        target = [c for c in choices if is_target(c, template)]
        extern = [c for c in choices if "triton" not in c.name.lower()]
        if not target:
            raise RuntimeError(f"no {template} candidates; choices={[c.name for c in choices]}")
        return original_call(self, name, target + extern, input_nodes, layout, *args, **kwargs)

    AlgorithmSelectorCache.log_results = patched_log
    AlgorithmSelectorCache.__call__ = patched_call

    A = torch.randn((M, K), dtype=torch.bfloat16, device="xpu")
    B = torch.randn((K, N), dtype=torch.bfloat16, device="xpu")

    @torch.compile(mode="max-autotune-no-cudagraphs")
    def fn(x, y):
        return torch.mm(x, y)

    output = fn(A, B)
    torch.xpu.synchronize()
    reference = torch.mm(A.float(), B.float()).to(torch.bfloat16)
    torch.testing.assert_close(output, reference, rtol=0.02, atol=0.02)

    if not captured:
        raise RuntimeError("AlgorithmSelector did not produce candidate timings")
    timings = captured[-1]
    target_timings = {
        name: time for name, time in timings.items()
        if name.startswith("triton_mm") and (
            template == "triton_mm" or f"bmg_{template.removeprefix('bmg_')}" in name
        )
    }
    if not target_timings:
        raise RuntimeError("captured timing does not contain target template")
    best_name, best_time = min(target_timings.items(), key=lambda item: item[1])
    result = {
        "shape": [M, N, K],
        "template": template,
        "raw_search": raw,
        "requested_configs": len(configs),
        "benchmarked_configs": len(target_timings),
        "best_name": best_name,
        "best_time_ms": best_time,
        "all_target_timings_ms": target_timings,
        "all_timings_ms": timings,
        "correct": True,
    }
    print("RESULT:" + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True, help="M,N,K")
    parser.add_argument("--template", required=True, choices=TEMPLATES)
    parser.add_argument("--raw", action="store_true", help="include structurally invalid combinations")
    args = parser.parse_args()
    M, N, K = (int(x) for x in args.shape.split(","))
    os.environ.setdefault("XE2_ENABLE_BMG_FLOAT_TEMPLATES", "1")
    os.environ.setdefault("XE2_MM_TUNED_CONFIGS", "1")
    run(M, N, K, args.template, args.raw)


if __name__ == "__main__":
    main()
