"""Measure the ATen/oneDNN choice through Inductor's candidate timer.

The Triton candidate is retained only as a benchmark anchor so that Inductor
runs its normal AlgorithmSelector benchmark path.  The returned value is the
internal timing of the non-Triton ``mm`` choice, not an outer torch.xpu.Event
measurement.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, "/home/sdp/meng/pytorch")


def _parse_config(text: str) -> tuple[int, int, int, int, int]:
    values = tuple(int(value) for value in text.split(","))
    if len(values) != 5:
        raise ValueError("config must be BM,BN,BK,num_stages,num_warps")
    return values


def _configure_anchor(config: tuple[int, int, int, int, int]) -> None:
    from torch._inductor.heuristics.template import get_template_heuristic
    from torch._inductor.heuristics.template.triton import GemmConfig
    from torch._inductor.kernel.mm import mm_template

    heuristic = get_template_heuristic(mm_template.uid, "xpu", "mm")
    anchor = GemmConfig(*config)
    heuristic.mm_configs = [anchor]
    heuristic.exhaustive_configs = [anchor]
    heuristic.should_scale_configs = False


def worker(M: int, N: int, K: int, anchor_config: tuple[int, int, int, int, int]) -> None:
    import torch
    import torch._inductor.config as inductor_config
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    logging.getLogger("torch._inductor.autotune_process").setLevel(logging.WARNING)
    logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.WARNING)
    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm = True
    inductor_config.fx_graph_cache = False
    inductor_config.autotune_local_cache = False
    _configure_anchor(anchor_config)

    captured: list[dict[str, float]] = []
    original_log = AlgorithmSelectorCache.log_results
    original_call = AlgorithmSelectorCache.__call__

    @staticmethod
    def patched_log(
        name,
        input_nodes,
        timings,
        elapse,
        precompile_elapse,
        prescreening_elapse=None,
        hint_override=None,
        is_collective=False,
    ):
        captured.append({choice.name: timing for choice, timing in timings.items()})
        return original_log(
            name,
            input_nodes,
            timings,
            elapse,
            precompile_elapse,
            prescreening_elapse,
            hint_override,
            is_collective,
        )

    def patched_call(self, name, choices, input_nodes, layout, *args, **kwargs):
        target = [
            choice
            for choice in choices
            if choice.name.lower().startswith("triton_mm_")
            and "bmg_" not in choice.name.lower()
        ]
        extern = [choice for choice in choices if "triton" not in choice.name.lower()]
        if len(target) != 1:
            raise RuntimeError(
                f"expected one Triton anchor, got {len(target)}: "
                f"choices={[choice.name for choice in choices]}"
            )
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
        raise RuntimeError("Inductor did not report candidate timings")
    timings = captured[-1]
    extern = {name: value for name, value in timings.items() if "triton" not in name.lower()}
    if not extern:
        raise RuntimeError(f"Inductor did not report an ATen/mm timing: {timings}")
    name, timing_ms = min(extern.items(), key=lambda item: item[1])
    print(
        "RESULT:" + json.dumps(
            {
                "shape": [M, N, K],
                "anchor_config": list(anchor_config),
                "onednn_name": name,
                "onednn_timing_ms": timing_ms,
                "timings_ms": timings,
                "correct": True,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--config", default="128,128,32,3,16")
    args = parser.parse_args()
    os.environ.setdefault("XE2_ENABLE_BMG_FLOAT_TEMPLATES", "1")
    os.environ.setdefault("XE2_MM_TUNED_CONFIGS", "1")
    worker(args.M, args.N, args.K, _parse_config(args.config))


if __name__ == "__main__":
    main()
