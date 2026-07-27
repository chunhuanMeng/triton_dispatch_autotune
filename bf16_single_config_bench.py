"""Benchmark one real Inductor Triton template/config for one BF16 GEMM shape.

The requested config is injected into the selected Inductor XPU heuristic, so
the caller can tune the original Cartesian search space one config at a time.
The kernel implementation and candidate benchmark remain entirely inside
Inductor; this script does not reimplement a Triton template.

The selected Triton candidate is benchmarked through the normal
AlgorithmSelector -> ChoiceCaller.benchmark -> TritonBenchmarkRequest ->
Triton do_bench path.  The ATen/oneDNN choice is retained as a fallback so that
Inductor does not short-circuit the single-choice path and skip candidate
benchmarking.  Only one Triton template/config remains.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, "/home/sdp/meng/pytorch")


def parse_config(text):
    values = tuple(int(x) for x in text.split(","))
    if len(values) != 5:
        raise argparse.ArgumentTypeError(
            "config must be BLOCK_M,BLOCK_N,BLOCK_K,num_stages,num_warps"
        )
    return values


def template_matches(choice, template):
    name = choice.name.lower()
    if template == "triton_mm":
        return name.startswith("triton_mm_") and not any(
            x in name for x in ("bmg_persistent", "bmg_tiled2d")
        )
    if template == "bmg_persistent":
        return "bmg_persistent" in name
    if template == "bmg_tiled2d":
        return "bmg_tiled2d" in name
    raise ValueError(f"unknown template: {template}")


def config_matches(choice, config):
    description = str(getattr(choice, "description", ""))
    fields = {
        "BLOCK_M": config[0],
        "BLOCK_N": config[1],
        "BLOCK_K": config[2],
        "num_stages": config[3],
        "num_warps": config[4],
    }
    return all(
        re.search(rf"\b{key}={value}\b", description)
        for key, value in fields.items()
    )


def configure_requested_template(template, config):
    """Make the requested config a real candidate of the selected template."""
    from torch._inductor.heuristics.template import get_template_heuristic
    from torch._inductor.heuristics.template.triton import GemmConfig
    from torch._inductor.kernel.mm import (
        bmg_persistent_mm_template,
        bmg_tiled2d_mm_template,
        mm_template,
    )

    template_object = {
        "triton_mm": mm_template,
        "bmg_persistent": bmg_persistent_mm_template,
        "bmg_tiled2d": bmg_tiled2d_mm_template,
    }[template]
    heuristic = get_template_heuristic(template_object.uid, "xpu", "mm")
    candidate = GemmConfig(*config)
    # max-autotune and exhaustive paths use different attributes in the
    # heuristic implementation.  Set both so this worker has one stable
    # candidate regardless of which path the current Inductor version takes.
    heuristic.mm_configs = [candidate]
    heuristic.exhaustive_configs = [candidate]
    if hasattr(heuristic, "should_scale_configs"):
        # Preserve the exact config requested by the outer search.  In
        # particular, do not silently scale small BM/BN values to 16.
        heuristic.should_scale_configs = False


def worker(M, N, K, template, config):
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

    configure_requested_template(template, config)

    captured = []
    filter_info = {}
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
            if template_matches(choice, template)
            and config_matches(choice, config)
        ]
        extern = [
            choice
            for choice in choices
            if "triton" not in choice.name.lower()
        ]
        filter_info["before"] = [choice.name for choice in choices]
        filter_info["target"] = [choice.name for choice in target]
        filter_info["extern"] = [choice.name for choice in extern]

        if len(target) != 1:
            raise RuntimeError(
                f"expected exactly one target candidate, got {len(target)}; "
                f"template={template}, config={config}, choices={filter_info}"
            )

        # Keep ATen/oneDNN only to force AlgorithmSelector through its normal
        # benchmark_choices path. No other Triton template/config is retained.
        filtered = target + extern
        filter_info["after"] = [choice.name for choice in filtered]
        return original_call(self, name, filtered, input_nodes, layout, *args, **kwargs)

    AlgorithmSelectorCache.log_results = patched_log
    AlgorithmSelectorCache.__call__ = patched_call

    a = torch.randn((M, K), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((K, N), dtype=torch.bfloat16, device="xpu")

    @torch.compile(mode="max-autotune-no-cudagraphs")
    def fn(x, y):
        return torch.mm(x, y)

    output = fn(a, b)
    torch.xpu.synchronize()

    # Explicit correctness check; the candidate benchmark itself also follows
    # Inductor's normal output-buffer/correctness handling.
    reference = torch.mm(a.float(), b.float()).to(torch.bfloat16)
    torch.testing.assert_close(output, reference, rtol=0.02, atol=0.02)

    if not captured:
        raise RuntimeError(
            "AlgorithmSelector did not log candidate timing; this shape likely "
            "had a single-choice/decomposed path"
        )

    timings = captured[-1]
    target_names = filter_info["target"]
    target_name = target_names[0]
    result = {
        "shape": [M, N, K],
        "template": template,
        "config": list(config),
        "target_name": target_name,
        "target_timing_ms": timings[target_name],
        "timings_ms": timings,
        "filter": filter_info,
        "correct": True,
    }
    print("RESULT:" + json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument(
        "--template",
        required=True,
        choices=("triton_mm", "bmg_persistent", "bmg_tiled2d"),
    )
    parser.add_argument("--config", required=True, type=parse_config)
    args = parser.parse_args()

    os.environ.setdefault("XE2_ENABLE_BMG_FLOAT_TEMPLATES", "1")
    os.environ.setdefault("XE2_MM_TUNED_CONFIGS", "1")
    worker(args.M, args.N, args.K, args.template, args.config)


if __name__ == "__main__":
    main()
