#!/usr/bin/env python3
"""Run one Inductor int_mm template with exactly one fixed config.

Each invocation is intentionally a fresh process.  The caller must set
XE2_PARITY_TEMPLATE to one of triton_mm, bmg_persistent, or bmg_decode.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics

import torch
import triton.testing


def parse_config(value: str) -> tuple[int, int, int, int, int]:
    fields = tuple(int(x) for x in value.split(","))
    if len(fields) != 5:
        raise ValueError("config must be BM,BN,BK,num_stages,num_warps")
    return fields


def candidate_matches(choice, template_name, config):
    """Match the logical template/config against a real Inductor choice."""
    name = choice.name.lower()
    if template_name == "triton_mm":
        template_match = name.startswith("triton_mm_") and "bmg_" not in name
    elif template_name == "bmg_persistent":
        template_match = "bmg_persistent" in name
    else:
        # Historical name used by this script; the real Inductor template is
        # bmg_tiled2d_mm_template.
        template_match = "bmg_tiled2d" in name
    if not template_match:
        return False

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


def configure_one_template(
    template_name: str, config: tuple[int, int, int, int, int], op_name: str
) -> None:
    # Importing these modules registers the XPU template heuristics.
    from torch._inductor.heuristics.template import get_template_heuristic
    from torch._inductor.heuristics.template.triton import GemmConfig
    from torch._inductor.kernel.mm import (
        bmg_persistent_mm_template,
        bmg_tiled2d_mm_template,
        mm_template,
    )

    template = {
        "triton_mm": mm_template,
        "bmg_persistent": bmg_persistent_mm_template,
        "bmg_decode": bmg_tiled2d_mm_template,
    }[template_name]
    heuristic = get_template_heuristic(template.uid, "xpu", op_name)
    candidate = GemmConfig(
        config[0], config[1], config[2], config[3], config[4]
    )
    # Both lists are set because max-autotune and exhaustive paths use
    # different attributes in the template heuristic implementation.
    heuristic.mm_configs = [candidate]
    heuristic.exhaustive_configs = [candidate]
    # XPU heuristics normally scale small tiles up to a minimum BLOCK_M/N of
    # 16.  That is correct for normal autotune, but would silently change the
    # requested worker config (for example BM=8 -> BM=16).  Disable scaling
    # for this fixed-config parity experiment.
    heuristic.should_scale_configs = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True, help="M,N,K")
    parser.add_argument("--config", required=True, help="BM,BN,BK,num_stages,num_warps")
    parser.add_argument(
        "--dtype", choices=("int8", "bf16", "fp16"), default="int8"
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--timer",
        choices=("event", "do_bench", "candidate"),
        default="event",
        help=(
            "event uses fixed-count amortized timing; do_bench measures the "
            "compiled wrapper; candidate uses Inductor BenchmarkRequest/do_bench"
        ),
    )
    args = parser.parse_args()

    template_name = os.environ.get("XE2_PARITY_TEMPLATE")
    if template_name not in {"triton_mm", "bmg_persistent", "bmg_decode"}:
        raise RuntimeError("XE2_PARITY_TEMPLATE must select one template")

    # This fixed-config BF16 parity run is intended to exercise State B.  Do
    # not silently omit the BMG templates when the caller has not exported the
    # state variables; explicit caller values still take precedence.
    os.environ.setdefault("XE2_ENABLE_BMG_FLOAT_TEMPLATES", "1")
    os.environ.setdefault("XE2_MM_TUNED_CONFIGS", "1")

    M, N, K = (int(x) for x in args.shape.split(","))
    config = parse_config(args.config)
    if args.dtype == "int8":
        input_dtype = torch.int8
        output_dtype = torch.int32
        op_name = "int_mm"
    elif args.dtype == "bf16":
        input_dtype = output_dtype = torch.bfloat16
        op_name = "mm"
    else:
        input_dtype = output_dtype = torch.float16
        op_name = "mm"
    configure_one_template(template_name, config, op_name)

    from torch.compiler import config as compiler_config
    compiler_config.force_disable_caches = True
    compiler_config.assume_static_by_default = True

    # Exercise the max-autotune GEMM path even though this experiment exposes
    # exactly one BF16 candidate configuration.
    from torch._inductor import config as inductor_config
    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm = True

    # Candidate mode follows the same path as bf16_single_config_bench.py:
    # retain exactly one requested Triton choice plus ATen/oneDNN, then read
    # the timing returned by Inductor's BenchmarkRequest/do_bench.
    captured_candidate_timings = []
    if args.timer == "candidate":
        from torch._inductor.select_algorithm import AlgorithmSelectorCache

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
            captured_candidate_timings.append(
                {choice.name: timing for choice, timing in timings.items()}
            )
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
                if candidate_matches(choice, template_name, config)
            ]
            extern = [
                choice for choice in choices if "triton" not in choice.name.lower()
            ]
            if len(target) != 1:
                raise RuntimeError(
                    f"expected one target candidate, got {len(target)}: "
                    f"template={template_name}, config={config}, "
                    f"choices={[choice.name for choice in choices]}"
                )
            return original_call(
                self, name, target + extern, input_nodes, layout, *args, **kwargs
            )

        AlgorithmSelectorCache.log_results = patched_log
        AlgorithmSelectorCache.__call__ = patched_call

    if input_dtype == torch.int8:
        A = torch.randint(-128, 127, (M, K), dtype=input_dtype, device="xpu")
        B = torch.randint(-128, 127, (K, N), dtype=input_dtype, device="xpu")
        reference_fn = torch._int_mm
    else:
        A = torch.randn((M, K), dtype=input_dtype, device="xpu")
        B = torch.randn((K, N), dtype=input_dtype, device="xpu")
        reference_fn = torch.mm
    reference = reference_fn(A, B)

    compiled = torch.compile(
        lambda x, y: reference_fn(x, y),
        backend="inductor",
        fullgraph=True,
        dynamic=False,
    )

    # Compilation is intentionally outside the timed region.
    output = compiled(A, B)
    torch.xpu.synchronize()
    if args.dtype == "int8":
        wrong = torch.count_nonzero(output != reference).item()
        max_abs_diff = (output - reference).abs().max().item()
        correct = wrong == 0
    else:
        max_abs_diff = (output.float() - reference.float()).abs().max().item()
        correct = bool(torch.allclose(output, reference, rtol=2e-2, atol=2e-2))
        wrong = 0 if correct else 1

    times_us = []
    if args.timer == "candidate":
        if not captured_candidate_timings:
            raise RuntimeError("Inductor did not report candidate timing")
        timings = captured_candidate_timings[-1]
        target = [
            (name, time_ms)
            for name, time_ms in timings.items()
            if "triton" in name.lower()
        ]
        if len(target) != 1:
            raise RuntimeError(f"expected one Triton timing, got {timings}")
        target_name, target_ms = target[0]
        print(
            f"timer=candidate template={template_name} shape={(M, N, K)} "
            f"config={config} dtype={args.dtype} correct={int(correct)} "
            f"target_name={target_name} target_time_us={target_ms * 1000.0:.4f} "
            f"timings_ms={json.dumps(timings, sort_keys=True)}",
            flush=True,
        )
        return
    if args.timer == "do_bench":
        # Triton's arguments are time budgets in milliseconds, not iteration
        # counts.  Also, this implementation clears the benchmark cache before
        # every measured call.  This mode is diagnostic only: it is not
        # equivalent to bench_worker.py's warm-cache amortized loop.
        for _ in range(args.trials):
            time_ms = triton.testing.do_bench(
                lambda: compiled(A, B),
                warmup=max(1, args.warmup),
                rep=max(1, args.iterations),
                return_mode="mean",
            )
            times_us.append(time_ms * 1000.0)
    else:
        for _ in range(args.trials):
            for _ in range(args.warmup):
                compiled(A, B)
            torch.xpu.synchronize()
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                compiled(A, B)
            end.record()
            torch.xpu.synchronize()
            times_us.append(start.elapsed_time(end) / args.iterations * 1000.0)

    time_us = statistics.median(times_us)
    element_bytes = torch.tensor([], dtype=input_dtype).element_size()
    output_bytes = torch.tensor([], dtype=output_dtype).element_size()
    tops = 2.0 * M * N * K / (time_us * 1e-6) / 1e12
    bw = (
        (M * K + K * N) * element_bytes + M * N * output_bytes
    ) / (time_us * 1e-6) / 1e9
    print(
        f"timer={args.timer} template={template_name} shape={(M, N, K)} config={config} "
        f"dtype={args.dtype} correct={int(correct)} wrong={wrong} max_abs_diff={max_abs_diff} "
        f"times_us={[round(x, 4) for x in times_us]} median_us={time_us:.4f} "
        f"TOPS={tops:.3f} BW_GBPS={bw:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
