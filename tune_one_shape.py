#!/usr/bin/env python3
"""Exhaustively tune one GEMM shape across all Triton templates."""
import argparse
import json
from pathlib import Path

from bench_worker import bench_one_template, bench_onednn
from search_space import (
    TEMPLATES,
    GemmConfig,
    generate_valid_configs,
    is_valid_for_template,
)


def tune_shape(M, N, K, dtype="bf16", num_iters=100, stable_iters=1000):
    configs = generate_valid_configs(M, N, K)
    choices = [
        (template, config)
        for config in configs
        for template in TEMPLATES
        if is_valid_for_template(M, N, K, config, template)
    ]

    print(
        f"Shape ({M},{N},{K}), dtype={dtype}: "
        f"{len(configs)} valid configs, {len(choices)} config/template choices"
    )
    print(f"Search iterations per candidate: {num_iters}")

    results = []
    for index, (template, config) in enumerate(choices, 1):
        time_us = bench_one_template(
            M, N, K, config, template, num_iters=num_iters, dtype=dtype
        )
        results.append(
            {
                "template": template,
                "config": list(config.key),
                "time_us": time_us,
            }
        )
        status = "FAILED" if time_us is None else f"{time_us:.3f} us"
        print(
            f"[{index}/{len(choices)}] {template:15s} "
            f"BM={config.BLOCK_M:<3d} BN={config.BLOCK_N:<3d} "
            f"BK={config.BLOCK_K:<3d} NS={config.num_stages} "
            f"NW={config.num_warps:<2d} time={status}",
            flush=True,
        )

    valid_results = [r for r in results if r["time_us"] is not None]
    valid_results.sort(key=lambda r: r["time_us"])
    baseline = bench_onednn(M, N, K, num_iters=stable_iters, dtype=dtype)

    print("\n=== Search result ===")
    print(f"oneDNN stable baseline ({stable_iters} iters): {baseline:.3f} us")
    print("Top 20 search results:")
    for rank, result in enumerate(valid_results[:20], 1):
        speedup = baseline / result["time_us"]
        print(
            f"{rank:2d}. {result['time_us']:.3f} us "
            f"{result['template']:15s} config={result['config']} "
            f"vs_oneDNN={speedup:.4f}x"
        )

    print("\n=== Best per template ===")
    for template in TEMPLATES:
        template_results = [r for r in valid_results if r["template"] == template]
        if template_results:
            result = template_results[0]
            print(
                f"{template:15s} {result['time_us']:.3f} us "
                f"config={result['config']} "
                f"vs_oneDNN={baseline / result['time_us']:.4f}x"
            )

    output = {
        "shape": [M, N, K],
        "dtype": dtype,
        "search_num_iters": num_iters,
        "stable_baseline_num_iters": stable_iters,
        "onednn_time_us": baseline,
        "results": results,
    }
    output_path = Path(f"tune_{M}_{N}_{K}_{dtype}.json")
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved results to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="128,2048,768", help="M,N,K")
    parser.add_argument("--dtype", default="bf16", choices=("int8", "bf16", "fp16"))
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--stable-iters", type=int, default=1000)
    args = parser.parse_args()
    M, N, K = (int(value) for value in args.shape.split(","))
    tune_shape(M, N, K, args.dtype, args.num_iters, args.stable_iters)


if __name__ == "__main__":
    main()
