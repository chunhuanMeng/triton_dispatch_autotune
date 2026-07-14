#!/usr/bin/env python3
"""
Oracle Best Measurement — Find the best (config, template) for each shape.

Tests all 7 dispatch configs × all 3 templates on all 118 shapes,
then summarizes:
1. Per-shape oracle best (config, template, ratio)
2. Per-template distribution: which config wins most often
3. Oracle coverage: how many shapes can beat oneDNN with these 7 configs
"""
import json
import torch
import triton
import triton.language as tl
from pathlib import Path
from math import exp, log
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent))

from search_space import ALL_SHAPES, TEMPLATES, GemmConfig, is_valid_for_template, generate_valid_configs
from bench_worker import _bench_template, bench_onednn


# ═══ 7 Dispatch Configs (from tune_6) ═══
DISPATCH_CONFIGS = [
    GemmConfig(256, 128, 64, 3, 16),   # config_1
    GemmConfig(8, 512, 32, 2, 16),      # config_2
    GemmConfig(8, 512, 32, 2, 8),       # config_3
    GemmConfig(256, 256, 64, 2, 32),    # config_4
    GemmConfig(32, 256, 32, 2, 8),     # config_5
    GemmConfig(128, 512, 64, 2, 32),    # config_6
    GemmConfig(256, 256, 128, 2, 32),   # config_7
]

# Config names for display
CONFIG_NAMES = [f"config_{i+1}" for i in range(len(DISPATCH_CONFIGS))]


def config_index(config):
    """Return 1-based config index."""
    for i, c in enumerate(DISPATCH_CONFIGS):
        if (c.BLOCK_M == config.BLOCK_M and
            c.BLOCK_N == config.BLOCK_N and
            c.BLOCK_K == config.BLOCK_K and
            c.num_stages == config.num_stages and
            c.num_warps == config.num_warps):
            return i + 1
    return None


def shape_key(M, N, K):
    return f"{M},{N},{K}"


def measure_oracle_all_shapes(num_iters=100, output_file="oracle_results.json"):
    """
    Measure oracle best for all shapes using the 7 dispatch configs.
    
    Returns:
        oracle_results: {shape_key: {"config": idx, "template": name, "ratio": float, "time_us": float}}
        summary: {...}
    """
    print(f"Measuring oracle best for {len(ALL_SHAPES)} shapes...")
    print(f"Using {len(DISPATCH_CONFIGS)} dispatch configs × {len(TEMPLATES)} templates")
    print(f"num_iters={num_iters}")
    print()
    
    # Load baseline
    baseline_file = Path("state/onednn_baseline.json")
    if baseline_file.exists():
        baseline = json.loads(baseline_file.read_text())
        print(f"Loaded {len(baseline)} baseline entries")
    else:
        print("WARNING: No baseline found, will measure oneDNN first...")
        baseline = {}
    
    oracle_results = {}
    config_wins = defaultdict(int)  # config_idx -> win count
    template_wins = defaultdict(int)  # template -> win count
    config_template_wins = defaultdict(lambda: defaultdict(int))  # config_idx -> template -> wins
    
    total_shapes = len(ALL_SHAPES)
    
    for shape_idx, (M, N, K) in enumerate(ALL_SHAPES):
        sk = shape_key(M, N, K)
        print(f"[{shape_idx+1}/{total_shapes}] Shape ({M},{N},{K})")
        
        # Measure oneDNN baseline if not available
        if sk not in baseline:
            print(f"  Measuring oneDNN baseline...")
            onednn_time = bench_onednn(M, N, K, num_iters=num_iters)
            baseline[sk] = {"time_us": onednn_time}
            baseline_file.write_text(json.dumps(baseline, indent=2))
        else:
            onednn_time = baseline[sk]["time_us"]
        
        # Find oracle best across all configs × templates
        best_time = float('inf')
        best_config_idx = None
        best_template = None
        best_ratio = 0
        
        for config in DISPATCH_CONFIGS:
            config_idx = config_index(config)
            
            for template in TEMPLATES:
                if not is_valid_for_template(M, N, K, config, template):
                    continue
                
                time_us = _bench_template(
                    torch.randint(-128, 127, (M, K), dtype=torch.int8, device='xpu'),
                    torch.randint(-128, 127, (K, N), dtype=torch.int8, device='xpu'),
                    torch.zeros((M, N), dtype=torch.int32, device='xpu'),
                    M, N, K, config, template, num_iters
                )
                
                if time_us is not None:
                    ratio = onednn_time / time_us if time_us > 0 else 0
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_time = time_us
                        best_config_idx = config_idx
                        best_template = template
        
        if best_config_idx is not None:
            oracle_results[sk] = {
                "M": M, "N": N, "K": K,
                "config": best_config_idx,
                "template": best_template,
                "ratio": best_ratio,
                "time_us": best_time,
            }
            
            config_wins[best_config_idx] += 1
            template_wins[best_template] += 1
            config_template_wins[best_config_idx][best_template] += 1
            
            print(f"  Oracle: config_{best_config_idx} + {best_template} ratio={best_ratio:.4f}")
        else:
            print(f"  WARNING: No valid config+template for this shape!")
            oracle_results[sk] = None
    
    # Compute summary statistics
    ratios = [r["ratio"] for r in oracle_results.values() if r is not None]
    gmean = exp(sum(log(max(r, 0.01)) for r in ratios) / len(ratios)) if ratios else 0
    worst = min(ratios) if ratios else 0
    winning_shapes = sum(1 for r in ratios if r > 1.0)
    
    summary = {
        "total_shapes": total_shapes,
        "shapes_with_valid": len([r for r in oracle_results.values() if r is not None]),
        "gmean": gmean,
        "worst": worst,
        "winning_shapes": winning_shapes,
        "winning_pct": 100 * winning_shapes / total_shapes if total_shapes > 0 else 0,
        "config_wins": dict(config_wins),
        "template_wins": dict(template_wins),
        "config_template_wins": {str(k): dict(v) for k, v in config_template_wins.items()},
    }
    
    # Save results
    results_data = {
        "oracle_results": oracle_results,
        "summary": summary,
        "dispatch_configs": [c.to_dict() for c in DISPATCH_CONFIGS],
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return oracle_results, summary


def print_summary(oracle_results, summary):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("ORACLE SUMMARY")
    print("=" * 70)
    
    print(f"\n--- Overall ---")
    print(f"Total shapes: {summary['total_shapes']}")
    print(f"Valid shapes: {summary['shapes_with_valid']}")
    print(f"Oracle gmean ratio: {summary['gmean']:.4f}")
    print(f"Oracle worst ratio: {summary['worst']:.4f}")
    print(f"Shapes beating oneDNN: {summary['winning_shapes']}/{summary['total_shapes']} ({summary['winning_pct']:.1f}%)")
    
    print(f"\n--- Config Wins (which config gets most oracle bests) ---")
    sorted_config_wins = sorted(summary['config_wins'].items(), key=lambda x: -x[1])
    for config_idx, win_count in sorted_config_wins:
        config = DISPATCH_CONFIGS[config_idx - 1]
        print(f"  config_{config_idx} (BM={config.BLOCK_M}, BN={config.BLOCK_N}, BK={config.BLOCK_K}, NS={config.num_stages}, NW={config.num_warps}): {win_count} shapes")
    
    print(f"\n--- Template Wins (which template wins most) ---")
    sorted_template_wins = sorted(summary['template_wins'].items(), key=lambda x: -x[1])
    for template, win_count in sorted_template_wins:
        print(f"  {template}: {win_count} shapes")
    
    print(f"\n--- Config × Template Distribution ---")
    for config_idx in range(1, len(DISPATCH_CONFIGS) + 1):
        config = DISPATCH_CONFIGS[config_idx - 1]
        ct_wins = summary['config_template_wins'].get(str(config_idx), {})
        if ct_wins:
            template_strs = [f"{t}:{w}" for t, w in sorted(ct_wins.items(), key=lambda x: -x[1])]
            print(f"  config_{config_idx}: {', '.join(template_strs)}")
    
    print(f"\n--- Per-Shape Oracle Best (first 20) ---")
    sorted_results = sorted(oracle_results.items(), key=lambda x: x[1]["ratio"] if x[1] else 0)
    for sk, result in sorted_results[:20]:
        if result:
            print(f"  ({result['M']},{result['N']},{result['K']}): "
                  f"config_{result['config']} + {result['template']} "
                  f"ratio={result['ratio']:.4f} ({result['time_us']:.1f}us)")
    
    if len(sorted_results) > 20:
        print(f"  ... and {len(sorted_results) - 20} more shapes")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Oracle best measurement")
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations per benchmark")
    parser.add_argument("--output", default="oracle_results.json", help="Output file")
    args = parser.parse_args()
    
    oracle_results, summary = measure_oracle_all_shapes(
        num_iters=args.iters,
        output_file=args.output
    )
    
    print_summary(oracle_results, summary)