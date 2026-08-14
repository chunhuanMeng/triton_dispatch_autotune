"""
Triton GEMM Dispatch Autotune — Main Entry Point

Implements the worst-first greedy dispatch table construction:
  Step 0: Load/measure oneDNN baseline
  Step 1: Seed search on representative shapes
  Step 2-8: Iterative worst-first loop with promotion gates

Usage:
  python run_autotune.py --step baseline     # Step 0: measure oneDNN
  python run_autotune.py --step seed         # Step 1: seed search
  python run_autotune.py --step iterate      # Step 2-8: main loop
  python run_autotune.py --step report       # Generate final report
  python run_autotune.py                     # Run all steps (auto-resume)
"""
import os, sys, json, argparse, time
from math import exp, log
from pathlib import Path

from search_space import (
    ALL_SHAPES, TEMPLATES, GemmConfig, DispatchConfig,
    generate_valid_configs, generate_autotune_configs,
    get_shape_family, get_source_pattern, template_config_keys,
    is_valid_for_template,
)
from bench_worker import bench_config_all_templates, bench_one_template, bench_onednn


# The autotune state is dtype-specific.  Do not reuse INT8 timings for BF16 or
# FP16: the accumulator, output size, reference operator, and Triton lowering
# are different.  INT8 keeps the historical ``state`` directory for resume
# compatibility; floating-point runs use separate directories.
AUTOTUNE_DTYPE = os.environ.get("XE2_AUTOTUNE_DTYPE", "int8")
# ``generic`` preserves the original worker autotune behavior: use the search
# candidates declared in search_space.py and test every template.  ``exact``
# restricts the search to configs currently registered by Inductor and is
# useful only for a strict parity run.
SEARCH_SPACE_MODE = os.environ.get("XE2_AUTOTUNE_SEARCH_SPACE", "generic")

# ═══ Paths ═══
STATE_DIR = Path("state_v6") if AUTOTUNE_DTYPE == "int8" else Path(f"state_{AUTOTUNE_DTYPE}_v6")
BASELINE_FILE = STATE_DIR / "onednn_baseline.json"
DISPATCH_FILE = STATE_DIR / "dispatch_table.json"
SWEEP_FILE = STATE_DIR / "sweep_results.json"
SEARCH_CACHE_DIR = STATE_DIR / "search_cache"
LOG_FILE = STATE_DIR / "iteration_log.json"


def configure_dtype(dtype):
    """Select dtype and rebuild all state paths before any step runs."""
    global AUTOTUNE_DTYPE, STATE_DIR, BASELINE_FILE, DISPATCH_FILE
    global SWEEP_FILE, SEARCH_CACHE_DIR, LOG_FILE
    if dtype not in ("int8", "bf16", "fp16"):
        raise ValueError(f"unsupported dtype: {dtype}")
    AUTOTUNE_DTYPE = dtype
    STATE_DIR = Path("state_v6") if dtype == "int8" else Path(f"state_{dtype}_v6")
    BASELINE_FILE = STATE_DIR / "onednn_baseline.json"
    DISPATCH_FILE = STATE_DIR / "dispatch_table.json"
    SWEEP_FILE = STATE_DIR / "sweep_results.json"
    SEARCH_CACHE_DIR = STATE_DIR / "search_cache"
    LOG_FILE = STATE_DIR / "iteration_log.json"

# ═══ Parameters ═══
MAX_DISPATCH_SIZE = 15
CONVERGE_RATIO = 0.95
MAX_ROUNDS = 20
MAX_NO_PROMOTION_ROUNDS = 3

# Benchmark iteration policy:
# - Full candidate searches prioritize coverage and use fewer iterations.
# - A candidate that may be promoted is remeasured more carefully.
# Keep the baseline and dispatch sweeps at their existing stable setting.
FULL_SEARCH_NUM_ITERS = 50
PROMOTION_NUM_ITERS = 200

# Promotion gates
CORE_TARGET_GAIN = 0.05        # 5%
CORE_GMEAN_GAIN = 0.005       # 0.5%
CORE_TAIL_GAIN = 0.01         # 1%
CORE_MAX_REGRESSION = 0.03    # 3%

SPEC_TARGET_GAIN = 0.10       # 10%
SPEC_LOCAL_GMEAN_GAIN = 0.03  # 3%
SPEC_MIN_IMPROVED = 2
SPEC_MAX_REGRESSION = 0.05    # 5%
SPEC_GMEAN_FLOOR = -0.005     # -0.5%


def shape_key(M, N, K):
    return f"{M},{N},{K}"


def compute_metrics(ratios):
    """Compute gmean, tail, worst from ratio dict."""
    vals = list(ratios.values())
    if not vals:
        return {"gmean": 0, "tail": 0, "worst": 0}
    gmean = exp(sum(log(max(v, 0.01)) for v in vals) / len(vals))
    sorted_vals = sorted(vals)
    tail_n = max(1, len(vals) // 10)
    tail = sum(sorted_vals[:tail_n]) / tail_n
    worst = min(vals)
    return {"gmean": gmean, "tail": tail, "worst": worst}


# ═══ Step 0: Baseline ═══
def step_baseline():
    """Measure oneDNN baseline for all shapes."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing if available
    baseline = {}
    if BASELINE_FILE.exists():
        baseline = json.loads(BASELINE_FILE.read_text())
        print(f"Loaded {len(baseline)} existing baseline entries")

    remaining = [(M, N, K) for M, N, K in ALL_SHAPES
                 if shape_key(M, N, K) not in baseline]
    print(f"Measuring oneDNN baseline: {len(remaining)} shapes remaining")

    for i, (M, N, K) in enumerate(remaining):
        time_us = bench_onednn(M, N, K, num_iters=500, dtype=AUTOTUNE_DTYPE)
        if time_us is None:
            raise RuntimeError(
                f"oneDNN baseline benchmark failed for shape ({M},{N},{K}); "
                "see the worker error output"
            )
        baseline[shape_key(M, N, K)] = {"time_us": round(time_us, 2)}
        # Save incrementally
        if (i + 1) % 5 == 0 or i == len(remaining) - 1:
            BASELINE_FILE.write_text(json.dumps(baseline, indent=2))
            print(f"  [{i+1}/{len(remaining)}] ({M},{N},{K}): {time_us:.1f}us")

    print(f"Baseline complete: {len(baseline)} shapes")
    return baseline


# ═══ Step 1: Seed Search ═══
def step_seed():
    """Full search on seed shapes to initialize dispatch table."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SEARCH_CACHE_DIR.mkdir(exist_ok=True)

    # Pick 2 seed shapes: one compute-bound, one memory-bound
    seed_shapes = [
        (2048, 4096, 4096),   # compute-bound
        (4, 28672, 4096),     # memory-bound (large decode)
    ]

    dispatch_table = []
    if DISPATCH_FILE.exists():
        dispatch_table = [DispatchConfig.from_dict(d) for d in json.loads(DISPATCH_FILE.read_text())]
        if dispatch_table:
            print(f"Dispatch table already has {len(dispatch_table)} configs, skipping seed")
            return dispatch_table

    for M, N, K in seed_shapes:
        results = search_shape(M, N, K)
        if results:
            # Take top-2 from each seed
            for config, time_us in results[:2]:
                if config not in dispatch_table:
                    dispatch_table.append(config)
                    print(f"  Seed: added {config.template} BM={config.BLOCK_M} BN={config.BLOCK_N} "
                          f"BK={config.BLOCK_K} NS={config.num_stages} NW={config.num_warps} ({time_us:.1f}us)")

    # Save
    DISPATCH_FILE.write_text(json.dumps([c.to_dict() for c in dispatch_table], indent=2))
    print(f"Seed complete: {len(dispatch_table)} configs in dispatch table")
    return dispatch_table


def search_shape(M, N, K):
    """Full five-dimensional search on one shape. Uses a dtype-specific cache."""
    if SEARCH_SPACE_MODE == "persistent_group":
        raise ValueError(
            "XE2_AUTOTUNE_SEARCH_SPACE=persistent_group is no longer supported; "
            "GROUP_M is fixed at Inductor's default and is not a search dimension"
        )
    cache_file = SEARCH_CACHE_DIR / f"search_{M}_{N}_{K}.json"

    # Load cache
    cached_results = {}
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        cached_results = {
            tuple(d["config_key"]): d["time_us"] for d in data.get("results", [])
        }

    if SEARCH_SPACE_MODE == "exact":
        # Strict parity mode: only configs registered by the corresponding
        # Inductor heuristic are benchmarked.
        valid_configs = generate_autotune_configs(M, N, K, AUTOTUNE_DTYPE)
        choices = [
            (template, config)
            for config in valid_configs
            for template in TEMPLATES
            if config.key in template_config_keys(template, AUTOTUNE_DTYPE)
        ]
    else:
        # Generic mode searches every hardware-valid Cartesian configuration.
        # Do not apply the heuristic is_good_config() pruning here: it can
        # discard viable BF16 candidates such as BK=32 for large-K shapes.
        valid_configs = generate_valid_configs(M, N, K)
        choices = [
            (template, config)
            for config in valid_configs
            for template in TEMPLATES
            if is_valid_for_template(M, N, K, config, template)
        ]
    print(f"  Searching ({M},{N},{K}): {len(choices)} template/config choices, {len(cached_results)} cached")

    results = []
    new_count = 0
    failed = 0

    for i, (template, config) in enumerate(choices):
        key = [template, *config.key]
        if tuple(key) in cached_results:
            cached = cached_results[tuple(key)]
            if isinstance(cached, dict):
                time_us = cached.get("time_us")
            else:
                time_us = cached  # backward compat
            if time_us is not None:
                results.append((DispatchConfig(template, config), time_us))
            continue

        time_us = bench_one_template(
            M, N, K, config, template,
            num_iters=FULL_SEARCH_NUM_ITERS, dtype=AUTOTUNE_DTYPE
        )
        cached_results[tuple(key)] = {"time_us": time_us, "template": template}
        new_count += 1

        if time_us is not None:
            results.append((DispatchConfig(template, config), time_us))
        else:
            failed += 1

        # Save cache periodically
        if new_count % 10 == 0:
            _save_search_cache(cache_file, M, N, K, cached_results)
            print(f"    [{i+1}/{len(choices)}] {new_count} new, {failed} failed")

    # Final save
    _save_search_cache(cache_file, M, N, K, cached_results)
    print(f"  Done: {len(results)} successful, {failed} failed, {new_count} newly benchmarked")

    # Sort by time
    results.sort(key=lambda x: x[1])
    return results


def _save_search_cache(cache_file, M, N, K, cached_results):
    data = {
        "shape": {"M": M, "N": N, "K": K},
        "results": [{"config_key": list(k), "time_us": v.get("time_us") if isinstance(v, dict) else v,
                     "template": v.get("template") if isinstance(v, dict) else None}
                    for k, v in cached_results.items()],
    }
    cache_file.write_text(json.dumps(data))


# ═══ Step 2: Sweep ═══
def step_sweep(dispatch_table, baseline, round_num=0):
    """Sweep all shapes with current dispatch table.

    Reuse completed shape/config measurements from SWEEP_FILE.  This is
    important because a sweep can be interrupted after many shapes; the
    previous implementation only wrote the file after the whole sweep and
    always re-benchmarked every config on resume.
    """
    results = {}  # shape_key -> {config_key : time_us}
    ratios = {}

    cached_results = {}
    if SWEEP_FILE.exists():
        try:
            cached_results = json.loads(SWEEP_FILE.read_text()).get("results", {})
            print(f"Loaded sweep cache: {len(cached_results)} shapes")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: ignoring invalid sweep cache {SWEEP_FILE}: {exc}")

    print(f"Sweeping {len(ALL_SHAPES)} shapes × {len(dispatch_table)} configs...")
    for i, (M, N, K) in enumerate(ALL_SHAPES):
        sk = shape_key(M, N, K)
        old_shape_results = cached_results.get(sk, {})
        shape_results = {}

        for config in dispatch_table:
            config_key = str(config.key)
            if config_key in old_shape_results:
                time_us = old_shape_results[config_key]
            else:
                time_us = bench_one_template(
                    M, N, K, config.gemm, config.template,
                    num_iters=500, dtype=AUTOTUNE_DTYPE,
                )
            if time_us is not None:
                shape_results[config_key] = time_us

        if shape_results:
            best_time = min(shape_results.values())
            baseline_time = baseline.get(sk, {}).get("time_us", best_time)
            ratios[sk] = baseline_time / best_time if best_time > 0 else 0
        else:
            ratios[sk] = 0

        results[sk] = shape_results

        if (i + 1) % 10 == 0:
            metrics = compute_metrics(ratios)
            print(f"  [{i+1}/{len(ALL_SHAPES)}] gmean={metrics['gmean']:.4f} worst={metrics['worst']:.4f}")
            _save_sweep_results(results, ratios, round_num)

    _save_sweep_results(results, ratios, round_num)
    metrics = compute_metrics(ratios)
    print(f"Sweep done: gmean={metrics['gmean']:.4f} tail={metrics['tail']:.4f} worst={metrics['worst']:.4f}")
    return results, ratios, metrics


def _save_sweep_results(results, ratios, round_num):
    """Persist sweep progress to the resume cache and current round."""
    sweep_data = json.dumps({"results": results, "ratios": ratios}, indent=2)
    if round_num > 0:
        round_dir = STATE_DIR / f"round_{round_num}"
        round_dir.mkdir(exist_ok=True)
        (round_dir / "sweep_results.json").write_text(sweep_data)
    SWEEP_FILE.write_text(sweep_data)


# ═══ Step 3: Zero-Win Prune ═══
def step_prune(dispatch_table, sweep_results):
    """Remove configs that win zero shapes."""
    winners = {}  # config_key -> win_count
    for sk, shape_results in sweep_results.items():
        if shape_results:
            best_key = min(shape_results, key=shape_results.get)
            winners[best_key] = winners.get(best_key, 0) + 1

    pruned = []
    kept = []
    for config in dispatch_table:
        if str(config.key) in winners or len(dispatch_table) <= 1:
            kept.append(config)
        else:
            pruned.append(config)

    if pruned:
        print(f"  Pruned {len(pruned)} zero-win configs:")
        for c in pruned:
            print(f"    - BM={c.BLOCK_M} BN={c.BLOCK_N} BK={c.BLOCK_K}")

    return kept


# ═══ Step 7: Promotion ═══
def evaluate_promotion(candidate, dispatch_table, baseline, current_metrics, worst_shape_key):
    """Evaluate if candidate should be promoted. Returns ('core', 'specialist', or 'reject')."""
    # Quick sweep: only measure candidate on all shapes
    trial_ratios = {}
    for M, N, K in ALL_SHAPES:
        sk = shape_key(M, N, K)
        time_us = bench_one_template(
            M, N, K, candidate.gemm, candidate.template,
            num_iters=PROMOTION_NUM_ITERS, dtype=AUTOTUNE_DTYPE,
        )
        if time_us is not None:
            baseline_time = baseline.get(sk, {}).get("time_us", time_us)
            trial_ratios[sk] = baseline_time / time_us if time_us > 0 else 0

    # Merge with existing ratios (take max = best of old + new)
    # Load current sweep ratios
    if SWEEP_FILE.exists():
        sweep_data = json.loads(SWEEP_FILE.read_text())
        old_ratios = sweep_data.get("ratios", {})
    else:
        old_ratios = {}

    merged_ratios = {}
    for sk in set(list(old_ratios.keys()) + list(trial_ratios.keys())):
        old_r = old_ratios.get(sk, 0)
        new_r = trial_ratios.get(sk, 0)
        merged_ratios[sk] = max(old_r, new_r)

    new_metrics = compute_metrics(merged_ratios)

    # Compute gains
    target_gain = (merged_ratios.get(worst_shape_key, 0) /
                   max(old_ratios.get(worst_shape_key, 0.01), 0.01)) - 1
    gmean_gain = new_metrics["gmean"] / max(current_metrics["gmean"], 0.01) - 1
    tail_gain = new_metrics["tail"] / max(current_metrics["tail"], 0.01) - 1

    # Max regression
    max_regression = 0
    for sk in old_ratios:
        if sk in merged_ratios:
            regression = old_ratios[sk] - merged_ratios[sk]
            max_regression = max(max_regression, regression / max(old_ratios[sk], 0.01))

    # Improved count
    improved = sum(1 for sk in trial_ratios
                   if trial_ratios[sk] > old_ratios.get(sk, 0) * 1.01)

    print(f"    Candidate eval: target_gain={target_gain:.3f} gmean_gain={gmean_gain:.4f} "
          f"tail_gain={tail_gain:.4f} max_regr={max_regression:.4f} improved={improved}")

    # Core gate
    if (target_gain >= CORE_TARGET_GAIN and
        gmean_gain >= CORE_GMEAN_GAIN and
        tail_gain >= CORE_TAIL_GAIN and
        max_regression <= CORE_MAX_REGRESSION):
        return "core", new_metrics

    # Specialist gate
    if (target_gain >= SPEC_TARGET_GAIN and
        improved >= SPEC_MIN_IMPROVED and
        gmean_gain >= SPEC_GMEAN_FLOOR and
        max_regression <= SPEC_MAX_REGRESSION):
        return "specialist", new_metrics

    return "reject", new_metrics


# ═══ Main Loop ═══
def step_iterate():
    """Main worst-first iteration loop."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Load baseline.  An interrupted incremental baseline write may exist but
    # still be incomplete, so resume Step 0 rather than using partial ratios.
    baseline_complete = False
    if BASELINE_FILE.exists():
        existing_baseline = json.loads(BASELINE_FILE.read_text())
        baseline_complete = all(
            shape_key(M, N, K) in existing_baseline for M, N, K in ALL_SHAPES
        )
    if not baseline_complete:
        print("No baseline found, running step_baseline first...")
        step_baseline()
    baseline = json.loads(BASELINE_FILE.read_text())

    # Load or create dispatch table
    if not DISPATCH_FILE.exists():
        print("No dispatch table found, running step_seed first...")
        step_seed()
    dispatch_table = [DispatchConfig.from_dict(d) for d in json.loads(DISPATCH_FILE.read_text())]

    # Load iteration log
    log_entries = []
    if LOG_FILE.exists():
        log_entries = json.loads(LOG_FILE.read_text())

    no_promotion_count = 0
    round_num = len(log_entries)
    exhausted_shapes = set()  # Track shapes where all candidates failed promotion

    while True:
        round_num += 1
        print(f"\n{'='*60}")
        print(f"ROUND {round_num} | dispatch_table size = {len(dispatch_table)}")
        print(f"{'='*60}")

        # Step 2: Sweep
        sweep_results, ratios, metrics = step_sweep(dispatch_table, baseline, round_num)

        # Step 3: Prune
        dispatch_table = step_prune(dispatch_table, sweep_results)

        # Step 4: Convergence check
        if metrics["worst"] >= CONVERGE_RATIO:
            print(f"\n✓ CONVERGED: worst_ratio={metrics['worst']:.4f} >= {CONVERGE_RATIO}")
            break
        if len(dispatch_table) >= MAX_DISPATCH_SIZE:
            print(f"\n✓ FULL: dispatch_table has {len(dispatch_table)} configs")
            break
        if round_num >= MAX_ROUNDS:
            print(f"\n✓ MAX ROUNDS: {round_num}")
            break
        if no_promotion_count >= MAX_NO_PROMOTION_ROUNDS:
            print(f"\n✓ SATURATED: {no_promotion_count} rounds without promotion")
            break

        # Step 5: Worst-first select (skip exhausted shapes)
        available_shapes = {sk: r for sk, r in ratios.items() if sk not in exhausted_shapes}
        if not available_shapes:
            print(f"\n✓ ALL SHAPES EXHAUSTED: {len(exhausted_shapes)} shapes tested, no more candidates")
            print(f"  Exhausted shapes: {sorted(exhausted_shapes)}")
            break
        worst_sk = min(available_shapes, key=available_shapes.get)
        worst_ratio = ratios[worst_sk]
        M, N, K = [int(x) for x in worst_sk.split(",")]
        print(f"\n  Worst shape: ({M},{N},{K}) ratio={worst_ratio:.4f} (exhausted: {len(exhausted_shapes)})")

        # Step 6: Search on worst shape
        search_results = search_shape(M, N, K)
        if not search_results:
            print("  No valid configs found for worst shape!")
            no_promotion_count += 1
            continue

        # Step 7: Try promotion for top candidates
        promoted = False
        for candidate, cand_time in search_results[:5]:
            if candidate in dispatch_table:
                continue
            print(f"\n  Evaluating: template={candidate.template} BM={candidate.BLOCK_M} BN={candidate.BLOCK_N} "
                  f"BK={candidate.BLOCK_K} NS={candidate.num_stages} NW={candidate.num_warps} "
                  f"(time={cand_time:.1f}us on worst shape)")

            decision, new_metrics = evaluate_promotion(
                candidate, dispatch_table, baseline, metrics, worst_sk)

            if decision in ("core", "specialist"):
                dispatch_table.append(candidate)
                print(f"  ✓ PROMOTED as {decision}! New dispatch size: {len(dispatch_table)}")
                print(f"    gmean: {metrics['gmean']:.4f} → {new_metrics['gmean']:.4f}")
                print(f"    worst: {metrics['worst']:.4f} → {new_metrics['worst']:.4f}")
                promoted = True
                no_promotion_count = 0
                break
            else:
                print(f"  ✗ Rejected")

        if not promoted:
            no_promotion_count += 1
            print(f"  No promotion this round ({no_promotion_count} consecutive)")
            # Mark this shape as exhausted - all top candidates failed
            exhausted_shapes.add(worst_sk)
            print(f"  Marking shape {worst_sk} as exhausted ({len(exhausted_shapes)} total)")

        # Check saturation AFTER marking exhausted (so skip logic can take effect next round)
        if no_promotion_count >= MAX_NO_PROMOTION_ROUNDS:
            print(f"\n✓ SATURATED: {no_promotion_count} rounds without promotion")
            break

        # Save state
        DISPATCH_FILE.write_text(json.dumps([c.to_dict() for c in dispatch_table], indent=2))
        log_entries.append({
            "round": round_num,
            "worst_shape": worst_sk,
            "worst_ratio": worst_ratio,
            "dispatch_size": len(dispatch_table),
            "promoted": promoted,
            "metrics": metrics,
        })
        LOG_FILE.write_text(json.dumps(log_entries, indent=2))

    # Final save
    DISPATCH_FILE.write_text(json.dumps([c.to_dict() for c in dispatch_table], indent=2))
    print(f"\nFinal dispatch table: {len(dispatch_table)} configs")
    return dispatch_table


# ═══ Report ═══
def step_report():
    """Generate final report."""
    if not DISPATCH_FILE.exists():
        print("No dispatch table found!")
        return

    dispatch_table = [DispatchConfig.from_dict(d) for d in json.loads(DISPATCH_FILE.read_text())]
    baseline = json.loads(BASELINE_FILE.read_text()) if BASELINE_FILE.exists() else {}

    print(f"\n{'='*70}")
    print(f"FINAL DISPATCH TABLE ({len(dispatch_table)} configs)")
    print(f"{'='*70}")
    for i, c in enumerate(dispatch_table):
        print(f"  {i+1}. template={c.template:<15} BM={c.BLOCK_M:>3} BN={c.BLOCK_N:>3} "
              f"BK={c.BLOCK_K:>3} NS={c.num_stages} NW={c.num_warps:>2}")

    # Generate heuristic code
    print(f"\n{'='*70}")
    print("HEURISTIC CODE (paste into template_heuristics/triton.py):")
    print(f"{'='*70}")
    for template in TEMPLATES:
        print(f"\n# {template}")
        print("self.mm_configs = [")
        for c in dispatch_table:
            if c.template == template:
                print(f"    GemmConfig({c.BLOCK_M}, {c.BLOCK_N}, {c.BLOCK_K}, "
                      f"{c.num_stages}, {c.num_warps}),")
        print("]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["baseline", "seed", "iterate", "report", "all"],
                        default="all")
    parser.add_argument(
        "--search-shape",
        help="search one shape only, formatted as M,N,K; bypasses the multi-shape loop",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="with --search-shape, run search_shape() and do not run baseline/iterate/report",
    )
    parser.add_argument(
        "--dtype", choices=("int8", "bf16", "fp16"), default="int8",
        help="autotune dtype; state is isolated per dtype",
    )
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    configure_dtype(args.dtype)

    if args.search_shape:
        try:
            M, N, K = (int(value) for value in args.search_shape.split(","))
        except ValueError as exc:
            parser.error("--search-shape must be formatted as M,N,K")
        results = search_shape(M, N, K)
        print(f"\nSingle-shape search complete: ({M},{N},{K})")
        for rank, (config, time_us) in enumerate(results[:20], 1):
            print(
                f"{rank:2d}. {time_us:.3f} us "
                f"{config.template} config=({config.BLOCK_M},{config.BLOCK_N},"
                f"{config.BLOCK_K},{config.num_stages},{config.num_warps}"
                + ")"
            )
        if args.search_only:
            return

    if args.step == "baseline" or args.step == "all":
        step_baseline()
    if args.step == "seed" or args.step == "all":
        step_seed()
    if args.step == "iterate" or args.step == "all":
        step_iterate()
    if args.step == "report" or args.step == "all":
        step_report()


if __name__ == "__main__":
    main()
