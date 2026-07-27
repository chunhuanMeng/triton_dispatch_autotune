"""
A/B comparison for XPU BF16 GEMM in torch.compile(max-autotune):

  State A (original baseline): only `triton_mm` template, original generic
      default config search space (no XPU-specific tuning).
  State B (current/new state): 3 templates (triton_mm + bmg_persistent +
      bmg_decode), using the current v7 XPU heuristic config lists
      (15 BF16 Triton configs before shape filtering).

State is selected purely via environment variables read inside
torch/_inductor/heuristics/template/triton.py:

  State A: XE2_ENABLE_BMG_FLOAT_TEMPLATES=0  XE2_MM_TUNED_CONFIGS=0
  State B: XE2_ENABLE_BMG_FLOAT_TEMPLATES=1  XE2_MM_TUNED_CONFIGS=1

Methodology mirrors int8_gemm_optimization_xe2/scripts/int_mm_ab_compare.py
and int_mm_ab_fair.py:
  - each (shape, mode) benchmark runs in its own subprocess (avoids OOM from
    accumulated Triton compilation memory, and avoids any state leaking
    between the BaseHeuristicSingleton-cached heuristic classes across modes)
  - AlgorithmSelectorCache.log_results is monkeypatched to capture every
    candidate's autotune timing (oneDNN/ATen choice + every Triton choice),
    not just the one Inductor ultimately picks
  - for each shape, mode A and mode B are run back-to-back ("fair A/B") to
    reduce the impact of GPU clock/power-state drift between separate full
    A-run and full B-run passes
    - candidate timings use the same ``benchmarker.benchmark`` interface as
        production Inductor max-autotune. The statistic, iteration count, and
        time budget therefore come from the active Inductor benchmarker.

Usage:
  python bf16_ab_compare.py --run [--out FILE] [--timeout SECONDS]
      # run fair A/B across the current v7 shapes
  python bf16_ab_compare.py --show FILE
      # print the comparison report from saved results
  python bf16_ab_compare.py --worker M N K --mode A|B
      # internal: benchmark a single shape in a single state, print RESULT: json
"""
import argparse
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, "/home/sdp/meng/pytorch")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_RESULTS_PATH = os.path.join(
    SCRIPT_DIR, "state_bf16_v6", "sweep_results.json"
)
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "bf16_ab_compare_results.json")
DEFAULT_TIMEOUT = 240


def load_shapes():
    """Load current v7 shapes and exclude stale M=1 sweep entries."""
    from search_space import ALL_SHAPES

    with open(SWEEP_RESULTS_PATH) as f:
        data = json.load(f)["results"]
    available = set(data)
    return [shape for shape in ALL_SHAPES if ",".join(map(str, shape)) in available]


def mode_env(mode):
    """Environment overrides that select state A or state B."""
    if mode == "A":
        return {"XE2_ENABLE_BMG_FLOAT_TEMPLATES": "0", "XE2_MM_TUNED_CONFIGS": "0"}
    if mode == "B":
        return {"XE2_ENABLE_BMG_FLOAT_TEMPLATES": "1", "XE2_MM_TUNED_CONFIGS": "1"}
    raise ValueError(f"unknown mode: {mode}")


def worker_main(M, N, K, manual_iters=None):
    """Run in subprocess: benchmark one shape (BF16) and print JSON to stdout.

    The mode (A/B) is selected entirely via env vars set on this subprocess
    by the parent (see mode_env()); this function does not need to know
    which mode it is running under.
    """
    import logging

    import torch
    import torch._inductor.config as inductor_config
    from torch.compiler import config as compiler_config

    # Optional diagnostic used to resolve the numeric triton_mm_N labels in
    # captured autotune results back to their actual GemmConfig parameters.
    # Keep this disabled during normal A/B runs because it adds stdout noise.
    if os.environ.get("XE2_DUMP_TEMPLATE_CONFIGS") == "1":
        from torch._inductor.kernel_template_choice import KernelTemplateChoice

        original_choice_property = KernelTemplateChoice.choice

        def dump_choice(self):
            choice = original_choice_property.fget(self)
            if choice is not None and "triton" in choice.name.lower():
                print(
                    "CONFIG:"
                    + json.dumps(
                        {
                            "name": choice.name,
                            "template": self.template.uid,
                            "params": self.params.to_kwargs(),
                        },
                        default=str,
                    ),
                    flush=True,
                )
            return choice

        KernelTemplateChoice.choice = property(dump_choice)

    logging.getLogger("torch._inductor.autotune_process").setLevel(logging.WARNING)
    logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.WARNING)

    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm = True
    inductor_config.fx_graph_cache = False
    inductor_config.autotune_local_cache = False
    compiler_config.force_disable_caches = True
    compiler_config.assume_static_by_default = True

    # Capture every autotune candidate's timing, not just the winner.
    captured = []
    # Capture the choice list passed into every AlgorithmSelectorCache call too:
    # when there is only a single valid choice (e.g. all Triton templates get
    # filtered out for some tiny-M decode shapes), Inductor short-circuits
    # *before* ever calling log_results, so `captured` stays empty even though
    # autotuning "worked" (there was just nothing to compare against).
    choices_seen = []
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
        result = {}
        for choice, time_ms in timings.items():
            result[choice.name] = time_ms
        captured.append(result)
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
        choices_seen.append([c.name for c in choices])
        return original_call(self, name, choices, input_nodes, layout, *args, **kwargs)

    AlgorithmSelectorCache.log_results = patched_log
    AlgorithmSelectorCache.__call__ = patched_call

    a = torch.randn((M, K), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((K, N), dtype=torch.bfloat16, device="xpu")

    fn = torch.compile(
        lambda x, y: torch.mm(x, y),
        backend="inductor",
        fullgraph=True,
        dynamic=False,
    )

    output = fn(a, b)
    torch.xpu.synchronize()
    reference = torch.mm(a, b)
    torch.testing.assert_close(output, reference, rtol=0.02, atol=0.02)

    def measure_compiled_fn(num_iters):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        torch.xpu.synchronize()
        start.record()
        for _ in range(num_iters):
            fn(a, b)
        end.record()
        torch.xpu.synchronize()
        return start.elapsed_time(end) / num_iters

    # Optional probe measurement. This is separate from the autotune timings:
    # it measures the fully compiled selected fn(a, b), while captured timings
    # measure each individual candidate.
    measured_elapsed_ms = (
        measure_compiled_fn(manual_iters) if manual_iters is not None else None
    )

    if not captured:
        names = choices_seen[-1] if choices_seen else []
        if not names:
            # AlgorithmSelectorCache.__call__ was never invoked at all, i.e.
            # `tuned_mm` never ran. Confirmed root cause (via TORCH_LOGS=
            # +inductor) for M=1 shapes: Inductor decomposes torch.mm(1xK, KxN)
            # into elementwise-mul + reduction (two fused Triton pointwise/
            # reduction kernels), entirely bypassing the mm template/autotune
            # machinery. This means there is no MM candidate to compare, but
            # it does not prove that the generated kernels or runtime timing
            # are identical across states; report the measurements explicitly.
            NUM_ITERS = manual_iters or 200
            elapsed_ms = measured_elapsed_ms or measure_compiled_fn(NUM_ITERS)
            print(
                "RESULT:"
                + json.dumps(
                    {
                        "decomposed": True,
                        "note": "mm decomposed to elementwise+reduction; "
                        "no MM template/autotune choices involved",
                        "elapsed_ms": elapsed_ms,
                        "onednn": None,
                        "best_triton": None,
                        "num_choices": 0,
                        "num_triton_choices": 0,
                    }
                )
            )
            return

        # Only a single choice was available (Inductor skips benchmarking
        # entirely in that case). Manually time the single choice via
        # amortized event timing so this shape still yields a usable data
        # point instead of an error.
        NUM_ITERS = manual_iters or 200
        elapsed_ms = measured_elapsed_ms or measure_compiled_fn(NUM_ITERS)

        single_name = names[0]
        is_triton = "triton" in single_name.lower()
        print(
            "RESULT:"
            + json.dumps(
                {
                    "onednn": None if is_triton else elapsed_ms,
                    "onednn_name": None if is_triton else single_name,
                    "best_triton": elapsed_ms if is_triton else None,
                    "best_triton_name": single_name if is_triton else None,
                    "num_choices": len(names),
                    "num_triton_choices": 1 if is_triton else 0,
                    "single_choice": True,
                }
            )
        )
        return

    timings = captured[-1]
    non_triton = [
        (name, t) for name, t in timings.items() if "triton" not in name.lower()
    ]
    triton_entries = [
        (name, t) for name, t in timings.items() if "triton" in name.lower()
    ]

    onednn_name, onednn_t = (
        min(non_triton, key=lambda x: x[1]) if non_triton else (None, None)
    )
    best_triton_name, best_triton_t = (
        min(triton_entries, key=lambda x: x[1]) if triton_entries else (None, None)
    )

    result = {
        "elapsed_ms": measured_elapsed_ms,
        "onednn": onednn_t,
        "onednn_name": onednn_name,
        "best_triton": best_triton_t,
        "best_triton_name": best_triton_name,
        "num_choices": len(timings),
        "num_triton_choices": len(triton_entries),
        "captured_timings_ms": {
            name: time_ms for name, time_ms in timings.items()
        },
    }
    print(
        "RESULT:"
        + json.dumps(result)
    )


def run_one_shape(M, N, K, mode, base_env, timeout):
    """Call this script's --worker mode in a subprocess, return parsed dict."""
    env = dict(base_env)
    env.update(mode_env(mode))
    cmd = [
        sys.executable,
        "-u",
        __file__,
        "--worker",
        str(M),
        str(N),
        str(K),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
        json_line = None
        for line in proc.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("RESULT:"):
                json_line = line[len("RESULT:") :]
        if json_line:
            return json.loads(json_line)
        stderr_tail = "\n".join(proc.stderr.strip().split("\n")[-5:])
        return {
            "onednn": None,
            "best_triton": None,
            "error": f"no RESULT output (exit={proc.returncode}); stderr_tail={stderr_tail}",
        }
    except subprocess.TimeoutExpired:
        return {"onednn": None, "best_triton": None, "error": f"timeout({timeout}s)"}
    except Exception as e:
        return {"onednn": None, "best_triton": None, "error": str(e)}


def run_all(output_file, timeout):
    """Fair A/B: for each shape, run A then B back-to-back in subprocesses."""
    shapes = load_shapes()
    print(f"BF16 GEMM A/B Compare (state A: triton_mm only / state B: deployed v6 3 templates + 10 configs)")
    print(f"Total shapes: {len(shapes)}")
    print()

    results = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} shapes already done")

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = "/home/sdp/meng/pytorch"

    for i, (m, n, k) in enumerate(shapes):
        key = f"{m},{n},{k}"
        if key in results:
            continue

        print(f"[{i + 1}/{len(shapes)}] M={m}, N={n}, K={k}", end=" ", flush=True)

        ra = run_one_shape(m, n, k, "A", base_env, timeout)
        rb = run_one_shape(m, n, k, "B", base_env, timeout)
        results[key] = {"A": ra, "B": rb}

        at = ra.get("best_triton")
        bt = rb.get("best_triton")
        od = ra.get("onednn") or rb.get("onednn")
        if ra.get("decomposed") or rb.get("decomposed"):
            print(
                f"decomposed (mm->elementwise+reduction, no MM autotune choices): "
                f"A={ra.get('elapsed_ms')} B={rb.get('elapsed_ms')}"
            )
        elif at and bt and od:
            sp = at / bt
            print(
                f"A_triton={at:.4f}({ra.get('num_choices', '?')}cfg) "
                f"B_triton={bt:.4f}({rb.get('num_choices', '?')}cfg) "
                f"speedup={sp:.3f}x oneDNN={od:.4f}"
            )
        else:
            print(f"partial: A={ra} B={rb}")

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone! {len(results)} shapes -> {output_file}")


def show(path):
    with open(path) as f:
        data = json.load(f)

    print(
        f"{'#':>3} {'M':>5} {'N':>6} {'K':>6} | {'A_Triton':>9} {'B_Triton':>9} "
        f"{'T-speedup':>9} | {'oneDNN':>8} | {'best(A)':>9} {'best(B)':>9} "
        f"{'Sys-speedup':>11} | {'Acfg':>4} {'Bcfg':>4}"
    )
    print("-" * 130)

    triton_speedups = []
    system_speedups = []
    b_faster = same = a_faster = 0
    triton_win_a = triton_win_b = 0
    decomposed_count = 0

    idx = 0
    for key in sorted(data.keys(), key=lambda k: tuple(int(x) for x in k.split(","))):
        idx += 1
        m, n, k = key.split(",")
        a = data[key].get("A", {})
        b = data[key].get("B", {})
        at = a.get("best_triton")
        bt = b.get("best_triton")
        od = a.get("onednn") or b.get("onednn")
        ac = a.get("num_choices", "?")
        bc = b.get("num_choices", "?")

        if a.get("decomposed") or b.get("decomposed"):
            decomposed_count += 1
            et = a.get("elapsed_ms") or b.get("elapsed_ms")
            note = "decomposed(mm->elemwise+reduce, no MM autotune choices)"
            print(
                f"{idx:>3} {m:>5} {n:>6} {k:>6} | {note:>52} | "
                f"{et:>8.4f} ms" if et else f"{idx:>3} {m:>5} {n:>6} {k:>6} | {note}"
            )
            continue

        if not at or not bt or not od:
            print(
                f"{idx:>3} {m:>5} {n:>6} {k:>6} | {'N/A':>9} {'N/A':>9} {'N/A':>9} | "
                f"{'N/A':>8} | {'N/A':>9} {'N/A':>9} {'N/A':>11} | {str(ac):>4} {str(bc):>4}"
            )
            continue

        t_sp = at / bt  # Triton-only speedup, A -> B
        best_a = min(at, od)
        best_b = min(bt, od)
        sys_sp = best_a / best_b  # system-level speedup, A -> B

        triton_speedups.append(t_sp)
        system_speedups.append(sys_sp)

        if at < od:
            triton_win_a += 1
        if bt < od:
            triton_win_b += 1
        if sys_sp > 1.02:
            b_faster += 1
        elif sys_sp < 0.98:
            a_faster += 1
        else:
            same += 1

        mark = " <<" if sys_sp > 1.05 else (" >>" if sys_sp < 0.95 else "")
        print(
            f"{idx:>3} {m:>5} {n:>6} {k:>6} | {at:>9.4f} {bt:>9.4f} {t_sp:>8.3f}x | "
            f"{od:>8.4f} | {best_a:>9.4f} {best_b:>9.4f} {sys_sp:>10.3f}x | "
            f"{str(ac):>4} {str(bc):>4}{mark}"
        )

    def geomean(xs):
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 1.0

    print()
    print("=" * 80)
    print(f"Total: {len(triton_speedups)} shapes with complete data")
    if decomposed_count:
        print(
            f"Decomposed (mm->elementwise+reduction, no MM autotune choices, excluded from speedup): "
            f"{decomposed_count}"
        )
    print(
        f"System-level (min(Triton,oneDNN)) B faster (>2%): {b_faster}   "
        f"A faster (>2%): {a_faster}   Same (+/-2%): {same}"
    )
    print(f"Triton beats oneDNN: state A = {triton_win_a}   state B = {triton_win_b}")
    print(f"Geometric mean speedup, Triton only   (A -> B): {geomean(triton_speedups):.4f}x")
    print(f"Geometric mean speedup, system-level  (A -> B): {geomean(system_speedups):.4f}x")
    print("  (system-level = min(best_triton, oneDNN) per state; this reflects what")
    print("   the real autotune-driven dispatch would actually pick and run)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true", help="run fair A/B across all shapes")
    p.add_argument("--out", type=str, default=DEFAULT_OUTPUT, help="output JSON file")
    p.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT, help="per-shape subprocess timeout (s)"
    )
    p.add_argument("--show", type=str, help="print report from saved results JSON")
    p.add_argument("--worker", nargs=3, type=int, metavar=("M", "N", "K"))
    p.add_argument(
        "--probe",
        nargs=3,
        type=int,
        metavar=("M", "N", "K"),
        help="benchmark one shape and print selected-fn elapsed_ms plus all autotune timings",
    )
    p.add_argument("--mode", choices=("A", "B"), default="B", help="state for --probe")
    p.add_argument(
        "--manual-iters",
        type=int,
        default=1000,
        help="amortized iterations for --probe elapsed_ms (default: 1000)",
    )
    args = p.parse_args()

    if args.probe:
        os.environ.update(mode_env(args.mode))
        worker_main(*args.probe, manual_iters=args.manual_iters)
    elif args.worker:
        m, n, k = args.worker
        worker_main(m, n, k)
    elif args.show:
        show(args.show)
    elif args.run:
        run_all(args.out, args.timeout)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
