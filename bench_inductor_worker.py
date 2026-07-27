"""Inductor candidate-timing backend for the BF16 autotune controller.

The controller still owns search, resume/cache, sweep, promotion, and dispatch
table construction.  This module only replaces the old external Triton timing
function.  Each candidate runs in a fresh process through
``bf16_single_config_bench.py`` so Inductor's heuristic singletons, Triton
compilation state, and monkeypatches cannot leak between candidates.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
import fcntl
import math
import os
import subprocess
import sys
from pathlib import Path

from search_space import is_valid_for_template


SCRIPT = Path(__file__).with_name("bf16_single_config_bench.py")
ONEDNN_SCRIPT = Path(__file__).with_name("bench_inductor_onednn.py")
LOCK_FILE = Path(
    os.environ.get(
        "XE2_INDUCTOR_BENCH_LOCK",
        "/tmp/xe2_bf16_inductor_bench.lock",
    )
)
TEMPLATE_ALIASES = {
    "triton_mm": "triton_mm",
    "bmg_persistent": "bmg_persistent",
    # Keep the historical dispatch-table key stable while using the real
    # Inductor template name underneath.
    "bmg_decode": "bmg_tiled2d",
    "bmg_tiled2d": "bmg_tiled2d",
}


@contextmanager
def _gpu_benchmark_lock():
    """Serialize all Inductor GPU benchmark subprocesses on this host.

    ``subprocess.run`` already serializes calls made by one autotune parent,
    but it does not prevent two independently launched autotune jobs from
    benchmarking the same XPU concurrently.  ``flock`` is released by the
    kernel automatically if the owning process exits or is killed.
    """
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _parse_result(stdout: str):
    """Return the last RESULT payload emitted by the worker, if any."""
    result = None
    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            result = json.loads(line[len("RESULT:") :])
    return result


def _worker_env(num_iters: int) -> dict[str, str]:
    env = os.environ.copy()
    pytorch_root = "/home/sdp/meng/pytorch"
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        pytorch_root if not old_pythonpath
        else os.pathsep.join((pytorch_root, old_pythonpath))
    )
    # State B is required for the two BMG floating-point templates.  Explicit
    # values supplied by the caller are retained.
    env.setdefault("XE2_ENABLE_BMG_FLOAT_TEMPLATES", "1")
    env.setdefault("XE2_MM_TUNED_CONFIGS", "1")
    # Inductor's TritonBenchmarkRequest reads these during import.  The outer
    # tune controller's iteration count therefore controls candidate timing,
    # instead of being ignored as it was by the original proof-of-concept.
    env["TORCHINDUCTOR_DEFAULT_AUTOTUNE_REP"] = str(max(1, int(num_iters)))
    env.setdefault("TORCHINDUCTOR_DEFAULT_AUTOTUNE_WARMUP", "50")
    return env


def bench_one_template(M, N, K, config, template, num_iters=200, dtype="bf16"):
    """Benchmark one template/config using Inductor candidate timing.

    Returns microseconds, matching the legacy ``bench_worker`` API.  A failed
    compilation, a decomposed M=1 path, or a missing candidate returns None so
    the existing search/cache logic can mark that choice as failed.
    """
    if dtype != "bf16":
        raise ValueError(
            "bench_inductor_worker only supports dtype='bf16'; "
            "use bench_worker for other dtypes"
        )
    if not is_valid_for_template(M, N, K, config, template):
        return None

    real_template = TEMPLATE_ALIASES.get(template)
    if real_template is None:
        raise ValueError(f"unknown template: {template}")

    command = [
        sys.executable,
        "-u",
        str(SCRIPT),
        str(M),
        str(N),
        str(K),
        "--template",
        real_template,
        "--config",
        ",".join(str(value) for value in config.key),
    ]
    timeout = float(os.environ.get("XE2_INDUCTOR_BENCH_TIMEOUT", "240"))
    try:
        # Hold the host-wide lock for compilation and candidate timing, not
        # only for the final measurement.  Compilation can also consume GPU
        # resources and should not overlap another benchmark job.
        with _gpu_benchmark_lock():
            process = subprocess.run(
                command,
                cwd=str(SCRIPT.parent),
                env=_worker_env(num_iters),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        _debug_failure(template, M, N, K, config, f"timeout after {timeout:g}s")
        return None
    except OSError as exc:
        _debug_failure(template, M, N, K, config, str(exc))
        return None

    result = _parse_result(process.stdout)
    if process.returncode != 0 or result is None:
        detail = "no RESULT output"
        if result and result.get("error"):
            detail = result["error"]
        elif process.stderr.strip():
            detail = process.stderr.strip().splitlines()[-1]
        _debug_failure(template, M, N, K, config, detail)
        return None

    timing_ms = result.get("target_timing_ms")
    if timing_ms is None or not math.isfinite(float(timing_ms)) or timing_ms <= 0:
        _debug_failure(template, M, N, K, config, f"invalid timing: {timing_ms!r}")
        return None

    if os.environ.get("XE2_INDUCTOR_BENCH_VERBOSE") == "1":
        print(
            f"INDUCTOR candidate template={template} real={real_template} "
            f"shape=({M},{N},{K}) config={config} "
            f"candidate={result.get('target_name')} time={timing_ms * 1000:.3f}us",
            flush=True,
        )
    return float(timing_ms) * 1000.0


def bench_config_all_templates(M, N, K, config, num_iters=200, dtype="bf16"):
    """Benchmark all logical templates and return the fastest pair."""
    best_time = None
    best_template = None
    for template in ("triton_mm", "bmg_persistent", "bmg_decode"):
        time_us = bench_one_template(
            M, N, K, config, template, num_iters=num_iters, dtype=dtype
        )
        if time_us is not None and (best_time is None or time_us < best_time):
            best_time = time_us
            best_template = template
    return best_time, best_template


def bench_inductor_onednn(M, N, K, num_iters=200, dtype="bf16"):
    """Return the ATen/oneDNN candidate timing from Inductor's benchmark path."""
    if dtype != "bf16":
        raise ValueError("bench_inductor_onednn currently supports dtype='bf16' only")

    command = [
        sys.executable,
        "-u",
        str(ONEDNN_SCRIPT),
        str(M),
        str(N),
        str(K),
    ]
    timeout = float(os.environ.get("XE2_INDUCTOR_BENCH_TIMEOUT", "240"))
    try:
        with _gpu_benchmark_lock():
            process = subprocess.run(
                command,
                cwd=str(ONEDNN_SCRIPT.parent),
                env=_worker_env(num_iters),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
    except (subprocess.TimeoutExpired, OSError) as exc:
        _debug_failure("onednn", M, N, K, "inductor-candidate", str(exc))
        return None

    result = _parse_result(process.stdout)
    if process.returncode != 0 or result is None:
        detail = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "no RESULT output"
        _debug_failure("onednn", M, N, K, "inductor-candidate", detail)
        return None

    timing_ms = result.get("onednn_timing_ms")
    if timing_ms is None or not math.isfinite(float(timing_ms)) or timing_ms <= 0:
        _debug_failure("onednn", M, N, K, "inductor-candidate", f"invalid timing: {timing_ms!r}")
        return None
    return float(timing_ms) * 1000.0


def _debug_failure(template, M, N, K, config, detail):
    if os.environ.get("XE2_BENCH_DEBUG_ERRORS") == "1":
        print(
            f"INDUCTOR BENCH ERROR template={template} shape=({M},{N},{K}) "
            f"config={config}: {detail}",
            file=sys.stderr,
            flush=True,
        )