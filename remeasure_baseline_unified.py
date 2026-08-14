#!/usr/bin/env python3
"""Re-measure the oneDNN baseline through the same Inductor harness as the search.

`state_bf16_v6/onednn_baseline.json` mixes two timing methods: some entries were
recorded with a raw cache-hot event loop, while every Triton candidate in
`search_cache/` is timed by Inductor's `do_bench`, which flushes the cache.  For
cache-sensitive shapes the two differ by >25% -- (256,4096,4096) reads 116.5 us
one way and 147.2 us the other -- so any ratio built from them is meaningless.

This script rewrites the baseline using `bench_inductor_onednn.py`, i.e. exactly
the path `bench_one_template` uses for the Triton side.  The previous file is
kept as `onednn_baseline.pre_unified.json`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state_bf16_v6"
BASELINE = STATE / "onednn_baseline.json"
BACKUP = STATE / "onednn_baseline.pre_unified.json"
PY = sys.executable


def measure(M: int, N: int, K: int, repeats: int) -> float | None:
    """Return the best oneDNN timing in us, or None if Inductor never autotunes."""
    values = []
    for _ in range(repeats):
        proc = subprocess.run(
            [PY, "-u", "bench_inductor_onednn.py", str(M), str(N), str(K)],
            cwd=ROOT, capture_output=True, text=True,
        )
        line = next(
            (l for l in proc.stdout.splitlines() if l.startswith("RESULT:")), None
        )
        if line is None:
            # M=1 and similar shapes are decomposed into elementwise/reduction
            # kernels and never reach MM template autotuning.
            return None
        values.append(json.loads(line[len("RESULT:"):])["onednn_timing_ms"] * 1000.0)
    return min(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes", default="cache",
        help="'cache' for every shape in search_cache/, or M,N,K;M,N,K;...",
    )
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    if args.shapes == "cache":
        shapes = sorted(
            tuple(int(x) for x in p.stem.split("_")[1:])
            for p in (STATE / "search_cache").glob("search_[0-9]*.json")
        )
    else:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]

    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    if BASELINE.exists() and not BACKUP.exists():
        shutil.copy(BASELINE, BACKUP)

    print(f"{'shape':22s} {'old us':>10} {'unified us':>11} {'delta':>8}")
    for M, N, K in shapes:
        key = f"{M},{N},{K}"
        old = baseline.get(key, {}).get("time_us")
        new = measure(M, N, K, args.repeats)
        if new is None:
            print(f"{key:22s} {old if old else '-':>10} {'skipped':>11} {'':>8}")
            sys.stdout.flush()
            continue
        baseline[key] = {"time_us": round(new, 2), "method": "inductor_do_bench"}
        BASELINE.write_text(json.dumps(baseline, indent=2))
        delta = f"{new / old:.3f}x" if old else "-"
        print(f"{key:22s} {old if old else '-':>10} {new:11.2f} {delta:>8}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
