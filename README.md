# Triton GEMM Dispatch Table Auto-Tune

## Problem Definition

**Objective**: Select ≤15 Triton kernel configs from a large search space to achieve optimal performance relative to the oneDNN baseline across the active LLM GEMM shapes (101 after excluding M=1 decomposed cases).

### Formalization

Given:
- **S** = {s₁, s₂, ..., s₁₁₈} — 118 GEMM shapes (M, N, K)
- **T** = {triton_mm, bmg_persistent, bmg_decode} — 3 Triton templates
- **D** = BLOCK_M × BLOCK_N × BLOCK_K × num_stages × num_warps — 5D search space
- **baseline(s)** = oneDNN amortized time on shape s

Find:
- **C** = a config set, |C| ≤ 15
- Each config c ∈ C is (template, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)

Optimization Objective:
```
maximize  gmean( ratio(s) for all s ∈ S )
where     ratio(s) = baseline(s) / min{ time(s, c) | c ∈ C }
subject to:
  min{ ratio(s) } ≥ 0.95        (worst shape no worse than 95% of oneDNN)
  |C| ≤ 15                       (config count limit)
```

### Search Space

| Dimension | Candidate Values | Count |
|-----------|------------------|-------|
| BLOCK_M | 4, 8, 16, 32, 64, 128, 256 | 7 |
| BLOCK_N | 32, 64, 128, 256, 512 | 5 |
| BLOCK_K | 32, 64, 128, 256 | 4 |
| num_stages | 1, 2, 3, 4 | 4 |
| num_warps | 4, 8, 16, 32 | 4 |

Raw combinations: 7×5×4×4×4 = **2240** per template, 3 templates = **6720** total

After constraint pruning (~40% valid): **~2700** valid configs total

### Metric Definitions

```
ratio(s) = onednn_time(s) / our_best_time(s)
  > 1.0: we are faster
  = 1.0: tied
  < 1.0: oneDNN is faster

gmean  = geometric_mean(all ratios)        # overall level
tail   = mean(worst 10% ratios)            # average of worst batch
worst  = min(all ratios)                   # single worst case
```

### Constraints

| Constraint | Value | Reason |
|------------|-------|--------|
| |C| ≤ 15 | 15 configs | torch.compile autotune ≈ 15×3s = 45s/shape |
| worst ratio ≥ 0.95 | convergence condition | worst shape must not be significantly slower than oneDNN |
| amortized N=200 | measurement method | exclude event overhead, fair comparison with oneDNN |

### Input Files

| File | Content |
|------|---------|
| `shapes.csv` | 118 shapes + family + source_pattern |
| `onednn_baseline*.json` | oneDNN time_us per shape; BF16 Inductor-timed data uses `onednn_baseline_inductor.json` |
| `search_space.py` | candidate value definitions + validity checker |

### Output Files

| File | Content |
|------|---------|
| `dispatch_table.json` | final ≤15 configs |
| `per_shape_report.csv` | per shape: winner config, time, ratio, efficiency |
| `iteration_log.json` | per round: worst shape, candidate, promotion result |
| `heuristic_configs.py` | GemmConfig list for PyTorch Inductor (paste-ready) |

---

## Workflow

### Step 0: Measure oneDNN Baseline

```
Input: active shapes (101 for BF16)
Method: Inductor candidate timing, N=500 repetitions
Output: `onednn_baseline_inductor.json` for BF16 (`onednn_baseline.json` for legacy INT8)
     {shape_key: {time_us, tops, bw_gbs}}
```

Completed (onednn_ms field in phase1_amortized_118.json).

### Step 1: Full Search on Seed Shapes

```
Selection: 1 representative compute-bound shape (e.g. 2048×4096×4096)
         + 1 representative memory-bound shape (e.g. 4×28672×4096)
Method: For each seed shape, run 3 templates × ~500 valid configs
        Use Triton cache: first shape requires compilation, subsequent only need benchmark
Output: Initial dispatch_table (top-3 winners across both seeds)
```

### Step 2: Sweep All Shapes

```
Input: Current dispatch_table (K configs)
Method: Amortized benchmark on 118 shapes × K configs
        (using Triton cache, each config compiled only once)
Output: 
  - per_shape_results[shape] = {config: time_us}
  - per_shape_winner[shape] = best config
  - ratios[shape] = baseline / winner_time
  - gmean, tail, worst
```

### Step 3: Zero-Win Prune

```
For each config in dispatch_table:
  win_count = |{s | winner(s) == config}|
  if win_count == 0:
    remove from dispatch_table

Output: Updated dispatch_table (may be smaller)
```

### Step 4: Convergence Check

```
if worst_ratio >= 0.95:        → done (converged)
if |dispatch_table| >= 15:     → done (full)
if round >= 20:                → done (timeout)
if 3 consecutive rounds no promotion: → done (saturated)
```

### Step 5: Worst-First Select

```
worst_shape = argmin{ ratio(s) | s ∈ S }
Record: This shape has been searched (avoid duplicates)
```

### Step 6: Search on Worst Shape

```
Input: worst_shape
Method: Amortized benchmark on 3 templates × all valid configs
        (most configs already cached, only need re-benchmark on new shape)
Output: top-5 candidate configs (sorted by time)
```

### Step 7: Promotion Gate

For each candidate:

```
trial_table = dispatch_table + [candidate]
sweep all shapes with trial_table
compute new gmean, tail, worst

Core Gate:
  target_gain >= 5%
  gmean_gain >= 0.5%
  tail_gain >= 1%
  max_regression <= 3%
  → PROMOTE as core

Specialist Gate:
  target_gain >= 10%
  local_gmean_gain >= 3% (same family shapes)
  improved_count >= 2
  max_regression <= 5%
  gmean_gain >= -0.5%
  → PROMOTE as specialist

Neither:
  → REJECT
```

### Step 8: Loop

```
→ Back to Step 2
```

---

## Resume Mechanism

Intermediate results from each step are saved to files:

```
state/
├── onednn_baseline.json        # Step 0 output
├── search_cache/               # Full search results per shape (resumable)
│   ├── search_4_28672_4096.json
│   ├── search_2048_4096_4096.json
│   └── ...
├── dispatch_table.json         # Current dispatch table
├── sweep_results.json          # Most recent sweep results
├── iteration_log.json          # Historical round records
└── round_N/                    # Snapshot per round
    ├── dispatch_table.json
    ├── sweep_results.json
    └── promotion_eval.json
```

Scripts check state/ directory on startup and automatically resume from last checkpoint.

---

## Time Estimation

| Step | Time |
|------|------|
| Step 0 (baseline) | Completed |
| Step 1 (seed search, 2 shapes) | ~35 min (first compilation) + ~5 min (second shape, cached) |
| Step 2 (sweep 118 shapes × 3 configs) | ~15 min |
| Step 6 (search on 1 shape, cached) | ~5 min |
| Per round (Step 2-7) | ~20 min |
| Expected 10-15 rounds to converge | **~4-5 hours total** |

After using Triton cache, except for the first seed search, subsequent runs are only benchmarks (no compilation).

---

## BF16 Inductor Autotune (Current Workflow)

BF16 tuning uses the original `run_autotune.py` control flow, but its Triton
candidate benchmark is routed through the real Inductor XPU templates:

```text
run_autotune.py
  -> bench_worker.bench_one_template()
  -> bench_inductor_worker.bench_one_template()
  -> bf16_single_config_bench.py (fresh subprocess)
  -> AlgorithmSelector candidate timing
```

The logical template names are:

| Dispatch-table name | Inductor template |
|---|---|
| `triton_mm` | `mm_template` |
| `bmg_persistent` | `bmg_persistent_mm_template` |
| `bmg_decode` | `bmg_tiled2d_mm_template` |

### Environment setup

From the workspace root:

```bash
source env.sh
cd /home/sdp/meng/int8_gemm_optimization_xe2/triton_dispatch_autotune
export PYTHONPATH=/home/sdp/meng/pytorch:$PWD
export XE2_ENABLE_BMG_FLOAT_TEMPLATES=1
export XE2_MM_TUNED_CONFIGS=1
```

`env.sh` selects the `chunhuan` environment and BMG/Xe2 target. If the
environment is already configured, only the `PYTHONPATH` and State B exports
are required.

### Recommended execution

Run the complete resumable BF16 workflow:

```bash
python run_autotune.py --dtype bf16 --step all
```

Or run individual stages:

```bash
python run_autotune.py --dtype bf16 --step baseline
python run_autotune.py --dtype bf16 --step seed
python run_autotune.py --dtype bf16 --step iterate
python run_autotune.py --dtype bf16 --step report
```

The stages are resumable. BF16 state is stored separately under
`state_bf16_v7/`, so it does not reuse INT8 timings or the previous BF16 v6
experiment.

### Timing policy

| Stage | Inductor warmup | Inductor repetitions |
|---|---:|---:|
| baseline (`torch.mm`) | 50 | 500 Inductor candidate-timer repetitions |
| full search / seed | 50 | 50 |
| sweep | 50 | 500 |
| promotion | 50 | 200 |

The BF16 baseline measures the non-Triton `mm` choice through the same Inductor
AlgorithmSelector candidate-timing path used for Triton candidates. A single
Triton anchor choice is retained to ensure that Inductor executes the normal
choice benchmark flow; only the captured non-Triton `mm` timing is returned.
The resulting data is written to `onednn_baseline_inductor.json`; the previous
`onednn_baseline.json` remains the legacy XPU-Event baseline. The iteration
counts above are controlled by `run_autotune.py` and are passed to Inductor as
`TORCHINDUCTOR_DEFAULT_AUTOTUNE_REP`.

### Search-space mode

The default mode is the original pruned generic search space:

```bash
python run_autotune.py --dtype bf16 --step seed
```

To restrict candidates to configs currently registered in the corresponding
Inductor heuristics:

```bash
XE2_AUTOTUNE_SEARCH_SPACE=exact python run_autotune.py --dtype bf16 --step seed
```

### Single-candidate validation

To validate one template/config without starting the full controller:

```bash
XE2_PARITY_TEMPLATE=bmg_persistent \
python run_inductor_fixed_config.py \
  --shape 128,1536,2048 \
  --config 128,128,32,4,32 \
  --dtype bf16 \
  --timer candidate
```

The historical `bmg_decode` name maps to Inductor's real `bmg_tiled2d`
template. `bf16_single_config_bench.py` is the worker used by the controller;
it is normally called indirectly through `bench_inductor_worker.py`.

### GPU serialization

Every Inductor candidate runs in a fresh subprocess, but subprocesses are
serialized by the host-wide Linux `flock` file:

```text
/tmp/xe2_bf16_inductor_bench.lock
```

This prevents two independently launched autotune jobs using this backend from
benchmarking the same XPU concurrently. To use another lock location:

```bash
export XE2_INDUCTOR_BENCH_LOCK=/tmp/xe2_bf16_xpu0.lock
```

Do not use different lock paths for jobs that target the same XPU.

### Debugging and backend override

To print failed worker details:

```bash
export XE2_BENCH_DEBUG_ERRORS=1
export XE2_INDUCTOR_BENCH_VERBOSE=1
```

The BF16 Inductor backend is the default. The old external Triton backend can
be selected temporarily with:

```bash
export XE2_BF16_BENCH_BACKEND=legacy
```

Use the legacy override only for comparison; it does not measure the real
Inductor candidate timing.
