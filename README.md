# Triton GEMM Dispatch Table Auto-Tune

## Problem Definition

**Objective**: Select ≤15 Triton kernel configs from a large search space to achieve optimal performance relative to the oneDNN baseline across 118 LLM GEMM shapes.

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
| `onednn_baseline.json` | oneDNN amortized time_us per shape |
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
Input: 118 shapes
Method: amortized N=500 event timing
Output: onednn_baseline.json
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
