# Triton GEMM Dispatch Table Auto-Tune

## 问题定义

**目标**: 从大搜索空间中选出 ≤15 个 Triton kernel configs，使其在 118 个 LLM GEMM shapes 上的性能相对 oneDNN baseline 最优。

### 形式化

给定：
- **S** = {s₁, s₂, ..., s₁₁₈} — 118 个 GEMM shapes (M, N, K)
- **T** = {triton_mm, bmg_persistent, bmg_decode} — 3 个 Triton templates
- **D** = BLOCK_M × BLOCK_N × BLOCK_K × num_stages × num_warps — 5 维搜索空间
- **baseline(s)** = oneDNN amortized time on shape s

求解：
- **C** = 一个 config 集合, |C| ≤ 15
- 每个 config c ∈ C 是 (template, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)

优化目标：
```
maximize  gmean( ratio(s) for all s ∈ S )
where     ratio(s) = baseline(s) / min{ time(s, c) | c ∈ C }
subject to:
  min{ ratio(s) } ≥ 0.95        (最差 shape 不低于 oneDNN 95%)
  |C| ≤ 15                       (config 数量上限)
```

### 搜索空间

| 维度 | 候选值 | 数量 |
|------|--------|------|
| BLOCK_M | 4, 8, 16, 32, 64, 128, 256 | 7 |
| BLOCK_N | 32, 64, 128, 256, 512 | 5 |
| BLOCK_K | 32, 64, 128, 256 | 4 |
| num_stages | 1, 2, 3, 4 | 4 |
| num_warps | 4, 8, 16, 32 | 4 |

原始组合: 7×5×4×4×4 = **2240** per template, 3 templates = **6720** total

约束裁剪后 (~40% valid): **~2700** valid configs total

### 度量定义

```
ratio(s) = onednn_time(s) / our_best_time(s)
  > 1.0: 我们快
  = 1.0: 持平
  < 1.0: oneDNN 快

gmean  = geometric_mean(all ratios)        # 整体水平
tail   = mean(worst 10% ratios)            # 最差一批的平均
worst  = min(all ratios)                   # 单点最差
```

### 约束

| 约束 | 值 | 理由 |
|------|-----|------|
| |C| ≤ 15 | 15 个 config | torch.compile autotune ≈ 15×3s = 45s/shape |
| worst ratio ≥ 0.95 | 收敛条件 | 最差 shape 也不能明显慢于 oneDNN |
| amortized N=200 | 测量方式 | 排除 event overhead，和 oneDNN 公平对比 |

### 输入文件

| 文件 | 内容 |
|------|------|
| `shapes.csv` | 118 shapes + family + source_pattern |
| `onednn_baseline.json` | 每 shape 的 oneDNN amortized time_us |
| `search_space.py` | 候选值定义 + validity checker |

### 输出文件

| 文件 | 内容 |
|------|------|
| `dispatch_table.json` | 最终 ≤15 个 config |
| `per_shape_report.csv` | 每 shape: winner config, time, ratio, efficiency |
| `iteration_log.json` | 每轮: worst shape, candidate, promotion result |
| `heuristic_configs.py` | 可直接粘贴到 PyTorch Inductor 的 GemmConfig 列表 |

---

## 流程

### Step 0: Measure oneDNN Baseline

```
输入: 118 shapes
方法: amortized N=500 event timing
输出: onednn_baseline.json
     {shape_key: {time_us, tops, bw_gbs}}
```

已完成 (phase1_amortized_118.json 中的 onednn_ms 字段)。

### Step 1: Full Search on Seed Shape

```
选择: 1 个代表性 compute-bound shape (如 2048×4096×4096)
     + 1 个代表性 memory-bound shape (如 4×28672×4096)
方法: 对每个 seed shape, 跑 3 templates × ~500 valid configs
     利用 Triton cache: 第一个 shape 需编译, 后续只需 benchmark
输出: 初始 dispatch_table (top-3 winners across both seeds)
```

### Step 2: Sweep All Shapes

```
输入: 当前 dispatch_table (K configs)
方法: 对 118 shapes × K configs 做 amortized benchmark
     (利用 Triton cache, 每 config 只编译一次)
输出: 
  - per_shape_results[shape] = {config: time_us}
  - per_shape_winner[shape] = best config
  - ratios[shape] = baseline / winner_time
  - gmean, tail, worst
```

### Step 3: Zero-Win Prune

```
对 dispatch_table 中每个 config:
  win_count = |{s | winner(s) == config}|
  if win_count == 0:
    remove from dispatch_table

输出: 更新后的 dispatch_table (可能变小)
```

### Step 4: Convergence Check

```
if worst_ratio >= 0.95:        → 结束 (达标)
if |dispatch_table| >= 15:     → 结束 (满额)
if round >= 20:                → 结束 (超时)
if 连续 3 轮无 promotion:      → 结束 (饱和)
```

### Step 5: Worst-First Select

```
worst_shape = argmin{ ratio(s) | s ∈ S }
记录: 该 shape 已被搜索过 (避免重复)
```

### Step 6: Search on Worst Shape

```
输入: worst_shape
方法: 对 3 templates × all valid configs 做 amortized benchmark
     (大部分 config 已 cached, 只需 re-benchmark on new shape)
输出: top-5 candidate configs (按 time 排序)
```

### Step 7: Promotion Gate

对每个 candidate:

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
  local_gmean_gain >= 3% (同 family shapes)
  improved_count >= 2
  max_regression <= 5%
  gmean_gain >= -0.5%
  → PROMOTE as specialist

Neither:
  → REJECT
```

### Step 8: Loop

```
→ 回到 Step 2
```

---

## Resume 机制

每个 step 的中间结果保存到文件:

```
state/
├── onednn_baseline.json        # Step 0 输出
├── search_cache/               # 每 shape 的全量搜索结果 (可 resume)
│   ├── search_4_28672_4096.json
│   ├── search_2048_4096_4096.json
│   └── ...
├── dispatch_table.json         # 当前 dispatch table
├── sweep_results.json          # 最近一次 sweep 的完整结果
├── iteration_log.json          # 历史轮次记录
└── round_N/                    # 每轮快照
    ├── dispatch_table.json
    ├── sweep_results.json
    └── promotion_eval.json
```

脚本启动时检查 state/ 目录，自动 resume 到上次断点。

---

## 时间估算

| 步骤 | 时间 |
|------|------|
| Step 0 (baseline) | 已完成 |
| Step 1 (seed search, 2 shapes) | ~35 min (第一次编译) + ~5 min (第二个 shape, cached) |
| Step 2 (sweep 118 shapes × 3 configs) | ~15 min |
| Step 6 (search on 1 shape, cached) | ~5 min |
| 每轮 (Step 2-7) | ~20 min |
| 预计 10-15 轮收敛 | **~4-5 小时总计** |

利用 Triton cache 后，除第一次 seed search 外，后续都只是 benchmark (无编译)。
