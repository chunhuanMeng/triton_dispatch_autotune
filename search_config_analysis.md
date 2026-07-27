# Search Cache 配置规律深入分析

- 日期：2026-07-24
- 数据源：[search_top15_report.md](search_top15_report.md) 中的三个 cache 目录
- 目的：理解五维配置的作用，找出适用于 `is_good_config` 的、尽量不误剪枝的规则
- 本文只做分析，不修改 [search_space.py](search_space.py)

## 1. 分析方法和数据边界

我把每个 JSON 文件作为一个独立实验，不直接把不同版本的 `time_us` 混合排序：

1. 同一个文件内，相同 config 取最小 `time_us`。
2. 按 `time_us` 升序取 Top 15。
3. 分别统计 BF16 v6、BF16 v7 和旧版 `state`。
4. 对重复 shape 比较参数范围，而不是只比较绝对时间。
5. 用理论上的 tile 数量、边界浪费、寄存器/共享内存压力解释观测结果。

实际数据是：BF16 v7 有 8 个文件，BF16 v6 有 15 个文件，旧版 `state` 有 7 个文件，共 30 个文件、22 个不同 shape。

重要数据质量结论：

- BF16 v7 的 `(1,1024,4096)` 文件全部是 `null`，不能用于推导规则。
- BF16 v6 的 `(2048,6144,16384)` 只有 100 个有效结果，属于未完成/部分搜索，不能和 v7 的 1716 个有效结果等权。
- 旧版 `state` 是五维 cache，部分结果没有可靠 template 字段，且 timing/search 版本不同；它可以用于观察参数范围，但不应覆盖 BF16 v6/v7 的完整结果。
- 小 shape 的 Top 15 时间通常非常接近。例如很多文件的 Top15/Top1 只有 $1.01\sim1.07\times$。这意味着排名对测量噪声、编译状态和 cache 状态敏感，不能把第 15 名之外的配置直接判定为坏配置。

## 2. 五维配置分别是什么

配置是：

```text
(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
```

### 2.1 `BLOCK_M` 和 `BLOCK_N`：输出 tile 的二维形状

GEMM 为：

$$
C_{M\times N}=A_{M\times K}B_{K\times N}
$$

一个 Triton program 通常负责一个 `BLOCK_M × BLOCK_N` 的输出 tile。粗略的 program 数量是：

$$
P \approx \left\lceil\frac{M}{BLOCK_M}\right\rceil
       \left\lceil\frac{N}{BLOCK_N}\right\rceil
$$

它们的主要作用是：

- tile 越大，program 数量越少，调度/边界开销可能越低；
- tile 越大，累加器越大，寄存器压力越高；
- `BLOCK_M` 太大时，小 M 会产生大量无效行；
- `BLOCK_N` 太小时，N 方向 program 太多；
- `BLOCK_M × BLOCK_N` 太大时，必须增加 warp 数量，否则每个线程承担的 accumulator 太多。

小 M、大 N 的 LLM decode 形状具有明显的非对称性：`BLOCK_M` 应小，而 `BLOCK_N` 可以较大。这也是为什么不能简单使用 `BLOCK_M≈M` 或把 `BLOCK_N` 固定为 256。

### 2.2 `BLOCK_K`：K 方向的 reduction tile

K 方向需要大约：

$$
R_K=\left\lceil\frac{K}{BLOCK_K}\right\rceil
$$

个 reduction iteration。

- 较大的 `BLOCK_K` 减少 K-loop、同步和循环控制次数；
- 但会增加一次迭代的寄存器/共享内存占用；
- 对 K 较小或 memory-bound shape，`BLOCK_K=32/64` 通常足够；
- 对 K 很大、compute-bound shape，`BLOCK_K=64/128/256` 更常进入 Top 15。

历史数据不支持把 `BLOCK_K=256` 全局删除：它在 `(4,4096,1536)`、`(2048,6144,16384)` 和 `(2048,7168,28672)` 等 shape 的 Top 15 中出现。

### 2.3 `num_stages`：K-loop 的软件流水深度

`num_stages` 表示预取/流水线中同时在飞的 K tile 数量。它的典型 trade-off 是：

- stages 增大，可以隐藏 global memory latency；
- stages 增大，也会近似增加 shared memory 和部分寄存器占用；
- 小 M 或小工作量时，深流水线的收益通常不稳定；
- 大 compute shape 不一定需要大 stages，因为某些 Triton kernel 的最佳路径反而是 `num_stages=1`。

因此不能只按 M 设置“大的 M 必须 stages=4”。在 BF16 v7 的 `(2048,6144,16384)` 中，Top 15 全部是 `num_stages=1`；而 `(2048,4096,4096)` 更集中在 `2/3/4`。

### 2.4 `num_warps`：一个 program 使用的 warp 数量

`num_warps` 增大可以提高一个 tile 内的并行度，但会增加资源消耗和调度粒度。它与 tile 面积强耦合，不能单独看：

$$
Q=\frac{BLOCK_M\times BLOCK_N}{num\_warps}
$$

可以把 $Q$ 理解为每个 warp 需要承担的输出元素量的粗略指标。现有 `is_valid_config` 的约束实际上已经把 $Q$ 约束在一个较窄区间内：大致为 $256\le Q\le2048$。Top 15 也几乎全部落在这个区间。

观察到的趋势：

- M≤4：$Q$ 多数为 256 或 512，`num_warps` 主要是 4/8，最多到 16；
- M=32：$Q$ 通常为 256/512/1024，但某些低 K shape 会出现 2048；
- M=512：$Q$ 主要为 512/1024/2048；
- M≥1024 且为完整 BF16 compute 搜索时，Top 15 多数为 $Q=1024/2048$，`num_warps` 主要是 8/16/32。

## 3. 从 Top 15 得到的 shape 家族

下面的范围是“历史 Top 15 曾出现过的范围”，不是声称每个 shape 都必须搜索整个范围。

| Shape 家族 | `BLOCK_M` | `BLOCK_N` | `BLOCK_K` | stages | warps | 主要现象 |
|---|---:|---:|---:|---:|---:|---|
| M≤4 | 4–32 | 64–512 | 32–256 | 1–4 | 4–16 | 小 M、大 N；小 BM、大 BN；高延迟/内存约束 |
| 4<M≤32 | 8–64 | 32–512 | 32–256 | 1–4 | 4–32 | 形状差异最大，M=32、K较小时可用大 BN/NW |
| 32<M≤512 | 16–256 | 64–512 | 32–256 | 1–4 | 4–32 | BM 逐渐增大，Triton 开始在 M=512 占主导 |
| M≥1024，K约4K–7K | 64–256；旧 decode 数据可到16 | 128–512 | 32–64为主 | 2–4；旧 decode 可为1 | 8–32 | 大 tile、较高 warp；persistent/triton 竞争 |
| M≥1024，K≥16K | 64–256（完整 v7） | 128–512 | 64–256 | 完整 v7 主要1 | 8–32 | compute-bound；更大的 BK，低 stages，较大 tile |

### 3.1 M≤4：不能再使用“BM≈4、BN=256”的过窄规则

历史 Top 15 实际包含：

- `(1,1024,4096)`：BM=8/16/32，BN=64/128/256；
- `(1,3584,2560)`：BM=8/16，BN=64/128/256；
- `(4,28672,4096)`：BM=4/8/16，BN=128/256/512；
- `(4,3584,2560)`：BM=16/32，BN=32/64/256/512；
- `(4,4096,1536)`：BM=4/8/16/32，BN=64/128/256/512。

因此，可靠结论不是“BM必须为4”或“BN必须为256”，而是：

- `BM > 32` 在已有 M≤4 Top 15 中没有出现；
- `num_warps > 16` 在已有 M≤4 Top 15 中没有出现；
- `BN` 必须随 N 保留足够宽度，N很大时 256/512 更常见；
- `BK=32/64/128/256` 都可能有用，取决于 K 和具体 template；
- stages 不能收窄到单一值：不同 M=4 shape 的最佳范围分别偏向 1/2 或 2/4。

### 3.2 4<M≤32：最适合用宽范围，而不是固定单点

`(32,3584,2560)` 的 Top 15 同时出现 BM=16/32/64、BN=32/64/256/512、BK=32/64/128、warps=4/8/16/32。

`(32,32768,6144)` 则完全不同：Top 15 的 BM 全部为16，BN主要为128/256，stages主要为1，warps主要为4。

这说明 N 和 K 会改变 M 方向的最佳 tile。仅按 M 分段会误剪枝，至少要保留：

- 小 M、大 N：BM小、BN中到大；
- M=32且 K 较小：可探索较大 BN 和较大 warp；
- M=32且 N极大：优先 BM=16、BN=128/256、低 stages/低 warp，但仍保留少量 fallback。

### 3.3 32<M≤512：M=512 是一个明显转折点

`(128,1536,2048)`、`(128,2048,768/1408)` 的 Top 15 仍有很多 `bmg_decode`，BM主要是16/32。

`(512,1536,2048)` 则 Top 15 主要是 `triton_mm`，BM=64/128/256，BN=64/128/256，BK几乎以32为主，warps=4/8。

所以可以把 M=512 视为 template/资源利用的转折点，但不建议仅用 `M>512 => BM>=16` 这种规则来决定全部配置；它只能作为一个候选优先级规则。

### 3.4 M≥1024：需要按 K 再分支

完整 BF16 v7 数据显示：

- `(2048,4096,4096)`：BM=64/128/256，BN=128/256/512，BK=32/64，stages=2/3/4，warps=8/16/32；
- `(2048,6144,16384)`：BM=64/128/256，BN=128/256，BK=64/128/256，stages全部为1；
- `(2048,7168,28672)`：BM=64/128/256，BN=128/256/512，BK=64/128/256，stages主要为1；
- `(2048,32768,6144)`：BM=64/128，BN=128/256/512，BK=32/64，stages=2/3/4。

因此，`M≥1024` 不能只保留一套配置：

- K约4K–7K：BK=32/64、stages=2/3/4 更重要；
- K≥16K：BK=64/128/256、stages=1 更重要；
- N很大时 BN=512 仍然可能获胜；
- 完整 compute 搜索的每-warp tile 面积 $Q$ 基本不低于1024。

## 4. 多种剪枝假设的验证结果

我尝试了几类规则，而不是只看单个 Top1：

### 规则 A：只使用当前规则

当前规则已经能保留 BF16 v6/v7 所有有效文件的 Top 15，但它的主要问题是：

- 对 M≤512 基本没有额外剪枝；
- `BM` 的小 M 上限过宽；
- `N>4096` 时仍允许 `BN=32`；
- 对大 compute tile 没有利用 $Q$ 的规律。

### 规则 B：历史驱动的保守收紧

建议先考虑以下逻辑，仍然保守、template-agnostic：

```text
M <= 1:  BM <= 32, num_warps <= 8
M <= 4:  BM <= 32, num_warps <= 16
4 < M <= 32: BM <= 64
N > 4096: BLOCK_N >= 64
K >= 8192: BLOCK_K >= 64
其余维度保留当前 valid_config 的范围
```

在已有数据上：

- BF16 v7 Top 15 覆盖率：100%；
- BF16 v6 Top 15 覆盖率：100%；
- 旧版 `state` 的覆盖率仍为96.2%，漏掉的是旧 cache `(128,6144,16384)` 中的 BK=32 配置。这个旧 cache 与 BF16 v6/v7 的 timing/template 语义不同，不建议为了兼容它而放弃 BF16 规则；如果必须兼容所有历史，则保留 `K>=8192` 的 BK=32 fallback。

在当前 generic `generate_valid_configs()` 候选集上，规则 B 的候选数变化示例：

| Shape | 原 valid | 规则 B | 缩减 |
|---|---:|---:|---:|
| (1,1024,4096) | 736 | 368 | 50.0% |
| (4,28672,4096) | 736 | 448 | 39.1% |
| (32,2048,768) | 1104 | 736 | 33.3% |
| (128,2048,768) | 1104 | 1104 | 0% |
| (2048,4096,4096) | 1104 | 960 | 13.0% |
| (2048,6144,16384) | 1104 | 600 | 45.7% |
| (2048,7168,28672) | 1104 | 600 | 45.7% |

这里的结果很重要：只按 M/N/K 的硬阈值，无法把所有 shape 都缩小很多。M=128/512 的历史 Top15 覆盖范围本来就很宽，强行收窄会有明显误剪枝风险。

### 规则 C：增加 compute-bound 的 $Q\ge1024$

对于 $M\ge1024$ 且算术强度大于当前代码阈值的 shape，增加：

```text
BLOCK_M * BLOCK_N / num_warps >= 1024
```

完整 BF16 v7 数据的 Top 15 全部满足这个条件，因此它很有理论依据：大 compute shape 需要更大的 work/warp 才能摊薄调度和 tile 开销。

但它不能无条件应用：

- BF16 v6 的 `(2048,6144,16384)` 只有100个结果，Top15包含大量 BM=16、小 Q 配置；
- 旧版 `state` 中存在 bmg_decode 风格的 BM=16 配置；
- 这些结果更像部分搜索/旧 template 的特殊路径。

所以规则 C 更适合以后增加 `template` 参数后，只对 `triton_mm`/`bmg_persistent` 的大 compute 候选启用，而不是直接放进当前无 template 参数的 `is_good_config`。

### 规则 D：收窄 stages

尝试按 M 或 K 删除 stages：

- `M<=32 => stages<=2` 会误删多个 M=4/M=32 Top15；
- `M>=1024 => stages>=2` 会误删 BF16 v7 `(2048,6144,16384)` 的全部 Top15；
- `K>=8192 => stages=1` 会误删 persistent/template fallback。

结论：在 template-agnostic 的 `is_good_config` 中，不建议继续剪 `num_stages`。它更适合由 template-specific heuristic 处理。

## 5. 推荐的实际规则层次

### 第一层：放进当前 `is_good_config` 的安全规则

这些规则在 BF16 v6/v7 Top15 上有完整覆盖，并且有明确形状依据：

```text
1. M <= 1 且 BM > 32：剪掉
2. M <= 1 且 num_warps > 8：剪掉
3. M <= 4 且 BM > 32：剪掉
4. M <= 4 且 num_warps > 16：剪掉
5. 4 < M <= 32 且 BM > 64：剪掉
6. N > 4096 且 BN < 64：剪掉
7. K >= 8192 且 BK < 64：保留当前已有规则
```

`BLOCK_N` 的上限仍交给 `is_valid_config` 的 `bn > N` 判断；不要把 BN 固定为256。

`BLOCK_K` 只建议使用 K 的下限规则，不建议删除256；历史上大 K shape 的 BK=256 有实际 Top15 记录。

### 第二层：template-aware 的优先级规则

如果未来允许把 template 传给判断函数，建议用“优先级/候选排序”，而不是立即删除：

```text
M <= 128 且 memory-bound:
    优先 bmg_decode；BM 16/32、BN 64/128/256/512、BK 32/64/128

M >= 512 且 compute-bound:
    优先 triton_mm；BM 64/128/256、BN 64/128/256/512

M >= 1024 且大 K:
    保留 triton_mm + bmg_persistent；BM 64/128/256、BK 64/128/256
```

旧 cache 表明 `bmg_decode` 在大 M 上也可能测到较好结果，因此不建议第一轮直接删除该 template；可以先减少其配置数量或降低优先级。

### 第三层：基于 $Q$ 的 compute-bound 规则

在确认 cache 是完整搜索、并且 template 已知后，可以尝试：

```text
if M >= 1024 and arithmetic_intensity > 513:
    require 1024 <= BLOCK_M * BLOCK_N / num_warps <= 2048
```

上限2048与当前寄存器压力约束一致，下限1024来自完整 BF16 v7 的 Top15 分布和 tile 摊销理论。

## 6. 最终结论

最稳定的普适规律不是某一个固定配置，而是三条结构性规律：

1. **小 M 采用窄 BM、较宽 BN**：小 M 的 BM 上限可以安全收紧，但 BN 不能固定为256。
2. **K 越大，BK 的下限越应提高**：K≥8192 时优先 BK=64/128/256；但 BK=256不能全局删除。
3. **大 compute shape 需要更大的 tile-work/warp**：完整 BF16 数据支持 $Q\ge1024$，但这个规则应在 template-aware 或确认搜索完整后使用。

最推荐的推进顺序：

- 先采用“规则 B”的小 M、N、K 三类保守收紧；
- 新增搜索时记录每个文件的候选总数、失败数和搜索完整度；
- 重新收集 M=128/512 和 M≥1024 的完整结果；
- 再决定是否把 $Q\ge1024$ 和 template-specific 规则加入第二层剪枝。

当前不建议：

- 固定 `BN=256`；
- 全局删除 `BK=256`；
- 按 M 强行固定 `num_stages`；
- 在没有 template 参数的 `is_good_config` 中删除某个 template；
- 把旧 `state` 的绝对时间和 BF16 v6/v7 时间直接比较。
