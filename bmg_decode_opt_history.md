# bmg_decode Kernel 优化实验记录

## 背景

目标 shape: `(M, N, K) = (128, 2048, 768)`, dtype = BF16。

结论回顾（详见 [tune_one_shape_128_2048_768.log](tune_one_shape_128_2048_768.log)）：

- 预分配输出、kernel-only benchmark 下，oneDNN（约 13.6–16.6 μs）比当前 Triton
  最优配置（`bmg_decode`, `BLOCK_M=32, BLOCK_N=256, BLOCK_K=32, num_stages=2,
  num_warps=8`, 约 21.5 μs）快约 1.3–1.5×。
- 几乎所有 `BM=16~128` 范围内的候选 config，无论怎么调
  `BLOCK_N/BLOCK_K/num_stages/num_warps`，时间都聚集在 21–25 μs 这个窄区间，
  说明已经触到某种硬件/kernel 结构上的固定开销地板，而不是单纯的 tile 参数问题。

## 根因分析

查看 [triton_bmg_tiled2d_mm.py.jinja](../../pytorch/torch/_inductor/kernel/templates/triton_bmg_tiled2d_mm.py.jinja)
与 bench_worker.py 中的 `kernel_bmg_decode` 复刻实现：

1. **`bmg_decode` 模板被限制为 `BLOCK_M ≤ 32`**（见
   [search_space.py](search_space.py) 的 `is_valid_for_template`），这是历史上
   为"纯 decode（M≤32）"场景设计的假设，未必适合当前 M=128 的 shape。
2. **B 矩阵被重复读取**：kernel 的 2D grid `(pid_m, pid_n)` 中，B 的
   `block_ptr` 只依赖 `pid_n`，与 `pid_m` 无关。也就是说，对于固定的
   `pid_n`，M 方向上的每一个 `pid_m` 程序实例都会独立地把 B 的这一列条带完整
   读一遍。
   - 当前最优 config：`BLOCK_M=32` → M 方向切成 `128/32=4` 个 tile。
   - 理论上 B 应该只需要读一次（`768×2048×2B ≈ 3.14MB`），但由于上述结构，
     实际可能被读了 4 遍（`≈12.6MB`），如果 GPU L2 没有很好地在并发
     workgroup 间复用这些数据。
   - 理论最小 DRAM 流量（A+B+C 各读/写一次）对应的最短时间约为 8.5 μs
     （在 456GB/s 峰值带宽下），而 4 倍 B 流量对应约 29 μs，与观测到的
     21–25 μs 高度吻合（说明确实存在明显但不完全的重复读取，可能有部分
     L2 命中）。

## 实验计划（按优先级）

1. **P1** — 重构 kernel：把 grid 从 `(pid_m, pid_n)` 二维改为只按 `pid_n`
   一维划分，kernel 内部循环遍历所有 M-tile；对每个 K-tile，B 只从
   DRAM 加载一次，在寄存器中立即被多个 M-tile 复用后再丢弃
   （循环顺序：outer=K-tile，inner=M-tile）。
2. **P2** — 放开 `is_valid_for_template` 里 `bmg_decode` 的 `BLOCK_M ≤ 32`
   限制，专门为 M=128 生成更大 `BLOCK_M` 的候选。
3. **P3** — 尝试增大 `BLOCK_K`，减少 K 循环次数和 barrier/同步开销。
4. **P4** — 用硬件计数器（vtune/hw_metrics）验证 P1 假设是否命中真实瓶颈。
5. **P5** — 如果收益有限，考虑更大层面的 kernel 融合/批处理。

以下按顺序记录每一步的实验设置与实际结果。

---

## 实验 1（P1）：M 内循环 + B 复用 kernel

### 设计

新增 `kernel_bmg_decode_mloop4`（见 [bench_worker.py](bench_worker.py)），
针对 `M=128, BLOCK_M=32`（恰好 4 个 M-tile）硬编码：

- grid 从 `(pid_m, pid_n)` 二维改为只有 `(pid_n,)` 一维，共 `N/BLOCK_N` 个
  程序实例；
- 每个程序实例内部维护 4 个独立的累加器 `acc0..acc3` 和 4 个 A 的
  `block_ptr`；
- 外层循环遍历 K-tile，每次循环只从 DRAM `tl.load` 一次 B 的
  `(BLOCK_K, BLOCK_N)` tile，然后立刻对 4 个 M-tile 依次复用这份寄存器中的
  B 数据做 `tl.dot`，再统一 `advance` 所有指针。

（注：Triton 的 AST 前端不支持通过列表 `append`/索引赋值来维护
"循环携带的张量列表"，因此 4 个累加器/指针必须写成显式命名变量，而不是
`for` 循环 + 列表结构，这也是本次实现放弃"任意 M-tile 数量"通用实现、改为
针对该 shape 硬编码 4 个 tile 的原因。）

### 正确性验证

```text
correctness max_abs_diff = 0.0000
```

与 `torch.mm(A, B)` 参考结果完全一致，kernel 逻辑正确。

### 性能结果（预分配输出、2000 次/轮，5 轮）

```text
mloop4 (新 kernel):  380.09 / 382.46 / 380.68 / 380.76 / 380.71 us   median ≈ 380.76 us
bmg_decode (原始):    20.44 /  20.62 /  20.38 /  20.41 /  20.46 us   median ≈  20.43 us
```

### 结论：**严重负向结果（约 18.6× 变慢），假设不成立**

原本预期"减少 B 矩阵重复读取"应带来收益，但实测反而暴增 18 倍。根因推断：

- 同时保留 4 个 `(BLOCK_M=32, BLOCK_N=256)` 的 float32 累加器，每个占用
  `32×256×4B = 32KB`，4 个合计 `128KB` 寄存器/共享内存占用，远超 Xe2
  单个 Xe-core 的可用寄存器文件容量，导致大量寄存器溢出（register
  spilling）到较慢的内存层级；
- 4 路 `tl.dot` 展开后完全串行执行（同一份 B 数据被依次用于 4 次矩阵乘），
  破坏了原本 `tl.dot` 内部的软件流水线和 DPAS 调度优化。

**结论**：显式的"B 只读一次、内部循环复用"重构，在寄存器/共享内存资源
有限的 Xe2 硬件上，资源压力代价远大于减少内存读取带来的收益。这个方向
在当前 tile 大小下不可行，**已放弃**。

---

## 实验 2（交叉验证）：BLOCK_M > 32 是否能突破 21–24 μs 的地板

`bmg_decode` 模板被限制为 `BLOCK_M ≤ 32`（[search_space.py](search_space.py)
`is_valid_for_template`），但 **`triton_mm` 和 `bmg_persistent` 两个模板并没有
这个限制**，[tune_one_shape_128_2048_768.log](tune_one_shape_128_2048_768.log)
里已经包含了大量 `BLOCK_M=64` 和 `BLOCK_M=128` 的实测数据，可以直接复用
这些数据回答"放开限制是否有意义"这个问题，不需要额外跑实验：

```text
BM=32  bmg_decode      BN=256 BK=32  NS=2 NW=8   time=21.551 us   (当前最优)
BM=64  triton_mm       BN=128 BK=64  NS=3 NW=8   time=22.586 us
BM=64  bmg_persistent  BN=256 BK=128 NS=2 NW=16  time=21.944 us
BM=128 triton_mm       BN=64  BK=64  NS=2 NW=4   time=23.070 us
BM=128 bmg_persistent  BN=32  BK=32  NS=2 NW=8   time=22.272 us
```

**结论**：`BLOCK_M=64/128` 并没有比 `BLOCK_M=32` 更快，反而基本持平或略慢。
说明"放开 `bmg_decode` 的 `BLOCK_M≤32` 限制"（原计划 P2）**不会带来收益**，
已经被现有搜索数据证伪，无需专门重新实现放开限制的版本。

---

## 实验 3（交叉验证）：BLOCK_K 增大是否有帮助

同样复用已有日志数据（`BM=32, BN=256` 固定，只看 `BLOCK_K`）：

```text
BK=32   NS=2 NW=8   time=21.551 us  （全局最优）
BK=64   NS=4 NW=4   time=21.698 us
BK=128  NS=1 NW=8   time=21.681 us
BK=256  NS=2 NW=32  time=21.641 us
```

**结论**：`BLOCK_K` 从 32 增大到 256，时间几乎不变（都在 21.5–21.7 μs 区间内
波动）。说明增大 `BLOCK_K` 减少 K 循环次数/barrier 次数（原计划 P3）
**对当前瓶颈没有实质影响**，也已经被现有数据覆盖，无需重新测试。

---

## 实验 4（P4）：验证"固定调度开销地板"假设

由于实验 1-3 都显示：无论怎么调整 `BLOCK_M / BLOCK_N / BLOCK_K /
num_stages / num_warps`，几乎所有候选都卡在 **21–24 μs** 的窄区间，
怀疑主要瓶颈是 Xe2/Triton 的固定 kernel 调度/发射开销，而不是 tile 参数或
内存带宽。用两个极简 Triton kernel（几乎不做计算，只做 load+add+store）
测量"纯发射地板"：

### 4a. 4 个 program 实例（对应极小 grid）

```python
grid = (4,)   # 每个 block 处理 256 个 float32 元素，共 4KB 数据
```

结果（2000 次/轮 × 5 轮）：

```text
15.010 / 14.594 / 14.503 / 14.493 / 14.444 us   median ≈ 14.50 us
```

### 4b. 32 个 program 实例（对应真实 GEMM 的 grid 大小：4×8=32）

```python
grid = (32,)  # 每个 block 处理 1024 个 float32 元素，共 128KB 数据
```

结果：

```text
30.24 / 29.24 / 30.16 / 29.32 / 29.20 us   median ≈ 29.31 us
```

### 解读

- 即便是几乎不做任何计算的 kernel，4-block 的极小 grid 也需要约
  **14.5 μs** 才能完成一次调用——这本身就已经接近甚至超过 oneDNN 处理这个
  真实 GEMM 所需的全部时间（13.6–16.6 μs）。
- 32-block 版本因为多做了真实的 128KB elementwise load/store（不是纯发射
  开销，还叠加了朴素 1D 访存模式的内存流量），达到 29.3 μs，比真实 GEMM 最优
  配置的 21.5 μs 还要高——说明这个对照本身不是"更干净的地板"，但至少可以
  确认：**在 32-way 并行、需要跨 2 波（20 个 Xe-core）调度的场景下，
  单纯的调度 + 简单内存访问开销就能轻易达到 GEMM 实际总耗时的量级**。

### 4c. 追加验证：这个地板是不是 Triton 专属？是不是编译时间？多跑几轮是否稳定？

针对这三个问题补充了一组对照实验（10 轮 × 2000 次/轮，warmup 20 次）：

```text
1) Triton no-op（4-block grid）:
   15.161 / 14.635 / 14.606 / 14.803 / 15.473 / 14.570 / 14.410 / 28.686 / 28.846 / 28.898
   median ≈ 14.98 us（注意：第 8–10 轮突然跳到 ~29 us，见下方解读）

2) oneDNN/ATen 极小 4x4x4 matmul（torch.mm(A, B, out=C)）:
   15.710 / 15.384 / 15.324 / 14.400 / 13.859 / 13.800 / 13.883 / 14.466 / 13.843 / 13.916
   median ≈ 14.16 us

3) 纯 host 循环（不提交任何 GPU kernel，仅做 Event record/sync）:
   0.186 / 0.142 / 0.048 / 0.048 / 0.048 / 0.046 / 0.047 / 0.047 / 0.048 / 0.050
   median ≈ 0.048 us

4) 同一个 Triton no-op，用 time.perf_counter() 而非 XPU Event 交叉验证:
   14.296 us/iter（与 Event 计时结果一致）
```

**解读（回答"是不是 Triton 专属、是不是编译时间、多跑几次是否稳定"三个问题）**：

- **不是 Triton 专属**：oneDNN 的极小 `4x4x4` matmul（对照 2）耗时
  ~14.16 μs，与 Triton no-op kernel（对照 1）的 ~14.98 μs 几乎相同。说明
  这个地板是 host 提交 + GPU 排队调度 + 同步这条通路的**通用开销**，与
  用 Triton 还是 oneDNN 无关。
- **不是编译时间**：每次计时前都做了 20 次 warmup，Triton 的 JIT 编译
  只发生在第一次调用（已经被 warmup 吸收）。对照 3（不提交任何 kernel
  的纯 Python 循环）耗时仅 ~0.048 μs，说明只要不提交真实 GPU 工作，
  Python/Event 记录本身的开销可以忽略——一旦提交 kernel（哪怕几乎不做
  计算），耗时立刻跳到 ~14–15 μs。
- **多跑几轮后确实不稳定**：把轮数从 5 增加到 10 后，发现第 8–10 轮
  从 ~14.5 μs 突然跳到 ~29 μs，之后保持在高位。这与此前反复观察到的
  oneDNN 在 13–16 μs 和 27–29 μs 之间跳变是**同一类现象**——大概率是
  GPU 时钟/功耗状态（例如 boost 频率与基础频率之间）在测试过程中发生了
  切换，属于设备层面的状态变化，与具体用哪个库/kernel 无关。

**结论（修正版）**：现有最优 Triton 配置（21.5 μs）里，大部分时间是
"kernel 提交 + host/GPU 调度同步"这条通路上的**通用固定开销**
（约 14–15 μs，Triton 和 oneDNN 共享同一个地板），真正花在 GEMM
计算/数据搬运上的净增量大约是 `21.5 - 14.5 ≈ 7 μs`。而 oneDNN
处理这个真实 GEMM 的总耗时（13.6–16.6 μs）基本等于这个通用地板本身，
说明它的实际计算时间几乎可以忽略不计、完全被地板"盖住"了。这与之前
INT8 优化经验中记录的"数据量 < 3MB 的 kernel 会卡在 ~30 μs GPU 最小
调度延迟"是同一类现象（见 user memory `xe2_gemm_kernel_integration.md`），
且现在有了跨 Triton/oneDNN 的直接对照证据，而不只是单一 kernel 的推测。

### 4d. 为什么 oneDNN 能被地板"盖住"，而 Triton 不行？（推断，未做硬件计数器验证）

背后的机制是**流水线提交**：host 在提交 1000–2000 次连续调用时，通常会
在 GPU 还在执行第 i 个 kernel 的同时，就把第 i+1 个 kernel 提交进队列。
这种情况下，每次迭代实测耗时约等于：

```text
每次迭代耗时 ≈ max(host 提交 + 排队开销, GPU 真实执行时间)
```

- oneDNN 的真实 GEMM 执行时间比 ~14–15 μs 的提交/排队地板还短，因此完全
  被下一次提交的等待窗口"藏"住了——测出来的总时间基本等于地板本身。
- Triton 即使是调到最优的 config，真实 GPU 执行时间也超过了这个地板，
  超出的约 7 μs 没有东西可以掩盖，只能体现在总耗时里。

即：**不是 oneDNN 有什么特殊的"掩盖"机制，而是它的 kernel 本身跑得足够
快，快到能被本来就存在的排队开销盖住；Triton 的 kernel 盖不住，多出来的
部分就露出来了。**

背后为什么 oneDNN 真实计算时间更短，推断（未经硬件计数器验证）可能是：

1. oneDNN 用的是针对 Xe2/BMG 手工调优（甚至汇编级）的 GEMM kernel
   （类似 XeTLA），寄存器分配、DPAS 矩阵引擎利用率、软件流水线深度都是
   针对这块具体硬件精细打磨过的；
2. Triton 生成的代码要经过更通用的编译后端（IGC/SPIR-V），在指令调度、
   寄存器分配上不一定能做到和 vendor 库同等激进的优化，尤其是 DPAS 这种
   专用矩阵单元的调度；
3. `bmg_decode` 仍然存在实验 1 结构分析里提到的 B 矩阵被 `pid_m` 重复
   读取的问题（虽然验证下来不是主瓶颈），也可能让真实执行时间比理论
   最优多出一点。

**局限性说明**：以上"流水线提交、彼此重叠"的解释目前只有**间接证据**
（oneDNN 总时间 ≈ 地板，Triton 总时间 = 地板 + 额外量），并没有直接
用 vtune/hw_metrics 之类的 GPU 时间线工具验证"两次提交确实发生了重叠"
这件事本身。这部分仍标记为推断，如果要坐实需要进一步做硬件层面的
时间线分析。

---

## 实验 5（P4 深入）：用 unitrace 直接抓 GPU 侧真实执行时间，定位差距来源

实验 4d 的解释仍停留在推断层面。这里用
`/home/sdp/yifeng/pti-gpu/tools/unitrace/build/unitrace` 的
`--device-timing`（真实 GPU kernel 执行时间）和
`--device-timeline`（每次调用的 append/submit/start/end 时间戳）直接抓取
硬件层面的证据，脚本见
[unitrace_bf16_kernel.py](unitrace_bf16_kernel.py)（固定 shape
`(128,2048,768)` BF16，分别跑 `kernel_bmg_decode` 最优 config 和
`torch.mm(A,B,out=C)`，10 次 warmup + 20 次 profiled）。

### 5a. Device Timing Summary（GPU 真实执行时间，30 次调用统计）

```text
oneDNN gemm_kernel:        avg 7791 ns (7.79 μs)  min 6666  max 10625  SIMD16
Triton kernel_bmg_decode:  avg 8687 ns (8.69 μs)  min 8020  max 11979  SIMD16
```

**两者真实 GPU 执行时间几乎一样**，只差约 0.9 μs（~11%），远小于之前
host-side Event 计时看到的 6–8 μs 差距。也就是说：**Triton 生成的 kernel
在 GPU 上算得并不比 oneDNN 慢多少**，之前怀疑的"IGC 编译器不如 vendor
库激进优化"这个因素即使存在，影响也很有限。

### 5b. Device Timeline（连续调用之间的真实周期）

从 timeline 里取稳态区间（排除 warmup 与 sync 之后的跳变），逐次计算
"下一次 kernel start - 上一次 kernel start"：

```text
oneDNN:  周期 ≈ 27.0 μs = 执行 ~7.9 μs + host 调度/空闲间隔 ≈ 19.1 μs
Triton:  周期 ≈ 36.4 μs = 执行 ~9.1 μs + host 调度/空闲间隔 ≈ 27.3 μs
```

两者 GPU 执行时间只差 ~1.2 μs，但 **host 端调度/空闲间隔相差约 8 μs**
——这才是总耗时差距的主要来源。（注：unitrace 插桩本身会给两条路径都
引入额外开销，因此这里的绝对周期数值比不插桩时的 XPU Event 结果更大，
但两条路径在同一插桩条件下的**相对差距**仍然是有效、可比的证据。）

### 结论（修正版，取代实验 4d 的推断）

之前"通用调度地板"的方向没错，但现在有更精确的证据：**并不是 GPU
排队/执行的开销对 Triton 和 oneDNN 完全一样大**，而是：

- GPU 上真正执行 GEMM 计算的时间，两者几乎相同（~7.9 μs vs ~8.7 μs）；
- 差距主要来自 **host 端的调度路径**——Triton 的 Python kernel launcher
  （grid 计算、JIT 参数打包、编译变体缓存查找等）比 oneDNN/ATen 更直接的
  C++ dispatch 路径多花了约 8 μs 的 host 端开销。

这也解释了为什么实验 1–3（调 `BLOCK_M/N/K`、循环结构）都无法突破 ~21 μs
的地板：**瓶颈根本不在 kernel 内部**，而在 kernel 外面的 host 端调度
路径，调 tile 参数从一开始就触及不到这部分开销。如果要真正缩小与 oneDNN
的差距，方向应该是**减少 Triton 每次调用的 host 端开销**（例如通过
`torch.compile` 的 CUDA-graph 风格捕获、减少 Python 层参数打包/缓存
查找的开销），而不是继续调 kernel 内部的 tile 参数或循环结构。

---

## 总结与最终结论

| 优先级 | 实验方向 | 方法 | 结果 | 结论 |
|---|---|---|---|---|
| P1 | M 内循环、B 只读一次复用 | 新写 `kernel_bmg_decode_mloop4`，硬编码 4 个 M-tile | 380.76 μs（比原来慢 18.6×） | **失败**：4 份累加器同时驻留导致寄存器溢出，资源压力代价远超收益。已放弃该方向。 |
| P2 | 放开 `BLOCK_M≤32` 限制 | 复用已有日志中 `triton_mm`/`bmg_persistent` 在 BM=64/128 的数据 | 21.9–23.1 μs，与 BM=32 基本持平 | **证伪**：放开限制不会带来收益，因此没有必要重新实现放开限制版本的 `bmg_decode`。 |
| P3 | 增大 `BLOCK_K` | 复用已有日志中 BK=32→256 的数据（固定 BM=32,BN=256） | 21.55–21.70 μs，几乎不变 | **证伪**：`BLOCK_K` 对当前瓶颈没有实质影响。 |
| P4 | 验证固定调度开销地板假设 | 用几乎无计算的极简 Triton kernel 测量纯发射/小 grid 开销；追加 oneDNN 极小 matmul 对照、纯 host 循环对照、10 轮稳定性复测 | 4-block ≈14.5–15.0 μs（Triton）≈14.16 μs（oneDNN 极小 matmul），纯 host 循环 ≈0.048 μs，32-block(含 128KB IO) ≈29.3 μs；10 轮中有 3 轮跳变到 ~29 μs | **支持"存在固定地板"，但后续 unitrace 实验（见下）精确定位到差距主要来自 host 端调度，而非 GPU 执行本身**。 |
| P4+ | 用 unitrace 直接抓 GPU 真实执行时间，定位差距来源 | `--device-timing` + `--device-timeline`，对比 `kernel_bmg_decode` 最优 config 与 `torch.mm(out=)` | GPU 真实执行时间几乎相同（oneDNN 7.79 μs vs Triton 8.69 μs，差 ~0.9 μs）；host 端调度/空闲间隔相差约 8 μs（19.1 μs vs 27.3 μs） | **精确定位**：GEMM 计算本身两者几乎一样快，差距的主要来源是 Triton 的 Python kernel launcher（grid 计算、参数打包、缓存查找）比 oneDNN/ATen 更直接的 C++ dispatch 多花的 host 端开销，不是 kernel 内部效率问题。 |

### 最终判断

对 `(128, 2048, 768)` BF16 这个具体 shape：

1. **继续在当前 kernel 结构基础上微调 `BLOCK_M/N/K`、`num_stages`、
   `num_warps` 几乎没有进一步收益空间**——2720 个候选里已经把这些维度
   扫得很充分，最优值和绝大多数候选都卡在 21–24 μs 的同一个地板上。
2. **"B 矩阵被重复读取"这个结构性问题确实存在**（结构分析层面成立），但
   **不是当前的主导瓶颈**——用寄存器换取减少重复读取的尝试（P1）反而让
   情况变得更差，说明当前性能已经不是被内存带宽卡住。
3. **用 unitrace 精确测量后确认：瓶颈根本不在 GPU kernel 本身**
   （oneDNN 与 Triton 真实执行时间只差 ~0.9 μs），而在 **host 端的调度
   路径**——Triton 的 Python kernel launcher（grid 计算、JIT 参数打包、
   编译变体缓存查找）比 oneDNN/ATen 更直接的 C++ dispatch 多花了约 8 μs。
   这意味着继续调 `BLOCK_M/N/K`、`num_stages`、`num_warps` 这些 kernel
   内部参数，从原理上就无法触及这部分开销，P1–P3 的失败/无效结果也由此
   得到了更根本的解释。
4. 如果要进一步缩小这个差距，应该把方向从"调 kernel 内部 tile 参数"
   转移到**减少 host 端调度开销**本身，例如：
   - 把该 GEMM 和相邻的算子（如激活函数、残差加法、下一层的输入准备等）
     融合进同一个 kernel/图，用 `torch.compile` 的图捕获减少 Python 层
     每次调用的参数打包/缓存查找开销；
   - 在更大的批处理/多请求场景下，让同一个 kernel 处理更多请求，摊薄每次
     发射的固定成本；
   - 如果确实需要独立的单次 GEMM 调用且无法优化 host 端开销，`oneDNN`
     （约 13.6–16.6 μs）在这个具体 shape 上已经比目前所有 Triton 候选更
     优，直接使用 oneDNN 是更现实的选择。
5. 本次记录的 `kernel_bmg_decode_mloop4` 实现（[bench_worker.py](bench_worker.py)）
   以及 unitrace 分析脚本 [unitrace_bf16_kernel.py](unitrace_bf16_kernel.py)
   都保留在代码里，前者作为负向实验记录，后者作为可复用的 GPU 侧真实
   执行时间分析工具。

