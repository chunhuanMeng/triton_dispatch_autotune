# Xe2 / BMG GEMM 性能分析方法论与实验记录

记录日期：2026-08-11 创建，2026-08-12 补充实验 8–10 与预算表，2026-08-13 重排结构
硬件：Intel Arc Pro B60（BMG-G21，Xe2），驱动 `1.15.38646+6`
软件：PyTorch `2.14.0a0+gitc4f86df`，Triton `3.7.1`，oneDNN `3.12.0`

---

## 怎么读这篇

**第一、二部分是正文，读完就够用了。第四部分是这些结论的出处，不必读。**

| 部分 | 内容 | 必读？ |
|---|---|---|
| **第一部分** | Xe2 执行模型：dpas / slot / 四条管线 / 峰值怎么算 | ✅ 所有数字的定义都在这里 |
| **第二部分** | 瓶颈判定方法论：拿到一个 shape 该怎么做 | ✅ 干活时查这里 |
| 第三部分 | 采集命令、汇编导出 | 用到时再看 |
| 第四部分 | 10 个实验，第一、二部分每条结论的**实测出处** | ❌ 只在你不信某个数字时翻 |
| 附录 | 常量速查 | 查表 |

具体要找什么：

| 你想干什么 | 直接跳到 |
|---|---|
| 拿到一个 shape，想知道该期待什么数字 | **2.1 预算表**（部分不用跑 profiler） |
| 想快速看一个 kernel 有没有明显毛病 | **2.2 体检表**（8 项比值，多数不用 profiler） |
| 手上有 profiler 数据，想定位瓶颈 | **2.8 时间预算分解**（最实用） |
| 想知道什么时候可以停手 | **2.4 停手条件** |
| 想搞懂 dpas / slot / 峰值怎么算 | **第一部分** |
| 想知道哪些数字可信、哪些是推测 | **2.11 证据强度表** |
| 想看某个判定的原始计数器读数 | **2.12 实测数据汇总** |
| 只想查个常数 | **附录** |
| 不信某个结论，想看它是怎么测出来的 | **第四部分**（实验 1–10） |

> **为什么实验放在最后？** 因为它们是**验证**，不是推导。
> 第一部分的执行模型和第二部分的判定流程都是自洽的 —— 公式、常量、阈值、
> 反例数据全部就地给出。第四部分回答的是「你凭什么说峰值是 98.3 而不是 117」
> 这类问题，以及记录我们踩过的坑（其中 4 个坑让之前几个月的结论全部作废）。
>
> 正文里凡是写「（实验 N）」的地方，都只是**溯源标记**，跳过不影响理解。

> ### 动手前先记住一条
>
> **`unitrace` 的 metric 用 `-q -g ComputeBasic`，时间用 `-d`，两次分开跑。**
> `-q` 会让 kernel 真的慢 1.14–1.55 倍且每个 config 不同，用它的 `GpuTime` 比较
> 两个 config 会得到颠倒的结论。详见第三部分的工具警告。

---

## 第一部分：Xe2 执行模型

### 1.1 硬件层级

```
Arc Pro B60 (BMG-G21)                    @ 2.4 GHz
└── 20 x Xe core
    ├── 8 x XVE (Xe Vector Engine)       SIMD16，每个最多 8 硬件线程
    └── 8 x XMX (Xe Matrix Engine)       脉动阵列，与 XVE 一一配对
```

**XVE 和 XMX 是并列的两组引擎**，Intel 的官方描述是每个 Xe core 含 8 个 Vector Engine 和 8 个 Matrix Engine。它们：

- 物理上是**独立的执行单元**，可以同时工作
- 共享同一个**指令流**和**寄存器堆（GRF）**：一个线程发射 `add` 走 XVE，发射 `dpas` 走 XMX
- 在性能计数器里，XMX 被暴露为 XVE 的 **ALU2 管线**，所以计数器名字都带 `XVE_` 前缀

并行性证据：`XVE_MULTIPLE_PIPE_ACTIVE[%]` 定义为「至少两条管线（ALU0/ALU1/ALU2）同时执行」，实测 Triton kernel 为 17.3%，说明标量运算和矩阵运算确实在重叠执行。

全卡规模：$20 \times 8 = 160$ 个 XVE，同样 $20 \times 8 = 160$ 个 XMX，$160 \times 8 = 1280$ 个硬件线程。

两个 160 数值相同但含义不同：后文 1.7 节的 ALU2 slot 推导用的是 **XMX 的 160**（得到 80 slots/clk），
而标量管线上限 160 slots/clk 用的是 **XVE 的 160**。
注意后者是 **ALU0/ALU1 执行槽**的上限，**不是指令发射的上限**（见 1.4 节与实验 10）。

### 1.2 Shared Function 与 LSU

除了 XVE/XMX，每个 Xe core 还有一组 **Shared Function（共享功能单元）**，被该 core 内
**8 个 XVE 共用**（注意：不是每 XVE 一个）：

```
Xe core
├── 8 x XVE ──┐
├── 8 x XMX   │  发 SEND 指令
└── Shared Functions   <── 整个 Xe core 共用
    ├── LSU (Load/Store Unit)  ← 访存单元
    │   └── L1 数据缓存（unitrace 里叫 "Load Store Cache"）
    ├── Sampler
    └── Barrier / Message
```

**LSU = Load/Store Unit，访存单元。** 数据流：

```
XVE 发一条 SEND（load_block2d）
   ↓
LSU 收到消息，拆成若干次 cache line 访问
   ↓
命中 L1 → 直接返回；未命中 → L2（18 MB）→ DRAM
```

一条 block2D 消息会**展开成多次 L1 访问**（二维块跨越多条 cache line）。所以消息数和
访问数是两个量级的东西：

| 计数器 | 含义 | $(256,4k,4k)$ Triton |
|---|---|---:|
| `XVE_INST_EXECUTED_SEND_ALL` | XVE 发出的**消息**数 | 3.42/clk |
| `LOAD_STORE_CACHE_ACCESS` | LSU 对 L1 的**访问**数 | 78.57/clk |
| `XVE_SHARED_FUNCTION_ACCESS_HOLD` | XVE 因等 Shared Function 而停顿的时间占比 | 40.7% |

平均每条消息展开成约 23 次访问（oneDNN 约 13 次）。

**为什么它会成为瓶颈**：LSU 是每 core 一个的共享资源，8 个 XVE 抢它。
Triton 的 L1 访问数是 oneDNN 的 3.4 倍，但 DRAM 流量几乎相同 ——
多出来的访问全部命中 L1（命中率 94%），**不产生显存流量，但照样排队占用 LSU 带宽**。
干净口径下 Triton 的 L1 访问速率是 **89.70 次/clk**，oneDNN 只有 26.47 —— **3.38 倍**。
（LSU 没有可信上限，所以只能这样横比，不能说「占了百分之多少」，见 2.8 节。）

> **关于 LSU 上限**：没有对应的 `UTILIZATION` 计数器，**真实上限未知**。
> 373 个采样中 `LOAD_STORE_CACHE_ACCESS/clk` 封顶在 **82.07**。
>
> **但 82 不是硬件上限的估计，只是「profiling 口径下见过的最高值」。**
> 这些采样全部来自 `-q -g ComputeBasic`，而它会把时间抬高 1.14–1.55 倍（见第三部分警告），
> 每 clk 速率就被相应压低。同一个 $(256,4096,4096)$ Triton kernel：
>
> | 口径 | clk | LSU 速率 |
> |---|---:|---:|
> | profiling 下 | 320,289 | 78.58/clk |
> | `unitrace -d`（干净） | 280,582 | **89.70/clk** |
>
> **硬件至少能做 89.70，已经超过了那个「82 封顶」。** 同理 `ISSUED` 的 80.25 应为约 91.6。
>
> **使用规则：上限估值和被测 kernel 必须来自同一口径。**
> 既然 82 不是硬件上限，2.8 节已改为**只报 LSU 速率、不报百分比**。
>
> 对比 ALU2：它的 80 是**厂商常数**（驱动 metric 定义里写死的，见 1.7 节），
> 不是观察出来的，所以不存在"被干扰压低"的问题，任何口径下都能用。
>
> 所以判定 LSU-bound **不要依赖百分比**，而要看三个不依赖上限的信号：
> 1. `XVE_SHARED_FUNCTION_ACCESS_HOLD` 高（>25%）
> 2. L1 访问数明显高于参照实现，而 DRAM 流量相同
> 3. ALU2 和 DRAM 都未饱和

### 1.3 一个线程可发射的四条管线

前三条来自 XVE，ALU2 实际落到配对的 XMX 上，但计数器统一用 `XVE_` 前缀暴露：

| 管线 | 物理单元 | 承载指令 | 计数器 |
|---|---|---|---|
| ALU0 | XVE | 浮点/整数向量运算 | `XVE_INST_EXECUTED_ALU0_ALL` |
| ALU1 | XVE | 整数/逻辑（`shl`/`shr`/`and`/`bfn`/地址计算） | `XVE_INST_EXECUTED_ALU1_ALL` |
| **ALU2** | **XMX** | **脉动阵列（`dpas`）** | `XVE_INST_EXECUTED_ALU2_ALL` |
| SEND | XVE | 访存消息（`load_block2d`/`store`/prefetch） | `XVE_INST_EXECUTED_SEND_ALL` |

### 1.4 execution slot 的定义

`XVE_INST_EXECUTED_*` 统计的是 **execution slot（发射槽）**，不是指令条数：

- 普通 SIMD16 指令（`mov`/`add`/`shl`）→ 占 **1 个 slot**，指令数与 slot 数重合
- `dpas.8x8`（repeat count = 8）→ 占 **8 个 slot**，每个 repeat 行一个

这一点很关键，否则会把 ALU2 的读数误解成 dpas 指令数。

**但 `XVE_INST_ISSUED_ALL` 是个例外，它数的是指令条数。** 官方定义就不同：

| 计数器 | 定义原文 | 单位 |
|---|---|---|
| `XVE_INST_EXECUTED_ALU0/1/2_ALL` | *Number of **execution slots** taken by instructions executed on ALUx pipe* | **slot** |
| `XVE_INST_ISSUED_ALL` | *Number of **instructions** issued (decoded) to any pipe* | **指令** |
| `XVE_INST_EXECUTED_SEND_ALL` | *Number of **instruction dispatches** executed on SEND Pipe* | 消息 |

对账验证（看 dpas 该按 1 条还是 8 个 slot 计入 `ISSUED`）：

| | $8192^3$ oneDNN | $(256,4k,4k)$ oneDNN | $(256,4k,4k)$ Triton |
|---|---:|---:|---:|
| ALU2 events（slot） | 2,147,483,648 | 16,777,216 | 16,777,216 |
| ÷8 = dpas 指令数 | 268,435,456 | 2,097,152 | 2,097,152 |
| ALU0+ALU1+SEND | 137,386,300 | 1,148,928 | 23,908,352 |
| **若 dpas 算 1 条** | 405,821,756 | 3,246,080 | 26,005,504 |
| 若 dpas 算 8 个 slot | 2,284,869,948 | 17,926,144 | 40,685,568 |
| `ISSUED_ALL` 实测 | **401,595,744** | **3,181,568** | **24,623,112** |
| 偏差 | 1.1% / 469% | 2.0% / 463% | 5.6% / 65% |

即：

$$\texttt{ISSUED} \approx \frac{\texttt{ALU2}}{8} + \texttt{ALU0} + \texttt{ALU1} + \texttt{SEND} + \text{控制流}$$

残余的 1–6% 是 `jmpi`/`cmp`/`sync` 这类控制流指令 —— 被发射了，但不落在这四条管线上。

**实际影响**：三条上限单位不同，搞混会差好几倍。

| 上限 | 单位 | 数值 | 对应计数器 | 证据 |
|---|---|---:|---|---|
| XMX 吞吐 | **slot**/clk | 80 | `ALU2_EXECUTED` | 高（计数器反推） |
| 标量管线吞吐 | **slot**/clk | **160** | `ALU0/ALU1_EXECUTED` | 中高（厂商常数，见 1.7 节） |
| 指令发射 | **指令**/clk | **未知** | `ISSUED_ALL` | 无 |

> **160 属于 ALU0/ALU1 执行槽，不属于 `ISSUED`。** 这是个容易搞错的地方。
> 下表两个 kernel"都反推出 160"**是恒等式**（`ALU1_UTILIZATION` 的定义分母就是 160），
> 不构成验证；列在这里只是说明 160 对应的是**哪个**计数器：
>
> | kernel | ALU1 slots/clk | `ALU1_UTILIZATION` | 反推上限 |
> |---|---:|---:|---:|
> | 纯标量探针（`probe_issue_ceiling.py`） | 116.28 | 72.7% | **159.9** |
> | $(256,4k,4k)$ Triton | 69.0 | 43.2% | **159.7** |
>
> `ISSUED` 与 `ALU1` 差约 2 倍（61.98 vs 116.28），因为多数指令是 SIMD32，
> **占 2 个执行槽但只发射 1 次**。把 160 套到 `ISSUED` 上怎么算都对不上。

### 1.5 `dpas.8x8` 一条指令做多少事

指令名中的 `8x8` = **repeat count 8 × systolic depth 8**：

```
MAC = 8 (repeat 行) × 16 (SIMD16 通道) × 8×2 (depth × bf16 打包) = 2048
```

- repeat = 8 → 输出 tile 有 8 行
- SIMD16 → 每行 16 列，每个通道负责一列
- systolic depth = 8，bf16 每步打包 2 个元素 → 沿 K 方向一次吃 **16** 个元素

即一条 `dpas.8x8` 计算一个 `8×16` 的输出块，沿 K 累加 16 步。

**本项目里见到的全部是 `dpas.8x8`，没有第二种变体。**
扫了 22 个已 dump 的 kernel（oneDNN nGEN 三个、Triton 各种写法十几个），无一例外：

| 来源 | 循环体 dpas 条数 | 变体 |
|---|---:|---|
| oneDNN nGEN | 384 | `dpas.8x8` |
| Triton `make_block_ptr` | 32 | `dpas.8x8` |
| Triton tensor-of-pointers | 16 | `dpas.8x8` |
| Triton TensorDescriptor（四种组合） | 32 / 64 | `dpas.8x8` |
| Triton `bmg_persistent` / `triton_mm` | 8 / 16 | `dpas.8x8` |

两个维度都没得选：

- **systolic depth = 8 是硬件固定的**，Xe2 的脉动阵列就是 8 级
- **repeat count = 8 是 ISA 允许的最大值**，每条指令摊到的开销最小。
  repeat 取 1/2/4 只会出现在边界余数处理或寄存器压力逼迫时

所以后文把「每条 dpas = 8 个 slot」当常量用是安全的。

> **但 `8x8` 里的两个 8 跟数据类型无关 —— 变的是每个 32-bit 槽塞几个元素。**
>
> | dtype | 每 32-bit 槽 | 每条 dpas 吃的 K 步 | MAC/指令 | **MAC/slot** |
> |---|---:|---:|---:|---:|
> | bf16 / fp16 | 2 | $8\times2=16$ | 2048 | **256** |
> | int8 | 4 | $8\times4=32$ | 4096 | **512** |
>
> **本文所有公式里的 256 都是 bf16 专用常量。**
> 换成 int8，$\texttt{ALU2\_min} = MNK/512$，峰值也翻倍（$20\times4096\times2.4\text{G} = 196.6$ TOPS）。
>
> **int8 这一行已用计数器标定（2026-08-13）**，不是推的。
> `unitrace -q -g ComputeBasic python validate_int8_slot.py`，`torch._int_mm` 跑四个 shape：
>
> | shape | $MNK/\texttt{ALU2\_events}$ | UTIL% | TOPS |
> |---|---:|---:|---:|
> | $(8192,8192,8192)$ | **512.0000** | 93.78 | 184.33 |
> | $(4096,4096,4096)$ | **512.0000** | 87.22 | 171.45 |
> | $(2048,7168,28672)$ | **512.0000** | 96.49 | **189.70** |
> | $(2048,5120,5120)$ | **512.0000** | 89.11 | 175.15 |
>
> **读表须知（各列含义与陷阱）**：
>
> | 列 | 怎么来的 | 有信息量吗 |
> |---|---|---|
> | $MNK/\texttt{ALU2\_events}$ | 纯事件计数 | ✅ **唯一真正的证据** |
> | UTIL% | `ALU2_UTILIZATION` 直读 | ⚠️ `-q` 口径，欲得干净口径需按 2.8 节重建 |
> | TOPS | $2MNK/t$，$t$ 取自 `-q` | ⚠️ `-q` 口径，真值更高 |
>
> 原表还有 `ceiling` 和 `有效%` 两列，**已删除**——都是恒等式：
> `ceiling` $=\frac{\texttt{events}/\texttt{clk}}{\texttt{UTIL}}$ 必为 80（驱动定义）；
> 无浪费时 $\text{有效\%}=\frac{2MNK/t}{20\cdot4096\cdot f}=\frac{MNK}{40960\,tf}=\texttt{UTIL}$，
> 原表里 93.78/93.81 那种"一致"只是舍入，不是交叉验证。
>
> 三个结论：
> 1. **512 是精确整数**，4/4 shape，和 bf16 的 256 同等强度
> 2. **80 slots/clk 的天花板 int8 和 bf16 共用** —— XMX 的瓶颈在 slot 数，不在数据类型
> 3. 附带一个有用的推论：$(2048,7168,28672)$ 的有效% 达 **96.53%**，而它用的是 `-q` 口径的时间。
>    效率不可能超 100%，所以**这个 kernel 被 profiling 拖慢不超过 3.6%** —— 大 oneDNN GEMM 受干扰很小。
> 4. 旧峰值 234 TOPS 站不住：`ALU2_UTILIZATION` 读 96.49%，而 $189.70/234 = 81.1\%$。
>    **注意这不是独立测量** —— 该计数器的定义分母就隐含了 $20\times4096$ OPS/clk，
>    所以它说的其实是「**Intel 自己的驱动认为峰值是 196.6，不是 234**」。
>    真正独立的部分是 $MNK/\texttt{ALU2\_events} = 512.0000$ 这个纯计数结果。
>
> 复现脚本：`validate_int8_slot.py`。

### 1.6 从 slot 推到 FLOP

```
1 dpas.8x8 = 8 slots = 2048 MAC
1 slot     = 2048 / 8 = 256 MAC = 512 FLOP
```

物理含义：**1 个 slot = dpas 的 1 个 repeat 行** = 16 通道 × 16 个 K 步 = 256 MAC。

这个 256 是**量出来的，不是推的**（8192³ GEMM）：

```
M·N·K / ALU2_events = 8192³ / 2,147,483,648 = 256.0
```

整数 256.0，无小数残余。

### 1.7 「80 slots/clk」是怎么来的（2026-08-13 重写：原推导是循环论证）

> **⚠ 先说结论的性质。** 本节旧版写「80 是从两个计数器反推出来的」，**那是错的**。
>
> 逐行验证发现 $\dfrac{\texttt{ALU2\_events}/\texttt{clk}}{\texttt{ALU2\_UTILIZATION}}$ 在**每一行**
> 都精确等于 80.000000（12 行，std = 0.000003）——因为驱动就是按
> $\texttt{ALU2\_UTILIZATION} = \dfrac{\texttt{ALU2\_events}}{\texttt{clk}\times 80}$ 算的。
> **这是恒等式，不是测量。** 它只说明 Intel 的驱动认为上限是 80。
>
> **80 的真实性质：厂商常数**（写死在驱动的 metric 定义里），不是我们量出来的。
> 同理 ALU0/ALU1 的 160 也是驱动常数，「两个 kernel 独立反推出 160」同样是恒等式。
>
> **真正独立的证据有两条：**
> 1. $MNK / \texttt{ALU2\_events} = 256.0000$ —— **纯事件计数**，不涉及 UTILIZATION，这条是硬的
> 2. Intel 公布 B580 @2.85 GHz 为 117 TF/s $\Rightarrow \frac{117\text{e}12}{20\times2.85\text{e}9}=2052\approx 2048$
>    FLOP/clk/core，与「80 slots/clk ÷ 20 core × 512 FLOP/slot = 2048」**一致**
> 3. 旁证：oneDNN $8192^3$ 实测 96.83 TF/s = 98.3 的 98.5%。若真实峰值明显更高，
>    厂商调优的大 GEMM 不至于只用到这么少
>
> 所以 **98.3 TF/s 这个数仍然成立**，但它的依据是「256 实测 + 80 厂商常数 + 频率实测」，
> 不是「两个计数器凑出整数」。下面保留原推导过程，但请按上述性质理解。

实测量（8192³ GEMM，只有这四个是直接读的）：

| 量 | 值 |
|---|---:|
| `XVE_INST_EXECUTED_ALU2_ALL` | 2,147,483,648 |
| `GpuCoreClocks` | 27,251,018 |
| `XVE_INST_EXECUTED_ALU2_ALL_UTILIZATION` | 98.50 % |
| `AvgGpuCoreFrequencyMHz` | 2399 MHz |

**第 1 步：实际达到的 slot 速率**

```
ALU2_events / GpuCoreClocks = 2,147,483,648 / 27,251,018 = 78.8038 slots/clk
```

这是全 GPU 每个时钟周期退休的 ALU2 slot 数。`GpuCoreClocks` 是全局核心周期，
不是每个单元各算一遍，所以分子分母口径一致。

**第 2 步：80 就是这么来的**

`ALU2_UTILIZATION` 这个独立计数器说，上面这个速率占硬件上限的 98.50%。于是：

```
ceiling = 78.8038 / 0.9850 = 80.0000 slots/clk
```

得到**精确整数**。两个计数器（一个事件计数、一个时间占比）本来没理由凑出整数，
凑出来了就是模型正确的证据。

**第 3 步：每 core 4 个 slot**

```
80 slots/clk ÷ 20 cores = 4.0 slots/clk/core     (整数)
4 × 512 FLOP            = 2048 FLOP/clk/core
```

**第 4 步：拆到单个 XMX**

注意 ALU2 是 **XMX** 管线，所以这里的除数是 XMX 的数量，不是 XVE ——
两者都是每 Xe core 8 个，全卡都是 160，数值巧合相同，但概念不同：

```
80 slots/clk ÷ 160 XMX = 0.5 slot/clk/XMX
                       = 每个 XMX 每 2 个周期出 1 行
                       = 256 MAC / 2 clk = 128 MAC/clk/XMX = 256 FLOP/clk/XMX
每 core 8 个 XMX      → 8 × 256 = 2048 FLOP/clk/core ✓
```

推论：一条 `dpas.8x8`（8 行）占住一个 XMX **16 个周期**。

**旁证 —— 但要分清哪条是独立的**

| # | 旁证 | 独立吗 |
|---|---|---|
| 1 | `256.0000`（MAC/slot）是整数 | ✅ **独立**，纯事件计数 $MNK/\texttt{ALU2\_events}$ |
| 2 | `80.0000`（上限）、`4.0`（每 core）是整数 | ❌ **恒等式**，驱动定义如此，必然是整数 |
| 3 | 反算 117 TF/s：$\frac{117\text{e}12}{20\times2.85\text{e}9}=2052.6\approx 2048$ | ✅ **独立**，来自 Intel 公布的 B580 规格，与计数器无关 |
| 4 | 所有采样从未越过 80 | ⚠️ 弱，等价于说 `UTILIZATION` 从未超过 100% |

**真正支撑 98.3 TF/s 的是第 1 和第 3 条**：
我们实测 1 slot = 256 MAC，Intel 的规格隐含 2048 FLOP/clk/core，两者一致；
再乘实测频率 2.4 GHz。第 2、4 条只是自洽性检查。

对照：ALU0/ALU1 走 XVE，**执行槽**上限是每 XVE 每周期 1 个，全卡 **160 slots/clk**，
除数才是 160 个 XVE。两条上限互相独立，这也是 ALU1-bound 能和 compute-bound 分开判定的原因。
（注：这里说的是**执行槽**，不是指令发射；`ISSUED` 的上限仍然未知，见实验 10。）

### 1.8 完整峰值推导链

```
16 lanes × 16 K = 256 MAC = 512 FLOP / slot
      × 4 slots/clk/core   = 2048 FLOP/clk/core
      × 20 cores           = 40960 FLOP/clk
      × 2.4 GHz            = 98.3 TF/s
```

---

## 第二部分：瓶颈判定方法论

第一部分给了四个硬件常量（256 MAC/slot、80 slots/clk、160 slots/clk、98.3 TF/s），
这一部分把它们变成可操作的流程：**拿到一个 shape 和一个 config，怎么知道它卡在哪里、
还有没有搞头。**

顺序是有意的：

```
2.1  先算预算（一半不用跑 profiler）
2.2  kernel 体检表          <- 不做 bound 判定也能发现的问题
2.3  避开五个已知陷阱
2.4  搞清楚「做完了」的定义        <- 最容易误判的一节
2.5  确定优化目标
2.6  四条天花板的公式与可信度
2.7  采集后的判定流程
2.8  时间预算分解            <- 实战里用得最多的
2.9  模板缺陷的四种形态
2.10 prefill / decode 为何分化
2.11 哪些结论可信、哪些是推测
2.12 所有实测数据（本文一切百分比的出处）
```

本部分的公式、阈值、反例数据全部就地给出，不需要翻第四部分。
文中的「（实验 N）」只是告诉你那个数字是在哪次实验里量出来的。

### 2.1 拿到一个 shape，先列预算表

**在跑任何 profiler 之前**，只凭 $(M,N,K)$ 和 config $(BM,BN,BK,ns,nw)$ 就能把一部分数字算出来；
其余的需要一次采集。每一栏都标明要不要 profiler。

#### 第一栏：DPAS 预算

$$\texttt{ALU2\_min} = \frac{M\cdot N\cdot K}{256}
\qquad
\texttt{ALU2\_pred} = \frac{\lceil \tfrac{M}{BM}\rceil BM \cdot \lceil \tfrac{N}{BN}\rceil BN \cdot \lceil \tfrac{K}{BK}\rceil BK}{256}$$

$$\text{预测浪费} = \frac{\texttt{ALU2\_pred}}{\texttt{ALU2\_min}}
\qquad
\text{实测浪费} = \frac{\texttt{ALU2\_events}}{\texttt{ALU2\_min}}$$

**这两个不是同一回事，别混用。**

| | 要 profiler？ | 能发现什么 |
|---|---|---|
| 预测浪费 | **不要**，纯算术 | 只有 **tile 边缘 padding** |
| 实测浪费 | **要** | 全部，含调度层面的浪费 |

差别有多大？拿实验 8 的 10 个 shape（oneDNN，假设 tile $256\times256$、$BK{=}32$）对比：

| shape | 边长能整除？ | 预测浪费 | 实测浪费 | 解析式漏掉 |
|---|---|---:|---:|---:|
| $(8192,8192,8192)$ | 是 | 1.0000 | 1.0000 | — |
| $(4096,4096,4096)$ | 是 | 1.0000 | **1.0156** | **1.0156** |
| $(4096,11008,4096)$ | 是 | 1.0000 | **1.0156** | **1.0156** |
| $(2048,2048,2048)$ | 是 | 1.0000 | **1.0938** | **1.0938** |
| $(3072,3072,3072)$ | 是 | 1.0000 | **1.0417** | **1.0417** |
| 其余 5 个 | 是 | 1.0000 | 1.0000 | — |

**解析式对 10 个 shape 全部预测 1.0000**（边长都能被 256 整除），但实测有 4 个不是。

漏掉的是**波数量化**：oneDNN 用 **20 个常驻 workgroup**
（`gemm_kernel[SIMD16 {20;1;1} ...]`，每个 Xe core 一个），每个 workgroup 循环
$\lceil T/20\rceil$ 次（$T$ = tile 总数）。$T$ 不是 20 的整数倍时，末波只有部分
workgroup 有真活，其余照样发 dpas，结果写回时丢弃：$\text{浪费} = \dfrac{20\lceil T/20\rceil}{T}$。
这跟 tile 边缘对不齐无关 —— 解析式里根本没有「有多少个 workgroup」这个维度。

> **这个信号该怎么用 —— 注意别用错。**
> 它**不是**「对手在做无用功，我不做就赢了」。普通 kernel 用同样的 tile 也要跑
> $\lceil T/20\rceil$ 波，末波空闲的 core 只是**不计数**而已，时间照付。
> $\dfrac{20\lceil T/20\rceil}{T}$ 同时是 dpas 浪费系数**和**时间代价。
>
> 它真正的用途是**检测负载不均衡**：算对手的**有效 MAC/slot**
> $= \dfrac{MNK}{\texttt{ALU2\_events}}$，不等于 256 就说明它的 tile 数没落在波边界上。
>
> **但别用它估收益。** $(2048,2048,2048)$ 上 oneDNN 的调度损失是 9.4%，
> 实测 Triton 换 tile 后只拿到 **2.5%**（理论余量的 28%），
> 而且排序主要由 `num_warps` 决定而不是 $T$。完整实测与教训见实验 8b。

**其实是三个量，不是两个 —— 第三个两个计数器都看不见：**

| 量 | 怎么得到 | 覆盖什么浪费 | 对谁成立 |
|---|---|---|---|
| **预测浪费** $\dfrac{\texttt{ALU2\_pred}}{\texttt{ALU2\_min}}$ | 纯算术 | 只有 **tile 边缘 padding** | 所有实现 |
| **实测浪费** $\dfrac{\texttt{ALU2\_events}}{\texttt{ALU2\_min}}$ | 计数器 | tile padding **+ 被真正执行的废 tile** | 所有实现 |
| **调度代价** $\dfrac{20\lceil T/20\rceil}{T}$ | 纯算术（$T$ 由 tile 定） | **末波空转** | 所有实现 |

> **最容易搞错的一点：实测浪费只对 persistent kernel 能看见调度损失。**
>
> oneDNN 是 20 个常驻 workgroup，末波补出来的 tile **真的执行了 dpas**，所以计数器看得见（1.0938）。
> Triton 是非 persistent 的，直接启动 $T$ 个 workgroup，末波空闲的 core **什么都不做** ——
> 计数器读数永远是 **1.0000**，可它照样要等那一波跑完。
>
> 实测（$2048^3$，Triton 三个 tile 全部 `ALU2_events` $=MNK/256$ 精确 1.0000）证实了这点。
> **所以对 Triton 这类实现，调度损失必须用 $\dfrac{20\lceil T/20\rceil}{T}$ 算，不能指望计数器。**

**用法：**

- **预测浪费** —— 搜索阶段的**前置剪枝**：便宜、能批量算，跑 benchmark 之前
  就能把 $M{=}4$ 配 $BM{=}64$ 这种明显不匹配的 config 剔掉
- **调度代价** —— 同样**不用 profiler**，也该进剪枝：$T$ 落在 20 的整数倍附近最好。
  但**它只能排序、不能估收益**（实验 8b：预测 7.7%，实测只拿到 2.5%）
- **实测浪费** —— 诊断**对手**用的：$\ne 256$ 说明对方在算废 tile；
  诊断自己（非 persistent）时它恒等于 1.000，没有信息量
- **预测浪费与实测浪费的差值**：只在 persistent kernel 上有意义，
  差值 $>0$ 即末波空转量；对非 persistent 实现这个差恒为 0

#### 第二栏：每线程每 K 迭代的 dpas 条数

$$D = \frac{BM \cdot BN \cdot BK}{nw \cdot 2048}$$

推导：一条 `dpas.8x8` 覆盖 $8$ 行 $\times$ $16$ 列 $\times$ $16$ 个 K 步 $= 2048$ 个输出-K 组合，
一个 workgroup 的 tile 有 $BM\cdot BN\cdot BK$ 个，分给 $nw$ 个线程。

与 ASM 实数对照，**两个 config 都精确吻合**：

| config | 公式 $D$ | ASM 数出来 |
|---|---:|---:|
| $BM{=}128, BN{=}64, BK{=}16, nw{=}8$ | 8 | **8** |
| $BM{=}256, BN{=}256, BK{=}32, nw{=}32$ | 32 | **32** |

#### 第三栏：指令预算

指令数没有算法下界，但有**实现下界**。

先看一个真实的 K 循环长什么样 —— 从 oneDNN 的 `(256,4096,4096)` kernel 里数出来的
**18 条非 dpas 指令**：

| 类别 | 条数 | 具体 |
|---|---:|---|
| 真实 block2D 加载 | 3 | 1 个 A（`d16`）+ 2 个 B（`d16v` VNNI） |
| **prefetch** | **2** | `load_block2d` 目标为 `null` |
| 指针 / descriptor 推进 | 6 | 5 `add` + 1 `mov` |
| **systolic 同步** | **4** | 3 `sync.nop` + 1 `sync.allwr` |
| **workgroup 同步** | **2** | `sync.bar` + `send.gtwy` |
| 循环控制 | 1 | `jmpi` |
| **合计** | **18** | |

> 注意 prefetch、脉动阵列同步、workgroup 屏障这三类共 8 条，占了开销的近一半。
> **它们不是可选项** —— 没有 prefetch 就无法隐藏 DRAM 延迟，没有 `sync` 就无法保证
> 脉动阵列的数据依赖。任何「理论下限」如果不含这些，都是不可达的。

于是每条 dpas 摊到的指令数下限。**先把这 18 条按「随循环展开复制与否」分两类**：

| 类别 | 条数 | 展开 $U$ 倍后 |
|---|---:|---|
| 真实 block2D 加载 | 3 | ×$U$（每个 K 步都要读） |
| prefetch | 2 | ×$U$ |
| 指针 / descriptor 推进 | 6 | ×$U$ |
| systolic 同步 | 4 | ×$U$ |
| workgroup 同步 | 2 | ×$U$ |
| **小计 $C_{\text{var}}$** | **17** | **不摊薄** |
| 循环控制 `jmpi` | 1 | ×1（整个循环体只付一次） |
| **小计 $C_{\text{fix}}$** | **1** | **摊薄** |

$$\left(\frac{\texttt{instr}}{\texttt{dpas}}\right)_{\min} = 1 + \underbrace{\frac{C_{\text{var}}}{D}}_{\text{每个 K 步都要付}} + \underbrace{\frac{C_{\text{fix}}}{U \cdot D}}_{\text{每个循环体付一次}}, \qquad C_{\text{var}}=17,\ C_{\text{fix}}=1$$

$C=18$ 取自上表的 oneDNN 实测（$256\times256$ tile）。$D$ 越大（tile 越大），
这 18 条固定开销被摊得越薄 —— 这就是大 tile 的价值。
$C$ 会随 tile 增大略有上升（加载消息变多），但同步和控制部分基本固定。

> **关于循环展开 $U$：它几乎不动下限。**
> 编译器常会把 K 循环展开（实验 9 的 `make_block_ptr` 是 $U=2$，TensorDesc LHS 是 $U=4$），
> 但**只有循环控制这一条被摊薄** —— 加载、prefetch、坐标推进、同步全部跟着复制。
>
> ASM 证据（`make_block_ptr` 的 K 循环）：第 782 行用坐标 `r169.8` 加载 A，
> 784 行把坐标改成 `r169.9`，790 行**再加载一次** A；4 条 prefetch = 2 个 K 步 × 2 条；
> 真正只出现一次的只有 5 条（`add` 计数 + `cmp` + 2×`jmpi` + 末尾 `mov`）。
> $U=2$ 本身也有硬证据：循环计数器 `add r254.7 += 2`，退出条件 `cmp r254.7 == 448`，
> 而 $448 = K/BK = 14336/32$。
>
> 数值上，$D=16$ 时：$U{=}1 \to 2.125$、$U{=}2 \to 2.094$、$U{=}4 \to 2.078$，相差 2%。
> **所以实用上直接用 $1+18/D$ 就行**，不必为了 $U$ 去 dump 汇编。
> （早期版本写过 $1+C/(U{\cdot}D)$，**那是错的** —— 它把 17 条不能摊薄的开销也算成了可摊薄，
> 会把下限压得太低、把膨胀倍数抬得太高。）

#### 第四栏：与实测对账

从计数器算实际值：

$$\frac{\texttt{instr}}{\texttt{dpas}} = \frac{\texttt{XVE\_INST\_ISSUED\_ALL}}{\texttt{XVE\_INST\_EXECUTED\_ALU2\_ALL} / 8}$$

**这个比值也可以直接从汇编算**（循环体总指令数 ÷ 循环体 dpas 数），两条路径互相印证：

| kernel | ASM 直接算 | 计数器算 | 偏差 |
|---|---:|---:|---:|
| oneDNN | $50/32 = 1.5625$ | 1.52 | 2.7% |
| Triton | $95/8 = 11.875$ | 11.74 | 1.1% |

> **ASM 值总是略高于计数器值**，因为 `sync.nop` / `goto` / `join` 这类指令写在汇编里
> 但不产生发射。偏差通常 1–7%，逐项能对上（推导见实验 9 的说明框）。
> **判定用计数器值，定位用 ASM 值。**

实测结果（下限按 $1 + 17/D + 1/(U{\cdot}D)$）：

| kernel | $D$ | $U$ | 下限 | 实测 instr/dpas | 超出倍数 |
|---|---:|---:|---:|---:|---:|
| $8192^3$ oneDNN | — | — | — | **1.50** | — |
| $(256,4k,4k)$ oneDNN | 32 | 1 | 1.56 | **1.52** | **0.97x** |
| $(256,4k,4k)$ Triton | 8 | 1 | 3.25 | **11.74** | **3.61x** |
| $(1024,4k,14336)$ Triton `make_block_ptr` | 16 | 2 | 2.09 | **2.45** | **1.17x** |
| $(1024,4k,14336)$ Triton TensorDesc LHS | 16 | 4 | 2.08 | **19.84** | **9.55x** |
| $(4,4k,4k)$ Triton | 8 | 1 | 3.25 | **26.89** | **8.27x** |
| $(4,4k,4k)$ oneDNN | — | — | — | 8.18 | — |

注意最后两行的 $U$ 对下限几乎没影响（2.09 vs 2.08）—— 如前所述，展开只摊薄了 1 条循环控制。

**oneDNN 精确落在自己的下限上（0.97x）**，说明 $C=18$ 就是这类 kernel 的真实开销，
不是拍脑袋的数。Triton 高 3.6–8.3 倍。

这就是「指令过多」的量化判据 —— 之前只能靠 ASM 目测 DPAS 密度，现在有了可以直接从
计数器算、不用 dump 汇编的数字。

#### 第五栏：多出来的指令到底是什么

把两边的 K 循环拆开对比，差距全部集中在 **descriptor 维护**上：

| | oneDNN | Triton |
|---|---:|---:|
| 访存消息数 | 5 | 4 |
| 地址 / descriptor 维护指令 | **6** | **76** |
| 每条消息摊到 | **1.2** | **19.0** |
| 比值 | — | **15.8x** |

oneDNN 的 6 条是 5 个 `add` + 1 个 `mov`，**纯粹是增量更新坐标** ——
descriptor 在循环外建好，循环内只把 X/Y 坐标往前推。

Triton 的 76 条里，有 **15 条是把写死的立即数写进 descriptor 寄存器**：

```
(W) mov (1|M0)  r118.3<1>:ud  31:w          <- block width-1
(W) mov (1|M0)  r118.4<1>:ud  8191:w        <- surface width-1 = 4096*2-1
(W) mov (1|M0)  r118.6<1>:ud  0:w           <- offset
(W) mov (1|M0)  r118.7<1>:ud  0x1F0F:uw     <- block height/width 编码
(W) mov (1|M0)  r35.3<1>:ud   15:w
(W) mov (1|M0)  r35.4<1>:ud   8191:w
...共 15 条，分属 r118 / r35 / r119 / r120 四个 descriptor
```

`8191` 出现 4 次，那是 B 矩阵的行跨距（字节）减一，**从头到尾就没变过**。
四个 descriptor 寄存器每次 K 迭代都从零重建一遍。

代价：

$$15\ \text{条} \times \underbrace{256}_{K/BK} \times \underbrace{1024}_{\text{线程}} = 3.93\times10^6\ \text{条纯浪费指令}$$

**准确的表述**：oneDNN 把 descriptor 当**持久对象**——建一次，循环内只推坐标；
Triton 把它当**临时值**——每轮重新算全部字段，包括那些从头到尾不变的常量。

这也解释了为什么 config 调不动这个问题：$BM$/$BN$/$BK$ 改变的是 $D$，
而这 76 条维护指令基本不随 config 变。唯一的缓解手段是把 $D$ 做大让它们摊薄 ——
这正是实验 6 中 $(256,256,32,2,32)$ 比 $(128,64,16,3,8)$ 快的原因（$D$ 从 8 变 32）。

#### 预算表算完之后

上面四栏给出的是**这个 shape 应该花多少**。把它和实测一比，就得到一组「做了多少倍多余的活」
的比值 —— 那是下一节的内容。**2.2 节的体检表是本节四栏的直接产物**，可以当作
2.1 的执行清单来用。

### 2.2 kernel 体检表：不做 bound 判定也能发现的问题

「这个 kernel 是什么 bound」是个**很难回答、而且经常不必回答**的问题：
它需要干净时间、需要所有天花板都可信（LSU 的就不可信）、还容易被表面症状带偏。

有一类问题**不需要知道 bound 就能发现**，而且发现了就直接对应到修法 ——
就是「同样的算法，它比理论最少的多做了多少」。全部写成**比值**，理想值都是 **1.00**。

先算一个不属于体检项的量：**算术强度** $\text{AI} = \dfrac{2MNK}{2(MK+KN+MN)}$，
与机器平衡点 242 比较，得到「这个 shape 理论上该落在哪一侧」。它不是浪费，是预期。

| # | 体检项 | 怎么算 | 超标说明 | 要 profiler | 直接对应的修法 |
|---|---|---|---|---|---|
| 1 | **DPAS 工作量比** | $\dfrac{\texttt{ALU2\_events}}{MNK/256}$ | tile padding / 算废 tile | 要 | 减小 $BM$/$BN$ |
| 1' | 同上的**纯算术预测** | $\dfrac{\texttt{ALU2\_pred}}{\texttt{ALU2\_min}}$ | 只覆盖 tile 边缘 padding | **不要** | 搜索阶段前置剪枝 |
| 2 | **调度代价** | $\dfrac{20\lceil T/20\rceil}{T}$，$T$ = tile 总数 | 末波空转 | **不要** | 调 tile 让 $T$ 贴近 20 的倍数 |
| 3 | **并行度** | $\dfrac{T}{20}$ | $<1$ 就有 core 全程闲置 | **不要** | 减小 tile 把 grid 撑到 $\ge$ 20 |
| 4 | **DRAM 流量比** | $\dfrac{\texttt{BYTE\_READ}}{2(MK+KN+MN)}$ | L2 blocking 差，B 被重复读 | 要 | 增大 $BM$ / 调 `GROUP_M` |
| 5 | **指令比** | $\dfrac{\texttt{instr/dpas}}{1+17/D+1/(UD)}$（2.1 第三栏） | 地址/descriptor 冗余 | 要 | 外提 descriptor（backend） |
| 6 | **访存碎片度** | $\dfrac{\texttt{L1\_ACCESS}}{\texttt{SEND}}$ | 一条消息展开成太多次 L1 访问（oneDNN $\approx$ 13） | 要 | 增大 $BK$ / 去冗余 prefetch |
| 7 | **SLM 流量** | `SLM_BYTE_READ` | $\ne 0$ 就说明没走 block2D 直出 | 要 | 换回 `make_block_ptr` |
| 8 | **寄存器溢出** | `n_spills` | $\ne 0$ 直接废 | **不要**（编译期） | 换 config |

第 1'、2、3、8 项**不跑任何东西**就能算，适合在搜索阶段批量剪枝；
第 1、4、5、6、7 项需要**一次** `unitrace -q -g ComputeBasic` 采集，一次全出。

> ### 为什么这些比 bound 判定更好用
>
> **第 1、4、5、6、7 项的分子分母都是同一次运行里的事件计数，第 1'、2、3、8 项是编译期常量。**
> **全都不含时间。**
>
> 所以 `-q -g ComputeBasic` 那个 1.14–1.55x 的减速**在比值里被约掉了**：
> 这些体检项在 `-q` 口径下**可以直接用，甚至可以跨 config 比较** ——
> 而 `*_UTILIZATION`、`XVE_STALL`、`SF_HOLD` 都不行（见 2.8 节）。
>
> **唯一需要干净时间的是最后那一步**：
> $\dfrac{t_{\text{实测}}}{\max(\text{各资源下界})}$ —— 那一步才是「bound 分析」。

**两者的分工：**

```
体检表（1–8）    回答「有没有做多余的事」   -> 每一项直接对应一个修法
bound 判定       回答「先修哪一项、能省多少」-> 需要干净时间 + 可信天花板
体检全 1.00 还慢 -> 才轮到 latency / 并行度这类「没做多余的事但也没喂饱」的解释
```

#### 四个实例：不做任何 bound 判定，问题已经一目了然

先说清楚这四个是什么 —— 全是 bf16 GEMM，$C_{M\times N} = A_{M\times K}\cdot B_{K\times N}$，
config 记法为 $(BM,\ BN,\ BK,\ \texttt{num\_stages},\ \texttt{num\_warps})$：

| 代号 | shape $(M,N,K)$ | 什么场景 | Triton config | 备注 |
|---|---|---|---|---|
| **①** | $(256,\ 4096,\ 4096)$ | 小 batch prefill | $(128,64,16,\ ns{=}3,\ nw{=}8)$ | 与 oneDNN 只差 1% |
| **②** | $(4,\ 4096,\ 4096)$ | decode，$M{=}4$ | $(16,256,32,\ ns{=}2,\ nw{=}8)$ | 比 oneDNN 慢 1.85x |
| **③** | $(1024,\ 4096,\ 14336)$ | FFN，计算受限 | $(128,256,32,\ ns{=}3,\ nw{=}32)$，`tl.make_block_ptr` | 85.4% peak |
| **④** | $(1024,\ 4096,\ 14336)$ | **与 ③ 同 shape 同 tile** | 同 ③，只把 A（`tl.dot` 的 LHS）换成 `tl.make_tensor_descriptor` | 慢 3.57x |

**③ 和 ④ 是一组对照实验**：shape、tile、`num_warps` 全都一样，**只改了 A 矩阵的访存写法**，
所以两者的差异可以完全归因到那一处改动（实验 9）。
复现脚本 `triton_tensordesc_dot_lhs_repro.py`（③ = `--variant none`，④ = `--variant lhs`）。

| 体检项 | ① $(256,4096,4096)$ | ② $(4,4096,4096)$ | ③ `make_block_ptr` | ④ TensorDesc LHS |
|---|---:|---:|---:|---:|
| 1 DPAS 工作量比 | 1.00 | **4.00** ❌ | 1.00 | 1.00 |
| 2 调度代价 | 1.09 | 1.25 | 1.09 | 1.09 |
| 3 并行度 $T/20$ | 6.4 | **0.8** ❌ | 6.4 | 6.4 |
| 5 指令比 | **3.7x** ❌ | — | 1.17x | **9.55x** ❌ |
| 6 L1/SEND | **23**（oneDNN 13） ❌ | — | — | — |
| 7 SLM 流量 | 0 | 0 | 0 | **3758 MB** ❌ |
| 实际慢多少 | 仅 1% | **1.85x** | — | **3.57x** |

几个数怎么来的（都能手算，不用查表）：

- **② 的 DPAS 4.00**：$BM{=}16$ 去装 $M{=}4$，$16/4 = 4$ 倍全是 padding 出来的空行
- **② 的并行度 0.8**：$T=\lceil 4/16\rceil \times \lceil 4096/256\rceil = 1\times16 = 16$ 个 workgroup，
  而 GPU 有 20 个 Xe core → **4 个核全程闲置**
- **① 的并行度 6.4**：$T = \lceil 256/128\rceil \times \lceil 4096/64\rceil = 2\times 64 = 128$，$128/20 = 6.4$ 波
- **① 的指令比 3.7x**：$D = \frac{128\times64\times16}{8\times2048} = 8$ → 下限 $1+\frac{17}{8}+\frac{1}{2\times8} \approx 3.19$，
  实测 `instr/dpas` $=11.74$ → $11.74/3.19 = 3.7$

② 的两个 ❌ 和 ④ 的两个 ❌ **不用任何天花板、不用干净时间**就能看出来，修法也是现成的
（② 减小 $BM$ 到 4 并把 $BN$ 减到 128 让 grid 涨到 32；④ 换回 `make_block_ptr`）。

而 ① 有两项超标却只慢 1% —— **这正是体检表管不了的部分**：
浪费是真的，但被 roofline 拐点的空隙吸收了。
「超标到底值不值得修」，那才需要 bound 分析，从 2.7 节开始。

### 2.3 五个常见陷阱

这些都是实际踩过的坑，每一条都有对应的实测反例。

#### 陷阱 1：只看 `ALU2_UTILIZATION`

- 对 compute-bound shape：它确实等于 achieved/peak，是有效指标
- 对 memory-bound shape：它**必然**很低，低不代表有优化空间

反例：$(4,4096,4096)$ 上 Triton 和 oneDNN 的 `ALU2_UTILIZATION` **都是 3%**，
但性能差 1.85 倍。真正的差别在 DRAM 速率（197 vs 358 GB/s）。

**先分类，再选指标。**

#### 陷阱 2：把「指令多」直接判成 issue-bound

指令拖慢性能有两种机制，只有第一种才配叫 X-bound：

| 机制 | 需要资源饱和吗 |
|---|---|
| **吞吐**：占满某个单元的带宽 | 要 |
| **延迟/依赖**：处在关键路径上，后面的必须等 | **不要** |

反例：$(256,4096,4096)$ Triton 的 `ALU1_UTILIZATION` 只有 43.2%（远未饱和），
但 `XVE_STALL` 有 28.1%。该 shape 实际停在 roofline 拐点（ALU2/DRAM 各约 74%），
多出来的 ALU1 和 L1 访问被空隙吸收了。指令多是**事实**，issue-bound 是**误判**。

要判「指令太多」，用 `ALU1_UTILIZATION`（分母 160 已验证），
**不要用 `ISSUED`**（上限未知）。

#### 陷阱 3：拿没有算法下界的资源当「做完了」的依据

DRAM 和 DPAS 有由 shape 唯一确定的下界，LSU 访问数和指令数没有。
在后两者上饱和，本身就说明写法有问题 —— 详见 2.4 节。

#### 陷阱 4：跨口径比较时间

L2 全热的裸循环 vs 带 cache flush 的 `do_bench`，在小 shape 上能差 **5 倍**
（实验 3）。所有比值必须来自同一进程、同一 harness。

#### 陷阱 5：用错峰值常数

本卡 BF16 峰值是 **98.3 TF/s**，不是 117（那是 2.85 GHz 的 B580）。
DRAM 用 **407 GB/s** 而不是理论值 456。用错会让效率数字系统性偏低 19%。

### 2.4 什么时候才算「优化完了」

「饱和」不等于「做完了」。停手需要**同时**满足两个条件：

```
停手 = (瓶颈资源已饱和) AND (该资源上的工作量已经是算法下界)
```

只满足前者是最常见的误判：一个 kernel 可以把某个资源用到 98%，但它在那个资源上做的
工作有一大半是多余的。

关键区别在于**哪些资源有算法下界**：

| 资源 | 算法下界 | 饱和能否作为停手依据 |
|---|---|---|
| DRAM | $2(MK+KN+MN)$ 字节 | **能**（前提：实测流量 ≈ 下界） |
| DPAS | $\dfrac{MNK}{256}$ slots | **能**（前提：无 padding 浪费） |
| LSU / L1 | **没有** | **永远不能** |
| 指令发射 | **没有** | **永远不能** |

DRAM 和 DPAS 的下界由 shape 唯一确定，跟怎么写 kernel 无关，所以「用满且用得不冤」
就是真的到头了。

LSU 访问次数和指令条数完全由代码决定，没有理论下界。**在这两者上饱和，本身就说明
写法有问题** —— 你把一个不该成为瓶颈的资源用爆了。

实测对照（浪费倍数 = 实测 ÷ 算法下界）：

| case | DPAS 浪费 | DRAM 浪费 | 瓶颈资源 | 判定 |
|---|---:|---:|---|---|
| $8192^3$ oneDNN | 1.00x | — | DPAS 98.5% | **做完了** |
| $(1024,4k,14336)$ Triton `make_block_ptr` | 1.00x | — | DPAS 85.4% | **看着像做完，其实没有**：K 循环每轮 28 条 prefetch descriptor 重建（实验 9） |
| $(4,4096,4096)$ oneDNN | 2.00x | 0.96x | DRAM 88% | **基本做完** |
| $(4,4096,4096)$ Triton | **4.00x** | 0.98x | 都不饱和 | 有空间（实测慢 1.85x） |
| $(256,4096,4096)$ oneDNN | 1.00x | 0.92x | 都约 69% | 接近拐点 |
| $(256,4096,4096)$ Triton | 1.00x | 0.94x | 拐点（ALU2/DRAM 各 ~74%） | **有空间**：L1 访问是 oneDNN 的 3.4 倍，只是在拐点被空隙吸收 |

最后一行是典型例子：Triton 的 L1 访问次数是 oneDNN 的 **3.4 倍**，而 DRAM 流量几乎相同 ——
多出来的访问全命中 L1，是**纯粹的浪费**。它现在被拐点的空隙掩盖着，所以只慢 1%，
但工作量摆在那里，换到 compute-bound 的大 shape 上就会暴露。远没到头。

### 2.5 所以目标是什么

不是「把 kernel 变成 memory-bound 或 compute-bound」——bound 类型由 shape 的算术强度决定，
不是你能选的。正确的表述是：

```
目标 = 让 shape 回到它本来该在的 bound 上，并且在那个资源上不浪费
```

具体到三步：

1. **消除工作量浪费**：DPAS 浪费 → 调 tile；DRAM 浪费 → 调 `BLOCK_M`/`GROUP_M`
2. **消除虚假瓶颈**：LSU 饱和或 issue 饱和 → 减少访存消息数和地址指令
3. **确认落回真实瓶颈**：$\text{AI} < 242$ 应落在 DRAM 上，$\text{AI} > 242$ 应落在 DPAS 上

三步走完，若瓶颈资源饱和且工作量已是下界，才是真的停手。

还剩最后一层：**换算法**。roofline 只对「这个算法」成立，改变算法可以移动下界本身
（量化降低流量、算子融合省掉中间张量、split-K 改变并行结构）。这属于另一个层面的优化。

### 2.6 四个天花板加一个开放项

对给定 shape 和 config，算出下面四个下界，实际时间不可能低于其中最大的那个：

| 天花板 | 公式 | 证据强度 |
|---|---|---|
| $T_{\text{compute}}$ | $\dfrac{2MNK}{98.3\times10^{12}}$ | **中高**：256 MAC/slot 为实测，80 slots/clk 为厂商常数（见 1.7 节） |
| $T_{\text{memory}}$ | $\dfrac{2(MK + g_m \cdot KN + MN)}{407\times10^9}$ | **高**：407 GB/s 由 stream 基准实测 |
| $T_{\text{ALU1}}$ | $\dfrac{\texttt{ALU1 slots}}{160 \times f}$ | **中高**：160 是厂商常数（驱动 metric 定义），非独立实测 |
| $T_{\text{LSU}}$ | $\dfrac{\texttt{L1访问次数}}{82 \times f}$ | **中**：上限未知，82 是实测封顶值，配合 `SF_HOLD` 使用 |
| $T_{\text{latency}}$ | — | 开放项：无解析式，看 occupancy 与 `XVE_STALL` |

> **不要把这几条当同等可靠。** 前三条有硬证据，$T_{\text{LSU}}$ 的分母是实测封顶而非真实上限，
> 所以它的百分比只能做横向对比，不能当绝对饱和度。详见 2.11 节。
>
> 既往版本列过一条 $T_{\text{issue}}$（分母 160），**已删除** —— 实验 10 证明 160 属于
> ALU0/ALU1 执行槽，`ISSUED` 的上限仍然未知。要判「指令太多」请用 $T_{\text{ALU1}}$。

机器平衡点：

$$\frac{98.3\times10^{12}}{407\times10^{9}} = 242\ \text{FLOP/byte}$$

算术强度 $\text{AI} = \dfrac{2MNK}{2(MK+KN+MN)}$，$\text{AI} < 242$ 为内存受限。

### 2.7 判定流程

```
第 0 步：算 AI，与 242 比较，得到「理论上应该是什么 bound」

第 1 步：采集 unitrace -q -g ComputeBasic

第 2 步：检查 DPAS 是否被浪费
        ALU2_events / (M·N·K/256)
        > 1.2  -> tile padding 浪费，先改 BLOCK_M/N/K

第 3 步：检查 DRAM 流量是否被放大
        GPU_MEMORY_BYTE_READ / 理论下界
        > 1.3  -> L2 blocking 差，改 BLOCK_M 或 GROUP_M

第 4 步：分类判定
```

| 观察 | 判定 | 该做什么 |
|---|---|---|
| `ALU2_UTILIZATION` > 85% | **compute 饱和** | 若 DPAS 浪费 $>1.0$：改 tile 消除 padding；<br>浪费已是 1.0 → **config 到头**，只剩换算法 |
| DRAM 速率 > 350 GB/s（>86% of 407） | **memory 饱和** | 若 DRAM 浪费 $>1.0$：改 `BLOCK_M`/`GROUP_M`；<br>浪费已是 1.0 → **config 到头**，只剩量化/融合 |
| `SF_HOLD` > 25% **且** L1 访问数远高于参照 **且** DRAM 流量相同 | **LSU-bound** | 减少 L1 访问（更大 tile、更大 `BK`、去冗余 prefetch）<br>**永远有空间**，LSU 没有算法下界 |
| `ALU1_UTILIZATION` > 85% | **ALU1-bound**（标量管线饱和） | 减少地址/descriptor 指令<br>**永远有空间**，指令数没有算法下界。<br>这就是口语里说的 issue-bound，但请用 `ALU1_UTILIZATION`（分母 160 已验证），**不要用 `ISSUED`**（上限未知） |
| 上述都不饱和 + `XVE_STALL` 高（occupancy 高低都可能） | **latency-bound** | **先看 occupancy 再决定**：<br>低（<50%）→ 加并发（更多 workgroup、减寄存器、加深 prefetch）<br>高（>80%）→ 并发不是瓶颈，必须**缩短依赖链本身**（去掉 SLM 中转等长链），config 层面通常无解<br>实测案例见实验 9，occupancy 81.8% 属于后者 |

> ### ⚠ 这五类里，只有三类有实测样本
>
> | 判定 | 实测案例 | 证据强度 |
> |---|---|---|
> | compute 饱和 | $8192^3$ oneDNN（UTIL 98.5%，浪费 1.00） | ✅ 确凿 |
> | memory 饱和 | $(4,4096,4096)$ oneDNN（DRAM 88%，达访存 roofline 91.5%） | ✅ 确凿 |
> | **LSU-bound** | **无**。唯一候选 $(256,4096,4096)$ 已改判为拐点 | ❌ **纯推演** |
> | **ALU1-bound** | **无**。实测最高 49.3%，从未饱和 | ❌ **纯推演** |
> | latency-bound | TensorDesc LHS（实验 9，occupancy 高那一支）<br>$(4,4096,4096)$ Triton（并行度不足那一支，16 WG < 20 core） | ✅ 两支各一例 |
>
> **后两类的阈值（`SF_HOLD > 25%`、`ALU1_UTILIZATION > 85%`）是按定义推的，没有校准过。**
> 遇到疑似情形，别直接套结论，按 2.8 节做完整的时间预算分解。
>
> ### 三种浪费：不是 bound 类型，是叠加项
>
> 还有一类东西**不在上面那张表里，但比 bound 本身更常见**。它们不回答「卡在哪个资源」，
> 只回答「同样的活为什么干得更多」，可以叠加在任何一种 bound 之上：
>
> | 浪费 | 判据 | 例子 |
> |---|---|---|
> | **DPAS 工作量浪费** | `ALU2_events/(MNK/256) > 1.2` | $(4,4096,4096)$ Triton：$BM{=}16$ 装 $M{=}4$ → 4.0x |
> | **DRAM 流量浪费** | `BYTE_READ / 理论下界 > 1.3` | $BM$ 太小 → B panel 被重复读 $g_m$ 次 |
> | **指令浪费** | `instr/dpas ÷ 下限 > 1.2`（下限见 2.1） | Triton 的 descriptor 每轮重建，稳定 1.45–1.49x |
>
> 判定流程的第 2、3 步就是查前两种，别跳过 —— 一个 kernel 完全可能是
> 「并行度不足 + DPAS 浪费 4 倍」两个问题并存（$(4,4096,4096)$ Triton 就是）。

> ### ⚠ 指令浪费（descriptor 重建）**尚未**在任何实例中成为主约束
>
> 「descriptor 每轮重建 → 指令变多」是 Triton XPU backend 已确认的缺陷，
> 但它是**指令浪费**，不是一种 bound。它有没有变成瓶颈要另外判：
>
> | 案例 | instr/dpas | 超出下限 | ALU1 | **实际判定** | 指令浪费吃掉多少 |
> |---|---:|---:|---:|---|---|
> | $(256,4096,4096)$ Triton | 11.74 | — | 49.3% | roofline 拐点 | **~1%**，被拐点的空隙吸收 |
> | $(1024,4k,14336)$ `make_block_ptr` | 2.45 | 1.17x | 6.1% | compute-bound | ~15% |
> | $2048^3$ Triton $64\times64$ | 3.11 | 1.49x | 9.93/clk | compute-bound（仍**赢** oneDNN 2.5%） | 14.8% |
> | $(1024,4k,14336)$ TensorDesc LHS | 19.84 | **9.55x** | 17.6% | **latency-bound** | 指令不是瓶颈，SLM 依赖链才是 |
>
> **四个实例，指令浪费一次都没成为主约束。** 最极端的 TensorDesc LHS 指令多出 9.55 倍，
> 瓶颈却是依赖链延迟 —— **指令多 $\ne$ issue-bound / ALU1-bound**（见 2.3 节陷阱 2）。
> 这也正是 LSU-bound / ALU1-bound 两类至今没有实测样本的原因。
>
> 正确用法：**先做时间预算分解定 bound，再用 `instr/dpas` 量化「还剩多少能捡」**。
> 反过来（看见指令多就判 ALU1-bound）是本文档记录过的典型误判。

> **「饱和」不等于「到头」。** 前两行的饱和只说明**这个资源用满了**，接下来要看
> **用得冤不冤**：如果工作量还没到算法下界（有 padding 或重复读），减少工作量依然能加速，
> 而且往往就是改 config；只有工作量也到下界了，config 层面才真的到头。
> 完整规则见 2.4 节的两级停手条件。
>
> 后两行不一样：LSU 访问数和指令数**没有算法下界**，在它们上面饱和本身就是写法问题，
> 永远不构成「到头」。

### 2.8 判据速查

#### 最实用的做法：时间预算分解

**先说采集规程 —— 这一步做错，后面全错：**

```
第 1 步  时间：unitrace -d（或进程内 300 次摊销）      <- 绝不用 -q 的 GpuTime
第 2 步  事件：unitrace -q -g ComputeBasic            <- 只取事件计数
第 3 步  自己重算利用率，别直读：
         UTILIZATION_干净 = events / (t_干净 × f × ceiling)
```

> **核心规则：能重建的自己重建，不能重建的只做定性对比。**
>
> | 类别 | 指标 | 怎么用 |
> |---|---|---|
> | **事件计数** | `ALU2`/`ALU1`/`ISSUED`/`SEND`/`LOAD_STORE_CACHE_ACCESS`/`GPU_MEMORY_BYTE_READ` | 直接用，跨口径不变 |
> | **可重建** | 所有 `*_UTILIZATION[%]`、任何 `events/clk` | **必须用干净时间重算**，`-q` 直读会系统性偏低 |
> | **不可重建** | `XVE_STALL`、`SF_HOLD`、`occupancy`、`XVE_ACTIVE` | 独立的时间占比测量，无干净口径对应物；**只用于大幅度定性对比** |
>
> `-q` 直读的值**没有错**，它如实描述了那次被拖慢的执行；
> 只是那次执行不是你要优化的那次。所以是**重建**，不是弃用。
>
> 差别有多大：$2048^3$ 上 $128\times128$ 的三个 config，
> `-q` 直读 ALU2U% = 55.3 / 59.9 / 60.8，重建后 **85.8 / 73.0 / 80.5** —— **排序相反**。

> **为什么必须分两次跑。** `unitrace -d` 与进程内摊销互相验证到 **0.4%**
> （$2048^3$ oneDNN：212.82 vs 213.69 μs），可以放心用。
> 但 `-q -g ComputeBasic` 会把 kernel **真的拖慢 1.14–1.55 倍，且每个 config 不同**，
> 它报的 `GpuTime` 是那次被拖慢的真实耗时 —— 数值没错，但不是你要的那个量。
> 详见第三部分的工具警告。

然后把干净耗时按各资源的**最低需求**拆开：

| 资源 | 需要的 clk | 上限可信度 |
|---|---|---|
| ALU2（XMX） | $\texttt{ALU2\_events} / 80$ | **高**，反推得出，任何口径可用 |
| ALU1（标量） | $\texttt{ALU1\_events} / 160$ | **高**，同上 |
| DRAM | $\texttt{GPU\_MEMORY\_BYTE\_READ} / 407\text{e}9 \times f$ | **高**，stream 实测（未 profiling） |
| LSU（L1） | **没有可信上限** | 只报**速率**（次/clk），横向对比，不要算百分比 |

$(256,4096,4096)$ 实例（干净时间 117.3 / 116.1 μs，事件取自 `-q` 采集）：

| 资源 | Triton | oneDNN | 说明 |
|---|---:|---:|---|
| ALU2 | **74.7%** | **75.3%** | 几乎一样 |
| DRAM | **74.4%** | **73.6%** | 几乎一样 |
| ALU1 | **49.3%** | 1.1% | Triton 是 45 倍 |
| LSU 速率 | **89.70 次/clk** | 26.47 次/clk | **3.38 倍** |
| `SF_HOLD` | 40.7% | 1.9% | 21 倍 |

**读法（这个结论是 2026-08-13 修正过的）**：

两个实现的 **ALU2 和 DRAM 都在 74–75%**，谁都没饱和，耗时也几乎相同（117.3 vs 116.1）。
该 shape 的 $\text{AI} = 227.6$，正好压在机器平衡点 242 上 —— **两者都停在 roofline 拐点**，
剩下的 25% 是计算与访存重叠不完美。

Triton 额外背了 **3.4 倍的 L1 访问**和 **45 倍的 ALU1**，却只慢 1% ——
**因为拐点处 ALU2/DRAM 各有 25% 空闲，这些额外开销被塞进空隙藏住了**（机制见 2.10 节）。

> **旧版本这里写「Triton 是 LSU-bound，占耗时 95.8%」，那个数字不能用了。**
> 它是拿 profiling 口径的时间除以 profiling 口径的封顶值 82 得到的 ——
> 两边口径自洽所以算式没错（干净口径重算是 95.7%），
> 但 **82 根本不是 LSU 的硬件上限**：干净口径下 Triton 自己就跑到 89.70 次/clk。
> 没有可信上限，就没法说「LSU 饱和了百分之多少」。
>
> **不依赖上限、依然成立的结论**：Triton 的 L1 访问是 oneDNN 的 3.38 倍，
> 而两者 DRAM 流量几乎相同（35.5 vs 34.8 MB）—— 多出来的访问全命中 L1。
> 这是**真实的浪费**，也是这个 shape 上唯一够得着的优化点；
> 只是在拐点处它被空隙吸收了，换到 compute-bound 的大 shape 上就会暴露（见 2.10 节）。

#### 单项指标阈值

| 指标 | 饱和阈值 | 含义 |
|---|---|---|
| `XVE_INST_EXECUTED_ALU2_ALL_UTILIZATION[%]` | > 85% | 算力打满（XMX）。**必须用干净口径重建值**，`-q` 直读会系统性偏低 |
| `GPU_MEMORY_BYTE_READ_RATE[GBpS]` | > 350 | 带宽打满（DRAM）。**注意这也是速率**：`-q` 直读会偏低，而参照值 407 GB/s 是干净口径实测的，两者不可直接比 —— 用 $\texttt{GPU\_MEMORY\_BYTE\_READ} / t_{\text{干净}}$ 自己算 |
| `XVE_SHARED_FUNCTION_ACCESS_HOLD[%]` | > 25% | 卡在 LSU 等共享单元 —— **LSU-bound 的主判据** |
| `LOAD_STORE_CACHE_ACCESS / GpuCoreClocks` | **无可信上限** | 干净口径实测已达 89.70；只报速率、只做横比，**不要算百分比** |
| `XVE_INST_EXECUTED_ALU1_ALL_UTILIZATION[%]` | > 85% | 标量管线打满（分母 160，厂商常数）。**同样要用重建值** |
| `XVE_INST_ISSUED_ALL / GpuCoreClocks` | ？ | 上限未知，实测最高 80.25。**不要除以 160** |
| `XVE_THREADS_OCCUPANCY_ALL[%]` | < 50% | 并发不足。**注意反向不成立**：occupancy 高**不能**排除 latency-bound（实验 9 的 latency-bound 案例 occupancy 有 81.8%），它只决定「加并发」这条路通不通 |
| `XVE_STALL[%]` | > 60% | 等待为主 |
| `SLM_BYTE_READ[bytes]` | > 0 | GEMM 里出现 SLM 流量通常是 layout 转换退化的信号；**SLM 不计入 `LOAD_STORE_CACHE_ACCESS`**，别被低 LSU 读数误导 |
| `ALU2_events / (M·N·K/256)` | > 1.2 | tile padding 浪费 |
| `GPU_MEMORY_BYTE_READ / 理论下界` | > 1.3 | 重复读 B |

实测参考值（完整数据见 2.12 节）：

| kernel | LSU | ALU2 | ALU1 | DRAM | SF_HOLD | 判定 |
|---|---:|---:|---:|---:|---:|---|
| $8192^3$ oneDNN | 43.2% | **98.5%** | 低 | 低 | 0.7% | compute-bound |
| $(256,4096,4096)$ oneDNN | 26.5/clk | 75.3% | 1.1% | 73.6% | 1.9% | 拐点，重叠不完美 |
| $(256,4096,4096)$ Triton | 89.7/clk | 74.7% | 49.3% | 74.4% | 40.7% | 拐点，L1 访问 3.4x |
| $(1024,4k,14336)$ TensorDesc LHS | 低 | 23.9% | 低 | 12% | 5.9% | **latency-bound** |
| $(4,4096,4096)$ oneDNN | 低 | 3.0% | 低 | **88%** | 0.9% | memory-bound |

### 2.9 模板写得差，会把 memory-bound 变成 compute-bound 吗？

**会，但严格说是变成另外两种 bound**，需要分三种情况：

#### 情况 A：DPAS 工作量放大 → 真的变成 compute-bound

`BLOCK_M >> M` 或 `BLOCK_N >> N` 时，padding 出来的部分照样跑 dpas。

实验 7 中 $M=4$ 用 $BM=16$ → DPAS 量变 4 倍。若浪费倍数足够大，$T_{\text{compute}}$ 会超过 $T_{\text{memory}}$，本该内存受限的 shape 变成算力受限。

判据：`ALU2_events / (M·N·K/256) > 1.2`
修复：减小 `BLOCK_M`，或换 decode 专用模板

#### 情况 B：DRAM 流量放大 → 仍是 memory-bound，但天花板抬高

`BLOCK_M` 太小 → $g_m = \lceil M/BM \rceil$ 变大 → B 被重复读 $g_m$ 次。

| BM | $g_m$ | 最坏 DRAM | 访存下界 |
|---:|---:|---:|---:|
| 32 | 8 | 272.6 MB | 597.9 μs |
| 64 | 4 | 138.4 MB | 303.5 μs |
| 128 | 2 | 71.3 MB | 156.4 μs |
| 256 | 1 | 37.8 MB | 82.8 μs |

L2（18 MB）配合 `GROUP_M` 分组可以吸收一部分，但 B panel 超过 L2 时就吸收不了。

判据：`GPU_MEMORY_BYTE_READ / 理论下界 > 1.3`
修复：增大 `BLOCK_M` 或调整 `GROUP_M`

#### 情况 C：指令/访存消息膨胀 → 变成 LSU-bound 或 ALU1-bound（最容易误判成 compute-bound）

> **⚠ 这一类目前只是「有嫌疑」，还没有确证的实例。**
> 唯一的候选 $(256,4096,4096)$ 在改用干净口径后被重新判定为**拐点**（见 2.8 节），
> ALU1-bound 则从未观测到（最高只到 49.3%）。
> 下面描述的是**机制推演 + 已确认的工作量浪费**，不是已坐实的瓶颈类型。

descriptor 每次 K 迭代重建 → 大量 `mov`/`shl`/`and`，同时产生更零碎的 block2D 访存消息。
**工作量的浪费是确凿的**（L1 访问 3.4 倍、ALU1 45 倍），
但它有没有真的成为**主约束**，取决于 shape 有没有空隙把它藏住。

**表面症状很像 compute-bound**：`XVE_ACTIVE` 高（Triton 59.2% vs oneDNN 36.2%），EU 看起来很忙。

但 $(256,4096,4096)$ 的实测证明，**真正绑住性能的是 L1/LSU，不是标量管线**
（干净口径的时间预算分解）：

| 资源 | Triton | oneDNN |
|---|---:|---:|
| ALU2 | 74.7% | 75.3% |
| DRAM | 74.4% | 73.6% |
| ALU1 | **49.3%** | 1.1% |
| LSU 速率（无可信上限） | **89.70/clk** | 26.47/clk |
| `SHARED_FUNCTION_ACCESS_HOLD` | **40.7%** | 1.9% |

Triton 的 L1 访问次数是 oneDNN 的 3.4 倍（25.2M vs 7.37M），而 DRAM 流量几乎相同——
说明多出来的访问全部命中 L1（命中率 94%），但依然占用 LSU 带宽。每条 SEND 消息展开成的
 L1 访问数：Triton 约 23，oneDNN 约 13。

**区分方法**：

| | compute-bound | LSU-bound | ALU1-bound |
|---|---|---|---|
| `ALU2_UTILIZATION` | 高（>85%） | 中等（50–70%） | 中等 |
| LSU 时间预算 | 任意 | **接近 100%** | 中等 |
| `ALU1_UTILIZATION` | 极低（~1%） | 中等 | **>85%** |
| `SF_HOLD` | ~1% | **>25%** | 中等 |
| DRAM 速率 | 任意 | 明显低于可达值 | 明显低于可达值 |

（**这张表目前没有实测样本**：$(256,4096,4096)$ Triton 已改判为拐点，ALU1-bound 从未观测到。
表中的阈值是按各自的定义推的，用之前请自己验证。）

修复：LSU-bound 需要减少 L1 访问次数（更大 tile、更大 `BLOCK_K`、去掉冗余 prefetch）；
ALU1-bound 需要减少地址/descriptor 指令。两者的根源都在 Triton XPU backend 的 block2D
lowering（见三个已提交 issue），而且**同源** —— descriptor 每轮重建既抬高了 ALU1 占用，
也把访存拆得更碎，所以修一处会同时改善两者。config 层面对 LSU-bound 还有部分手段（加大 $D$）。

#### 小结

```
本应 memory-bound 的 shape，可能被模板搞成：
  A. compute-bound         <- tile padding 放大 DPAS 工作量
  B. 更差的 memory-bound    <- L2 blocking 差，B 被重复读
  C. LSU-bound / ALU1-bound <- 访存消息与地址指令膨胀（最隐蔽）
  D. latency-bound          <- SLM 中转等依赖链（见实验 9）
```

反过来，本应 compute-bound 的 shape **不会**被模板变成 memory-bound（流量只会增不会减），
但会被 C 或 D 拖累。

### 2.10 为什么 prefill 和 decode 表现分化

同样的指令膨胀，在两类 shape 上后果完全不同：

- **decode / memory-bound**：DPAS 管线本来就 3% 占用，额外指令塞进空隙，
  被内存延迟掩盖 → wall time 几乎不变
- **prefill / compute-bound**：DPAS 已经 90%+ 占用，没有空隙，
  额外指令无处可藏 → wall time 线性恶化

> 注意措辞：这里说的是额外指令**无法被隐藏**，不是「挤占发射带宽」——
> 实验 10 证明发射带宽并未饱和。真正的机制是它们占用 ALU1 执行槽并且处在依赖链上，
> 当 DPAS 没有空闲时，这些时间就无处重叠。

$(256,4096,4096)$ 恰在拐点（AI = 227.6 < 242，微弱内存受限），DPAS 有 30% 空闲，
所以 43% 的 ALU1 开销只吃掉最后几个百分点，最终只差 1%。
（该 shape 的主约束其实是 LSU，见 2.8 节的预算分解；ALU1 是同一根因的另一个表现。）

### 2.11 哪些结论是验证过的，哪些不是

这一节专门记录本文档各项判据的证据强度，避免把推测当结论用。

| 上限 / 判据 | 值 | 证据 | 可信度 |
|---|---:|---|---|
| ALU2（XMX）吞吐 | **80 slots/clk** | **厂商常数**：驱动的 `ALU2_UTILIZATION` 就按 $\frac{\text{events}}{\text{clk}\times80}$ 定义，反推是恒等式。旁证：Intel 公布 B580 117 TF/s @2.85GHz $\Rightarrow$ 2048 FLOP/clk/core，与之一致 | **中高**（厂商口径，非独立实测） |
| 标量管线（ALU0/ALU1）吞吐 | **160 slots/clk** | 同为**厂商常数**（`ALU1_UTILIZATION` 的定义分母）；「两个 kernel 独立反推」也是恒等式 | **中高**（厂商口径） |
| 1 slot = 256 MAC | **256**（bf16） | $MNK/\texttt{ALU2\_events}$ 精确等于 256.0000，10/10 shape | **高** |
| 1 slot = 512 MAC | **512**（int8） | 同法标定，精确等于 512.0000，4/4 shape；`torch._int_mm`，ceiling 仍为 80.000 | **高** |
| DRAM 可达带宽 | **~407 GB/s** | stream 类基准实测（copy 398、add 400、纯读 448、33.6 MB bf16 读 407.6） | **高** |
| instr/dpas 下限 $1+18/D$ | $C_{\text{var}}{=}17,\ C_{\text{fix}}{=}1$ | 从 oneDNN K 循环逐条数出并按「是否随展开复制」分类；oneDNN 实测 1.52 vs 下限 1.56 = 0.97x | **中高**（$C$ 随 tile 尺寸变化非常数；展开系数 $U$ 影响 $<2\%$） |
| LSU 上限 | **未知** | 无 `UTILIZATION` 计数器。profiling 口径封顶 82.07，**但干净口径实测已达 89.70** —— 82 只是被干扰压低的观察值，不是硬件上限 | **低** |
| `ISSUED` 上限 | **未知** | 同上，profiling 封顶 80.25，干净口径已达 87.76。**已不再使用这个判据** | — |
| 时间派生指标的**跨 config 比较** | **不可用** | `-q -g ComputeBasic` 会颠倒快慢排序（$BK{=}16$ vs $32$：未 profiling 203.77/239.27，profiling 下 316.7/292.4）；且对 Triton 干扰 33%、对 oneDNN 只 5.5% | **已证伪** |

**两个需要特别注意的点：**

1. **LSU-bound 这个类别目前没有确证实例。** 唯一的候选 $(256,4096,4096)$ Triton
   在改用干净口径后，ALU2 和 DRAM 都是 74–75%，判定改为**roofline 拐点**（见 2.8 节）。
   它的 L1 访问确实是 oneDNN 的 3.4 倍（**工作量浪费是真的**），
   但在拐点处被空隙吸收了，没有成为主约束。
   加上 LSU 没有可信上限，「LSU 饱和 xx%」这种说法本身也站不住。

2. **ALU1-bound 尚未在数据集中观测到。** 上限（160 slots/clk）已由实验 10 验证，
   但实测到的最高占用是 $(256,4096,4096)$ Triton 的 **43.2%**，远未饱和。
   看到高 `ALU1_UTILIZATION` 就下 ALU1-bound 结论是一个常见误判（见 2.3 节陷阱 2），
   应该做完整的时间预算分解再下结论。

**剩下的缺口：**

- LSU 上限：构造一个访存密集但计算极少的 kernel（例如极小 `BLOCK_K` 的 GEMM），
  把 L1 访问率推高，观察它在哪里封顶
- `ISSUED` 上限：实验 10 的探针里 `ISSUED` 只到 61.98/clk 而 ALU1 已到 72.7%，
  说明当时卡在 ALU1 而不是发射。要测它需要一个指令混合更均匀（ALU0/ALU1/SEND 分摊）的 kernel。
  不过这个缺口**不影响实用**：`ALU1_UTILIZATION` 已经能回答「指令是不是太多」。

### 2.12 实测数据汇总

本文所有百分比、比值、判定都来自下面这几组数据。它们是**同一次采集**里读出来的，
所以彼此可比；换硬件或换驱动后需要按第三部分的命令重新采集。

**A. $(256,4096,4096)$ bf16 — Triton `(128,64,16,3,8)` vs oneDNN**

（`ALU2_UTILIZATION` / `ALU1_UTILIZATION` 等百分比为 **`-q` 直读**，未重建；判定所用的干净口径值见 2.8 节）

| 指标 | Triton | oneDNN |
|---|---:|---:|
| GpuTime | 133.9 μs | 125.1 μs |
| GpuCoreClocks | 320,311 | 300,111 |
| `ALU2` events | 16,777,216 | 16,777,216 |
| `ALU2_UTILIZATION` | 65.5% | 69.9% |
| `ALU1` events | 22.1 M | 0.49 M |
| `ALU1_UTILIZATION` | 43.2% | 1.0% |
| `ISSUED_ALL` | 24.6 M | 3.18 M |
| `LOAD_STORE_CACHE_ACCESS` | 25.2 M | 7.37 M |
| `GPU_MEMORY_BYTE_READ` | 35.5 MB | 34.8 MB |
| `SF_HOLD` | 40.7% | 1.9% |
| `XVE_ACTIVE` / `XVE_STALL` | 59.2% / 28.1% | 36.2% / 36.9% |
| instr/dpas | 11.74 | 1.52 |
| 有效 MAC/slot | 256（无浪费） | 256（无浪费） |

时间预算分解 —— **干净口径**（时间取 `unitrace -d` 的 117.3 / 116.1 μs，
事件取自 `-q` 采集；上表 GpuTime 133.9/125.1 是 profiling 口径，**不要用来算百分比**）：

| 资源 | Triton | oneDNN |
|---|---:|---:|
| ALU2 | 74.7% | 75.3% |
| DRAM | 74.4% | 73.6% |
| ALU1 | **49.3%** | 1.1% |
| LSU（速率，无可信上限） | **89.70/clk** | 26.47/clk |
| **判定** | **两者同在 roofline 拐点**（AI=227.6 ≈ 平衡点 242）；Triton 的 3.4x L1 访问与 45x ALU1 被 25% 的空隙吸收，故只慢 1% |

**B. $(4,4096,4096)$ bf16 — Triton `(16,256,32,2,8)` vs oneDNN**

（`ALU2_UTILIZATION` / `ALU1_UTILIZATION` 等百分比为 **`-q` 直读**，未重建；判定所用的干净口径值见 2.8 节）

| 指标 | Triton | oneDNN |
|---|---:|---:|
| GpuTime | 167.45 μs | 90.31 μs |
| `ALU2` events | 1,048,576 | 524,288 |
| DPAS 浪费（下界 262,144） | **4.0x** | 2.0x |
| `ALU2_UTILIZATION` | 3.3% | 3.0% |
| DRAM 读速率 | 197 GB/s | **358 GB/s** |
| `XVE_STALL` | 66.8% | 89.4% |
| grid（workgroup 数） | **16**（$1\times16$） | — |
| 访存 roofline（33.6 MB / 407 GB/s = 82.6 μs）达成率 | **49.3%** | **91.5%** |
| **判定** | **两个问题并存**：① DPAS 白算 4 倍（$BM{=}16$ 装 $M{=}4$）② **并行度不足** —— 只有 16 个 workgroup 却有 20 个 core，4 个核全程闲置，喂不饱 DRAM | **memory-bound，已做完**（91.5% roofline） |

> **注意 `XVE_STALL` 在这里是反的**：oneDNN 停顿 89.4% 却更快。
> 内存受限的 shape 本来就该大部分时间在等 DRAM ——
> **`XVE_STALL` 高不是病，DRAM 速率低才是。**
> 这个 case 的修法很具体：把 $BN$ 从 256 减到 128，grid 从 16 变 32，
> 同时 $BM$ 从 16 减到 8 或 4 消除 padding 浪费。

**C. $(1024,4096,14336)$ bf16 — 同 tile $(128,256,32)$、`nw=32`，只改访存方式**

（`ALU2_UTILIZATION` / `ALU1_UTILIZATION` 等百分比为 **`-q` 直读**，未重建；判定所用的干净口径值见 2.8 节）

| 指标 | `make_block_ptr` | TensorDescriptor LHS |
|---|---:|---:|
| GpuTime | 1432 μs | **5117 μs** |
| 吞吐 | 83.96 TF/s（85.4% peak） | 23.50 TF/s（23.9%） |
| `ALU2_UTILIZATION` | **85.7%** | 23.9% |
| `ALU1_UTILIZATION` | 6.1% | 17.6% |
| DRAM 速率 | 243 GB/s | 47.6 GB/s |
| LSU accesses/clk | 40.60 | 7.05 |
| SEND/clk | 2.28 | **12.21** |
| `SLM_BYTE_READ` | **0 MB** | **3758 MB** |
| `XVE_STALL` | 43.4% | 50.3% |
| **occupancy** | 89.7% | **81.8%** |
| instr/dpas | 2.45 | 19.84 |
| K 循环体 dpas（展开后） | 32 | 64 |
| 超出 instr/dpas 下限 | **1.17x** | **9.55x** |
| **判定** | **compute-bound，但 K 循环仍有 28 条/轮的 prefetch descriptor 重建，未做完** | **latency-bound（occupancy 已 82%，加并发无效）** |

**D. $8192^3$ bf16 oneDNN（峰值标定基准）**

| 指标 | 值 |
|---|---:|
| GpuTime | 11.356 ms |
| 吞吐 | 96.83 TF/s |
| `ALU2` events | 2,147,483,648 |
| `ALU2_UTILIZATION` | **98.5%** |
| `ISSUED_ALL` | 401,595,744 |
| instr/dpas | 1.50 |
| **判定** | **compute-bound（做完了）** |

---

## 第三部分：工具与命令

### 采集硬件指标

```bash
source /opt/intel/oneapi/setvars.sh
unitrace -q -g ComputeBasic python your_script.py
# 纯设备时间（无指标干扰）
unitrace -d python your_script.py
```

> ## ⚠ 最重要的一条工具警告：**metric 用 `-q`，时间用 `-d`**
>
> *（2026-08-13 实测确认，并经同事独立证实）*
>
> **`-q -g ComputeBasic` 会让 kernel 真的变慢，它报的 `GpuTime` 不能当计时用。**
> 而且不只是抬高——它会把不同 config 的快慢排序**直接颠倒**。
>
> 实测（$2048^3$，Triton $128\times128$ nw=8）：
>
> | | 未 profiling | `-q -g ComputeBasic` 下 |
> |---|---:|---:|
> | $BK{=}16$ | **203.77 μs**（快） | 316.7 μs（慢）|
> | $BK{=}32$ | 239.27 μs（慢） | **292.4 μs**（快）|
>
> 而且干扰幅度不均：同一次采集里 Triton 被抬高 33%，oneDNN 只抬 5.5%。
>
> **先澄清：`-q` 报出来的数值本身都是对的。**
> `UTILIZATION = events/(clk × ceiling)` 这个定义没问题，也没有别的合理定义；
> kernel 在 profiling 下确实变慢了，管线确实就没那么忙，计数器如实反映了那次执行。
> **问题不在数值对不对，在于它描述的是「被拖慢的那次执行」，不是你想优化的那次。**
>
> **解决办法：公式不变，只换 `clk` 的来源。**
> 事件计数在两种口径下相同（同一个程序、同样的指令），所以可以直接重建：
>
> $$\texttt{UTILIZATION}_{\text{干净}} = \frac{\texttt{events}_{\texttt{-q}}}{t_{\texttt{-d}} \times f \times \text{ceiling}}$$
>
> | | events 来源 | clk 来源 | 描述哪次执行 |
> |---|---|---|---|
> | `-q` 直接读 | `-q` | `-q`（被拖慢） | profiling 下那次 |
> | **重建** | `-q` | **`-d`（干净）** | **你关心的那次** |
>
> $2048^3$ 上 $128\times128$ 的三个 config：`-q` 直接读是 55.3 / 59.9 / 60.8，
> 重建后是 **85.8 / 73.0 / 80.5** —— **排序完全相反**。所以要重建，不是要弃用。
>
> **一个诚实的保留**：「事件计数跨口径相同」对**指令类**（`ALU2`/`ALU1`/`ISSUED`/`SEND`）
> 基本可确定；对**访存类**（`GPU_MEMORY_BYTE_READ`、`LOAD_STORE_CACHE_ACCESS`）
> 是合理假设但**无法验证** —— 计数器只在 `-q` 下存在，没有干净口径可对照。
>
> **无法重建的指标**：`XVE_STALL[%]`、`SF_HOLD[%]`、`occupancy[%]`、`XVE_ACTIVE[%]`
> 不是事件计数的换算，而是独立的时间占比测量，**换算不到干净口径**。
> 它们同样是「那次执行的真实值」，只能用于**大幅度的定性对比**
> （`SF_HOLD` 40.7% vs 1.9% 是 21 倍，任何合理失真都解释不了；13.9% vs 17.1% 则无意义）。

### 重新验算执行模型

硬件、驱动或 oneDNN 版本变更后，用这个重新确认 256 / 80 / 2048 三个常量：

```bash
unitrace -q -g ComputeBasic python validate_exec_model.py 12 > /tmp/uni_validate.log
# 然后按实验 8 的两条命题解析：ceiling 必须恒为 80.000
```

### 导出汇编

```bash
# oneDNN（nGEN，绕过 IGC）
python .github/skills/xe2-gemm-asm/scripts/dump_onednn_asm.py M N K --dtype bf16 -o /tmp/asm

# Triton（IGC）—— 必须用全新 TRITON_CACHE_DIR，命中缓存就没有 dump
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=$PWD/dump \
TRITON_CACHE_DIR=/tmp/fresh python your_kernel.py

# 对比 K 循环
python .github/skills/xe2-gemm-asm/scripts/compare_gemm_asm.py --triton A.asm --onednn B.asm
```

### 统一基线

```bash
python remeasure_baseline_unified.py --repeats 2
# 旧文件备份为 state_bf16_v6/onednn_baseline.pre_unified.json
```

---

## 第四部分：实验记录（结论的出处）

> **这一部分不是必读的。** 前两部分已经自洽 —— 所有公式、常量、阈值都就地给出了。
> 放这些实验是为了回答两类问题：
>
> 1. **「你凭什么？」** 比如「峰值凭什么是 98.3 而不是规格书上的 117」、
>    「80 slots/clk 是查来的还是算来的」。每个常量都能追到具体哪次采集的哪个计数器。
> 2. **「有哪些坑？」** 实验 1–4 全是**我们自己的测量错误**，不是 Triton 的问题。
>    这四个坑让之前几个月的搜索结论全部作废，值得单独记一笔。
>
> 想复现或质疑某个数字时再翻这里，按「实验 N」对应正文的溯源标记。

### 背景：这些实验是怎么来的

**项目目标**：让 PyTorch Inductor 在 Intel Xe2 上生成的 Triton GEMM 追平 oneDNN。

**原有做法**：对每个 shape 做五维穷举搜索（`BLOCK_M/N/K` × `num_stages` × `num_warps`），
在三个模板（`triton_mm`、`bmg_persistent`、`bmg_tiled2d`）上各跑一遍，
把每个 shape 的最优配置记进 Inductor 的启发式表。搜索结果缓存在
`state_bf16_v6/search_cache/`，oneDNN 的对照基线存在 `onednn_baseline.json`。

**遇到的墙**：搜索做到后期，大 shape 上 Triton 稳定停在 oneDNN 的 96–97%，
再怎么调 config 都动不了。试过改模板、调 `num_stages`、开大 GRF、去掉
`boundary_check`、换 tile 形状 —— 全部无效。**看起来像是遇到了 config 层面之外的瓶颈。**

**于是转向底层**：开始 dump 汇编、读硬件计数器。结果发现的第一批问题不是 Triton 的，
**而是我们自己的测量方法**：

| 实验 | 起因 | 发现 |
|---|---|---|
| 1 | 想知道「96% of oneDNN」离硬件天花板还有多远 | **峰值常数用错了**，效率被系统性低估 19% |
| 2 | roofline 需要一个可信的带宽上限 | 理论值 456 不可达，实际 ~407 |
| 3 | 想核实基线数据 | **基线和候选的计时口径不一致**，小 shape 差 5 倍 |
| 4 | 用正确口径重测最差的那个 shape | 「落后 17.6%」是假象，实际是 1.02x |
| 5 | 建立汇编与计数器的对应关系 | 可以从 ASM 精确预测计数器读数 |
| 6 | oneDNN 的 tile 长什么样？ | **搜索空间漏掉了最优配置** |
| 7 | decode shape 为什么这么慢 | tile padding 浪费可以纯算术预测 |
| 8 | 前面的模型要不要在更多 shape 上验证 | 10/10 通过 |
| 8b | 验证时发现 4 个 shape 数字对不上 | oneDNN 的 tile 数没落在**波边界**上；据此换 tile，Triton 在 $2048^3$ 上 **首次反超 oneDNN（1.025x）** |
| 9 | `TensorDescriptor` 是不是更好的写法 | 反而慢 3.57 倍，退化成 SLM 中转 |
| 10 | 「指令太多」需要一个可验证的判据 | 标定出标量管线上限 160 slots/clk |

**给读者的提示**：实验 1–4 的教训是**先确认自己的测量是对的**，再去优化。
这几个坑很容易踩，而且一旦踩了，后面所有结论都是错的 —— 我们据此做了几个月的
config 搜索，追的其实是一个不存在的 19% 差距。

### 实验 1：BF16 峰值不是 117 TF/s，是 98.3 TF/s

**起因**：搜索结果显示目标 shape 的 Triton 只有「81% of peak」，看起来还有很大空间，
于是一直在调 config。但先得确认这个 81% 的分母是对的。

**方法**：`unitrace -q -g ComputeBasic` 采集 8192³ `torch.mm` bf16。
用一个已知高度优化的 kernel（oneDNN 大 GEMM）去反推硬件真实上限。

| 口径 | 数值 |
|---|---:|
| 实测吞吐 | 96.83 TF/s |
| 理论峰值 @2399 MHz | 98.26 TF/s |
| 计算得效率 | **98.5%** |
| `ALU2_events / GpuCoreClocks` ÷ 80 | 78.80/80 = **98.5%** |
| `ALU2_UTILIZATION` 计数器直读 | **98.5%** |

> **⚠ 更正（2026-08-13）：这三行不是三个独立来源，是同一个恒等式。**
> 设无 padding 浪费（本例成立，$\texttt{ALU2\_events}=MNK/256$）：
>
> $$\frac{\text{achieved}}{\text{peak}} = \frac{2MNK/t}{20\cdot2048\cdot f}
> = \frac{MNK}{20480\,t f} = \frac{\texttt{ALU2\_events}/\texttt{clk}}{80} = \texttt{ALU2\_UTILIZATION}$$
>
> 三者恒等，"一致"是必然的。
>
> **这张表真正验证的是**：$\texttt{ALU2\_events}$ 精确等于 $MNK/256$
> （否则三者就会差一个浪费系数）—— 这条是纯事件计数，站得住。
> 至于"峰值是 98.3 而不是 117"，靠的是 1.7 节那两条独立证据（256 实测 + Intel 公布的 per-core 吞吐），不是这张表。

大 GEMM 上的横向验证：

| shape | TF/s | % of 98.3 |
|---|---:|---:|
| $(2048,7168,28672)$ | 97.51 | 99.2% |
| $(8192,8192,8192)$ | 96.93 | 98.6% |
| $(4096,11008,4096)$ | 94.93 | 96.6% |
| $(4096,4096,4096)$ | 89.83 | 91.4% |

**结论**：117 TF/s 是 Arc B580 在 2.85 GHz boost 下的数字。Arc Pro B60 是 2.4 GHz 工作站卡，`xpu-smi` 报 `Core Clock Rate: 2400 MHz`，负载中实测 2392–2399 MHz。

**影响**：此前所有 "% of peak" 被低估 $117/98.3 = 1.19$ 倍。目标 shape $(2048,7168,28672)$ 的 Triton 实际是 **96.4% of peak**，不是 81%。这解释了为什么改模板、stage、GRF、boundary_check 全部无效——早就贴着天花板了。

### 实验 2：可达 DRAM 带宽约 400–450 GB/s

**起因**：要做 roofline 分析，需要一个可信的带宽上限。规格书写 456 GB/s，
但理论值通常达不到，得实测。

| kernel | GB/s | % of 456 |
|---|---:|---:|
| copy (r+w) | 398.1 | 87.3% |
| add (2r+w) | 399.7 | 87.7% |
| sum (纯读) | 448.0 | 98.2% |
| 33.6 MB bf16 读 | 407.6 | 89.4% |

**结论**：roofline 计算用 **407 GB/s** 作为实际读带宽上限，不要用理论值 456。
GEMM 读 A/B 写 C，接近「读为主」的模式，所以取 bf16 大块读的实测值最有代表性。

### 实验 3：基线计时口径不统一，小 shape 数据全错

**起因**：搜索报告里 $(256,4096,4096)$ 显示 Triton 只有 oneDNN 的 0.824x，
是所有 shape 里最差的。但同一个 shape 在别的口径下看着又没那么差 —— 于是去查基线数据。

问题：`onednn_baseline.json` 里的存量值是 **L2 全热的裸循环**测出来的，
而 Triton 候选一直走 Inductor `do_bench`（**带 cache flush，冷 L2**）。
两个口径相除，比值就没有意义了。

下表**两列都是 oneDNN 自己的时间**（`remeasure_baseline_unified.py` 只测 oneDNN，
取 `bench_inductor_onednn.py` 的 `onednn_timing_ms`），Triton 不参与：

| shape | 旧口径：裸循环，L2 热 (μs) | 新口径：`do_bench`，L2 冷 (μs) | 偏差 |
|---|---:|---:|---:|
| $(4,3584,2560)$ | 17.08 | 87.13 | **5.10x** |
| $(128,2048,1408)$ | 14.39 | 27.34 | **1.90x** |
| $(32,2048,1408)$ | 14.76 | 25.88 | 1.75x |
| $(256,4096,4096)$ | 116.50 | 146.20 | 1.26x |
| $(512,4096,1536)$ | 86.90 | 97.50 | 1.12x |
| $(2048,7168,28672)$ | 8648.11 | 8627.92 | 1.00x |
| $(1024,7168,28672)$ | 4332.77 | 4338.54 | 1.00x |

**同一个 kernel、同一台机器，只换计时口径，小 shape 上差 5 倍。**

规律：大 shape 完全不受影响（工作集 $\gg$ L2，本来就装不下），小 shape 错得离谱。

$(4,3584,2560)$ 的旧值 17.08 μs 物理上不可能 —— B 矩阵 18.35 MB，
17 μs 读完需要 1074 GB/s，超 DRAM 峰值 2.4 倍。这类数字可以用 roofline 直接证伪。

**规则**：所有 Triton/oneDNN 比值必须来自**同一进程、同一 harness**。
`bf16_single_config_bench.py` 的返回 JSON 已含 `timings_ms["mm"]`，用它。
跨文件比值只能作参考。

**冷热该用哪个？** 取决于场景：LLM 推理中权重每层都要重读、L2 被别的算子挤占，
**冷口径更接近真实**。热口径反映的是「数据已经在 L2 里」的理想情况。
关键是**比较双方必须一致**。

### 实验 4：$(256,4096,4096)$ 的真实情况

**起因**：实验 3 发现基线口径有问题，那么那个「最差的 shape」到底差在哪？
用三种不同口径各测一遍，看结论是否一致。

之前报告 0.824x，是实验 3 的假象。同口径重测：

| 口径 | Triton | oneDNN | ratio |
|---|---:|---:|---:|
| 旧（跨方法） | 141.5 | 116.5 | 0.824x |
| 同进程 Inductor harness | 144.4 | 147.1 | **1.02x** |
| unitrace 设备时间 | 117.3 | 116.1 | **0.99x** |

roofline（98.3 TF/s，407 GB/s）：

- 计算下界 $8.59\ \text{GFLOP} / 98.3 = 87.4\ \mu s$
- 访存下界 $37.75\ \text{MB} / 407 = 92.7\ \mu s$
- **访存下界更高 → 该 shape 微弱内存受限**

实际 116–117 μs，约 79% roofline，Triton 与 oneDNN 一致。

### 实验 5：ASM 与硬件计数器互相印证

**起因**：已经有两套工具 —— dump 汇编（能看到每条指令）和硬件计数器（能看到总量）。
如果两者能对上，就可以用汇编**预测**性能，而不必每次都跑 profiler；
反过来，计数器对不上预测值时就知道有没数到的东西。这一节建立这座桥。

K 循环对比（`compare_gemm_asm.py`）：

| kernel | 循环指令 | dpas | 密度 |
|---|---:|---:|---:|
| oneDNN nGEN | 50 | 32 | 64.0% |
| Triton `(128,64,16,3,8)` | 95 | 8 | 8.4% |

**从 ASM 反推计数器**（不看实测值，纯计算）。先把 K 循环的 95 条按管线分类：

| 类别 | 条数 | 明细 |
|---|---:|---|
| dpas | 8 | `dpas.8x8` |
| **ALU0/ALU1** | **76** | `mov` 46、`add` 9、`and` 8、`shl` 4、`shr` 4、`bfn` 3、`or` 1、`sel` 1 |
| SEND | 4 | 2 prefetch + 1 `d16`（A）+ 1 `d16v`（B，VNNI） |
| 控制流 / flag | 6 | `cmp` 4、`jmpi` 2 |
| sync | 1 | `sync.allwr` |
| **非 dpas 合计** | **87** | |

> 注意 `cmp` 走 flag 管线，**不计入** `ALU0/ALU1_EXECUTED`，所以 ALU 类是 76 而不是 87。

**ALU2 预测（精确）**：

```
线程数     = 128 workgroup × 8 warp = 1024
K 迭代     = 4096 / 16 = 256
每迭代 dpas = 8（ASM 数出来）
每 dpas    = 8 个 slot（repeat count = 8）
ALU2_slots = 1024 × 256 × 8 × 8 = 16,777,216
```

实测：**16,777,216**，比值 **1.0000**，分毫不差。

**ALU0/ALU1 预测（近似）**：

```
预测 = 1024 × 256 × 76 = 19,922,944
实测 = ALU0 685,056 + ALU1 22,127,616 = 22,812,672    比值 1.145
```

**为什么一个精确一个差 14.5%？**

- `dpas.8x8` 的展开系数是**固定的 8**（repeat count 写在指令里），所以 ASM 条数能精确换算
- 标量指令的展开系数**随执行宽度变化**：SIMD16 占 1 个槽，**SIMD32 占 2 个槽**
  （见实验 10）。ASM 里数不出每条指令的执行宽度分布，所以只能近似。
  剩余差额来自 prologue/epilogue。

**结论**：ASM 与计数器之间存在可用的桥梁，但**精度取决于展开系数是否固定**。
DPAS 相关的量可以放心用 ASM 预测；标量指令量只能作数量级参考。

完整指标对比（$(256,4096,4096)$）：

| 指标 | Triton | oneDNN | 含义 |
|---|---:|---:|---|
| `ALU2` events | 16,777,216 | 16,777,216 | **矩阵工作量完全相同** |
| `ALU2_UTILIZATION` | 65.5% | 69.9% | DPAS 管线占空比 |
| `ISSUED_ALL` | 24.6 M | 3.18 M | 总发射指令，7.7x |
| `ALU1` | 22.1 M | 0.49 M | 地址算术，**45x** |
| `ALU1_UTILIZATION` | 43.2% | 1.0% | Triton 43% 时间在算地址 |
| `GPU_MEMORY_BYTE_READ` | 35.5 MB | 34.8 MB | **DRAM 流量相同**，均接近下界 |
| `LOAD_STORE_CACHE_ACCESS` | 25.2 M | 7.37 M | L1 访问 3.4x（命中率 94%） |
| `SHARED_FUNCTION_ACCESS_HOLD` | 40.7% | 1.9% | 卡在 load/store 单元 |
| `XVE_ACTIVE` | 59.2% | 36.2% | **Triton 更忙但更慢** |
| `XVE_STALL` | 28.1% | 36.9% | 所有管线全空 |

> **关于「`ALU2_UTILIZATION` 恰好等于 achieved/peak」**：这**不是巧合也不是验证，是恒等式**
> —— 无 padding 浪费时两者代数上就是同一个表达式（推导见实验 1 的更正框）。
> 有浪费时两者才会差一个浪费系数，**那个差值**倒是有信息量（见实验 8b）。

### 实验 6：搜索空间漏掉了 BLOCK_M=256

**起因**：既然有 profiler，不如看看 oneDNN 自己选了什么 tile。
unitrace 会把 kernel 的 grid 和 local size 印在名字里，可以反推出 tile 尺寸。

搜索空间代码里，`SMALL_M_SPACE` 对 $M<512$ 把 `BLOCK_M` 封顶在 128（当初为了控制搜索量）。
而 unitrace 显示 oneDNN 用的是 `gemm_kernel[SIMD16 {16;1;1} {64;8;1}]`
—— 16 个 workgroup，即 $1\times16$ 个 $256\times256$ tile，$\text{grid}_m=1$，B 只读一次。
**$BM=256$ 恰好在封顶线外，从来没被搜到过。**

补测（unitrace 设备时间）：

| config | Triton | oneDNN | ratio |
|---|---:|---:|---:|
| $(128,64,16,3,8)$ 原搜索最优 | 121.1 μs | 115.9 μs | 0.956 |
| $(128,512,32,3,32)$ | 120.0 μs | 116.6 μs | 0.972 |
| $(256,256,32,2,32)$ **新** | **117.3 μs** | 116.1 μs | **0.990** |

已修复 `search_space.py`，受影响的缓存 shape 仅 3 个（`is_valid_config` 的 `bm > max(M*8,64)` 规则使 $M\le4$ 不受影响，$M\ge512$ 本来就用 `BIG_M_SPACE`）：

| shape | 新增候选 |
|---|---:|
| $(256,4096,4096)$ | +800 |
| $(128,2048,1408)$ | +800 |
| $(32,2048,1408)$ | +500 |

### 实验 7：tile padding 造成的 DPAS 浪费可以精确预测

**起因**：decode shape（$M$ 极小，如 $M=4$）一直是 Triton 相对 oneDNN 最差的一类。
这类 shape 的 $M$ 远小于任何可用的 `BLOCK_M`，怀疑 tile padding 在做无用功 ——
想知道能不能**不跑 profiler 就预测出浪费量**。

decode shape $(4,4096,4096)$，Triton config $(16,256,32,2,8)$：

| 指标 | Triton | oneDNN |
|---|---:|---:|
| GpuTime | 167.45 μs | 90.31 μs |
| `ALU2` events | 1,048,576 | 524,288 |
| 算法下界 $MNK/256$ | 262,144 | 262,144 |
| **浪费倍数** | **4.0x** | 2.0x |
| padding 公式预测 | **1,048,576** | — |
| `ALU2_UTILIZATION` | 3.3% | 3.0% |
| DRAM 读速率 | **197 GB/s** | **358 GB/s** |
| `XVE_STALL` | 66.8% | 89.4% |

预测公式（与实测完全一致）：

$$\texttt{ALU2\_min} = \frac{M\cdot N\cdot K}{256}, \qquad
\texttt{ALU2\_actual} = \frac{\lceil M/BM \rceil BM \cdot \lceil N/BN \rceil BN \cdot \lceil K/BK \rceil BK}{256}$$

$M=4$ 用 $BM=16$ → 4 倍浪费，公式预测 1,048,576，实测 1,048,576。

**这个实验也说明了为什么不能只看 DPAS 占用率**：两个 kernel 的 `ALU2_UTILIZATION` 都是 3%，
但性能差 1.85 倍。这个 shape 上的判据是 DRAM 速率：oneDNN 358 GB/s（88% of 407，已饱和），
Triton 197 GB/s（48%）。该看哪个判据取决于 shape，统一方法见 2.8 节的时间预算分解。

### 实验 8：在 10 个形状上交叉验算执行模型

**起因**：前面几个实验推出了一套常数（256 MAC/slot、80 slots/clk、2048 FLOP/clk/core），
但都是在**少数几个 shape** 上得到的。如果这套模型是对的，它应该在**任意** shape 上都成立。
这一节是对已有结论的回归测试。

`unitrace -q -g ComputeBasic` 跑 10 个不同规模的 `torch.mm`，每个 5 次预热 + 12 次计时，
检验两条命题：

> **⚠ 2026-08-13 更正：原来的「两条命题」都是恒等式，不构成验证。**
>
> - 命题 1（ceiling = 80）：驱动就是按 $\texttt{UTIL}=\frac{\texttt{events}}{\texttt{clk}\times 80}$ 算的，
>   反推当然回到 80（逐行 std = 0.000003）。见 1.7 节。
> - 命题 2（两条路径一致）：代数展开后
>   $\frac{\texttt{UTIL}}{\text{achieved/peak}} = \frac{\texttt{events}/(t f \cdot 80)}{2MNK/(t\cdot 20\cdot 2048\cdot f)} = \frac{256\,\texttt{events}}{MNK}$
>   —— $t$ 和 $f$ 全约掉，**恒等于路径 A**。表里那 0.0005 只是舍入。
>
> **这次采集真正证明的只有一条**（但它很硬，因为是纯事件计数）：
>
> $$\dfrac{M\cdot N\cdot K}{\texttt{ALU2\_events}} \text{ 在无浪费的 shape 上精确等于 } 256.0000$$
>
> 有浪费的 4 个 shape 给出 234.06 / 245.76 / 252.06 等非 256 值，
> 而这些值能被**波数量化模型**（$T$、20 个 workgroup，与计数器无关）精确复现 ——
> **那才是真正的独立验证**，见实验 8b。

> **读表前先澄清一个容易误读的点**：下表的「有效 MAC/slot」= $\dfrac{M\cdot N\cdot K}{\texttt{ALU2\_events}}$，
> 分子是**算法需要的** MAC，分母是**硬件实际执行的** slot 数。
>
> 物理上 **1 个 slot 恒等于 256 MAC**（$16$ 个 SIMD 通道 $\times$ $8\times2$ 个 K 步），这是硬件常量。
> 所以算出 234.06 **不是**说某个 slot 只做了 234 个 MAC，而是说每个 slot 照样做满 256 个，
> 但平均只有 234.06 个落在真实矩阵范围内，剩下的算的是 padding，写回时被丢弃。
>
> 注意这个平均值不区分浪费的**形态**：可能是 slot 内部只用了一部分通道，
> 也可能是整个 slot 都在算废 tile。要区分得靠 tile 尺寸和 grid 去推。

**表格各列的含义**：

| 列 | 含义 | 来源 |
|---|---|---|
| **有效 MAC/slot** | $\dfrac{M\cdot N\cdot K}{\texttt{ALU2\_events}}$ | **纯事件计数，本表唯一真正的测量**。无浪费时应为 256 |
| 浪费 | $\dfrac{256}{\text{有效 MAC/slot}}$ | 由上一列换算 |
| **UTIL%** | DPAS 管线占空比 | `ALU2_UTILIZATION` 直读（**`-q` 口径**，欲得干净口径需按 2.8 节重建） |

> **UTIL% 和有效% 的区别是本表的核心。** UTIL% 数的是「管线忙不忙」——
> 包含算 padding 的那部分，硬件并不知道哪些结果会被丢弃；
> 有效% 数的是「有用的 FLOP」—— 只算落在真实矩阵里的。
> 两者相除就把白跑的比例隔离出来，这正是浪费 B 的定义。
>
> 无浪费时两者相等（如 $8192^3$ 的 98.51 vs 98.54）；
> 有浪费时 UTIL% 会高于有效%（如 $(2048,2048,2048)$ 的 83.86 vs 76.70）。

| shape | 有效 MAC/slot | 浪费 | UTIL%（`-q` 口径） |
|---|---:|---:|---:|
| $(8192,8192,8192)$ | **256.00** | 1.0000 | 98.51 |
| $(4096,4096,4096)$ | 252.06 | **1.0156** | 92.13 |
| $(2048,7168,28672)$ | **256.00** | 1.0000 | 99.14 |
| $(4096,11008,4096)$ | 252.06 | **1.0156** | 97.80 |
| $(2048,2048,2048)$ | 234.06 | **1.0938** | 83.86 |
| $(1024,4096,4096)$ | **256.00** | 1.0000 | 88.00 |
| $(6144,6144,6144)$ | **256.00** | 1.0000 | 77.53 |
| $(2048,5120,5120)$ | **256.00** | 1.0000 | 96.96 |
| $(512,4096,4096)$ | **256.00** | 1.0000 | 73.84 |
| $(3072,3072,3072)$ | 245.76 | **1.0417** | 94.30 |

> 原表还有 `ceiling`、`浪费 B`、`差`、`有效%` 四列，**已删除**——全是恒等式：
> `ceiling` 必为 80（驱动定义）；`浪费 B` $\equiv$ `浪费 A`；`差` 只是舍入；
> `有效%` $=$ `UTIL%` $/$ `浪费`，由前两列完全决定。
> 需要"有用算力效率"时自己除一下即可，例如 $(2048,2048,2048)$：$83.86/1.0938 = 76.7\%$。

**有效 MAC/slot：6 个 shape 精确等于 256.0000，4 个不是**——后者由波数量化解释（实验 8b）。
（ceiling 与「浪费 A/B 一致」两列是恒等式，留在表里只作自洽性检查，不构成证据。）

### 实验 8b：oneDNN 自己也会浪费 DPAS（以及 Triton 能否据此反超）

**起因**：实验 8 本来只是想验证模型，结果发现 4 个 shape 的「有效 MAC/slot」不是 256。
起初以为是模型错了，查下去才发现**是 oneDNN 在做多余的 dpas**。
这个意外发现比原本的验证目标更有价值。

上表里 4 个形状的有效 MAC/slot 不是 256，说明 oneDNN 做了超出算法下界的 dpas：

| shape | 浪费 | UTIL%（管线忙不忙） | 有效%（有用算力） | 差距 |
|---|---:|---:|---:|---:|
| $(2048,2048,2048)$ | **1.0938** | 83.86 | 76.70 | **7.2 pt** |
| $(3072,3072,3072)$ | 1.0417 | 94.30 | 90.53 | 3.8 pt |
| $(4096,4096,4096)$ | 1.0156 | 92.13 | 90.72 | 1.4 pt |
| $(4096,11008,4096)$ | 1.0156 | 97.80 | 96.32 | 1.5 pt |

> 这里保留「有效%」是因为**它就是本节要说的事**：管线看着忙，但有一部分白跑。
> 注意它是**派生列**（$=\texttt{UTIL\%}/\text{浪费}$），不是独立测量 ——
> 真正的测量只有「浪费」那一列背后的 $MNK/\texttt{ALU2\_events}$。
> 实验 8 的大表里同一列已删除，因为那里所有 shape 摆在一起时它没有额外信息。

机制：oneDNN 用 **20 个常驻 workgroup**（`gemm_kernel[SIMD16 {20;1;1} ...]`，每个 Xe core 一个），
每个 workgroup 循环 $\lceil T/20 \rceil$ 次（$T$ = tile 总数）。当 $T$ 不是 20 的整数倍时，
最后一波只有部分 workgroup 有真活干，**其余的照样发 dpas，只是结果在写回时被丢弃**。

$$\text{浪费} = \frac{20 \lceil T/20 \rceil}{T}$$

从浪费系数反解 $T$，四个形状**全部精确复现**：

| shape | 反解 $T$ | $\lceil T/20\rceil$ 波 | 末波占用 | 预测浪费 | 实测浪费 |
|---|---:|---:|---:|---:|---:|
| $(2048,2048,2048)$ | 128 | 7 | 8/20 | 1.09375 | **1.09375** |
| $(3072,3072,3072)$ | 96 | 5 | 16/20 | 1.04167 | **1.04167** |
| $(4096,4096,4096)$ | 256 | 13 | 16/20 | 1.01562 | **1.01562** |
| $(4096,11008,4096)$ | 256 | 13 | 16/20 | 1.01562 | **1.01562** |

> **这跟 tile 边缘对不齐完全无关。** 这四个形状的边长都能被 tile 整除，
> 解析式（$\lceil M/BM\rceil BM \cdots$）对它们全部预测 1.0000。
> 浪费来自**调度粒度**：$T$ 落不到 20 的整数倍上。

#### 那 Triton 有机会反超吗？—— 有，但不是「让 tile 整除」

先澄清一个陷阱：**Triton 的浪费系数会是 1.00，但这本身不带来任何加速。**

普通（非 persistent）kernel 直接启动 $T$ 个 workgroup，硬件调度到 20 个 core 上，
耗时同样是 $\lceil T/20 \rceil$ 波。区别只是末波里空闲的 core **什么都不做**（计数器看不见），
而 oneDNN 的 persistent kernel **在算废 tile**（计数器看得见）。
**两者付的时间一样。** 所以「有效 MAC/slot = 256」不等于「更快」。

换句话说：$\dfrac{20\lceil T/20\rceil}{T}$ 既是 oneDNN 的 dpas 浪费系数，
**也是任何用同样 tile 的实现所付的时间代价** —— 只是后者以 idle 的形式出现，不被计数。
所以这个数的真正用途是**检测负载不均衡**，不是「对手在做无用功而我不做」。

真正的杠杆是**换一个让 $T$ 更接近 20 整数倍的 tile**。以 $(2048,2048,2048)$ 为例：

| tile | $T$ | 波数 | 末波占用 | 调度代价 |
|---|---:|---:|---:|---:|
| $256\times256$ | 64 | 4 | 4/20 | **1.250** |
| $256\times128$（oneDNN 实际用的） | 128 | 7 | 8/20 | **1.094** |
| $128\times128$ | 256 | 13 | 16/20 | **1.016** |
| $64\times64$ | 1024 | 52 | 4/20 | **1.016** |

#### 实测结果：Triton 确实反超了，但只有 2.5%，不是预测的 7.7%

`probe_wave_quantization.py`，同进程交错测量 7 轮 × 300 次（std < 0.5 μs）：

| 实现 | $T$ | $D$ | 调度代价 | 中位 μs | vs oneDNN |
|---|---:|---:|---:|---:|---:|
| oneDNN | 128 | — | 1.094 | 214.22 | 1.000 |
| Triton $256\times256$ nw=32 | 64 | 32 | 1.250 | 239.58 | 0.892 |
| Triton $256\times128$ nw=16 | 128 | 32 | 1.094 | 240.45 | 0.891 |
| Triton $128\times128$ nw=8 | 256 | 32 | 1.016 | 236.39 | 0.906 |
| Triton $128\times128$ nw=16 | 256 | 16 | 1.016 | 214.31 | 1.000 |
| Triton $64\times64$ nw=4 | 1024 | 16 | 1.016 | **209.05** | **1.025** |

**三条结论，一条比一条重要：**

**1. 预测 #1 精确成立。** Triton 所有配置的 `ALU2_events` $= 33{,}554{,}432 = MNK/256$，
浪费**精确 1.000**；oneDNN 是 $36{,}700{,}160$，浪费 1.094。Triton 确实一个废 tile 都没算。

**2. 预测 #2 落空了。** 预测 $128\times128$ 能拿到 7.7%，实测它和 oneDNN **打平**（0.9996x）；
真正赢的是 $64\times64$，也只有 **2.5%**。

理论空间是 $214.22 / 1.09375 = 195.9\ \mu s$（oneDNN 完美均衡时的时间），
即 18.3 μs 的余量，Triton 只吃到 5.2 μs —— **28%**。
剩下的 72% 被 Triton 自己的开销（instr/dpas、LSU）吃掉了。
**这正好印证了上面那句警告：浪费系数 1.00 不换算成时间。**

**3. 调度代价这一列排不出顺序 —— `num_warps` 才是更大的杠杆。**
七个调度代价同为 1.016 的配置，实测从 209 到 242 μs，差 16%。
最刺眼的是 $128\times128$：tile、$T$、调度代价全相同，光把 `num_warps` 从 8 改成 16
就快了 **11%**（236.39 → 214.31）。

> **这张表有个方法论缺陷，必须说明**：$T = \frac{MN}{BM\cdot BN}$ 而
> $D = \frac{BM\cdot BN\cdot BK}{nw\cdot 2048}$ —— **改 tile 必然同时改 $T$ 和 $D$**，
> 上表把三个变量（$T$、$D$、`nw`）一起动了，严格说**无法把收益归因给调度**。
>
> 试过做单变量隔离（`probe_wave_isolated.py`：config 完全固定，只改 $N$ 让 $T \bmod 20$ 变化），
> **失败了** —— $N$ 从 1792 变到 2688 时工作集从 22 MB 涨到 29 MB，跨过 18 MB 的 L2，
> 缓存效应（±27%）比要测的波数量化（1–7%）大一个数量级，测量 std 也有 30%。
> $N{=}2560$ 调度代价 1.0000 却是最慢的一个，完全被 L2 盖住。
>
> **结论：目前没有干净隔离波数量化的办法。** 要测它需要一个能在工作集不变的前提下
> 改变 $T \bmod 20$ 的设计，暂时没想到。所以下面的归因只能靠时间预算分解，不靠这张表。

#### 那没拿到的 5% 去哪了？—— 指令开销

用时间预算分解（`probe_residual_gap.py`，未受 profiling 干扰的时间，2.4 GHz）：

$$T_{\text{floor}} = \frac{MNK/256}{80 \times 2.4\text{G}} = \frac{33{,}554{,}432}{192\times10^9} = 174.8\ \mu s$$

这是 ALU2 满载且零浪费时的时间，**两个实现共用同一个 floor**（算法工作量相同）。

| | 实测 μs | 有用算力 / floor | 离 floor 的损失 |
|---|---:|---:|---:|
| oneDNN | 214.22 | **81.6%** | 18.4% |
| Triton $64\times64$ | 209.05 | **83.6%** | **16.4%** |

Triton 那 16.4% 里，调度只占 **1.6%**，**剩下约 14.8% 是指令与访存开销**：

| kernel | instr/dpas | 自身下限 | 超出 | ALU1/clk | LSU/clk | SF_HOLD% |
|---|---:|---:|---:|---:|---:|---:|
| oneDNN | **1.65** | 1.56 | **1.06x** | 2.72 | 24.95 | 2.5 |
| Triton $64\times64$ nw4 | **3.11** | 2.09 | **1.49x** | 9.93 | 36.83 | 17.2 |
| Triton $128\times128$ nw16 | 3.11 | 2.09 | 1.49x | 9.71 | 31.63 | 13.5 |
| Triton $256\times128$ nw16 | 2.26 | 1.56 | 1.45x | 5.59 | 23.38 | 21.0 |

oneDNN 精确落在自己的实现下限上（1.06x），Triton 稳定高出 **1.45–1.49x**，
ALU1 是 oneDNN 的 3.7 倍，`SF_HOLD` 是 6.9 倍 ——
**就是实验 5 和实验 9 里那个 descriptor 每轮重建的缺陷**，换 tile 换不掉。

**所以那 7.7% 拿不到的原因是：调度省下来的时间，被 Triton 自己的指令开销吃回去了。**
反过来说，**如果 backend 把 descriptor 外提修好，这个 shape 上应该能逼近 $174.8/209.05 = 1.19$ 倍的额外空间。**

顺带一个正面证据：Triton 内部从 $256\times128$（$T{=}128$，代价 1.094）换到
$64\times64$（$T{=}1024$，代价 1.016），**instr/dpas 反而变差**（2.26 → 3.11），
但仍快了 15%（240.45 → 209.05 μs）—— 调度改善盖过了指令劣化。
**所以提升 wave efficiency 确实有用**，只是单独用它估不出收益。

> **给方法论的教训（三条）**：
> 1. 波数量化是**正确但很弱**的预测器：方向对（细粒度 tile 确实更好），
>    但**不能用它估收益**——预测 7.7%，实测 2.5%。
> 2. **tile 扫描天生无法归因**，因为 $T$ 和 $D$ 由同一组参数决定。
>    要归因只能靠时间预算分解。
> 3. 真正卡住 Triton 的仍是 **descriptor 重建**（instr/dpas 高出下限 1.49x）。
>    调度是可以用 config 摸到的那 1.6%，指令是够不着的那 14.8%。
>
> 顺带：这是目前**唯一一个 Triton 在大 compute-bound shape 上跑赢 oneDNN 的实测案例**。

#### 未解：`num_warps` / `BK` 那 17% 是什么？（已排除三个假设）

同一 tile $128\times128$、同一 $T=256$、同一调度代价，只改 `num_warps` 或 $BK$：

| config | acc/lane | $D$ | $BK$ | μs |
|---|---:|---:|---:|---:|
| nw=8 | 128 | 32 | 32 | **239.27** |
| nw=8 | 128 | 16 | **16** | **203.77** |
| nw=16 | 64 | 16 | 32 | 217.09 |
| nw=16 | 64 | 32 | 64 | 211.88 |

**已排除的假设：**

| 假设 | 怎么证伪的 |
|---|---|
| 寄存器压力 / large-GRF 模式 | `n_regs=256`、`n_spills=0` 对快慢两组**完全相同**（`probe_numwarps.py`）。<br>唯一真正 spill 的是 $128\times128$ nw=4：**11840 字节**，该 config 应直接剔除 |
| 累加器规模 acc/lane | $BK{=}16$ nw=8 的 acc 也是 128（"慢组"特征），却是**全场最快** |
| 软件流水被压垮 | ASM 显示两个 K 循环**结构完全相同**：5 条 block2D 全在第一条 dpas 之前，dpas 连成单段。<br>而且 $BK{=}32$ 的 instr/dpas **更好**（1.94 vs 2.81），总指令**少 45%**（3968 vs 5760），却更慢 |

**唯一剩下的线索**：DRAM 读取量（物理计数，可信）
$BK{=}16$ 22.3 MB vs $BK{=}32$ 26.6 MB，比值 1.192，与时间比 1.174 **相差 1.5%**。
但 nw=16 那组对不上（DRAM 比 1.224，时间比只有 1.065），所以最多是部分解释。
三个配置的有效带宽都只有 110–126 GB/s，远低于 407 —— **都不是 DRAM-bound**。

**为什么卡住**：想用计数器进一步定位，却发现 `-q -g ComputeBasic` 会**颠倒快慢排序**
（见第三部分的工具警告），时间派生指标全部不可用于跨 config 比较。
要继续查，需要一个不依赖时间派生指标的方法。

复现脚本：`validate_exec_model.py`（计数器）、`probe_wave_quantization.py`（tile 扫描）。

### 实验 9：TensorDescriptor 作为 `tl.dot` LHS 会退化成 SLM 中转

**起因**：Triton 有三种写法可以描述张量访存 —— 普通指针、`tl.make_block_ptr`、
`tl.make_tensor_descriptor`。后者是较新的 API，看上去更贴近硬件的 block2D 语义，
自然会想它是不是更快。这一节把三种写法在**完全相同的 shape / tile / warp** 下对比。

同一个 shape $(1024,4096,14336)$、同一个 tile $(128,256,32)$、`num_warps=32`，
**只改访存方式**（复现脚本 `triton_tensordesc_dot_lhs_repro.py`）：

| 指标 | `make_block_ptr` | TensorDescriptor LHS |
|---|---:|---:|
| GpuTime | 1432 μs | **5117 μs**（3.57x 慢） |
| 吞吐 | 83.96 TF/s（85.4% peak） | 23.50 TF/s（23.9%） |
| `ALU2_UTILIZATION` | **85.7%** | 23.9% |
| DRAM 速率 | 243 GB/s | 47.6 GB/s |
| LSU accesses/clk | 40.60 | **7.05** |
| `ISSUED/clk` | 21.03 | 47.44 |
| `SF_HOLD` | 8.8% | 5.9% |
| **SLM 读** | **0 MB** | **3758 MB** |
| `XVE_STALL` | 43.4% | **50.3%** |
| instr/dpas | 2.45 | **19.84** |
| 有效 MAC/slot | 256（无浪费） | 256（无浪费） |

**先澄清一个常见误解**：tensor-of-pointers 路径在 XPU 上 **仍然会被 lower 成 block2D**
（K 循环里有 4 条），只是外围地址计算多。真正不走 block2D 的是 TensorDescriptor 作为
LHS 的路径 —— 它改用 SLM 中转，每次 K 迭代 256 条 `load.slm`。

三条路径的 K 循环对比：

| 路径 | K 循环 | dpas | block2D | SLM | instr/dpas |
|---|---:|---:|---:|---:|---:|
| `make_block_ptr` | 82 | 32 | 8 | 0 | 2.56 |
| tensor-of-pointers | 103 | 16 | 4 | 0 | 6.44 |
| TensorDescriptor LHS | 1358 | 64 | 8 | **268** | 21.22 |

> **本表的 instr/dpas 和后面计数器算的不是同一个数，别搞混。**
>
> | | 本表（21.22） | 计数器（19.84） |
> |---|---|---|
> | 来源 | 反汇编，静态数循环体 | $\texttt{ISSUED\_ALL}/(\texttt{ALU2\_events}/8)$ |
> | 范围 | **只有 K 循环体** | **整个 kernel**，含 prologue/epilogue |
> | 数的对象 | ASM 里写的每一行 | 硬件真正**发射**出去的指令 |
>
> 差 7% 的原因可以逐项对上（重新提取该循环得 1350 条 / 64 dpas）：
>
> | 扣除项 | 条数 | instr/dpas |
> |---|---:|---:|
> | 原始 | 1350 | 21.09 |
> | $-$ `sync.nop`（调度指示，不产生发射） | 55 | 20.23 |
> | $-$ `goto` / `join`（结构化控制流标记） | 16 | **19.98** |
> | 计数器实测 | | **19.84** |
>
> 同一规律在 `make_block_ptr` 上复现：$(82-2\ \texttt{sync.nop}-2\ \texttt{sync.allwr})/32 = 2.44$，
> 计数器 2.45。
>
> **用法分工**：计数器值**用于判定**（真实发射量，无假设，可直接和上限/下限比）；
> ASM 值**用于定位**（能指出多出来的是哪几条指令、哪个寄存器、哪个立即数）。
> 两者一致到几个百分点，本身就是「K 循环占了绝大部分时间」这个前提成立的证据；
> 若差很多，说明 prologue/epilogue 或某个分支吃掉了不可忽略的时间，得单独看。

**它属于哪种 bound？** 四条上限**一条都没饱和**：ALU2 23.9%、DRAM 12%、
LSU 7.05/clk（比 baseline 的 40.60 还低）、`ISSUED` 47.44/clk。
但 `XVE_STALL` 高达 50.3%，SLM 读 3758 MB（baseline 是 0）。

> LSU 反而**变低**是个容易看漏的细节：`LOAD_STORE_CACHE_ACCESS` 只统计走 L1 的访问，
> **SLM 有独立的访存通路，不计入这个计数器**。所以 LSU 读数低不代表访存轻松，
> 得看 `SLM_BYTE_READ` 和 SEND 消息数（12.21/clk vs baseline 2.28/clk，5.4 倍）。

→ **latency-bound**：SLM 往返（写进 SLM 再逐元素读回做 layout 转换）形成长依赖链，
后面的 dpas 必须等前面的 SLM 读完成，硬件没有足够的独立工作去填这个空隙。

**这是 latency-bound 里「occupancy 已高、加并发无用」那一支的实测案例**
（另一支「并行度不足」的例子是 $(4,4096,4096)$ Triton，只有 16 个 workgroup，见 2.11 组 B）。
它也是「指令多 $\ne$ issue-bound」
的典型例证。量化一下：

这里的 $D$ 是**每个线程每个 K 迭代要发的 dpas 条数**，$D = \dfrac{BM\cdot BN\cdot BK}{nw\cdot 2048}$
（完整推导见 2.1 节第二栏）。本例 $D = \dfrac{128\times256\times32}{32\times2048} = 16$。
每条 dpas 摊到的指令数有个**实现下限** $1 + 17/D + 1/(U{\cdot}D)$，
其中 17 是每个 K 步都要付的开销（加载、prefetch、坐标推进、同步），
1 是每个循环体只付一次的循环控制（见 2.1 节第三栏）。

得先确定展开系数 $U$：ASM 里 `make_block_ptr` 的循环体有 32 条 dpas（$U=2$），
TensorDescriptor LHS 有 64 条（$U=4$）。但**展开几乎不动下限** ——
只有那 1 条循环控制被摊薄，17 条加载/同步开销都跟着复制：

| 路径 | 循环体 dpas | 展开 $U$ | 下限 | 实测 instr/dpas | 超出下限 |
|---|---:|---:|---:|---:|---:|
| `make_block_ptr` | 32 | 2 | $1+\tfrac{17}{16}+\tfrac{1}{32} = 2.09$ | **2.45** | **1.17x** |
| TensorDescriptor LHS | 64 | 4 | $1+\tfrac{17}{16}+\tfrac{1}{64} = 2.08$ | **19.84** | **9.55x** |

表中的 **2.46 和 19.84 就是上面对比表最后一行的 instr/dpas**，由计数器算出：
$\texttt{ISSUED\_ALL} / (\texttt{ALU2\_events}/8)$，分别是
$72{,}063{,}264 / 29{,}360{,}128$ 和 $582{,}553{,}616 / 29{,}360{,}128$。
两者相除 $19.84/2.45 = 8.1$ 倍，就是换个写法带来的指令膨胀。

**而 `ISSUED` 只有 48.53/clk，发射带宽远未打满** —— 指令多到 15 倍于下限，却依然不是
issue-bound，这正是陷阱 2 说的：拖慢它的是依赖链，不是吞吐。

#### 那 latency-bound 该怎么优化？

**第一步永远是看 occupancy，它决定了有没有「加并发」这条路。**

| | `make_block_ptr` | TensorDescriptor LHS |
|---|---:|---:|
| `XVE_THREADS_OCCUPANCY_ALL` | 89.7% | **81.8%** |
| `XVE_STALL` | 43.4% | 49.9% |
| `ALU1_UTILIZATION` | 6.1% | 17.6% |
| SEND/clk | 2.28 | **12.21** |

本例 occupancy 已经 **81.8%**，硬件手上并不缺可切换的线程 —— **「增加并发」这条路是堵死的**。
延迟没被掩盖，不是因为线程不够，而是因为**每个线程自己的依赖链太长**：
`store.slm` → barrier → 256 条 `load.slm` → dpas，中间没有可以插进去的独立工作。

于是按优先级排下来：

| 手段 | 本例适用？ | 说明 |
|---|---|---|
| **1. 砍掉依赖链本身** | ✅ **唯一有效** | 别让 LHS 走 SLM 中转，改回 `make_block_ptr` 直出 block2D。这是**写法/后端层面**的修复，不是 config 能调的 |
| 2. 提高 ILP（更多独立累加器、加大展开） | ⚠️ 有限 | Triton 不直接暴露；加大 $BM$/$BN$ 能间接增加独立 dpas，但改不了 SLM 往返这条主链 |
| 3. 加深流水（`num_stages`） | ❌ | 只能掩盖**访存**延迟；这里的链在 SLM 读之后，prefetch 提前也没用 |
| 4. 增加并发（更多 workgroup / 更少寄存器） | ❌ | occupancy 已 82%，没有空间 |

**结论**：latency-bound 里 occupancy 高的那一类，config 层面基本无解，
必须从**数据通路**下手 —— 这也是为什么这条最终变成了一个 backend issue
（`TRITON_ISSUE_tensordesc_dot_lhs.md`）而不是一个 tuning 结论。

#### 顺带：`make_block_ptr` 这一路也没做完

它跑到 85.4% of peak、DPAS 浪费 1.00，看着很接近 2.4 节的停手条件。但**没到**——
K 循环里仍有大量可以外提的 descriptor 重建。

从 `dump_blockptr/OCL_asm975fc2ecce6bbf5a_simd16_entry_0001.asm` 第 772–890 行数出来
（82 条指令 / 32 条 dpas，50 条非 dpas）：

| 用途 | 条数 |
|---|---:|
| 真实数据加载：4 条 `load_block2d` + 5 条增量推坐标 | 9 |
| **4 个 prefetch descriptor 的重建**（r3 / r4 / r69 / r70） | **28** |
| 同步（`sync.nop` ×2、`sync.allwr` ×2） | 4 |
| 循环控制（`cmp` ×1、`jmpi` ×2、计数 `add` ×1） | 4 |
| 其余 | 5 |

**真实加载路径是干净的** —— 和 oneDNN 一样只用 `or`/`add` 增量推坐标。
**问题全在 prefetch**：它的 4 个 descriptor 每轮从零重建，其中 12 条写的是编译期常量：

```
(W) mov (1|M0)  r3.3<1>:ud    1023:w      <- M-1 = 1024-1
(W) mov (1|M0)  r3.4<1>:ud    28671:w     <- A 行跨距 14336*2-1
(W) mov (1|M0)  r4.3<1>:ud    14335:w     <- K-1
(W) mov (1|M0)  r4.4<1>:ud    8191:w      <- B 行跨距 4096*2-1
(W) mov (1|M0)  r3.7<1>:ud    0x31F:uw    <- block 高宽编码
(W) mov (1|M0)  r4.7<1>:ud    0x71F:uw
...r69 / r70 是 r3 / r4 的完整复制品，同样 6 条
```

代价：

$$28\ \text{条} \times \underbrace{448}_{K/BK} \times \underbrace{4096}_{\text{线程}} = 5.14\times10^7\ \text{条}$$

**这和实验 5 里 $(256,4096,4096)$ 的 15 条立即数是同一个 backend 缺陷**，只是这里
表现在 prefetch 路径上。区别在于本例 tile 大（$D = 16$ vs 8），固定开销摊得薄，
所以恶化程度从 3.61x 降到 1.17x —— **但缺陷本身没变。**

> **能省多少时间？暂时不知道。** `ALU1_UTILIZATION` 只有 6.1%，标量管线远未饱和，
> 所以砍指令**不一定**线性转化成时间。真正的机制推测是这 28 条挡在循环头和 prefetch
> 发射之间，推迟了 prefetch 的下发时机。这个推测**尚未验证**，需要手写一个把 descriptor
> 外提的对照 kernel 才能定论。
>
> 但可以确定的是：**85.4% 不等于「做完了」**，这里有一个明确、可定位、非 config 能解决的缺口。

### 实验 10：标定标量管线上限 = 160 slots/clk

**目的**：确定 ALU0/ALU1 的吞吐上限，以及 `ISSUED_ALL` 能否作为判据。

**方法**：`probe_issue_ceiling.py` —— 8 条独立累加链的纯标量 kernel，
刻意避开依赖阻塞，让吞吐成为唯一约束。

| 计数器 | 实测 | 利用率 | 反推上限 |
|---|---:|---:|---:|
| `ALU1` 执行槽 | **116.28/clk** | 72.7% | $116.28/0.727 =$ **159.9** |
| `ISSUED` 指令 | 61.98/clk | — | 未饱和 |
| `ALU0` | 4.01/clk | 2.5% | — |

用 $(256,4096,4096)$ Triton 独立验证：$\dfrac{22.1\text{M}}{320{,}311} = 69.0$ slots/clk，
$\dfrac{69.0}{0.432} = 159.7$。**两个毫不相关的 kernel 都反推出 160。**

**结论 1**：$\texttt{ALU1\_UTILIZATION} = \texttt{ALU1 slots/clk} / 160$，上限已确定。

**结论 2**：160 属于 **ALU0/ALU1 执行槽**，**不属于 `ISSUED`**。两者差约 2 倍
（61.98 vs 116.28），因为多数指令是 SIMD32，**占 2 个执行槽但只发射 1 次**。
`ISSUED` 的真实上限仍未知，因此**不要用它做判据**，改用 `ALU1_UTILIZATION`。

**结论 3**：这给了「指令太多」一个可验证的量化口径：

| | ALU1 slots/clk | ÷160 |
|---|---:|---:|
| $(256,4k,4k)$ Triton | 69.0 | **43.2%** |
| $(256,4k,4k)$ oneDNN | 1.6 | **1.0%** |

Triton 有 43% 的标量执行槽被地址/descriptor 计算吃掉，oneDNN 只有 1%。
注意 43.2% 并未饱和 —— 它拖慢性能靠的是**依赖链**而非吞吐（见 2.3 节陷阱 2）。

复现脚本：`probe_issue_ceiling.py`。

---

## 附录：常量速查

| 量 | 值 | 来源 |
|---|---:|---|
| Xe core 数 | 20 | 规格 |
| XVE / Xe core | 8 | 规格 |
| XMX / Xe core | 8 | 规格 |
| 硬件线程总数 | 1280 | $160_{\text{XVE}} \times 8$ |
| 核心频率 | 2.4 GHz | `xpu-smi`，实测 2392–2399 |
| 1 ALU2 slot | 256 MAC = 512 FLOP | 计数器标定，10/10 shape 验证（**仅 bf16**）|
|  int8 对应值 | 512 MAC = 1024 OPS | 计数器标定，4/4 shape 验证（`validate_int8_slot.py`） |
| ALU2 slots/clk/core | 4 | $80 \div 20$，10/10 shape 验证 |
| FLOP/clk/core (bf16) | 2048 | $4\times512$ |
| **BF16 峰值** | **98.3 TF/s** | $20\times2048\times2.4\text{G}$ |
| **INT8 峰值** | **196.6 TOPS** | $20\times4096\times2.4\text{G}$，实测最高 189.70（96.5%）|
| 理论 DRAM BW | 456 GB/s | 规格 |
| **可达 DRAM BW** | **~407 GB/s** | 实测 |
| 机器平衡点 | 242 FLOP/byte | $98.3\text{T}/407\text{G}$ |
| ALU2 slot 上限 | 80 slots/clk | $160_{\text{XMX}} \times 0.5$ |
| 标量管线上限 | 160 slots/clk | ALU0/ALU1，两个 kernel 反推验证 |
| 指令发射上限 | **未知** | `ISSUED` 实测封顶 80.25，不要套 160 |
| L1/LSU 上限 | **未知** | profiling 口径封顶 82.07，干净口径实测已达 **89.70**；80 已被证伪 |
| 每线程每 K 迭代 dpas 数 | $D=\dfrac{BM\cdot BN\cdot BK}{nw\cdot 2048}$ | 与 ASM 实数吻合 |
| 循环展开系数 | $U=$ 循环体 dpas 条数 $\div D$ | 只能从 ASM 数；对下限影响 $<2\%$ |
| instr/dpas 下限 | $1+\dfrac{17}{D}+\dfrac{1}{U\cdot D} \approx 1+\dfrac{18}{D}$ | oneDNN K 循环实数；17 随 $U$ 复制，仅 1 条循环控制摊薄 |
| L2 (Device Cache) | 18 MB | 规格 |
