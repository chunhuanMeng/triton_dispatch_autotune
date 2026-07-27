# BF16/FP16 worker 与 Inductor parity 实验记录

## 实验范围

- 硬件：Intel Arc Pro B60 / BMG-G21，20 Xe cores
- 环境：`/home/sdp/miniforge3/envs/chunhuan/bin/python`，workspace PyTorch
- shape：`(M,N,K) = (1,2048,1408)`
- config：`(BLOCK_M,BLOCK_N,BLOCK_K,num_stages,num_warps) = (8,512,32,2,8)`
- 计时：XPU event，先 warmup，再固定调用次数；本记录中的小规模验证使用 20 次调用
- 当前日期：2026-07-15

---

## 实验一：将 bench worker 参数化为 INT8/BF16/FP16

### 实验目的

确认 raw Triton worker 是否可以从 INT8 专用实现扩展为 BF16 和 FP16，同时保留三个 template 的 grid、block pointer 和边界逻辑。

### 实验方法

修改 [bench_worker.py](bench_worker.py)：

1. 增加 dtype policy：`int8`、`bf16`、`fp16`。
2. 输入 tensor 根据 dtype 创建：
   - INT8 使用随机整数；
   - BF16/FP16 使用随机浮点数。
3. output dtype 根据 dtype 创建：
   - INT8：`int32`；
   - BF16：`bfloat16`；
   - FP16：`float16`。
4. accumulator 改为 `ACC_TYPE`：
   - INT8：`tl.int32`；
   - BF16/FP16：`tl.float32`。
5. `triton_mm` 补齐 `allow_tf32=ALLOW_TF32` 参数，当前实验设置为 `False`。
6. reference 根据 dtype 选择：
   - INT8：`torch._int_mm`；
   - BF16/FP16：`torch.mm`。

### 实验结果

修改后的 worker 通过实际 Python 环境语法检查。

INT8 回归测试 `test_kernel_compare.py`：

- `(512,4096,2048)`：结果匹配
- `(1,2048,1408)`：结果匹配
- `(2048,4096,14336)`：结果匹配

### 实验结论

worker 的 dtype、accumulator 和 output 已成功参数化；原有 INT8 路径仍然可以编译并得到正确结果。BF16/FP16 可以复用同一套 kernel 结构，但 accumulator 必须使用 FP32，不能简单地把输入 dtype 替换后继续使用 `tl.int32` accumulator。

---

## 实验二：验证标准 `triton_mm` 的 BF16/FP16 correctness 和 raw 性能

### 实验目的

确认通用 `triton_mm` template 在 BF16 和 FP16 下能够使用与 worker 相同的 config，并验证 `ACC_TYPE=tl.float32`、`ALLOW_TF32=False` 和 output cast 的正确性。

### 实验方法

对以下组合执行 worker direct launcher：

- dtype：BF16、FP16
- template：`triton_mm`
- shape：`(1,2048,1408)`
- config：`(8,512,32,2,8)`
- reference：`torch.mm(A,B)`
- correctness：`torch.allclose(..., rtol=2e-2, atol=2e-2)`

### 实验结果

| dtype | time (us) | correctness | max abs diff |
|---|---:|---:|---:|
| BF16 | 38.72 | PASS | 0.000122 |
| FP16 | 28.27 | PASS | 0.007812 |

### 实验结论

标准 `triton_mm` 在 BF16/FP16 下可以正确运行。当前 worker 的 dtype policy、FP32 accumulator、`EVEN_K` 逻辑和 `allow_tf32` 参数与 Inductor 标准 template 的语义一致。

---

## 实验三：验证 BMG persistent/tiled2d 的 BF16/FP16 correctness 和 raw 性能

### 实验目的

确认两个 BMG block pointer template 是否支持 BF16/FP16，并检查 block pointer 的 zero padding、边界 store 和 FP32 accumulator 是否正确。

### 实验方法

对同一个 shape/config 分别运行：

- `bmg_persistent`
- `bmg_decode`
- BF16、FP16
- `torch.mm` reference
- XPU event raw-kernel timing

### 实验结果

| dtype | template | time (us) | correctness | max abs diff |
|---|---|---:|---:|---:|
| BF16 | bmg_persistent | 28.49 | PASS | 0.000122 |
| BF16 | bmg_decode | 37.05 | PASS | 0.000122 |
| FP16 | bmg_persistent | 28.38 | PASS | 0.007812 |
| FP16 | bmg_decode | 28.40 | PASS | 0.007812 |

运行过程中仅出现 `tl.make_block_ptr` deprecated warning，没有 correctness 或编译错误。

### 实验结论

当前 BMG persistent 和 tiled2d kernel 在该 shape/config 上可以支持 BF16/FP16，并且结果符合 FP16/BF16 容差。BMG template 不需要单独的 `EVEN_K` 参数，因为 block pointer 的 `boundary_check` 已覆盖 K-tail；但必须把 `ACC_TYPE` 从 INT8 的 `tl.int32` 改为浮点路径的 `tl.float32`。

这一步只证明了 standalone worker 的 kernel 支持，尚不能证明 Inductor 普通 `aten.mm` 已经使用 BMG template。

---

## 实验四：将 BMG template 接入 Inductor 普通 BF16/FP16 MM

### 实验目的

验证 BF16/FP16 的普通 `torch.mm` 路径是否可以使用 BMG persistent/tiled2d template，并使 fixed-config runner 能够复用 worker 的 config。

### 实验方法

修改：

- [triton.py](../../pytorch/torch/_inductor/heuristics/template/triton.py)：为 BMG persistent/tiled2d 增加普通 `mm` heuristic；使用 BF16/FP16 可用的 BMG config 集合，并为 persistent 注入 runtime `NUM_SMS`。
- [mm.py](../../pytorch/torch/_inductor/kernel/mm.py)：通过 `XE2_ENABLE_BMG_FLOAT_TEMPLATES=1` opt-in 将 BMG templates 加入普通 MM；通过 `XE2_PARITY_TEMPLATE` 隔离单个 template。
- [run_inductor_fixed_config.py](run_inductor_fixed_config.py)：增加 `--dtype int8|bf16|fp16`，根据 dtype 选择 `torch._int_mm` 或 `torch.mm`，并使用对应 heuristic。

测试环境变量：

```text
XE2_ENABLE_BMG_FLOAT_TEMPLATES=1
XE2_PARITY_TEMPLATE=bmg_decode
TORCHINDUCTOR_MAX_AUTOTUNE=1
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON
```

### 实验结果

普通 Inductor `triton_mm` fixed-config 回归：

| dtype | correctness | time (us) |
|---|---:|---:|
| INT8 | PASS | 63.18 |
| BF16 | PASS | 72.95 |
| FP16 | PASS | 63.44 |

普通 Inductor BMG `bmg_decode` fixed-config：

| dtype | correctness | time (us) |
|---|---:|---:|
| BF16 | PASS | 62.71 |
| FP16 | PASS | 75.39 |

在未加入 `XE2_PARITY_TEMPLATE` 隔离时，autotune 日志曾显示 24 个 choices，并且实际 config 被 scaling 成 `BLOCK_M=16`；补充普通 MM 路径的 parity isolation 后，fixed-config runner 不再混入 generic template，且 correctness 通过。

### 实验结论

1. BMG persistent/tiled2d 已经可以被 Inductor 普通 BF16/FP16 MM 路径生成并执行。
2. `ACC_TYPE='tl.float32'`、`ALLOW_TF32=False`、`EVEN_K=True`、`GROUP_M=8` 和 `NUM_SMS=20` 在日志中符合预期。
3. fixed-config runner 的 dtype 扩展已经覆盖 INT8/BF16/FP16。
4. 这只是单 shape、单 config 的 parity 验证，不代表 BMG 浮点 config 在所有 shape 上都优于标准 `triton_mm`。
5. BMG 浮点路径目前由 `XE2_ENABLE_BMG_FLOAT_TEMPLATES=1` 控制，默认不会改变普通 Inductor BF16/FP16 autotune 行为。

---

## 总体结论

四步实验均完成：

- worker 已支持 INT8/BF16/FP16；
- 标准 `triton_mm` 的 BF16/FP16 correctness 已通过；
- BMG persistent/tiled2d 的 BF16/FP16 correctness 已通过；
- Inductor 普通 MM 已能在 opt-in 模式下生成并运行 BMG 浮点 template。

目前可以把 worker 测得的 config 复用到 Inductor，但必须同时满足：

1. 使用相同的 template；
2. 使用相同的 `BLOCK_M/N/K`、`num_stages`、`num_warps`；
3. 使用相同的 `ACC_TYPE` 和 `ALLOW_TF32`；
4. 使用相同的 `GROUP_M`；
5. persistent template 使用相同的 runtime `NUM_SMS`；
6. shape 对应的 `EVEN_K`、`INDEX_DTYPE` 和 grid 一致；
7. fixed-config 时关闭 Inductor config scaling；
8. raw worker 时间与 Inductor `compiled(A,B)` 时间分开记录，不能直接混用。

下一步建议：在多个 M/N/K shape 上重复 BF16/FP16 correctness，尤其覆盖 K 非 `BLOCK_K` 整除、M/N 非 tile 对齐以及大矩阵 compute-bound 场景，然后再建立 BF16/FP16 的正式 autotune 结果表。

---

## 实验五：扩展 `run_autotune.py` 的 dtype autotune 入口

### 实验目的

确认正式 autotune 驱动不会把 INT8 的 baseline、搜索缓存或 dispatch table
错误复用于 BF16/FP16，并让 autotune 使用已经参数化的 worker。

### 实验方法

修改 [run_autotune.py](run_autotune.py)：

1. 增加 `--dtype int8|bf16|fp16`。
2. 所有 baseline 和 template benchmark 调用显式传入 dtype。
3. INT8 继续使用历史目录 `state/`，保持已有结果可 resume。
4. BF16 和 FP16 分别使用 `state_bf16/`、`state_fp16/`，隔离：
   - oneDNN/MM baseline；
   - search cache；
   - dispatch table；
   - sweep results；
   - iteration log。
5. autotune benchmark 的固定调用次数提高到 500，降低小 kernel 的事件计时噪声。

### 实验结果

以下入口检查通过：

```text
python -m py_compile run_autotune.py bench_worker.py
python run_autotune.py --dtype bf16 --step report
```

第二条命令正确访问 `state_bf16/`，在没有结果时输出 `No dispatch table found!`，
没有误读现有 INT8 的 `state/dispatch_table.json`。

### 实验结论

现在可以分别运行：

```text
python run_autotune.py --dtype int8 --step all
python run_autotune.py --dtype bf16 --step all
python run_autotune.py --dtype fp16 --step all
```

BF16/FP16 运行前还应设置：

```text
XE2_ENABLE_BMG_FLOAT_TEMPLATES=1
```

这样 Inductor 普通 MM 路径才会包含 BMG 浮点 template。三种 dtype 的 autotune
结果不能混用；即使五元 config 相同，accumulator、output bytes、lowering 和
最优 template 也可能不同。

### 追加修正：对齐正式 autotune 的候选集合

进一步检查发现，仅增加 `--dtype` 还不够：原 `run_autotune.py` 使用通用笛卡尔
搜索空间，而 Inductor 对标准 `triton_mm` 和两个 BMG template 使用不同的候选
列表。这样可能选出 worker 测得很快、但 Inductor heuristic 根本不会注册的 config。

因此又完成以下修正：

- [search_space.py](search_space.py) 增加标准 INT8 config 列表；
- 增加标准 BF16/FP16 config 列表；
- 保留 BMG persistent/decode 的 template-specific config 列表；
- `generate_autotune_configs()` 返回三个 template 候选集合的 union；
- `bench_config_all_templates()` 只测当前 template 实际注册的 config；
- `run_autotune.py` 改用该 exact union，而不是 generic Cartesian product。

这一步是 config 可复用的必要条件：worker 现在不会因为某个 template 的非注册
config 在 standalone 中更快，就把它错误地写入 dispatch 结果。

---

## 实验六：将 autotune key 从五维扩展为六维

### 实验目的

原有 dispatch key 只有：

```text
(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
```

现在将 template 作为第六维，区分：

```text
triton_mm
bmg_persistent
bmg_decode
```

避免同一个五维 config 在不同 template 下的时间和 winner 被混在一起。

### 实验方法

修改：

- [search_space.py](search_space.py)：增加 `DispatchConfig(template, gemm)`，其
   `key` 为 `(template, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)`。
- [bench_worker.py](bench_worker.py)：增加 `bench_one_template()`，每次只测一个
   显式的 template/config 组合。
- [run_autotune.py](run_autotune.py)：
   - search 直接遍历 `template × five-dimensional config`；
   - cache key 改为六维；
   - sweep/promotion 只测 dispatch entry 指定的 template；
   - report 按 template 分组输出 heuristic config；
   - 使用 `state_v6/`、`state_bf16_v6/`、`state_fp16_v6/`，不读取旧五维 state。

### 实验结果

六维对象序列化验证通过：

```text
('bmg_decode', 8, 512, 32, 2, 8)
```

可以正确转换为 JSON 并恢复为 `DispatchConfig`。脚本通过语法检查，新的
`--dtype int8 --step report` 正确访问 v6 状态目录。

### 实验结论

现在的 autotune 结果明确表示为：

```text
(template, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
```

同一五维 config 可以在三个 template 中各自产生独立记录；最终 dispatch table
不会再丢失 winner template。旧五维 `state/` 结果不会被自动迁移，必须重新运行
六维 autotune 或显式编写迁移规则。

---

## 实验七：启动 118-shape BF16 全量 autotune

### 问题

首次启动命令使用了相对路径 `run_autotune.py`，但命令实际从 workspace 根目录
执行，Python 返回 exit code 2，`tuned_bf16.log` 没有生成有效结果。

### 实验方法

修正为先切换到 `int8_gemm_optimization_xe2/triton_dispatch_autotune`，并使用脚本
绝对路径启动。运行参数为 BF16、118 个 shape、三个 template，使用
`state_bf16_v6/`，并设置 `XE2_ENABLE_BMG_FLOAT_TEMPLATES=1`；全部输出重定向到
`tuned_bf16.log`。

### 实验结果（截至 2026-07-15）

修正后的进程已经在后台运行，CPU 处于计算状态，并已创建
`state_bf16_v6/onednn_baseline.json`。日志采用重定向方式，Python 默认缓冲输出，
因此运行期间日志可能暂时为空；不能据此判断进程失败。

### 结论

当前全量任务已使用正确工作目录和 BF16 专用 v6 状态目录运行，未发现新的
Python/Triton 错误。最终是否完成必须以进程退出状态、`tuned_bf16.log` 和
`state_bf16_v6/dispatch_table.json` 同时确认。

### 修改方式

保留原始代码不变，仅修正启动路径，并将输出绝对路径重定向到
`tuned_bf16.log`；后续若日志出现错误，将在本记录继续追加问题、实验结果、结论
和修改方式。

---

## 实验八：核对 BF16 搜索空间为何显示 28 个 choices

### 问题

日志中出现 `28 template/config choices`，看起来少于对六维参数做完整笛卡尔积
后的数量。

### 实验方法

对 shape `(2048,4096,4096)` 直接统计 `generate_autotune_configs()` 和三个
template 的候选集合，并执行与 `search_shape()` 相同的 choice 构造逻辑。

### 实验结果

- `triton_mm`：标准 BF16/FP16 heuristic 候选 20 个，经 shape validity 过滤后
   剩余 18 个；
- `bmg_persistent`：7 个固定候选；
- `bmg_decode`：3 个固定候选；
- 最终六维 template/config 组合：`18 + 7 + 3 = 28`；
- 五维 config 去重后的 union 为 25 个，但同一个五维 config 在不同 template
   下仍然作为不同的六维 choice 分别 benchmark。

### 结论

这里的“六维”表示一个 choice 的 key 是
`(template, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)`，并不表示对
六个字段执行当前候选值的完整笛卡尔积。当前搜索空间是有意限制为 Inductor
heuristic 实际注册的候选集合，以保证 worker 测出的 config 能直接复用到
Inductor；因此 28 个 choices 在现有代码逻辑下是预期值，不是 autotune 中途
漏掉了维度。

### 修改方式

本次未修改代码，仅通过实际统计验证搜索空间。若目标是探索更大的空间，需要
明确改为扩展 `STANDARD_FLOAT_CONFIGS`/BMG 候选集合或增加独立 exhaustive 模式，
不能把当前 exact-Inductor parity 搜索和完整笛卡尔搜索混为一谈。

---

## 实验九：恢复原始 generic autotune 搜索空间

### 问题

检查发现，之前的 `run_autotune.py` 使用的是
`generate_good_configs(M,N,K)`，它来自文件顶部声明的 generic 候选范围；后续
为了 Inductor parity 将其替换成了 `generate_autotune_configs()`，导致默认 BF16
搜索被限制为固定的标准/BMG candidate lists。

### 实验方法

恢复 generic 搜索为默认模式，同时保留严格 parity 模式：

- 默认：`XE2_AUTOTUNE_SEARCH_SPACE=generic`，使用原始 `generate_good_configs()`，
   对每个有效 config 和三个 template 形成六维 choice；
- parity：设置 `XE2_AUTOTUNE_SEARCH_SPACE=exact`，才使用当前 Inductor 已注册的
   固定 candidate lists。

### 实验结果

语法检查通过。统计结果：

- `(2048,4096,4096)`：960 个 generic config，2288 个 template/config choices；
- `(4,28672,4096)`：688 个 generic config，1872 个 template/config choices；
- 旧版固定候选对应的同类 shape 只有 28 个 choices。

### 结论

之前用户指出的候选数组确实是原始 autotune 的搜索空间；此前改动把默认行为错误
地收窄成了固定集合。现在默认行为已恢复，六维 key 仍然保留，不会丢失 template
winner。

### 修改方式

修改 [run_autotune.py](run_autotune.py)：增加
`XE2_AUTOTUNE_SEARCH_SPACE` 开关，默认 `generic`；只有显式设置 `exact` 才启用
Inductor parity candidate lists。未自动重启全量任务，因为 generic 模式的每个
shape 约有数千个 benchmark choices，运行时间显著高于此前的 28-choice 任务。

---

## 实验十：删除受限 BF16 结果并重新执行 generic autotune

### 问题

此前 BF16 结果来自错误的 fixed candidate 模式，每个 shape 仅测试约 28 个
template/config choices，不能代表原始 autotune 搜索空间，因此结果作废。

### 实验方法

已删除旧的 `state_bf16_v6/`、旧 BF16 state 目录和 `tuned_bf16.log`，并重新启动
118-shape BF16 全量任务，参数为 `XE2_AUTOTUNE_SEARCH_SPACE=generic`、
`XE2_ENABLE_BMG_FLOAT_TEMPLATES=1` 和 `PYTHONUNBUFFERED=1`，输出重定向到
`tuned_bf16.log`，使用 `state_bf16_v6/` 保存结果。

第一次使用 nohup 子 shell 启动时因该子 shell 没有执行 `conda init` 而失败，未
产生 autotune 结果；随后改为在已初始化 oneAPI 的当前 shell 中直接启动。

### 实验结果（2026-07-15）

当前 generic 重跑已正常运行，进程 CPU 占用正常，日志已经完成 baseline 的前
45/118 个 shape，并已重新创建 `state_bf16_v6/onednn_baseline.json`。尚未完成
搜索和 dispatch table，不能提前宣称最终性能结果。

### 结论

旧 fixed-mode BF16 结果不再使用；当前运行才是与原始 tune 脚本搜索范围一致的
BF16 全量实验。

### 修改方式

删除旧结果文件；恢复后的 [run_autotune.py](run_autotune.py) 默认使用 generic
搜索空间，并通过当前 shell 的已初始化环境重新启动任务。启动错误和修正过程
均保留在本记录中。

---

## 实验十一：修复 generic BF16 autotune 的 2271 个 false failed

### 问题

generic BF16 搜索在 `(2048,4096,4096)` 报告：

`Done: 17 successful, 2271 failed, 2288 newly benchmarked`

进一步检查发现这些不是 Triton kernel 真正失败。`run_autotune.py` 在 generic
模式下生成了 2288 个合法六维 choices，但 `bench_worker.py` 的
`bench_one_template()` 又额外执行 `config.key in template_config_keys(...)`，
把不在 Inductor 固定 candidate list 中的 generic config 直接返回 `None`。因此
几乎所有 generic config 都被错误统计为 failed。

### 实验方法

1. 读取 search cache，确认 2288 个条目中只有 17 个有时间，其余为 `None`；
2. 直接 launch 首个失败配置 `(16,64,32,1,4)`，三个 template 均可正常执行；
3. 直接调用 worker benchmark 后确认失败发生在入口过滤，而非 kernel 编译或计时；
4. 修改 [bench_worker.py](bench_worker.py)：显式 benchmark 不再检查 fixed
   Inductor candidate list；generic/exact 的候选筛选由 [run_autotune.py](run_autotune.py)
   负责；同时移除 `bench_config_all_templates()` 中同样不应存在的 fixed-list
   过滤。

### 实验结果

已删除错误 BF16 cache 和 state，并用修复后的 worker 重新启动 generic BF16 全量
任务。新进程正常运行，baseline 正在重新测量；旧的 `17 successful / 2271 failed`
统计不再使用。

### 结论

2271 个 failed 是逻辑误判，不是 BF16 kernel 支持率只有约 0.7%。之前的 BF16
性能结果和搜索结果均作废；修复后 generic 搜索才真正测试用户指定的候选空间。

### 修改方式

保留 `XE2_AUTOTUNE_SEARCH_SPACE=generic` 的完整搜索逻辑；将 fixed Inductor
candidate 检查限定在 `SEARCH_SPACE_MODE=exact` 的 choice 生成阶段。worker 现在
可以 benchmark 任意通过 shape/template validity 的 generic config。
