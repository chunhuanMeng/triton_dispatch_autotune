# Triton `triton_mm` 与 Inductor 性能差异实验记录

## 日期

2026-07-14

## 实验目标

验证手写 `bench_worker` 的 `triton_mm` kernel 是否能够复现 PyTorch Inductor 当前 XPU `int8_mm` 路径的性能。Inductor 当前启用的是 `triton_mm`，使用 11 个 `int8_mm_configs`。

## 初始现象

测试形状：`M=512, N=4096, K=4096`

初始结果：

- 手写 `bench_worker`：约 `953.71 us`
- Inductor：约 `154.93 us`
- 表面上 Inductor 快约 `6.2x`

此前还有一次约 `9x` 的差异。

## 第一个错误：比较了错误的 PyTorch 算子

之前的对照代码使用：

```python
torch.ops.aten.mm(A, B)
```

但 Inductor 的 11 个 `int8_mm_configs` 属于 `aten._int_mm` 路径，而不是普通 `aten.mm` 路径。

正确的对照算子是：

```python
torch._int_mm(A, B)
```

区别：

- `aten.mm(int8, int8)` 在当前 XPU 环境返回 `int8`
- `aten._int_mm(int8, int8)` 返回 `int32`
- Inductor 的 int8 Triton template 使用 `ACC_TYPE=tl.int32`

因此，使用 `aten.mm` 的结果不能用于验证 11 个 int8 Triton config。

## 第二个错误：手写 kernel 的 runtime 参数没有 specialize

之前手写 kernel 将以下参数作为普通 runtime 参数传入：

- `M/N/K`
- `stride_am/stride_ak`
- `stride_bk/stride_bn`
- `stride_cm/stride_cn`

这与 Inductor 生成的 kernel 不一致。

Inductor 生成的代码会把这些值直接固化为编译期常量，例如：

```python
M = 512
N = 4096
K = 4096
stride_am = 4096
stride_ak = 1
stride_bk = 4096
stride_bn = 1
```

runtime 参数导致：

- 地址计算不能完全常量折叠
- stride 乘法保留在生成代码中
- `M/N/K` 边界判断不能完全消除
- 生成的 kernel 与 Inductor 实际 kernel 不等价

这是造成主要性能差异的 root cause。

修复方式：将尺寸和 stride 声明为 `tl.constexpr`，并通过 keyword 参数传入，使每个 shape/config 生成与 Inductor 等价的 specialized kernel。

## 第三个差异：模板逻辑没有完全匹配

手写 kernel 初始版本还与 [triton_mm.py.jinja](../../pytorch/torch/_inductor/kernel/templates/triton_mm.py.jinja) 存在以下差异：

1. 没有 `EVEN_K` compile-time 分支
2. 无条件使用 mask load，即使 `K % BLOCK_K == 0`
3. `tl.dot` 始终使用 accumulator 参数
4. 没有匹配 `USE_FAST_ACCUM=False` 时的：

```python
acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
```

5. 没有完整匹配 `ACC_TYPE` 和 `ALLOW_TF32`

修复后，kernel 使用：

- `EVEN_K=(K % BLOCK_K == 0)`
- EVEN-K 时不生成 mask 和 `other=0`
- `USE_FAST_ACCUM=False`
- `ACC_TYPE=tl.int32`
- `tl.dot` 调用方式与模板一致
- rematerialize `rm/rn`
- `tl.max_contiguous(tl.multiple_of(...))`
- `tl.assume(pid_m >= 0)` 和 `tl.assume(pid_n >= 0)`

## 最终验证

运行环境：

```text
TORCHINDUCTOR_MAX_AUTOTUNE=1
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON
```

这样可以强制 Inductor 使用 Triton-only max autotune，避免 ATen/oneDNN 路径干扰。

最终结果：

- `bench_worker` 最优配置：约 `194.72 us`
- Inductor Triton-only `_int_mm`：约 `193.34 us`
- 性能比例：`0.99x`

最优配置：

```text
BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
num_stages=2, num_warps=8
```

结论：修复 runtime specialization、算子选择和模板逻辑后，手写 kernel 与 Inductor 性能一致。

## 经验总结

比较手写 Triton kernel 和 Inductor 时，必须同时满足：

1. 比较相同的 ATen 算子：int8 GEMM 使用 `_int_mm`，不能使用 `aten.mm`
2. 使用相同的 config 集合
3. 使用相同的 `EVEN_K`、`ACC_TYPE`、`USE_FAST_ACCUM` 和 `ALLOW_TF32`
4. 将 shape 和 stride 以相同方式 specialize 为编译期常量
5. 使用相同的输出 dtype
6. Inductor 需要显式限制 backend，才能确认实际使用的是 Triton，而不是 ATen/oneDNN

否则，表面上看似“同一个 Triton kernel”的性能比较实际上是不公平的。

## BMG persistent / tiled2d 模板审计

对 `bench_worker.py` 中另外两个 kernel 与 Inductor 的
`triton_bmg_persistent_mm.py.jinja`、`triton_bmg_tiled2d_mm.py.jinja` 进行逐项对比后，
发现原实现也不是等价实现：

1. `M/N/K` 和所有 stride 原本是 runtime 参数；现在改为 `tl.constexpr`，与 Jinja
	模板的 shape/stride specialization 一致。
2. persistent kernel 原本固定发射 20 个 program；现在使用
	`min(NUM_SMS, ceildiv(M, BLOCK_M) * ceildiv(N, BLOCK_N))`，与 Inductor 的 persistent
	grid 一致。
3. tiled2d kernel 原本对 C 使用 block-pointer store；Inductor 模板使用带边界 mask
	的线性 indexed store，现已改为后者。
4. 增加了 program-id 的 `tl.assume`，并保留 block-pointer 的边界检查和 zero padding。
5. BMG kernel 不能复用标准 `triton_mm` 的 11 个 config。现在 benchmark search space
	使用 Inductor 的 7 个 persistent config 和 3 个 tiled2d config。

### Correctness 验证

使用 `torch._int_mm(A, B)` 作为 reference，在以下 shape 上验证通过：

- `(1, 2048, 1408)`
- `(4, 2048, 1408)`

persistent 的 7 个 config 和 tiled2d 的 3 个 config 均输出 `int32` 且与 reference
完全一致。

### 手写 BMG kernel 测量

使用 100 次 amortized XPU event timing，单位为微秒：

| shape | tiled2d 最优 | persistent 最优 |
|---|---:|---:|
| `(1, 2048, 1408)` | 19.32 | 19.57 |
| `(4, 2048, 1408)` | 19.88 | 19.13 |

`(512, 4096, 4096)` 上 persistent 的最佳官方 config 为
`(256,256,64,2,32)`，约 `120.94 us`；但这只是手写 kernel 数据，不能替代
Inductor BMG 对照。

### Inductor 对照状态（修正环境后的结果）

之前没有执行 `source /home/sdp/meng/env.sh`，所以早先的某些命令使用了环境中
已安装的 PyTorch，不能代表 workspace checkout。随后加载 `env.sh` 后确认：

- Python：`/home/sdp/miniforge3/envs/chunhuan/bin/python`
- PyTorch：`/home/sdp/meng/pytorch/torch/__init__.py`
- 版本：`2.14.0a0+gitc4f86df`
- 设备：`Intel(R) Arc(TM) Pro B60 Graphics`
- `config.triton.enable_bmg_persistent_mm=True`

在该环境中重新运行 [test_kernel_compare.py](test_kernel_compare.py)，结果为：

- `(512,4096,2048)`：Inductor-style `867.98 us`，bench-style `896.67 us`
- `(1,2048,1408)`：Inductor-style `97.26 us`，bench-style `174.38 us`
- `(2048,4096,14336)`：Inductor-style `18822.09 us`，bench-style `22171.10 us`

但这个旧测试文件仍把 shape/stride 作为 runtime 参数，因此不能作为已经修复的
`bench_worker.py` BMG parity 结果。

进一步使用本地 PyTorch、`ENABLE_BMG_PERSISTENT_MM=1` 和 Triton-only autotune
重新运行 `_int_mm`，输出结果为：

- `(1,2048,1408)`：`36.60 us`，但 correctness 为 false（max diff `845399`）
- `(32,4096,4096)`：`390.14 us`，correctness 为 true
- `(512,4096,4096)`：`3633.41 us`，correctness 为 true

保存的 Inductor 输出代码/日志中仍未出现 `bmg_persistent` 或 `bmg_tiled2d`，只看到
标准 `triton_mm` 生成代码。因此这组数字仍不能证明 BMG 模板已被实际加入 autotune；
同时 `(1,2048,1408)` 的错误结果需要先作为 workspace Inductor 的独立 correctness
问题调查，不能用于性能结论。

结论修正为：手写 BMG kernel 的逻辑、官方 config 集合和独立 correctness 已对齐；
本地 PyTorch 已成功加载，但 Inductor BMG choices 尚未在实际 autotune 中出现，且
decode shape 存在 correctness failure，BMG 性能 parity 仍未完成。

## BMG template 实际启用与 correctness failure

后续使用 `source /home/sdp/meng/env.sh`，并设置
`torch.compiler.config.force_disable_caches=True` 重新编译，确认之前“没有使用
BMG template”的判断是 cache 导致的误判。当前 Inductor 日志明确出现：

- `triton_mm_bmg_persistent`
- `triton_mm_bmg_tiled2d`
- `(1,2048,1408)` 的 autotune choices 数量为 `21`

因此两个 template 的注册、heuristic 和 choices 生成路径都是正常的。

### 错误结果的实际来源

在 `(1,2048,1408)` 上，Inductor 选择了：

```text
triton_mm_bmg_tiled2d_18
BLOCK_M=16, BLOCK_N=256, BLOCK_K=32,
num_stages=2, num_warps=8
```

输出代码中可以看到：

```text
xindex = idx_n + 2048*idx_m
tl.store(out_ptr0 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), acc, mask)
```

最终 store 的 pointer 只保留了 `idx_n`，丢失 `idx_m` 行偏移。结果是同一输出
行被重复写入，出现 `1792` 个错误元素，最大差值约 `7e5`。标准
`triton_mm` 在关闭 BMG 后同一 shape correctness 为零差异；standalone
`bench_worker` BMG kernel 也正确。因此错误位于 **Inductor BMG template 的
输出 store codegen/后端 lowering**，不是 reference 或 GEMM 数学计算错误。

尝试让 BMG Jinja 使用不同的 output index 命名和 scalar/vector 分离，生成代码仍
出现相同的错误 store；删除 `val_shape` 则会触发
`TritonCSEVariable must have shape`。所以当前不能宣称仅修改 Jinja index 名称已经
修复该问题。

### 两个 BMG template 的单独隔离结果

为排除 autotune candidate 之间的影响，在 [mm.py](../../pytorch/torch/_inductor/kernel/mm.py)
中暂时只保留一个 template，并关闭标准 Triton 与另一个 BMG template：

- 仅保留 `bmg_tiled2d_mm_template`：`M=1,2,4,8,16,32` 全部错误，错误元素数
	分别为 `1792, 3584, 7168, 14336, 28672, 57344`。
- 仅保留 `bmg_persistent_mm_template`：同一组 shape 全部正确，错误元素数均为
	`0`。

所以小 shape 下真正出错的是 **BMG tiled2d template**；BMG persistent template
在当前测试范围内没有复现 correctness 问题。随后已单独修复 tiled2d 的 grid
问题，当前工作树保留 tiled2d template 作为唯一 BMG candidate 进行验证。

### tiled2d 修复

进一步检查生成代码发现，错误并非 output store。`bmg_tiled2d_mm_template`
使用 `pid_m = tl.program_id(0)` 和 `pid_n = tl.program_id(1)`，但它复用了标准
`mm_grid`。标准 `mm_grid` 会把所有 tile 展平为 `(grid_m * grid_n, 1, 1)`；例如
`M=1,N=2048,BLOCK_M=16,BLOCK_N=256` 实际启动的是 `(8,1,1)`，导致 `pid_m`
遍历 `0..7` 而 `pid_n` 永远为 `0`，最终只有第一列 tile 被正确写入。

新增 `bmg_tiled2d_grid`，使用真正的 `(grid_m, grid_n, 1)` 网格，并将
`bmg_tiled2d_mm_template.grid` 切换到该函数。修复后仅保留 tiled2d template，
`M=1,2,4,8,16,32` 的错误元素数全部为 `0`。

### 安全修复

在问题完全修复前，已将 [config.py](../../pytorch/torch/_inductor/config.py) 中
`ENABLE_BMG_PERSISTENT_MM` 的默认值从 `1` 改为 `0`。这样默认 max-autotune 不会
把错误的 BMG candidate 选入最终 kernel，避免返回错误结果。若显式设置
`ENABLE_BMG_PERSISTENT_MM=1`，仍可复现并继续调试 BMG output store 问题；不能把
该模式用于 correctness 或性能发布结果。
