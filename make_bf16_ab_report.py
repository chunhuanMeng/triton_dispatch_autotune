#!/usr/bin/env python3
"""Generate a Markdown report from bf16_ab_compare_results.json."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

PEAK_BF16_TFLOPS = 117.0
PEAK_BW_GBPS = 456.0
BF16_BYTES = 2


def gmean(values):
    values = [x for x in values if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in values) / len(values)) if values else None


def fmt(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def ratio(numerator, denominator):
    return numerator / denominator if numerator and denominator else None


def metrics(m, n, k, time_ms):
    if not time_ms:
        return {}
    flops = 2 * m * n * k
    bytes_moved = BF16_BYTES * (m * k + k * n + m * n)
    seconds = time_ms / 1000.0
    tflops = flops / seconds / 1e12
    gbps = bytes_moved / seconds / 1e9
    return {
        "flops": flops,
        "bytes": bytes_moved,
        "tflops": tflops,
        "tflops_eff": 100 * tflops / PEAK_BF16_TFLOPS,
        "gbps": gbps,
        "bw_eff": 100 * gbps / PEAK_BW_GBPS,
    }


def shape_key(item):
    return tuple(int(x) for x in item[0].split(","))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    with args.json_path.open() as f:
        data = json.load(f)

    normal = []
    decomposed = []
    errors = []
    for key, pair in sorted(data.items(), key=shape_key):
        m, n, k = shape_key((key, None))
        a, b = pair.get("A", {}), pair.get("B", {})
        if a.get("decomposed") or b.get("decomposed"):
            decomposed.append((m, n, k, a.get("elapsed_ms"), b.get("elapsed_ms")))
            continue
        at, bt = a.get("best_triton"), b.get("best_triton")
        od = a.get("onednn") or b.get("onednn")
        if not at or not bt or not od:
            errors.append((m, n, k, pair))
            continue
        ma, mb = metrics(m, n, k, at), metrics(m, n, k, bt)
        best_a, best_b = min(at, od), min(bt, od)
        normal.append({
            "m": m, "n": n, "k": k, "a": a, "b": b, "onednn": od,
            "a_ratio": ratio(od, at), "b_ratio": ratio(od, bt),
            "ab_triton": ratio(at, bt), "ab_system": ratio(best_a, best_b),
            "a_metrics": ma, "b_metrics": mb,
            "a_best": best_a, "b_best": best_b,
        })

    def mean_metric(state, name):
        return gmean([x[f"{state}_metrics"][name] for x in normal])

    b_ratios = [x["b_ratio"] for x in normal]
    a_ratios = [x["a_ratio"] for x in normal]
    ab_triton = [x["ab_triton"] for x in normal]
    ab_system = [x["ab_system"] for x in normal]
    b_win = [x for x in normal if x["b_ratio"] > 1]
    a_win = [x for x in normal if x["a_ratio"] > 1]
    b_faster = [x for x in normal if x["ab_system"] > 1.02]
    a_faster = [x for x in normal if x["ab_system"] < 0.98]
    same = len(normal) - len(b_faster) - len(a_faster)

    template_counts = Counter()
    for x in normal:
        name = x["b"].get("best_triton_name") or "unknown"
        template_counts[name] += 1

    lines = []
    add = lines.append
    add("# BF16 Triton A/B 性能报告")
    add("")
    add("> 数据来源：`bf16_ab_compare_results_v6_deployed.json`。本报告覆盖 state_bf16_v6 的 GEMM shape。")
    add("")
    add("## 1. 实验定义")
    add("")
    add("| 项目 | State A（baseline） | State B（current） |")
    add("|---|---|---|")
    add("| Triton 模板 | 仅 `triton_mm` | `triton_mm` + `bmg_persistent` + `bmg_tiled2d` |")
    add("| 搜索空间 | 原始通用配置 | state_bf16_v6 最终 10 个 curated Triton configs |")
    add("| oneDNN | `aten/mm` 候选同时参与比较 | `aten/mm` 候选同时参与比较 |")
    add("| 数据类型 | BF16 | BF16 |")
    add("| 选择方式 | Inductor max-autotune | Inductor max-autotune |")
    add("")
    add("State B 使用 state_bf16_v6 最终 dispatch table 中的 10 个 Triton configs，并已部署到 Inductor 的 `triton.py`。")
    add("")
    add("## 2. 重要口径")
    add("")
    add("- **Triton 相对 oneDNN 的加速比**：`oneDNN_time / Triton_time`；大于 1 表示 Triton 更快。")
    add("- **A→B 加速比**：`A_time / B_time`；大于 1 表示 State B 更快。")
    add("- 若存在 `M=1` shape 被 Inductor 分解为 elementwise + reduction，它们会单独列出，不纳入 template comparison 的几何平均。")
    add("")
    add("### Efficiency 口径")
    add("")
    add(f"- BF16 GEMM FLOPs：`2 × M × N × K`。")
    add(f"- 读写字节数估算：`2 × (M×K + K×N + M×N)`，其中 2 是 BF16 每元素字节数。")
    add(f"- 硬件参考峰值：BF16 **{PEAK_BF16_TFLOPS:.0f} TFLOP/s**，DRAM BW **{PEAK_BW_GBPS:.0f} GB/s**。")
    add("- 当前 JSON 中的时间是 Inductor autotune/host-observed 时间，不是 unitrace device kernel time。因此下表的 TFLOP/s、BW 和 efficiency 是**基于 host 时间的估算值**，不能等同于真实 kernel 硬件利用率；要得到严格 device efficiency，需要对每个 shape 采集 unitrace device execution time。")
    add("")
    add("## 3. 总结")
    add("")
    add("| 指标 | 结果 |")
    add("|---|---:|")
    add(f"| 总 shape 数 | {len(data)} |")
    add(f"| 分解 shape（不进入 MM template） | {len(decomposed)} |")
    add(f"| 完整 A/B template comparison | {len(normal)} |")
    add(f"| Triton 相对 oneDNN，State A 几何平均 | {gmean(a_ratios):.4f}x |")
    add(f"| Triton 相对 oneDNN，State B 几何平均 | {gmean(b_ratios):.4f}x |")
    add(f"| Triton 胜 oneDNN，State A | {len(a_win)}/{len(normal)} ({100*len(a_win)/len(normal):.1f}%) |")
    add(f"| Triton 胜 oneDNN，State B | {len(b_win)}/{len(normal)} ({100*len(b_win)/len(normal):.1f}%) |")
    add(f"| Triton-only A→B 几何平均 | {gmean(ab_triton):.4f}x |")
    add(f"| System-level A→B 几何平均 | {gmean(ab_system):.4f}x |")
    add(f"| System-level B 快于 A（>2%） | {len(b_faster)} |")
    add(f"| System-level A 快于 B（>2%） | {len(a_faster)} |")
    add(f"| System-level 差异在 ±2% 内 | {same} |")
    add("")
    add("### 结论")
    add("")
    add(f"1. 在 Triton-only 口径下，State B 几何平均为 **{gmean(ab_triton):.4f}x**，即相对 State A 约提升 **{(gmean(ab_triton)-1)*100:.2f}%**。")
    add(f"2. 由于 oneDNN 会参与 Inductor 选择，系统实际收益被 fallback 隐藏，system-level 几何平均为 **{gmean(ab_system):.4f}x**，约提升 **{(gmean(ab_system)-1)*100:.2f}%**。")
    add(f"3. State B 的 Triton 相对 oneDNN 几何平均为 **{gmean(b_ratios):.4f}x**；共有 **{len(b_win)}/{len(normal)}** 个完整 shape 上 Triton 快于 oneDNN。")
    add("")
    add("## 4. Triton 相对 oneDNN：逐 shape")
    add("")
    add("`ratio A/B = oneDNN_time / Triton_time`；`A→B = A_Triton_time / B_Triton_time`。时间单位为 ms。")
    add("")
    add("| Shape | A Triton | B Triton | oneDNN | ratio A | ratio B | A→B Triton | A→B system | B winner |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for x in normal:
        s = f"({x['m']},{x['n']},{x['k']})"
        add(f"| {s} | {x['a']['best_triton']:.4f} | {x['b']['best_triton']:.4f} | {x['onednn']:.4f} | {x['a_ratio']:.3f}x | {x['b_ratio']:.3f}x | {x['ab_triton']:.3f}x | {x['ab_system']:.3f}x | {x['b'].get('best_triton_name','N/A')} |")
    add("")
    add("## 5. State B Efficiency（host-time estimate）")
    add("")
    add("以下效率使用 State B 最优 Triton 时间计算；因为时间来源是 host-observed autotune latency，短 kernel 可能包含 dispatch/launcher 开销，故仅用于相对参考。")
    add("")
    add("| Shape | Time (ms) | TFLOP/s est. | Compute Eff. | BW est. (GB/s) | BW Eff. | Arithmetic Intensity (F/B) |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for x in normal:
        m = x["b_metrics"]
        intensity = m["flops"] / m["bytes"]
        add(f"| ({x['m']},{x['n']},{x['k']}) | {x['b']['best_triton']:.4f} | {m['tflops']:.2f} | {m['tflops_eff']:.1f}% | {m['gbps']:.1f} | {m['bw_eff']:.1f}% | {intensity:.2f} |")
    add("")
    add("### Efficiency 汇总")
    add("")
    add("| 指标 | 几何平均 | 算术平均 | 最大值 |")
    add("|---|---:|---:|---:|")
    for name, label, suffix in [
        ("tflops", "TFLOP/s", ""),
        ("tflops_eff", "Compute efficiency", "%"),
        ("gbps", "BW (GB/s)", ""),
        ("bw_eff", "BW efficiency", "%"),
    ]:
        values = [x["b_metrics"][name] for x in normal]
        add(f"| {label} | {gmean(values):.2f}{suffix} | {sum(values)/len(values):.2f}{suffix} | {max(values):.2f}{suffix} |")
    add("")
    add("## 6. State B winner 分布")
    add("")
    add("| Winner | Shapes |")
    add("|---|---:|")
    for name, count in template_counts.most_common():
        add(f"| {name} | {count} |")
    add("")
    add("## 7. 最大提升与回归")
    add("")
    add("### Triton-only 最大提升")
    add("")
    add("| Shape | A Triton (ms) | B Triton (ms) | A→B |")
    add("|---|---:|---:|---:|")
    for x in sorted(normal, key=lambda y: y["ab_triton"], reverse=True)[:10]:
        add(f"| ({x['m']},{x['n']},{x['k']}) | {x['a']['best_triton']:.4f} | {x['b']['best_triton']:.4f} | {x['ab_triton']:.3f}x |")
    add("")
    add("### Triton-only 最大回归")
    add("")
    add("| Shape | A Triton (ms) | B Triton (ms) | A→B |")
    add("|---|---:|---:|---:|")
    for x in sorted(normal, key=lambda y: y["ab_triton"])[:10]:
        add(f"| ({x['m']},{x['n']},{x['k']}) | {x['a']['best_triton']:.4f} | {x['b']['best_triton']:.4f} | {x['ab_triton']:.3f}x |")
    add("")
    add("## 8. 分解 shape")
    add("")
    add("这些 shape 不进入 MM template，A/B 配置变化理论上不影响其计算路径；本次仅保留其手工测得的 host 时间。")
    add("")
    add("| Shape | A elapsed (ms) | B elapsed (ms) |")
    add("|---|---:|---:|")
    for m, n, k, at, bt in decomposed:
        add(f"| ({m},{n},{k}) | {fmt(at)} | {fmt(bt)} |")
    add("")
    add("## 9. 限制与后续")
    add("")
    add("1. A/B 每个 shape 使用独立 subprocess，减少编译缓存、singleton heuristic 和跨 shape OOM 的影响；但 GPU clock/power state 仍可能造成短 kernel 的测量噪声。")
    add("2. oneDNN 时间来自同一轮 autotune candidate comparison；State B 的 system-level 结果按 `min(best_triton, oneDNN)` 计算，更接近实际 dispatch。")
    add("3. 若需要可发表/硬件利用率意义上的 compute/BW efficiency，应对 State A、State B 的最终 winner 和 oneDNN 重新进行 device-only profiling（例如 unitrace `--device-timing`），再替换本报告的 host-time estimates。")
    if errors:
        add("")
        add(f"> Warning: {len(errors)} 个非分解 shape 数据不完整，未纳入汇总。")

    args.output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
