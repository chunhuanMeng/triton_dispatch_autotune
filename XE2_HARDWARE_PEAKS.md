# BMG / Arc Pro B60 measured hardware limits (VERIFIED 2026-08-11)

Condensed reference. Full experiments and derivations: `XE2_PERF_METHODOLOGY.md`.

## BF16 peak is 98.3 TF/s, NOT 117 TF/s

- Arc Pro B60 = 20 Xe cores, `Core Clock Rate: 2400 MHz` (xpu-smi), observed 2392-2399 MHz under load
- peak = 20 x 2048 FLOP/clk x 2.4e9 = **98.3 TF/s**
- 117 TF/s corresponds to 2.85 GHz (Arc B580) and is wrong for this part

Verified by measurement (`torch.mm` bf16, hot loop):

| shape | TF/s | % of 98.3 |
|---|---:|---:|
| (2048,7168,28672) | 97.51 | 99.2% |
| (8192,8192,8192) | 96.93 | 98.6% |
| (4096,11008,4096) | 94.93 | 96.6% |
| (4096,4096,4096) | 89.83 | 91.4% |

**All earlier "% of peak" numbers in this repo are understated by 117/98.3 = 1.19x.**
e.g. (2048,7168,28672) Triton 8.88 ms = 94.8 TF/s = **96.4% of peak**, not 81%.

## Hardware counter proof of the peak model

`unitrace -q -g ComputeBasic` on an 8192^3 mm:

- `XVE_INST_EXECUTED_ALU2_ALL` = 2,147,483,648 events
- `M*N*K / events` = **256 MAC per ALU2 slot** (= 512 FLOP), exactly integral
- `ALU2 events / GpuCoreClocks` = 78.80, ceiling = 20 cores x 4 slots/clk = 80 -> 98.5%
- => 4 slots/clk/core x 512 FLOP = **2048 FLOP/clk/core**, confirming the peak formula
- `XVE_INST_EXECUTED_ALU2_ALL_UTILIZATION[%]` equals achieved/peak exactly
  (8192^3: 98.5%; (256,4096,4096) oneDNN 69.9%, Triton 65.5%)

Caveat: that utilization metric is only meaningful for compute-bound shapes. See
`xe2_bottleneck_methodology.md`.

## Achievable DRAM bandwidth ~400-450 GB/s (theoretical 456)

| kernel | GB/s | % of 456 |
|---|---:|---:|
| copy (r+w) | 398.1 | 87.3% |
| add (2r+w) | 399.7 | 87.7% |
| sum (read only) | 448.0 | 98.2% |
| 33.6 MB bf16 read | 407.6 | 89.4% |

Use **~407 GB/s** as the practical read roofline, not 456.

## Derived constants

| quantity | value |
|---|---:|
| Xe cores | 20 |
| XVE per Xe core | 8 |
| XMX per Xe core | 8 |
| hardware threads | 1280 |
| 1 ALU2 slot | 256 MAC = 512 FLOP |
| ALU2 slots/clk/core | 4 |
| FLOP/clk/core (bf16) | 2048 |
| BF16 peak | 98.3 TF/s |
| achievable DRAM BW | ~407 GB/s |
| machine balance | 242 FLOP/byte |
| instruction issue ceiling | 160 instr/clk |
| L2 (Device Cache) | 18 MB |
