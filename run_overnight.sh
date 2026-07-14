#!/bin/bash
cd /home/sdp/meng/int8_gemm_optimization_xe2/triton_dispatch_autotune
source /opt/intel/oneapi/setvars.sh 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate chunhuan

echo "=== Starting overnight autotune: $(date) ===" >> tune_1.log
python -u run_autotune.py --step seed >> tune_1.log 2>&1
echo "=== Seed done: $(date) ===" >> tune_1.log
python -u run_autotune.py --step iterate >> tune_1.log 2>&1
echo "=== Iterate done: $(date) ===" >> tune_1.log
python -u run_autotune.py --step report >> tune_1.log 2>&1
echo "=== All done: $(date) ===" >> tune_1.log
