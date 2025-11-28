#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# exp_1128: 啟動所有實驗 (並行)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 實驗矩陣:
# ┌──────────────┬─────────────┬─────────────┐
# │              │ Dist=0.05   │ Dist=0.1    │
# ├──────────────┼─────────────┼─────────────┤
# │ LoRA Rank=32 │ exp1 (GPU0) │ exp2 (GPU0) │
# │ LoRA Rank=64 │ exp3 (GPU1) │ exp4 (GPU1) │
# └──────────────┴─────────────┴─────────────┘
#
# 注意: 同一 GPU 上的實驗需要錯開時間執行

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1128

echo "=================================================="
echo "exp_1128: LoRA Rank + Distance Loss Grid Search"
echo "=================================================="
echo ""
echo "Baseline (exp_1126/1126-1):"
echo "  - LoRA Rank: 16, Distance Loss: 0.01"
echo "  - Code Distance: 5.38 -> 4.40 (-18%)"
echo "  - Feature Cosine: 0.34 -> 0.59 (+72%)"
echo ""
echo "New experiments:"
echo "  exp1: rank=32, dist=0.05 (GPU0)"
echo "  exp2: rank=32, dist=0.1  (GPU0) - after exp1"
echo "  exp3: rank=64, dist=0.05 (GPU1)"
echo "  exp4: rank=64, dist=0.1  (GPU1) - after exp3"
echo ""

# GPU 0: exp1 先跑
echo "Starting exp1 on GPU0..."
bash run_exp1_r32_dist0.05.sh

# GPU 1: exp3 先跑
echo "Starting exp3 on GPU1..."
bash run_exp3_r64_dist0.05.sh

echo ""
echo "=================================================="
echo "Two experiments started!"
echo ""
echo "After exp1 finishes, run: bash run_exp2_r32_dist0.1.sh"
echo "After exp3 finishes, run: bash run_exp4_r64_dist0.1.sh"
echo ""
echo "Monitor logs:"
echo "  tail -f experiments/lora_r32_dist0.05.log"
echo "  tail -f experiments/lora_r64_dist0.05.log"
echo "=================================================="
