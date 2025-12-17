#!/bin/bash
# ============================================================
# Exp1217 Phase 1: Loss 組合測試
# ============================================================
#
# 實驗矩陣:
# - Exp40: Feature=1.0, CE=0.5 (GPU 0)
# - Exp41: Feature=0.5, CE=1.0 (GPU 1)
# - Exp42: Feature=1.0, Triplet=0.5, CE=0.5 (GPU 2)
# - Exp43: Triplet=0.5, CE=1.0 (等待空閒 GPU)
#
# ============================================================

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1217

echo "============================================================"
echo "Launching Exp1217 Phase 1: Loss 組合測試"
echo "============================================================"
echo ""

# 確保腳本可執行
chmod +x run_exp40_feature_ce.sh
chmod +x run_exp41_ce_feature.sh
chmod +x run_exp42_balanced.sh
chmod +x run_exp43_ce_triplet.sh

echo "Starting Exp40 (GPU 0): Feature + CE..."
nohup bash run_exp40_feature_ce.sh > logs/exp40.log 2>&1 &
echo "  PID: $!"

sleep 5

echo "Starting Exp41 (GPU 1): CE + Feature..."
nohup bash run_exp41_ce_feature.sh > logs/exp41.log 2>&1 &
echo "  PID: $!"

sleep 5

echo "Starting Exp42 (GPU 2): Balanced..."
nohup bash run_exp42_balanced.sh > logs/exp42.log 2>&1 &
echo "  PID: $!"

echo ""
echo "============================================================"
echo "Phase 1 experiments launched!"
echo "============================================================"
echo ""
echo "Monitor with:"
echo "  tail -f logs/exp40.log"
echo "  tail -f logs/exp41.log"
echo "  tail -f logs/exp42.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Exp43 needs to wait for a free GPU."
echo "Run manually after one of Exp40-42 completes:"
echo "  nohup bash run_exp43_ce_triplet.sh > logs/exp43.log 2>&1 &"
