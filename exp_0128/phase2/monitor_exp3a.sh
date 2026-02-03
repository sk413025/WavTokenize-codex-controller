#!/bin/bash
while true; do
    clear
    echo "=== Exp 3a (λ=0.01) - Original ==="
    tail -5 exp_0128/phase2/entropy_regularization/run_exp3a_20260130_001754.log 2>/dev/null || echo "Not found"
    echo ""
    echo "=== Exp 3b (λ=0.05) - Latest ==="
    tail -5 exp_0128/phase2/entropy_regularization/run_exp3b_20260202_002130.log 2>/dev/null || echo "Not found"
    echo ""
    echo "=== Exp 3c (λ=0.10) - Latest ==="
    tail -5 exp_0128/phase2/entropy_regularization/run_exp3c_20260202_002209.log 2>/dev/null || echo "Not found"
    echo ""
    echo "=== GPU Usage ==="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    echo ""
    echo "Press Ctrl+C to exit. Refreshing in 10 seconds..."
    sleep 10
done
