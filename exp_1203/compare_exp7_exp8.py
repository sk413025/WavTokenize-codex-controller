#!/usr/bin/env python3
"""
比較 exp7 和 exp8 的訓練結果，分析為什麼差異不大
"""

import json
import numpy as np

# 載入數據
with open('experiments/feature_correct_vq/training_history.json') as f:
    exp7 = json.load(f)

with open('experiments/emb_distillation/training_history.json') as f:
    exp8 = json.load(f)

print("=" * 70)
print("exp7 vs exp8 結果比較")
print("=" * 70)

# 比較最終結果 (取相同 epoch 數)
epochs_to_compare = min(len(exp7['epochs']), 30)  # exp7 只有 30 epochs

print(f"\n比較前 {epochs_to_compare} epochs 的結果:")
print("-" * 70)

# Token Accuracy
exp7_train_acc = exp7['train_token_acc'][:epochs_to_compare]
exp7_val_acc = exp7['val_token_acc'][:epochs_to_compare]
exp8_train_acc = exp8['train_token_acc'][:epochs_to_compare]
exp8_val_acc = exp8['val_token_acc'][:epochs_to_compare]

print(f"\n📊 Token Accuracy:")
print(f"{'Metric':<25} {'exp7 (Feature+VQ)':<20} {'exp8 (EmbDist)':<20} {'差異':<15}")
print("-" * 70)
print(f"{'Train Acc (初始)':<25} {exp7_train_acc[0]*100:>17.2f}% {exp8_train_acc[0]*100:>17.2f}% {(exp8_train_acc[0]-exp7_train_acc[0])*100:>+12.2f}%")
print(f"{'Train Acc (最終)':<25} {exp7_train_acc[-1]*100:>17.2f}% {exp8_train_acc[-1]*100:>17.2f}% {(exp8_train_acc[-1]-exp7_train_acc[-1])*100:>+12.2f}%")
print(f"{'Val Acc (初始)':<25} {exp7_val_acc[0]*100:>17.2f}% {exp8_val_acc[0]*100:>17.2f}% {(exp8_val_acc[0]-exp7_val_acc[0])*100:>+12.2f}%")
print(f"{'Val Acc (最終)':<25} {exp7_val_acc[-1]*100:>17.2f}% {exp8_val_acc[-1]*100:>17.2f}% {(exp8_val_acc[-1]-exp7_val_acc[-1])*100:>+12.2f}%")

# 趨勢分析
print(f"\n📈 趨勢分析:")
print(f"{'Metric':<25} {'exp7':<20} {'exp8':<20}")
print("-" * 70)

# Train Acc 變化
exp7_train_change = (exp7_train_acc[-1] - exp7_train_acc[0]) * 100
exp8_train_change = (exp8_train_acc[-1] - exp8_train_acc[0]) * 100
print(f"{'Train Acc 變化':<25} {exp7_train_change:>+17.2f}% {exp8_train_change:>+17.2f}%")

# Val Acc 變化
exp7_val_change = (exp7_val_acc[-1] - exp7_val_acc[0]) * 100
exp8_val_change = (exp8_val_acc[-1] - exp8_val_acc[0]) * 100
print(f"{'Val Acc 變化':<25} {exp7_val_change:>+17.2f}% {exp8_val_change:>+17.2f}%")

# 最佳值
print(f"{'Train Acc 最佳值':<25} {max(exp7_train_acc)*100:>17.2f}% {max(exp8_train_acc)*100:>17.2f}%")
print(f"{'Val Acc 最佳值':<25} {max(exp7_val_acc)*100:>17.2f}% {max(exp8_val_acc)*100:>17.2f}%")

# Loss 分析
print(f"\n📉 Loss 分析:")
print("-" * 70)

# 注意：exp7 的 distance_loss 是 CE Loss (值 ~3.0)
# exp8 的 distance_loss 是 MSE Loss (值 ~0.03)
# 這兩個不能直接比較！

exp7_feature_loss = exp7['train_feature_loss'][:epochs_to_compare]
exp8_emb_loss = exp8['train_distance_loss'][:epochs_to_compare]  # 這是 emb_to_codebook loss

print(f"exp7 Feature Loss: {exp7_feature_loss[0]:.6f} → {exp7_feature_loss[-1]:.6f} (變化: {exp7_feature_loss[-1]-exp7_feature_loss[0]:+.6f})")
print(f"exp8 Emb Loss:     {exp8_emb_loss[0]:.6f} → {exp8_emb_loss[-1]:.6f} (變化: {exp8_emb_loss[-1]-exp8_emb_loss[0]:+.6f})")

print("\n" + "=" * 70)
print("關鍵發現")
print("=" * 70)

print("""
1. Token Accuracy 都很低 (< 15%)
   - exp7 Train: 30% → 10% (下降!)
   - exp8 Train: 26% → 14.5% (下降但較緩)
   - 兩者 Val Acc 都在 5-15% 範圍

2. 為什麼 exp7 和 exp8 結果差異不大？

   可能原因：
   a) 問題不在 Loss 設計，而在 LoRA 容量不足
   b) 問題不在 Loss 設計，而在任務本身太難
   c) 梯度雖然強 55 倍，但方向可能仍然不對
   d) 需要更多訓練時間才能看出差異
""")

# 計算關鍵指標
print("\n" + "=" * 70)
print("深入分析：為什麼結果差異不大？")
print("=" * 70)

# 假設 1: 檢查 Loss 是否真的在下降
exp7_loss_trend = np.polyfit(range(len(exp7_train_acc)), exp7_train_acc, 1)[0]
exp8_loss_trend = np.polyfit(range(len(exp8_train_acc[:epochs_to_compare])), exp8_train_acc[:epochs_to_compare], 1)[0]

print(f"\n假設檢驗:")
print("-" * 70)
print(f"1. Train Acc 趨勢斜率:")
print(f"   exp7: {exp7_loss_trend*100:.4f}% per epoch")
print(f"   exp8: {exp8_loss_trend*100:.4f}% per epoch")

if exp7_loss_trend < 0:
    print(f"   ⚠️  exp7 Train Acc 在下降！這是異常的")
if exp8_loss_trend < 0:
    print(f"   ⚠️  exp8 Train Acc 在下降！這是異常的")

# 假設 2: 初始 Token Acc 就很低
print(f"\n2. 初始 Token Accuracy 分析:")
print(f"   exp7 初始 Train Acc: {exp7_train_acc[0]*100:.2f}%")
print(f"   exp8 初始 Train Acc: {exp8_train_acc[0]*100:.2f}%")
print(f"   Baseline (未訓練的模型): 應該接近 1/4096 = 0.024%")
print(f"   → 初始值遠高於 baseline，說明 LoRA 一開始就有影響")

# 假設 3: Val Acc 波動分析
exp7_val_std = np.std(exp7_val_acc)
exp8_val_std = np.std(exp8_val_acc[:epochs_to_compare])
print(f"\n3. Val Acc 穩定性:")
print(f"   exp7 Val Acc std: {exp7_val_std*100:.2f}%")
print(f"   exp8 Val Acc std: {exp8_val_std*100:.2f}%")
if exp7_val_std > 0.02 or exp8_val_std > 0.02:
    print(f"   ⚠️  Val Acc 波動很大，說明模型泛化不穩定")

print("\n" + "=" * 70)
print("結論")
print("=" * 70)
print("""
exp7 和 exp8 結果差異不大的原因：

1. 【核心問題】任務難度太高
   - noisy audio → clean tokens 的映射非常複雜
   - 4096-way 分類問題
   - LoRA 只有 0.19% 參數可訓練

2. 【梯度強度不是唯一因素】
   - exp8 梯度強 55 倍，但仍然無法有效訓練
   - 說明問題不只是梯度強度，還有梯度方向

3. 【模型容量限制】
   - LoRA rank=64 可能不足以學習 noisy → clean 映射
   - 154,048 參數 vs 4096 classes × 512 dims = 2M 目標空間

4. 【建議下一步】
   - 嘗試增加 LoRA rank (128 或 256)
   - 或者使用 c_code 的 Transformer 架構 + Speaker Embedding
   - 驗證 baseline (未訓練模型) 的 Token Acc 是多少
""")
