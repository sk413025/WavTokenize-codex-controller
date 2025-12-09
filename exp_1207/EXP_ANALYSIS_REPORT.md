# Exp_1207 實驗分析報告

## 實驗概述

| 實驗 | 描述 | Feature Loss | Token Accuracy |
|------|------|--------------|----------------|
| feature_only | 純 Feature MSE Loss | 0.0354 → 0.0278 | 16% → 3% |
| exp16a_feature_ce_equal | Feature + CE Loss (1:1) | 177.8 → 137.8 | 17% → 1.3% |

## 關鍵發現

### 1. Feature Loss 下降但 Token Accuracy 反而下降

驗證結果（epoch 30 checkpoint）：
- **Feature Loss**: 0.072 → 0.045 (改善 37%)
- **Token Accuracy**: 4.99% → 0.91% (下降 82%!)

### 2. 根本原因：MSE 優化方向錯誤

使用 168,000 個樣本驗證：

```
Student 移動方向與正確方向的 cosine similarity:
  Mean:   0.061（幾乎正交！）
  Std:    0.584
  Median: 0.092

分布統計:
  正向移動比例 (cosine > 0):  54.4%（接近隨機 50%）
  強正向 (cosine > 0.5):      29.5%
  負向移動比例 (cosine < 0):  45.6%
  強負向 (cosine < -0.5):     22.4%
```

MSE Loss 只優化「整體距離」，不保證移動方向正確。
詳細數據見：`direction_analysis_evidence.json`

### 3. VQ 空間的幾何特性

```
Codebook: 4096 codes, 512 dimensions
平均最近鄰距離: 0.57
Voronoi cell 半徑: ~0.28
Noise 造成的位移: ~4.65
位移/半徑比: 16.4x
```

Noise 造成的特徵位移是 Voronoi cell 半徑的 16 倍！
要讓 token 正確，需要把特徵拉回正確的 cell，這需要極大的修正。

### 4. LoRA 容量不足

```
可訓練參數: 154,048 (0.19%)
總參數: 80,706,468
```

LoRA 只修改 encoder 的 4 個卷積層，容量太小無法做大幅度修正。

### 5. 梯度分析（非衝突但無意義）

```
MSE vs CE 梯度 cosine similarity: 0.993（高度對齊）
但：
  - CE 梯度量級: ~200
  - MSE 梯度量級: ~0.04
```

兩個 loss 方向一致，但都指向錯誤的方向。

## 結論

### 根本問題

1. **Feature MSE 不是正確的目標**：只優化距離不保證跨越 Voronoi 邊界
2. **LoRA 容量不足**：0.19% 的參數無法做大幅度修正
3. **Noise 幅度太大**：造成的位移遠超過 VQ 的容錯範圍

### 為什麼加上 CE Loss 也沒用？

CE Loss 的目標是讓 softmax(logits) 的正確類別機率最大化。
但當 encoder output 距離正確 code 太遠時（距離 4.21），
即使 CE Loss 試圖拉向正確方向，LoRA 的修正能力也不足以完成這個任務。

### 可能的改進方向

1. **增加 LoRA rank**：從 64 提升到 256 或更高
2. **擴大可訓練範圍**：不只訓練 encoder，也訓練部分 VQ 層
3. **使用不同的 loss**：例如 Triplet Loss、Contrastive Loss
4. **降低 noise 幅度**：使用更溫和的數據增強
5. **重新思考架構**：可能需要額外的 adapter 層

## 實驗文件

- `train.py`: Feature-Only 訓練腳本
- `train_with_ce.py`: Feature + CE Loss 訓練腳本
- `analyze_gradient_conflict.py`: 梯度衝突分析工具
- `gradient_analysis/`: 梯度分析結果
- `experiments/feature_only/`: Feature-Only 實驗結果
- `experiments/exp16a_feature_ce_equal/`: Feature+CE 實驗結果
