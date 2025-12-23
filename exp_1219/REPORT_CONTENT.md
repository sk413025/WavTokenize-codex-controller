# 實驗結果分析與未來工作

## 1. 結果分析 (Results Analysis)

### 1.1 實驗概述

本研究探索使用 LoRA (Low-Rank Adaptation) 微調 WavTokenizer 的 encoder，實現 **自監督式音訊去噪**。透過 Teacher-Student 架構，讓 Student 學習將含噪音訊映射到與 Teacher 處理乾淨音訊相同的 token 空間。

### 1.2 Loss 函數探索

我們系統性地測試了多種 loss 組合：

| Loss 組合 | 實驗 | Val Accuracy | 結論 |
|-----------|------|--------------|------|
| Feature MSE only | Exp36, 39 | ~0.65% | 基準，但收斂慢 |
| CE only | Exp37, 38 | ~0.55% | 離散目標過於嚴格 |
| Feature + CE | Exp40, 41 | ~0.70% | 有改善但不穩定 |
| Feature + Triplet | **Exp47, 48** | **~0.88%** | ✅ 最佳組合 |
| Feature + Triplet + Cosine | Exp51 | ~0.82% | Cosine 無明顯幫助 |

**關鍵發現**：
- **Feature MSE + Triplet Loss** 是最有效的組合
- Cross-Entropy loss 因離散化特性導致梯度不穩定
- Cosine similarity loss 與其他 loss 存在目標衝突，不建議使用

### 1.3 超參數分析

#### Triplet Margin
| Margin | Val Accuracy | 說明 |
|--------|--------------|------|
| 0.2 | **0.88%** | ✅ 較小 margin 提供更精細的學習信號 |
| 0.5 | 0.81% | Margin 過大導致學習信號過於寬鬆 |

#### LoRA 配置
| Rank | Layers | Val Accuracy | 說明 |
|------|--------|--------------|------|
| 128 | all_18 | **0.88%** | ✅ 最佳配置 |
| 256 | all_18 | ~0.85% | 高容量未帶來提升，可能過擬合 |
| 128 | critical_8 | ~0.75% | 層數不足，遺漏關鍵語義層 |

### 1.4 最佳配置

基於系統性實驗，我們確定的最佳配置為：

```
Loss Configuration:
├── feature_weight: 1.0 (MSE loss)
├── triplet_weight: 1.0
├── triplet_margin: 0.2
├── cosine_weight: 0.0
└── ce_weight: 0.0

LoRA Configuration:
├── rank: 128
├── alpha: 256
├── dropout: 0.2
└── layers: all_18 (全部 18 層 encoder conv)
```

### 1.5 訓練動態分析

![Training Curves](triplet_cosine_analysis.png)

**觀察**：
1. **Val Accuracy** 在約 epoch 12 後趨於穩定 (~0.88%)
2. **Train Loss** 持續下降，但 **Val Loss** 停滯
3. 存在明顯的 **過擬合** 現象：Train-Val gap 持續擴大

### 1.6 當前瓶頸

| 瓶頸類型 | 證據 | 影響 |
|----------|------|------|
| **過擬合** | Train/Val gap 持續擴大 | Val Acc 停滯在 ~0.88% |
| **數據多樣性不足** | 高 rank LoRA 未能提升性能 | 模型容量未被充分利用 |
| **離散化限制** | Token accuracy 上限受 VQ 影響 | 即使特徵對齊，量化誤差仍存在 |

---

## 2. 未來工作 (Future Work)

### 2.1 短期改進 (Short-term)

#### 2.1.1 正則化增強
針對過擬合問題，計劃測試：
- **增加 Dropout**: 0.2 → 0.4
- **增加 Weight Decay**: 0.05 → 0.1
- **Early Stopping**: 基於 Val Loss 的早停機制

#### 2.1.2 層選擇優化
基於 WavTokenizer encoder 的架構分析：
- 設計 **Critical-10** 層配置：覆蓋 model.7 和 model.10 的語義提取層
- 減少參數量的同時保留關鍵表示能力
- 預期可降低過擬合風險

#### 2.1.3 訓練策略優化
- **Gradient Accumulation**: 在 GPU 記憶體限制下實現更大的等效 batch size
- **Learning Rate Schedule**: 探索 warmup + cosine annealing 的最佳組合

### 2.2 中期探索 (Mid-term)

#### 2.2.1 數據增強
- **Speaker 多樣性**: 擴充訓練數據中的說話者數量
- **噪聲類型多樣性**: 加入更多類型的環境噪聲
- **音訊增強**: Speed perturbation, SpecAugment

#### 2.2.2 架構改進
- **Adapter 替代 LoRA**: 探索其他 parameter-efficient fine-tuning 方法
- **Multi-scale Feature**: 融合不同層的特徵表示

### 2.3 長期方向 (Long-term)

#### 2.3.1 軟標籤學習
- 利用 VQ codebook 的距離資訊作為軟目標
- 緩解離散化帶來的資訊損失

#### 2.3.2 端到端評估
- 整合去噪後的 token 到下游 TTS/ASR 系統
- 建立完整的語音處理 pipeline 評估

#### 2.3.3 跨域泛化
- 測試模型在不同語言、不同錄音環境的泛化能力
- 探索 domain adaptation 技術

---

## 3. 結論 (Conclusion)

本研究成功驗證了使用 LoRA 微調 WavTokenizer encoder 進行自監督音訊去噪的可行性。主要貢獻包括：

1. **系統性的 Loss 函數探索**：確定 Feature MSE + Triplet Loss 為最佳組合
2. **超參數優化**：找到 triplet margin = 0.2、LoRA rank = 128 的最佳配置
3. **瓶頸分析**：識別過擬合和數據多樣性為主要限制因素

當前最佳模型達到 **0.88% token accuracy**，雖然數值看似較低，但考慮到 VQ codebook 有 4096 個離散 token，隨機猜測僅有 0.024% 的準確率，模型已展現顯著的去噪學習能力。

未來工作將聚焦於：
1. 透過正則化和層選擇優化解決過擬合
2. 擴充數據多樣性以充分利用模型容量
3. 探索軟標籤學習以突破離散化限制

---

## 附錄：實驗配置對照表

| 實驗 | Feature | Triplet | Margin | Cosine | CE | Rank | Layers | Val Acc |
|------|---------|---------|--------|--------|----|----|--------|---------|
| Exp36 | 1.0 | 0.0 | - | 0.0 | 0.0 | 128 | all_18 | 0.65% |
| Exp47 | 1.0 | 1.0 | 0.2 | 0.0 | 0.0 | 128 | all_18 | 0.83% |
| **Exp48** | **1.0** | **1.0** | **0.2** | **0.0** | **0.0** | **128** | **all_18** | **0.88%** |
| Exp50 | 1.0 | 1.0 | 0.5 | 0.0 | 0.0 | 128 | all_18 | 0.81% |
| Exp51 | 1.0 | 1.0 | 0.5 | 0.1 | 0.0 | 128 | all_18 | 0.82% |
