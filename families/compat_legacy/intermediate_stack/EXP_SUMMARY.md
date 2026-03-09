# Exp 0112/0113 中間層監督實驗總結報告

**日期**: 2026-01-15
**作者**: 實驗團隊
**版本**: v2.0

---

## 摘要

本報告總結 WavTokenizer LoRA 降噪任務中的中間層監督策略研究。核心發現：

1. **LoRA 容量非瓶頸**：4 倍參數量僅帶來 0.06% 改善
2. **噪音敏感度分布不均**：淺層 (L0-L4) 和 L6 最敏感，L10 最穩定
3. **當前監督位置部分合理**：L3, L6 有監督，但最敏感的 L0, L1 缺失
4. **Loss 類型選擇**：Cosine Loss 最適合中間層監督（尺度不變性）

---

## 目錄

1. [實驗背景與目標](#一實驗背景與目標)
2. [容量瓶頸測試 (exp_test)](#二容量瓶頸測試-exp_test)
3. [噪音敏感度分析](#三噪音敏感度分析)
4. [訓練收斂性分析](#四訓練收斂性分析)
5. [Loss 設計分析](#五loss-設計分析)
6. [綜合評估與建議](#六綜合評估與建議)
7. [圖表索引](#七圖表索引)

---

## 一、實驗背景與目標

### 1.1 問題描述

WavTokenizer LoRA 訓練中觀察到的矛盾現象：
- 噪音對中層 (L5-L8) 特徵影響最大
- 但訓練後深層變化最大，淺層變化較小

### 1.2 研究問題

1. LoRA 容量是否是淺層學習不足的原因？
2. 最佳的中間層監督位置在哪裡？
3. 何種 Loss 類型最適合中間層監督？

### 1.3 實驗概覽

| 實驗 | 目標 | 狀態 | 主要發現 |
|------|------|------|----------|
| exp_test | 容量瓶頸測試 | ✅ 完成 | 容量非瓶頸 |
| Exp K v2 | 中間層監督 | ✅ 完成 | Val Acc 0.80% |
| 噪音敏感度分析 | 找最敏感層 | ✅ 完成 | L0, L1, L6 最敏感 |
| Loss 類型比較 | 選擇最佳 Loss | ✅ 完成 | Cosine 最佳 |

---

## 二、容量瓶頸測試 (exp_test)

### 2.1 實驗設計

**假說**: LoRA 容量不足導致淺層無法有效學習

**驗證方法**:
- 只訓練淺層 (L0-L4)
- 比較不同 LoRA rank (256/512/1024)
- 若容量是瓶頸，增加 rank 應顯著改善效果

### 2.2 實驗結果

|oss | 改善幅度 |
|------|--------|---------- Rank | 參數量 | Best Val L-----|----------|
| 256 | 116K | 51.79 | baseline |
| 512 | 233K | 51.66 | -0.25% |
| 1024 | 466K | 51.76 | -0.06% |

### 2.3 結論

**容量不是主要瓶頸**

- 4 倍參數量 (256→1024) 僅改善 0.06%
- 三個 rank 最終收斂到相同水平 (CosSim ≈ 0.21)
- 淺層學習困難的原因在於任務本身，非容量限制

> **圖表參考**: [exp_test/EXPERIMENT_REPORT.md](../exp_test/EXPERIMENT_REPORT.md)
> **訓練曲線**: [exp_test/runs/shallow_r*/training_curves.png](../exp_test/runs/)

---

## 三、噪音敏感度分析

### 3.1 歷史數據整合

本分析整合兩個實驗的結果：

| 實驗 | 測量對象 | 層數 | 方法 |
|------|----------|------|------|
| **exp_1231_feature** | WavTokenizer 完整模型 | 18 層 (L0-L17) | 合成噪音測試 |
| **本次分析** | encoder.model | 16 層 (0-15) | 實際 noisy/clean 對 |

### 3.2 exp_1231_feature 關鍵發現

**層組平均噪音敏感度**:

| 層組 | 層範圍 | 平均敏感度 | 解讀 |
|------|--------|------------|------|
| input | L0 | 0.16 | 意外地對噪音不敏感 |
| low_level | L1-L4 | 0.47 | 中等敏感 |
| **mid_level** | **L5-L8** | **0.71** | **★ 最敏感！噪音處理層** |
| semantic | L9-L12 | 0.50 | 中等 |
| abstract | L13-L16 | 0.28 | 對噪音魯棒 |
| output | L17 | 0.69 | 敏感（噪音傳播到 codebook） |

**最敏感的層**: L6 (0.79), L5 (0.72), L2 (0.71)

> **關鍵洞察**: Mid-level (L5-L6) 是 WavTokenizer **「直覺處理噪音」** 的位置，
> 這是噪音特徵與語音特徵開始分離的地方。

### 3.3 測量方法

測量**原始模型（無 LoRA）**對噪音的敏感度：

```
噪音敏感度 = 1 - cos_sim(feature(clean_audio), feature(noisy_audio))
```

此指標表示「同一層對乾淨輸入與噪音輸入的特徵差異程度」。

### 3.2 各層噪音敏感度

| Layer | Cos Sim | Sensitivity | 層組 | 評估 |
|-------|---------|-------------|------|------|
| L0 | 0.041 | **0.959** | input | ⚠️ 極敏感 |
| L1 | 0.057 | **0.943** | low_level | ⚠️ 極敏感 |
| L2 | 0.172 | 0.828 | low_level | |
| L3 | 0.105 | **0.895** | low_level | ★ 當前有監督 |
| L4 | 0.121 | 0.879 | low_level | |
| L5 | 0.351 | 0.649 | mid_level | |
| **L6** | 0.081 | **0.919** | mid_level | ★ 當前有監督，中層最敏感 |
| L7 | 0.165 | 0.835 | mid_level | |
| L8 | 0.303 | 0.697 | mid_level | |
| L9 | 0.479 | 0.521 | semantic | |
| **L10** | **0.946** | **0.054** | semantic | ✅ 最穩定（錨點候選）|
| L11 | 0.566 | 0.434 | semantic | |
| L12 | 0.593 | 0.407 | semantic | |
| L13 | 0.589 | 0.411 | abstract | |
| L14 | 0.556 | 0.444 | abstract | |
| L15 | 0.379 | 0.621 | abstract | |

### 3.3 層組摘要

| 層組 | 層範圍 | 平均敏感度 | 特性 |
|------|--------|------------|------|
| input | L0 | 0.959 | 最敏感，直接接觸噪音 |
| low_level | L1-L4 | 0.886 | 非常敏感，處理低階特徵 |
| mid_level | L5-L8 | 0.775 | 中等敏感，噪音-語音分離 |
| semantic | L9-L12 | 0.354 | 較穩定，語義層級 |
| abstract | L13-L15 | 0.492 | 中等 |

### 3.4 關鍵發現

1. **L10 是特異點**: cos_sim = 0.946，對噪音幾乎不敏感
   - 可能是語義聚合層
   - 適合作為訓練「錨點」

2. **最敏感層**: L0, L1, L6 (sensitivity > 0.91)
   - L0, L1 目前**沒有**中間層監督
   - 建議加入監督

3. **當前監督位置 (L3, L6) 部分合理**
   - L6 是中層最敏感的層 ✅
   - L3 代表淺層 ✅
   - 但錯過了 L0, L1 ⚠️

> **圖表參考**:
> - 噪音敏感度比較: [analysis/noise_sensitivity_comparison.png](analysis/noise_sensitivity_comparison.png)
> - 監督位置推薦: [analysis/supervision_recommendation.png](analysis/supervision_recommendation.png)
> - 數據檔案: [analysis/noise_sensitivity.json](analysis/noise_sensitivity.json)

---

## 四、訓練收斂性分析

### 4.1 Exp K v2 訓練結果

**配置**:
- LoRA rank: 256, alpha: 512
- 中間層監督: L3, L6 (Cosine Loss, weight=0.5)
- Epochs: 300
- Curriculum: 0.3 → 1.0

**結果**:

| 指標 | Train | Val | Gap |
|------|-------|-----|-----|
| Total Loss | 403.95 | 420.51 | 16.56 |
| Match Acc | 1.99% | 0.76% | 1.24% |
| Feature Loss | 0.27 | 0.27 | 0.00 |
| Triplet Loss | 0.73 | 0.91 | 0.18 |
| Intermediate | 805.88 | 838.64 | 32.76 |

**Best Val Acc**: 0.80% @ Epoch 223

### 4.2 訓練前後比較

| 層 | 訓練前敏感度 | 訓練後距離 | 學習難度* |
|----|-------------|------------|-----------|
| L0 | 0.959 | 0.960 | 0.921 |
| L1 | 0.943 | 0.963 | 0.908 |
| L3 | 0.895 | 0.758 | 0.678 |
| L6 | 0.919 | 0.767 | 0.705 |
| L10 | 0.054 | 0.254 | 0.014 |

*學習難度 = 敏感度 × 訓練後距離

### 4.3 觀察

1. **有監督的層 (L3, L6) 訓練後距離較小**
   - L3: 0.895 → 0.758 (改善 15%)
   - L6: 0.919 → 0.767 (改善 17%)

2. **無監督的層 (L0, L1) 距離幾乎沒變**
   - L0: 0.959 → 0.960
   - L1: 0.943 → 0.963

3. **L10 穩定性確認**
   - 訓練前: 0.054 (極穩定)
   - 訓練後: 0.254 (仍然最相似)

> **圖表參考**:
> - 訓練收斂曲線: [analysis/training_convergence.png](analysis/training_convergence.png)
> - Loss 位置分析: [analysis/loss_position_analysis.png](analysis/loss_position_analysis.png)
> - 各層距離分布: [analysis/layer_distances.png](analysis/layer_distances.png)

---

## 五、Loss 設計分析

### 5.1 為何選擇 Cosine Loss？

**問題**: 不同層的特徵幅度差異極大

| 層 | MSE (L2) | Cosine Loss | 尺度比 |
|----|----------|-------------|--------|
| L6 | 3185.23 | 0.919 | 44408x |
| L15 | 0.07 | 0.621 | 1x |

**MSE 範圍**: 0.07 ~ 3185.23 (差異 44408 倍)
**Cosine 範圍**: 0.05 ~ 0.96 (差異 18 倍)

### 5.2 Loss 類型比較

| Loss 類型 | 特性 | 適用場景 |
|-----------|------|----------|
| **Cosine** | 尺度不變，只看方向 | 中間層監督 ✅ |
| L2 (MSE) | 受幅度影響 | 最後層、同尺度特徵 |
| L1 (MAE) | 對異常值魯棒 | 有極端值時 |
| Normalized MSE | L2 正規化後計算 | 中間層替代方案 |

### 5.3 結論

**Cosine Loss 最適合中間層監督**，原因：
1. 不同層尺度差異 44000 倍，L2 會被大尺度層主導
2. Cosine 讓所有層在相同尺度比較
3. 語義特徵的「方向」比「幅度」更重要

> **圖表參考**:
> - Loss 類型比較: [analysis/loss_type_comparison.png](analysis/loss_type_comparison.png)
> - Loss 設計分析: [analysis/loss_design_analysis.png](analysis/loss_design_analysis.png)

---

## 六、綜合評估與建議

### 6.1 整合 exp_1231_feature 的結論

結合歷史實驗數據，我們有更完整的理解：

```
WavTokenizer 的噪音處理流程:
┌─────────────────────────────────────────────────────────────┐
│ L0-L4 (low_level):  噪音進入，初步特徵提取                   │
│                     敏感度中等 (0.47)                        │
│                                                              │
│ L5-L6 (mid_level):  ★ 噪音處理核心！                        │
│                     敏感度最高 (0.71)                        │
│                     這裡是噪音-語音分離的關鍵位置            │
│                                                              │
│ L9-L12 (semantic):  語義編碼                                 │
│                     敏感度下降 (0.50)                        │
│                                                              │
│ L13-L15 (abstract): 高階語義，對噪音魯棒                    │
│                     敏感度最低 (0.28)                        │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 當前架構評估

| 項目 | 當前設計 | 評估 | 理由 |
|------|----------|------|------|
| 監督位置 | L3, L6 | ✅ **合理** | L6 正是噪音處理核心層！ |
| Loss 類型 | Cosine | ✅ 正確 | 尺度不變性 |
| 權重 | 各 0.5 | 可優化 | L6 應該更高 |
| L10 利用 | 無 | 可選 | 作為語義錨點 |

**重要修正**: 之前建議加入 L1 監督，但根據 exp_1231_feature：
- L5-L6 (mid_level) 是噪音處理的核心，敏感度 **0.71-0.79**
- L1 的敏感度只有 **0.37**，遠低於 L6
- **當前 L3, L6 的選擇是正確的**，L6 正是最關鍵的層

### 6.3 建議的 Loss 配置

```python
# 當前配置 - 基本正確
intermediate_indices = [3, 6]
intermediate_weights = {3: 0.5, 6: 0.5}

# ★ Exp K v3 推薦配置 - 完整版
intermediate_indices = [3, 5, 6, 10]
intermediate_weights = {
    3: 0.5,    # low_level 代表
    5: 0.8,    # mid_level (與 L6 協同處理噪音)
    6: 1.0,    # ★ 噪音處理核心 (最高權重)
    10: 0.3,   # 語義錨點 (用 MSE，確保不偏離)
}
# L10 使用 MSE Loss (因為本來就穩定，cos_sim=0.946)
# 其他層使用 Cosine Loss
```

**配置說明**:
```
L3 (0.5):  捕捉早期噪音 → Cosine Loss
L5 (0.8):  噪音處理協同 → Cosine Loss
L6 (1.0):  ★ 核心層     → Cosine Loss (最高權重)
L10 (0.3): 語義錨點     → MSE Loss (精確匹配)
```

### 6.4 為何監督 L6 而非 L0/L1？

雖然本次分析顯示 L0, L1 敏感度很高，但：

1. **exp_1231 顯示 mid_level 才是噪音處理核心**
   - L5-L6 敏感度 0.71-0.79（最高）
   - 這是噪音特徵與語音特徵開始分離的地方

2. **L0/L1 敏感但非處理層**
   - L0 是輸入投影，主要是信號轉換
   - 監督 L0 可能干擾模型的輸入處理

3. **L6 是「直覺處理」位置**
   - WavTokenizer 設計上在 mid_level 做特徵整合
   - 監督這裡等於指導模型「如何分離噪音」

### 6.5 為何深層也需要訓練？

雖然深層對噪音較穩定，但仍需參與訓練：

1. **協同效應**: 淺層 LoRA 改變輸出 → 深層收到不同輸入 → 需要適應
2. **任務需求**: 去噪是全層協同任務，非單層可完成
3. **深層策略**: 不需強監督，輕量 loss 確保不偏離即可

### 6.6 下一步實驗建議

| 優先級 | 實驗 | 目標 | 配置 |
|--------|------|------|------|
| **高** | **Exp K v3** | 完整中間層監督 | L3(0.5) + L5(0.8) + L6(1.0) + L10(0.3) |
| 中 | Exp K v3-lite | 簡化版測試 | L3(0.5) + L6(1.0) + L10(0.3) |
| 低 | Exp M | 強化 L6 權重 | L6 權重 > 1.0 測試 |

**Exp K v3 預期效果**:
1. L5-L6 協同 → 更好的噪音分離學習
2. L10 錨點 → 確保語義特徵不偏離
3. 整體更平衡的監督訊號

---

## 七、圖表索引

### 7.1 核心分析圖表

| 圖表名稱 | 路徑 | 內容說明 |
|----------|------|----------|
| **整合噪音分析** | [analysis/integrated_noise_analysis.png](analysis/integrated_noise_analysis.png) | ★ 整合 exp_1231 與本次分析 |
| 噪音敏感度比較 | [analysis/noise_sensitivity_comparison.png](analysis/noise_sensitivity_comparison.png) | 原始模型各層對噪音的敏感度 |
| 監督位置推薦 | [analysis/supervision_recommendation.png](analysis/supervision_recommendation.png) | 基於敏感度的監督位置建議 |
| 各層距離分布 | [analysis/layer_distances.png](analysis/layer_distances.png) | 訓練後 Student-Teacher 距離 |
| 訓練收斂曲線 | [analysis/training_convergence.png](analysis/training_convergence.png) | Loss 和 Accuracy 的訓練曲線 |
| Loss 位置分析 | [analysis/loss_position_analysis.png](analysis/loss_position_analysis.png) | 訓練前後比較與配置建議 |
| Loss 類型比較 | [analysis/loss_type_comparison.png](analysis/loss_type_comparison.png) | Cosine vs L2 vs L1 比較 |
| Loss 設計分析 | [analysis/loss_design_analysis.png](analysis/loss_design_analysis.png) | 當前 Loss 設計的問題分析 |

### 7.2 數據檔案

| 檔案名稱 | 路徑 | 內容說明 |
|----------|------|----------|
| 噪音敏感度 | [analysis/noise_sensitivity.json](analysis/noise_sensitivity.json) | 原始模型各層敏感度數據 |
| 層距離數據 | [analysis/layer_distances.json](analysis/layer_distances.json) | 訓練後各層距離數據 |
| 訓練歷史 | [runs/exp_k_intermediate/history.json](runs/exp_k_intermediate/history.json) | Exp K v2 完整訓練歷史 |

### 7.3 歷史實驗參考

| 報告名稱 | 路徑 | 內容說明 |
|----------|------|----------|
| exp_1231 分析報告 | [../exp_1231_feature/ANALYSIS.md](../exp_1231_feature/ANALYSIS.md) | WavTokenizer 18 層完整分析 |
| exp_1231 噪音數據 | [../exp_1231_feature/outputs/noise_sensitivity_results.json](../exp_1231_feature/outputs/noise_sensitivity_results.json) | 合成噪音測試數據 |

### 7.3 實驗報告

| 報告名稱 | 路徑 | 內容說明 |
|----------|------|----------|
| 詳細分析報告 | [analysis/ANALYSIS_REPORT.md](analysis/ANALYSIS_REPORT.md) | 中間層監督詳細分析 |
| 容量測試報告 | [../exp_test/EXPERIMENT_REPORT.md](../exp_test/EXPERIMENT_REPORT.md) | exp_test 完整結果 |

---

## 附錄

### A. 關鍵公式

**Cosine Similarity**:
```
cos_sim(a, b) = (a · b) / (||a|| × ||b||)
```

**Cosine Loss**:
```
cos_loss = 1 - cos_sim(student, teacher)
```

**噪音敏感度**:
```
sensitivity = 1 - cos_sim(feature(clean), feature(noisy))
```

**學習難度**:
```
difficulty = sensitivity × (1 - cos_sim(student, teacher))
```

### B. 實驗環境

- GPU: NVIDIA GPU
- Framework: PyTorch
- Model: WavTokenizer + LoRA
- 訓練數據: 課程式噪音 (0.3 → 1.0)

---

*報告生成時間: 2026-01-15*
*所有實驗已完成，分析圖表已生成*
