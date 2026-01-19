# WavTokenizer LoRA 中間層監督實驗 - PPT 進度報告

**報告日期**: 2026-01-19
**實驗代號**: Exp 0112/0113 (Exp K)

---

## Slide 1: 標題頁

### WavTokenizer LoRA 降噪訓練
## 中間層監督策略研究

- **實驗期間**：2026-01-12 ~ 2026-01-19
- **狀態**：Exp K v4 進行中（修正版）

---

## Slide 2: 研究動機 - 觀察到的矛盾現象

### 問題：為什麼噪音敏感層學不好？

| 分析結果 | 訓練結果 |
|----------|----------|
| 噪音對 **中層 (L5-L8)** 影響最大 | 但深層變化最大 |
| 淺層直接接觸噪音 | 淺層幾乎沒學到 |

### 根本原因：梯度傳播的物理限制

```
標準反向傳播流程：

Loss (最後層計算)
    ↓ 梯度 100%
[L9-L17] 深層 ← 最先收到梯度，變化最大
    ↓ 梯度衰減
[L5-L8]  中層 ← 噪音破壞最嚴重，但梯度已衰減！
    ↓ 梯度繼續衰減
[L0-L4]  淺層 ← 梯度信號最弱
```

**核心矛盾**：
- 深層對噪音不敏感（本來就穩定）→ 但最先被更新
- 中層對噪音最敏感（需要學習）→ 但梯度已衰減

---

## Slide 3: 驗證步驟一 - 確認噪音敏感位置

### 方法：測量各層對噪音的敏感度

```
噪音敏感度 = 1 - cos_sim(feature(clean), feature(noisy))
```

### encoder.model 結構（重要！）

```
model[0]:  SConv1d (Input Conv)
model[1]:  SEANetResnetBlock (ResBlock 1)
model[2]:  ELU (激活函數)
model[3]:  SConv1d (Downsample 1)    ← 監督目標
model[4]:  SEANetResnetBlock (ResBlock 2) ← 監督目標
model[5]:  ELU (激活函數，監督無效!)
model[6]:  SConv1d (Downsample 2)    ← 監督目標
...
```

### 結果：各層噪音敏感度

| encoder.model | 類型 | 敏感度 | 解讀 |
|---------------|------|--------|------|
| model[0] | Input Conv | 0.16 | 魯棒 |
| model[1] | ResBlock 1 | 0.75 | 敏感 |
| **model[3]** | Downsample 1 | **0.67** | **監督目標** |
| **model[4]** | ResBlock 2 | **0.80** | **★ 最敏感！監督目標** |
| model[5] | ELU | 0.80 | (激活函數，無效) |
| **model[6]** | Downsample 2 | **0.79** | **★ 監督目標** |
| model[13] | LSTM | 0.25 | 魯棒 |

### 關鍵發現
- **model[4], model[6] 是噪音敏感區核心**（敏感度 0.79-0.80）
- **model[5] 是 ELU 激活函數，監督它沒有意義！**

**圖片**: `analysis/noise_sensitivity_correct.png`

---

## Slide 4: 驗證步驟二 - 排除容量瓶頸假說

### 假說：LoRA 容量不足導致淺層學習困難？

### 實驗設計 (exp_test)
- 只訓練淺層 (L0-L4)
- 比較不同 LoRA rank: 256 / 512 / 1024

### 結果

| Rank | 參數量 | Best Val Loss | 改善幅度 |
|------|--------|---------------|----------|
| 256 | 116K | 51.79 | baseline |
| 512 | 233K | 51.66 | -0.25% |
| 1024 | 466K | 51.76 | -0.06% |

### 結論
**容量不是主要瓶頸**
- 4 倍參數量 (116K → 466K) 僅改善 0.06%
- 三個 rank 最終收斂到相同水平

**→ 問題不在容量，在於梯度信號不足**

---

## Slide 5: 解決方案 - 中間層直接監督

### 核心思想
既然梯度傳播會衰減，那就**直接在敏感位置加 Loss**！

### 方法：Teacher-Student + 中間層 Loss

```
Teacher (Clean)              Student (Noisy)
    ↓                            ↓
[L0-L4] ─────── Loss₁ ───────→ [L0-L4 + LoRA]  ← 直接監督
    ↓                            ↓
[L5-L8] ─────── Loss₂ ───────→ [L5-L8 + LoRA]  ← 直接監督
    ↓                            ↓
[L9-L17] ──── Final Loss ────→ [L9-L17 + LoRA]
```

**Total Loss = Loss_final + Σ (wᵢ × Loss_intermediate)**

### 優勢
1. 噪音敏感層直接獲得監督信號
2. 不需等待梯度從深層傳播
3. 可針對不同層設計不同權重

**圖片**: `analysis/ppt_architecture_comparison.png` (左半)

---

## Slide 6: Loss 類型選擇 - 為何用 Cosine Loss？

### 問題：不同層特徵幅度差異極大

| 層 | MSE (L2) | Cosine Loss |
|----|----------|-------------|
| L6 | 3185.23 | 0.919 |
| L15 | 0.07 | 0.621 |

**MSE 範圍**: 0.07 ~ 3185 (差異 **44000 倍**)
**Cosine 範圍**: 0.05 ~ 0.96 (差異 **18 倍**)

### 結論：Cosine Loss 最適合中間層監督

| 特性 | MSE | Cosine |
|------|-----|--------|
| 尺度敏感 | ✗ 會被大尺度層主導 | ✓ 尺度不變 |
| 跨層比較 | ✗ 難以平衡 | ✓ 所有層同尺度 |
| 語義對齊 | 幅度匹配 | ✓ 方向匹配更重要 |

**圖片**: `analysis/loss_type_comparison.png`

---

## Slide 7: Exp K v2 實驗配置與結果

### 配置
| 項目 | 設定 |
|------|------|
| 監督位置 | L3, L6 |
| Loss 類型 | Cosine Loss |
| 權重 | 各 0.5 |
| Epochs | 285 |

### 結果

| 指標 | Train | Val | Gap |
|------|-------|-----|-----|
| Total Loss | 1.46 | 1.94 | 0.48 |
| Match Acc | 3.48% | 0.91% | 2.57% |

**Best Val Acc**: 0.93% @ Epoch 236

**圖片**: `runs/exp_k_v2_20260115_020445/training_curves.png`

---

## Slide 8: 效果驗證 - 有監督 vs 無監督層

### 訓練前後各層距離變化

| 層 | 訓練前 | 訓練後 | 變化 | 監督狀態 |
|----|--------|--------|------|----------|
| L0 | 0.959 | 0.960 | +0.1% | ❌ 無監督 |
| L1 | 0.943 | 0.963 | +2.1% | ❌ 無監督 |
| **L3** | 0.895 | **0.758** | **-15%** | ✅ **有監督** |
| **L6** | 0.919 | **0.767** | **-17%** | ✅ **有監督** |
| L10 | 0.054 | 0.254 | - | ❌ 無監督 |

### 結論
**中間層監督有效！**
- 有監督的層 (L3, L6)：距離下降 15-17%
- 無監督的層 (L0, L1)：幾乎沒有改善

**圖片**: `analysis/loss_position_analysis.png`

---

## Slide 9: 核心發現總結

### 四個關鍵結論

| # | 發現 | 證據 |
|---|------|------|
| 1 | **梯度傳播是瓶頸** | 深層先更新，敏感層梯度衰減 |
| 2 | **L5-L6 是噪音處理核心** | 敏感度 0.71-0.79 (最高) |
| 3 | **容量不是問題** | 4x 參數僅 0.06% 改善 |
| 4 | **中間層監督有效** | 有監督層改善 15-17% |

### 邏輯鏈條

```
觀察矛盾現象
    ↓
確認敏感位置 (L5-L6 是核心)
    ↓
排除容量瓶頸 (增加參數無效)
    ↓
提出解決方案 (中間層直接監督)
    ↓
驗證效果 (有監督層顯著改善)
```

**圖片**: `analysis/ppt_summary_dashboard.png`

---

## Slide 10: 當前實驗 - Exp K v4（修正版）

### 關鍵修正：發現 model[5] 是 ELU！

原本設計監督 `[3, 5, 6]`，但 **model[5] 是 ELU 激活函數**，監督無效！

修正後監督 `[3, 4, 6]`：

| encoder.model | 類型 | 權重 | 角色 |
|---------------|------|------|------|
| **model[3]** | SConv1d (Downsample) | 0.3 | 淺層錨點 |
| **model[4]** | SEANetResnetBlock | **1.0** | **★ 噪音敏感區核心** |
| **model[6]** | SConv1d (Downsample) | 0.5 | 中層監督 |

### v3 → v4 改進

| 項目 | v3 | v4 |
|------|----|----|
| 監督層 | [3, 5, 6, 10] | **[3, 4, 6]** |
| model[5] | ✗ ELU (無效!) | ✓ 移除 |
| model[4] | ❌ 無 | ✅ **新增 (ResBlock)** |
| L10 | w=0.3 | ✗ 移除 (效果存疑) |

### 預期效果
1. 修正監督到實際的卷積層
2. ResBlock 比 Downsample 有更多參數可學習
3. 聚焦噪音敏感區 (model[4], model[6])

**狀態**: 🔄 進行中 (Exp K v4 修正版)

**圖片**: `analysis/noise_sensitivity_correct.png`

---

## Slide 11: 下一步計畫

### 短期 (本週)

| 優先級 | 任務 | 狀態 |
|--------|------|------|
| 高 | 完成 Exp K v3 訓練 | 🔄 進行中 |
| 高 | v2 vs v3 效果對比 | 待 v3 完成 |
| 中 | 音頻品質評估 | PESQ, STOI |

### 中期

| 任務 | 目標 |
|------|------|
| 消融實驗 | 驗證各層貢獻 (L5 單獨效果) |
| 權重調優 | L6 > 1.0 測試 |
| 正則化 | 減少 Train-Val Gap |

---

## Slide 12: 圖表索引

### 核心圖表

| 用途 | 檔案 |
|------|------|
| **★ encoder.model 結構與監督層** | `analysis/noise_sensitivity_correct.png` |
| **★ 架構對比** | `analysis/ppt_architecture_comparison.png` |
| **★ 總結儀表板** | `analysis/ppt_summary_dashboard.png` |
| **★ Loss 類型比較** | `analysis/loss_type_comparison.png` |
| **★ Exp K v2 訓練曲線** | `runs/exp_k_v2_*/training_curves.png` |
| 訓練前後比較 | `analysis/loss_position_analysis.png` |
| 舊版敏感度分析 | `analysis/integrated_noise_analysis.png` |

---

*報告更新時間: 2026-01-19*
