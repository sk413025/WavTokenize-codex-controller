# WavTokenizer Encoder 特徵分析報告

**日期**: 2024-12-31
**實驗資料夾**: exp_1231_feature
**分析者**: Claude Code

---

## 1. 實驗目的

本實驗旨在深入了解 WavTokenizer encoder 的 18 層 conv 結構：

1. **Speaker Invariance 分析**: 不同說話者（boy1 vs girl2）說同一句話時，各層 feature map 的相似度如何？
   - 淺層是否提取聲學特徵（音色、音高）？
   - 深層是否提取語義特徵（內容、語言）？

2. **LoRA 訓練變化分析**: 訓練去噪任務後，哪些層的權重變化最大？
   - LoRA 主要在哪個層級學習去噪？
   - 是聲學級別修正還是語義級別修正？

---

## 2. 模型架構

### WavTokenizer Encoder 結構

18 層 Conv 分組如下：

| 層組 | 層索引 | 模組名稱 | 功能推測 |
|------|--------|----------|----------|
| **input** | L0 | model.0.conv | 輸入處理 |
| **low_level** | L1-L4 | model.1.*, model.3.* | 低階聲學特徵 |
| **mid_level** | L5-L8 | model.4.*, model.6.* | 中階特徵 |
| **semantic** | L9-L12 | model.7.*, model.9.* | 語義特徵 |
| **abstract** | L13-L16 | model.10.*, model.12.* | 抽象表徵 |
| **output** | L17 | model.15.conv | 輸出投影 |

### LoRA 配置

- **Rank (r)**: 256
- **Alpha**: 512
- **Target modules**: 全部 18 層 encoder conv
- **LoRA 參數量**: 14.24% of encoder

---

## 3. Speaker Invariance 分析結果

### 3.1 實驗設計

- **測試音檔**: 同一句話 (sentence 001) 由不同說話者錄製
  - boy1: 男孩聲音
  - girl2: 女孩聲音
- **指標**: Cosine Similarity（越高表示越相似，即越 speaker-invariant）

### 3.2 結果數據

| 層組 | 層索引 | 平均 Cosine Similarity | 解讀 |
|------|--------|------------------------|------|
| **input** | L0 | -0.011 | 完全不同（原始波形差異大） |
| **low_level** | L1-L4 | 0.021 | 幾乎無關（聲學差異仍大） |
| **mid_level** | L5-L8 | 0.068 | 略有相關 |
| **semantic** | L9-L12 | 0.117 | 開始收斂 |
| **abstract** | L13-L16 | **0.433** | **高度相似！** |
| **output** | L17 | 0.187 | 投影後略降 |

### 3.3 各層詳細數據

```
Layer   Cosine Sim   Type
L0      -0.011       input
L1      -0.011       low_level
L2       0.023       low_level
L3      -0.011       low_level
L4       0.068       low_level
L5       0.156       mid_level (peak in group)
L6       0.061       mid_level
L7       0.045       mid_level
L8       0.027       mid_level
L9       0.050       semantic
L10      0.143       semantic (peak in group)
L11      0.005       semantic
L12      0.274       semantic
L13      0.522       abstract ★ PEAK
L14      0.349       abstract
L15      0.672       abstract ★★ HIGHEST
L16      0.163       abstract
L17      0.187       output
```

### 3.4 發現

1. **L15 (model.10.shortcut) 有最高的 speaker invariance (0.672)**
   - 這意味著不同說話者說同一句話，在 L15 的特徵最相似
   - L15 提取的是內容/語義，而非說話者身份

2. **abstract 層組平均 0.433，遠高於其他層組**
   - 驗證了深層提取語義特徵的假設

3. **淺層 (L0-L4) 幾乎為 0 或負值**
   - 確認淺層保留說話者特有的聲學特徵

### 3.5 視覺化

![Speaker Invariance Summary](outputs/speaker_invariance_summary.png)

---

## 4. LoRA 權重變化分析

### 4.1 實驗設計

- **基準模型**: 原始 WavTokenizer (wavtokenizer_large_speech_320_24k)
- **訓練模型**: Exp67 curriculum VQ 最佳模型
- **指標**: Relative Change = ||W_trained - W_original|| / ||W_original||

### 4.2 結果數據

| 層組 | 平均相對變化 | 解讀 |
|------|--------------|------|
| **input** | 1.427 | 基礎變化 |
| **low_level** | 1.561 | 中等變化 |
| **mid_level** | 1.575 | 中等變化 |
| **semantic** | 1.560 | 中等變化 |
| **abstract** | **1.730** | **最大變化！** |
| **output** | 1.464 | 中等變化 |

### 4.3 各層詳細數據

```
Rank  Layer   Relative Change   Module
1     L13     1.973            model.10.block.1 ★ TOP
2     L14     1.891            model.10.block.3
3     L3      1.778            model.1.shortcut
4     L9      1.662            model.7.block.1
5     L12     1.641            model.9.conv
6     L16     1.621            model.12.conv
7     L7      1.620            model.4.shortcut
8     L5      1.611            model.4.block.1
9     L8      1.569            model.6.conv
10    L4      1.551            model.3.conv
11    L6      1.500            model.4.block.3
12    L10     1.490            model.7.block.3
13    L2      1.476            model.1.block.3
14    L17     1.464            model.15.conv
15    L11     1.446            model.7.shortcut
16    L15     1.436            model.10.shortcut
17    L1      1.441            model.1.block.1
18    L0      1.427            model.0.conv
```

### 4.4 發現

1. **L13 (model.10.block.1) 變化最大 (1.973)**
   - 這是 abstract 層組的第一層
   - LoRA 在此層學習最多

2. **abstract 層組整體變化最大 (1.730)**
   - 表明去噪任務主要在語義/抽象層級進行修正
   - 而非在聲學層級

3. **input 層變化最小 (1.427)**
   - 基礎波形處理變化較少

### 4.5 視覺化

![LoRA Weight Changes](outputs/lora_weight_changes_v2.png)

---

## 5. 綜合分析

### 5.1 核心發現

| 分析維度 | 最重要層 | 結論 |
|----------|----------|------|
| Speaker Invariance | L15 (abstract) | 深層提取與說話者無關的內容特徵 |
| LoRA 學習 | L13 (abstract) | 去噪主要在語義層級進行 |

### 5.2 機制推論

```
輸入波形 (noisy)
    ↓
[L0-L4: low_level] → 提取聲學特徵（音色、音高、噪音）
    ↓                  LoRA 變化：較小
[L5-L8: mid_level] → 整合局部特徵
    ↓                  LoRA 變化：中等
[L9-L12: semantic] → 開始抽象化，減少說話者差異
    ↓                  LoRA 變化：中等
[L13-L16: abstract] → ★ 核心語義表徵 ★
    ↓                  LoRA 變化：最大！
    ↓                  Speaker invariance：最高！
[L17: output] → 投影到 codebook space
    ↓
VQ Token (denoised representation)
```

### 5.3 ⚠️ LoRA 在 abstract 層學習最多是「問題」而非「正確行為」

**重新思考降噪任務的本質：**

```
降噪任務：
  Input:  noisy audio (有噪音)
  Output: clean audio (無噪音)

  語義內容：完全相同（說的是同一句話）
  聲學特徵：不同（一個有噪音，一個沒有）
```

**理論上正確的修正方向：**
- 淺層 → 應該大幅改變（修正噪音的聲學特徵）
- 深層 → 應該幾乎不變（語義本來就一樣！）

**但實際觀察：**
- 淺層變化最小 (1.43)
- 深層變化最大 (1.73)
- **這與預期完全相反！**

**可能的原因：**

1. **梯度傳播問題**
   - Loss 在 encoder output 計算
   - 深層先收到強梯度，淺層梯度衰減
   - 這是「技術限制」而非「正確的學習」

2. **模型在深層「硬記」而非「學習去噪」**
   - 淺層沒動 → 噪音特徵傳到深層
   - 深層被迫適應 noisy 輸入 → 大幅改變
   - 這是 overfitting 的徵兆！

3. **這可能解釋 Train/Val gap**
   - Train: 記住了訓練集特定噪音模式的語義映射
   - Val: 遇到新的噪音模式就無法正確映射

---

## 6. 對訓練策略的啟示

### 6.1 問題診斷

- ⚠️ 目前 LoRA 學習方向與任務需求不匹配
- ⚠️ 深層變化過大，可能導致 overfitting
- ⚠️ 淺層變化不足，無法有效去除噪音聲學特徵

### 6.2 建議的修正方向

1. **反轉 LoRA rank 分配**
   ```
   當前（問題）：
     所有層 rank = 256
     → 深層學太多，淺層學太少

   建議（修正）：
     淺層 (L0-L8):  高 rank (256) → 大幅修正噪音
     深層 (L9-L17): 低 rank (16)  → 限制變化幅度
   ```

2. **加入深層 regularization**
   - 對 abstract 層加入 L2 penalty
   - 防止深層過度適應訓練集噪音

3. **淺層 feature loss**
   - 在 mid_level (L5-L8) 加入中間層 loss
   - 強制淺/中層學習去噪

4. **Gradient scaling**
   - 對淺層放大梯度
   - 對深層縮小梯度
   - 補償反向傳播的梯度衰減

---

## 7. 實驗檔案

```
exp_1231_feature/
├── ANALYSIS.md              # 本報告
├── analyze_feature_maps.py  # Speaker invariance 分析腳本
├── analyze_lora_changes.py  # LoRA 變化分析 v1
├── analyze_lora_changes_v2.py # LoRA 變化分析 v2 (使用 merge_and_unload)
└── outputs/
    ├── speaker_invariance_summary.png
    ├── feature_maps_boy1_001.png
    ├── feature_maps_girl2_001.png
    ├── comparison_boy1_girl2_001.png
    ├── analysis_results.json
    ├── lora_weight_changes_v2.png
    └── lora_changes_v2.json
```

---

## 8. 結論

1. **WavTokenizer encoder 的深層 (L13-L16, abstract) 提取 speaker-invariant 的語義特徵**
   - 不同說話者說同一句話，在深層 feature map 高度相似 (cos_sim ≈ 0.43-0.67)
   - ✅ 這符合預期，深層應該保持語義不變

2. **⚠️ LoRA 訓練後，深層權重變化最大 - 這是問題！**
   - L13 變化 197%，但降噪任務中語義應該不變
   - 這表示模型可能在深層「硬記」訓練集的噪音映射

3. **降噪應該是「聲學級別修正」而非「語義級別修正」**
   - noisy → clean 只是去噪，語義內容不變
   - 應該在淺/中層修正噪音特徵
   - 深層應該保持穩定

4. **⚠️ 當前學習方向可能有問題**
   - 深層變化大 + 淺層變化小 = 與任務需求相反
   - 這可能是 Train/Val gap 的根本原因
   - 建議調整 LoRA rank 分配或加入層級 regularization

---

*Generated by Claude Code - 2024-12-31*
