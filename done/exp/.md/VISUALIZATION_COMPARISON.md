# ECAPA-TDNN Speaker Embedding 視覺化對比報告

**生成時間**: 2025-11-01 04:32
**目的**: 對比 Clean Audio 與 Noisy Audio 的 Speaker Embedding 視覺化結果

---

## 📊 視覺化文件總覽

### Clean Audio 視覺化（來自 [test_speaker_embedding.py](test_speaker_embedding.py)）
- `speaker_embedding_test/similarity_distribution.png` - 相似度分布圖
- `speaker_embedding_test/tsne_visualization.png` - t-SNE 視覺化
- `speaker_embedding_test/similarity_matrix.png` - Speaker 相似度矩陣

### Noisy Audio 視覺化（來自 [test_speaker_embedding_noisy.py](test_speaker_embedding_noisy.py)）

**基礎對比圖**:
- `speaker_embedding_noisy_test/clean_vs_noisy_comparison.png` - Clean vs Noisy 相似度對比
- `speaker_embedding_noisy_test/discrimination_comparison.png` - 跨材質區分度對比

**各材質詳細視覺化** (每種材質 3 張圖):
- `speaker_embedding_noisy_test/similarity_distribution_{material}.png` - 相似度分布
- `speaker_embedding_noisy_test/tsne_visualization_{material}.png` - t-SNE 視覺化
- `speaker_embedding_noisy_test/similarity_matrix_{material}.png` - Speaker 相似度矩陣

其中 `{material}` 包括: `box`, `papercup`, `plastic`

---

## 🔍 視覺化內容說明

### 1. Similarity Distribution（相似度分布圖）

**Clean Audio 版本**:
- 單一分布圖
- 綠色: 同一 speaker 不同音檔的相似度分布
- 紅色: 不同 speaker 之間的相似度分布
- 虛線: 平均值標記
- **期望**: 綠色分布應該偏右（高相似度），紅色分布應該偏左（低相似度）

**Noisy Audio 版本**:
- 為每種材質（box, papercup, plastic）生成獨立的分布圖
- 相同的配色方案（綠色 = 同 speaker，紅色 = 不同 speaker）
- 可以對比不同材質對 speaker discrimination 的影響

### 2. t-SNE Visualization（t-SNE 視覺化）

**目的**: 將高維 speaker embeddings (256-dim) 降維到 2D 平面

**Clean Audio 版本**:
- 單一 t-SNE 圖
- 每個 speaker 用不同顏色表示
- 每個點代表一個音檔的 embedding
- **期望**: 同一 speaker 的點應該聚集在一起，不同 speaker 應該分散開

**Noisy Audio 版本**:
- 為每種材質生成獨立的 t-SNE 圖
- 可以觀察不同材質的噪音是否影響 speaker 聚類效果
- PCA 預處理: 保留 99.5-99.6% 的變異量

### 3. Similarity Matrix（相似度矩陣熱圖）

**格式**: N×N 矩陣，N = speaker 數量

**Clean Audio 版本**:
- 顯示所有 speaker 配對的平均相似度
- 對角線: 同一 speaker 內部相似度（應該高，綠色）
- 非對角線: 不同 speaker 相似度（應該低，紅色）

**Noisy Audio 版本**:
- 為每種材質生成獨立的相似度矩陣
- 可以量化不同材質下的 speaker discrimination 能力
- 注意: 如果某 speaker 在該材質下沒有音檔，對應位置會顯示 0.000

---

## 📈 關鍵數據對比

### Clean Audio 結果
```
同一 speaker 平均相似度: 0.5543 ± 0.25
不同 speaker 平均相似度: 0.2015 ± 0.21
區分度 (Discrimination):  0.3528
```

### Noisy Audio 結果

#### Clean vs Noisy Embedding 差異
```
Box:       0.3987 ± 0.1401  (❌ 影響大)
Papercup:  0.4770 ± 0.1558  (❌ 影響大)
Plastic:   0.4244 ± 0.1745  (❌ 影響大)
```

**解讀**:
- 平均相似度 0.40-0.48，表示噪音顯著改變了 embeddings
- 但仍然保留了部分 speaker 特徵（不是完全破壞）

#### 各材質的 Speaker Discrimination
```
Material   | Same Speaker | Different | Discrimination | Status
-----------|--------------|-----------|----------------|-------
Box        | 0.5523       | 0.2337    | 0.3186         | ✅
Papercup   | 0.4759       | 0.1574    | 0.3185         | ✅
Plastic    | 0.5158       | 0.1861    | 0.3297         | ✅
```

**解讀**:
- 所有材質的區分度都在 0.31-0.33，非常接近
- 相比 Clean Audio (0.3528)，僅下降 10%（0.3528 → 0.32）
- **結論**: ECAPA-TDNN 在 noisy audio 上仍然能有效區分 speakers

#### 跨材質一致性
```
Material Pair       | Similarity | Status
--------------------|------------|-------
Box vs Papercup     | 0.5385     | ⚠️ 中等
Box vs Plastic      | 0.5975     | ⚠️ 中等
Papercup vs Plastic | 0.6076     | ✅ 良好
```

**解讀**:
- 同一 speaker 在不同材質下的 embeddings 相似度為 0.54-0.61
- 雖然有變化，但仍然保持一定的一致性

---

## 🎯 視覺化使用指南

### 場景 1: 驗證 ECAPA-TDNN 的基礎能力
**使用**: Clean Audio 視覺化
- 查看 `speaker_embedding_test/tsne_visualization.png`
- 確認不同 speaker 形成明顯的聚類

### 場景 2: 評估噪音影響
**使用**: Noisy Audio 對比圖
- 查看 `speaker_embedding_noisy_test/clean_vs_noisy_comparison.png`
- 觀察各材質的 clean-noisy 相似度分布
- **關鍵**: 雖然相似度下降，但只要維持在 0.4 以上就表示保留了主要特徵

### 場景 3: 評估跨材質 Speaker Discrimination
**使用**: Discrimination Comparison
- 查看 `speaker_embedding_noisy_test/discrimination_comparison.png`
- 確認各材質的區分度都在 0.3 以上（綠色 - 紅色 = 藍色）

### 場景 4: 深入分析特定材質
**使用**: 單一材質的 3 張圖
- 例如 Box 材質:
  - `similarity_distribution_box.png` - 相似度分布
  - `tsne_visualization_box.png` - speaker 聚類
  - `similarity_matrix_box.png` - speaker 配對相似度

### 場景 5: 對比不同材質的影響
**並排查看**:
- `tsne_visualization_box.png`
- `tsne_visualization_papercup.png`
- `tsne_visualization_plastic.png`

觀察:
- 聚類是否仍然清晰？
- 同一 speaker 的點是否仍然聚集？
- 是否有某些材質特別破壞 speaker 特徵？

---

## ✅ 關鍵發現

### 1. ECAPA-TDNN 對 Noisy Audio 有一定的魯棒性
- 雖然 clean-noisy 相似度只有 0.40-0.48
- 但 speaker discrimination 仍然維持在 0.31-0.33
- **原因**: ECAPA-TDNN 學習到的是 speaker-specific 特徵，對噪音有一定的抵抗力

### 2. 不同材質影響相似
- Box, Papercup, Plastic 的區分度都在 0.31-0.33
- 沒有哪種材質特別好或特別差
- **結論**: 材質類型對 ECAPA-TDNN 的影響是均勻的

### 3. 跨材質一致性中等
- 同一 speaker 在不同材質下的相似度為 0.54-0.61
- 雖然不如 clean audio 內部的 0.55，但仍然高於不同 speaker (0.20)
- **結論**: 可以用於 cross-material speaker recognition

### 4. 預測 Zero-Shot Denoising 效果
基於這些發現，我們預測:
```
Clean Audio (理論上限):  Val Acc 65-75%
Noisy Audio (實際情況):  Val Acc 55-65%  (10% 降級)
Baseline (無 speaker):   Val Acc 38%

預期提升: +44-71%
```

---

## 📁 完整文件清單

### Clean Audio (speaker_embedding_test/)
```
speaker_embedding_test/
├── similarity_distribution.png    (58 KB)
├── tsne_visualization.png         (117 KB)
└── similarity_matrix.png          (484 KB)
```

### Noisy Audio (speaker_embedding_noisy_test/)
```
speaker_embedding_noisy_test/
├── clean_vs_noisy_comparison.png      (58 KB)
├── discrimination_comparison.png       (47 KB)
│
├── similarity_distribution_box.png     (58 KB)
├── tsne_visualization_box.png          (117 KB)
├── similarity_matrix_box.png           (484 KB)
│
├── similarity_distribution_papercup.png (64 KB)
├── tsne_visualization_papercup.png     (117 KB)
├── similarity_matrix_papercup.png      (465 KB)
│
├── similarity_distribution_plastic.png  (64 KB)
├── tsne_visualization_plastic.png      (116 KB)
└── similarity_matrix_plastic.png       (475 KB)
```

**總計**: 14 張圖，約 2.1 MB

---

## 🚀 下一步

1. **訓練 Zero-Shot Denoising Transformer**
   - 創建 `train_zeroshot.py`
   - 創建 `run_zeroshot.sh`
   - 開始訓練

2. **監控指標**
   - 觀察 Val Acc 是否達到 55-65%
   - 觀察 Train-Val Gap 是否縮小到 15-25%

3. **如果效果不佳**
   - 考慮對 ECAPA-TDNN 進行 fine-tuning
   - 嘗試不同的 fusion 策略（concatenation, cross-attention）
   - 加入對比學習（contrastive learning）增強魯棒性

---

**總結**: 這些視覺化完整展示了 ECAPA-TDNN 在 clean 和 noisy audio 上的 speaker embedding 能力。雖然噪音會降低 embedding 質量，但 speaker discrimination 能力仍然保持良好（0.31-0.33），為 Zero-Shot Speaker Denoising 提供了堅實的基礎。
