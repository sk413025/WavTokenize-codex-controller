# ECAPA-TDNN Noisy Audio 驗證報告

**測試日期**: 2025-11-01
**測試音檔**: 54 個配對樣本（18 speakers × 3 音檔）
**材質**: box, papercup, plastic（3 種 noisy 材質）
**Encoder**: ECAPA-TDNN (預訓練，凍結)
**視覺化**: 14 張圖（詳見下方）

---

## 📊 視覺化圖表

### 基礎對比圖
1. [clean_vs_noisy_comparison.png](speaker_embedding_noisy_test/clean_vs_noisy_comparison.png) - Clean vs Noisy 相似度對比
2. [discrimination_comparison.png](speaker_embedding_noisy_test/discrimination_comparison.png) - 跨材質區分度對比

### 各材質詳細視覺化（每種材質 3 張圖）

**Box 材質**:
- [similarity_distribution_box.png](speaker_embedding_noisy_test/similarity_distribution_box.png) - 相似度分布
- [tsne_visualization_box.png](speaker_embedding_noisy_test/tsne_visualization_box.png) - t-SNE 視覺化
- [similarity_matrix_box.png](speaker_embedding_noisy_test/similarity_matrix_box.png) - Speaker 相似度矩陣

**Papercup 材質**:
- [similarity_distribution_papercup.png](speaker_embedding_noisy_test/similarity_distribution_papercup.png) - 相似度分布
- [tsne_visualization_papercup.png](speaker_embedding_noisy_test/tsne_visualization_papercup.png) - t-SNE 視覺化
- [similarity_matrix_papercup.png](speaker_embedding_noisy_test/similarity_matrix_papercup.png) - Speaker 相似度矩陣

**Plastic 材質**:
- [similarity_distribution_plastic.png](speaker_embedding_noisy_test/similarity_distribution_plastic.png) - 相似度分布
- [tsne_visualization_plastic.png](speaker_embedding_noisy_test/tsne_visualization_plastic.png) - t-SNE 視覺化
- [similarity_matrix_plastic.png](speaker_embedding_noisy_test/similarity_matrix_plastic.png) - Speaker 相似度矩陣

**對比 Clean Audio 視覺化**: 參見 [VISUALIZATION_COMPARISON.md](VISUALIZATION_COMPARISON.md)

---

## 🎯 關鍵發現

### ⚠️ Noisy Audio 對 ECAPA-TDNN 有顯著影響

| 材質 | Clean vs Noisy 相似度 | 影響程度 | 結論 |
|------|---------------------|---------|------|
| **box** | **0.4266 ± 0.15** | ❌ 大 | Embedding 變化顯著 |
| **papercup** | **0.5014 ± 0.17** | ⚠️ 中等 | Embedding 有一定變化 |
| **plastic** | **0.4442 ± 0.19** | ❌ 大 | Embedding 變化顯著 |

### ✅ 但仍能有效區分 Speakers

| 材質 | 同一 Speaker | 不同 Speaker | 區分度 | 結論 |
|------|-------------|-------------|--------|------|
| **box** | 0.5740 | 0.2637 | **0.3103** | ✅ 有效 |
| **papercup** | 0.4928 | 0.1813 | **0.3115** | ✅ 有效 |
| **plastic** | 0.5369 | 0.2276 | **0.3093** | ✅ 有效 |

**對比 Clean Audio** (之前的驗證):
- Clean 同一 speaker: 0.5543
- Clean 不同 speaker: 0.2015
- Clean 區分度: **0.3528**

**結論**:
- ✅ Noisy audio 的區分度（0.31）略低於 clean audio（0.35），但**仍然有效**
- ✅ 所有三種材質都能維持良好的 speaker discrimination

---

## 📊 詳細分析

### 1. Clean vs Noisy Embedding 差異

#### 最佳材質: Papercup (相似度 0.50)
```
平均相似度: 0.5014 ± 0.17
範圍: [0.01, 0.81]
影響: ⚠️ 中等

解釋:
- Papercup 的噪音相對較輕
- Embeddings 保留較多原始 speaker 信息
- 約 50% 的信息被保留
```

#### 較差材質: Box (相似度 0.43)
```
平均相似度: 0.4266 ± 0.15
範圍: [0.09, 0.68]
影響: ❌ 大

解釋:
- Box 的噪音較重（空間混響、共振）
- Embeddings 變化較大
- 約 43% 的信息被保留
```

#### 較差材質: Plastic (相似度 0.44)
```
平均相似度: 0.4442 ± 0.19
範圍: [-0.02, 0.75]
影響: ❌ 大

解釋:
- Plastic 的噪音特性複雜
- 某些樣本相似度甚至為負（-0.02）
- 約 44% 的信息被保留
```

#### 統計顯著性

**變化程度**:
- 從 clean 到 noisy，embedding 平均損失 **50-57%** 的相似度
- Clean-clean 自相似度: ~1.0
- Clean-noisy 相似度: 0.43-0.50
- 損失: 50-57%

**標準差分析**:
- Box: ±0.15 (相對穩定)
- Papercup: ±0.17 (中等波動)
- Plastic: ±0.19 (最不穩定)

---

### 2. 跨材質 Speaker Discrimination

#### Box 材質
```
同一 speaker (不同內容): 0.5740 ± 0.19
不同 speaker:           0.2637 ± 0.16
區分度:                 0.3103

分析:
✅ 區分度 0.31，接近 clean audio 的 0.35
✅ 同一 speaker 的相似度 > 0.5，仍然可識別
✅ 不同 speaker 的相似度 < 0.3，明確分離
```

#### Papercup 材質
```
同一 speaker (不同內容): 0.4928 ± 0.19
不同 speaker:           0.1813 ± 0.15
區分度:                 0.3115

分析:
✅ 區分度 0.31，最高！
⚠️  同一 speaker 相似度稍低（0.49），但仍可接受
✅ 不同 speaker 相似度最低（0.18），分離度最好
```

#### Plastic 材質
```
同一 speaker (不同內容): 0.5369 ± 0.19
不同 speaker:           0.2276 ± 0.14
區分度:                 0.3093

分析:
✅ 區分度 0.31，與其他材質一致
✅ 同一 speaker 相似度中等（0.54）
✅ 平衡的表現
```

#### 材質對比總結

| 指標 | Box | Papercup | Plastic | Clean (參考) |
|------|-----|----------|---------|-------------|
| 同一 speaker | 0.574 | 0.493 | 0.537 | 0.554 |
| 不同 speaker | 0.264 | 0.181 | 0.228 | 0.202 |
| **區分度** | **0.310** | **0.312** | **0.309** | **0.353** |
| 與 Clean 差距 | -12% | -12% | -12% | - |

**關鍵發現**:
- ✅ 所有材質的區分度都在 **0.31** 左右
- ✅ 與 clean audio (0.35) 相比，只降低了 **12%**
- ✅ 這個降低是**可接受的**，仍能有效區分 speakers

---

### 3. 跨材質一致性

#### Box vs Papercup (相似度 0.55)
```
平均相似度: 0.5542 ± 0.14
判斷: ⚠️ 中等

解釋:
- 兩種材質的 embeddings 有一定差異
- 但仍然保持約 55% 的一致性
- 同一 speaker 在不同材質下的 embedding 相關
```

#### Box vs Plastic (相似度 0.62) ✅
```
平均相似度: 0.6224 ± 0.11
判斷: ✅ 好

解釋:
- Box 和 Plastic 的 embeddings 較為一致
- 兩種材質的噪音特性可能較接近
- 62% 的一致性是較好的結果
```

#### Papercup vs Plastic (相似度 0.62) ✅
```
平均相似度: 0.6159 ± 0.13
判斷: ✅ 好

解釋:
- Papercup 和 Plastic 也保持良好一致性
- 說明 ECAPA 能提取相對穩定的 speaker features
```

#### 跨材質一致性矩陣

|             | Clean | Box | Papercup | Plastic |
|-------------|-------|-----|----------|---------|
| **Clean**   | 1.00  | 0.43| 0.50     | 0.44    |
| **Box**     | 0.43  | 1.00| 0.55     | 0.62    |
| **Papercup**| 0.50  | 0.55| 1.00     | 0.62    |
| **Plastic** | 0.44  | 0.62| 0.62     | 1.00    |

**觀察**:
1. **Clean vs Noisy**: 相似度 0.43-0.50（損失 50-57%）
2. **Noisy vs Noisy**: 相似度 0.55-0.62（較好）
3. **結論**: Noisy embeddings 之間更相似，可能形成一個"noisy space"

---

## 💡 對 Zero-Shot Denoising 的影響

### ⚠️ 潛在問題

#### 問題 1: Embedding Degradation
**現象**: Clean-Noisy 相似度只有 0.43-0.50

**影響**:
- Model 輸入的是 **noisy audio** 的 embedding
- 這個 embedding 與 clean audio 的差異達 50%
- Model 可能學習到"degraded speaker representation"

**預期後果**:
- 如果 model 過度依賴 speaker embedding，可能效果不佳
- 需要 model 學習從 noisy embedding 恢復的能力

#### 問題 2: Material-Specific Bias
**現象**: 不同材質的 embeddings 有 38-45% 的變化

**影響**:
- Model 可能學習到 material-specific patterns
- 對某些材質效果好，對其他材質效果差

**預期後果**:
- 訓練集包含 3 種材質，應該能學習材質不變性
- 但驗證集如果只有特定材質，可能泛化不佳

### ✅ 正面因素

#### 優勢 1: 區分度仍然保持
**現象**: Noisy audio 的區分度 0.31 vs Clean 0.35 (-12%)

**影響**:
- 雖然 embeddings 變化了，但 speaker 信息仍然存在
- Model 仍能利用這個信息進行 speaker-conditioned denoising

**預期效果**:
- Val Acc 應該能超過 baseline 的 38%
- 但可能達不到最理想的 70-75%

#### 優勢 2: 跨材質一致性
**現象**: Noisy-Noisy 相似度 0.55-0.62

**影響**:
- Model 會同時看到 3 種材質的 noisy embeddings
- 這些 embeddings 之間有一定一致性
- Model 可能學習到材質不變的 speaker representation

**預期效果**:
- 應該能在不同材質上保持穩定性能
- 不會出現"只對某種材質有效"的問題

---

## 🎯 調整後的預期

### 原始預期（基於 Clean Audio 驗證）
```
最佳: 70-75% Val Acc
一般: 60-70% Val Acc
最差: 50-60% Val Acc
```

### 調整後預期（基於 Noisy Audio 驗證）

#### 樂觀情況 (55-65% Val Acc)
**假設**: Model 能有效利用 degraded embeddings
```
條件:
- Token embeddings 提供主要的去噪信息
- Speaker embeddings 提供輔助的風格信息
- Fusion 策略有效

預期:
- Val Acc: 55-65%
- 比 baseline (38%) 提升 45-71%
- 達到「目標標準」
```

#### 中性情況 (50-55% Val Acc)
**假設**: Embedding degradation 有一定影響
```
條件:
- Speaker embeddings 的質量下降 50%
- Model 部分依賴 speaker 信息
- 仍比 baseline 好，但不如理想

預期:
- Val Acc: 50-55%
- 比 baseline (38%) 提升 32-45%
- 達到「最低標準」
```

#### 悲觀情況 (45-50% Val Acc)
**假設**: Model 過度依賴 speaker embeddings
```
條件:
- Degraded embeddings 嚴重影響性能
- Fusion 策略不理想
- 但仍比 baseline 好

預期:
- Val Acc: 45-50%
- 比 baseline (38%) 提升 18-32%
- 仍有改進，但不顯著
```

#### 最差情況 (38-45% Val Acc)
**假設**: Speaker embeddings 幾乎無用
```
條件:
- Noisy embeddings 太差
- Model 主要依賴 token embeddings
- 與 baseline 效果接近

預期:
- Val Acc: 38-45%
- 比 baseline (38%) 提升 0-18%
- 實驗失敗，需要重新設計
```

### 最可能的情況

**我的判斷**: **樂觀情況** (55-65% Val Acc)

**理由**:
1. ✅ 區分度仍有 0.31（只降低 12%）
2. ✅ 跨材質一致性好（0.55-0.62）
3. ✅ Model 會同時使用 token + speaker 信息
4. ✅ Training data 包含 3 種材質，應該能學習魯棒性

**風險**:
- ⚠️  如果 model 過度依賴 speaker embeddings → 降至中性情況
- ⚠️  如果 fusion 策略不佳 → 降至悲觀情況

---

## 🔧 建議的優化策略

### 策略 1: Noise-Robust Training（推薦）

**目標**: 讓 model 學習處理 degraded embeddings

```python
# 在訓練時對 speaker embeddings 添加噪音
speaker_emb_noisy = speaker_emb + torch.randn_like(speaker_emb) * 0.1

# 或使用 dropout
speaker_emb_dropped = F.dropout(speaker_emb, p=0.1, training=True)
```

**預期**: 提升 5-10% Val Acc

### 策略 2: Adaptive Fusion（如果效果不佳）

**目標**: 動態調整 token vs speaker 的權重

```python
# 學習融合權重
self.fusion_gate = nn.Linear(d_model * 2, 1)

# Dynamic fusion
gate = torch.sigmoid(self.fusion_gate(torch.cat([token_emb, speaker_emb], dim=-1)))
combined = gate * token_emb + (1 - gate) * speaker_emb
```

**預期**: 提升 3-5% Val Acc

### 策略 3: Multi-Material Training（已包含）

**目標**: 學習材質不變的 representation

```python
# Training data 應該包含所有材質
input_dirs = [
    'data/raw/box',
    'data/raw/papercup',
    'data/raw/plastic',
    'data/clean/box2'  # 也包含 clean
]
```

**預期**: 已經在使用，無需額外修改

### 策略 4: Contrastive Learning（進階）

**目標**: 微調 speaker encoder 使其對 noise 魯棒

```python
# Contrastive loss: 同一 speaker 的 clean 和 noisy 應該接近
loss_contrast = InfoNCE(
    anchor=clean_emb,
    positive=noisy_emb,
    negatives=other_speakers_emb
)
```

**預期**: 提升 10-15% Val Acc，但需要額外訓練

---

## 📊 實驗建議

### 階段 1: Baseline Zero-Shot（當前計劃）

**配置**:
```python
speaker_encoder = ECAPA-TDNN (frozen)
fusion = additive (token_emb + speaker_emb)
regularization = dropout 0.1
```

**預期**: 50-60% Val Acc

**成功標準**: > 50% Val Acc (提升 32%)

### 階段 2: 如果效果不佳 (< 50%)

**調整 A**: 降低 speaker embedding 的影響
```python
# 使用加權融合
combined = 0.7 * token_emb + 0.3 * speaker_emb
```

**調整 B**: 添加 noise augmentation
```python
speaker_emb = speaker_emb + torch.randn_like(speaker_emb) * 0.15
```

### 階段 3: 如果效果很好 (> 60%)

**優化 A**: 微調 speaker encoder
```python
speaker_encoder.freeze = False
optimizer = Adam([
    {'params': denoiser.parameters(), 'lr': 3e-4},
    {'params': speaker_encoder.parameters(), 'lr': 1e-5}
])
```

**優化 B**: 多任務學習
```python
# 加入 speaker classification
speaker_classifier = nn.Linear(d_model, num_speakers)
loss_total = loss_denoising + 0.3 * loss_speaker_cls
```

---

## 📈 與 Clean Audio 驗證的對比

| 指標 | Clean Audio | Noisy Audio | 變化 |
|------|-------------|-------------|------|
| **同一 speaker 相似度** | 0.5543 | 0.50-0.57 | -5% to +3% |
| **不同 speaker 相似度** | 0.2015 | 0.18-0.26 | -10% to +30% |
| **區分度** | **0.3528** | **0.31** | **-12%** |
| **預期 Val Acc** | 65-75% | 50-65% | -15% to -10% |

**關鍵結論**:
- ✅ Noisy audio 的區分度只降低 12%
- ✅ 仍然足夠進行 speaker-conditioned denoising
- ⚠️  但預期效果會比理想情況低 10-15%

---

## 🎓 最終結論

### ✅ 驗證通過，但需要調整預期

1. **ECAPA-TDNN 在 Noisy Audio 上仍然有效**
   - 區分度 0.31 (vs Clean 0.35)
   - 所有材質都能維持良好區分

2. **Embedding Degradation 是主要挑戰**
   - Clean-Noisy 相似度只有 0.43-0.50
   - 需要 model 處理 degraded embeddings

3. **調整後的預期更現實**
   - 樂觀: 55-65% Val Acc (vs 原預期 65-75%)
   - 仍能達到顯著改進 (vs Baseline 38%)

4. **建議的策略**
   - ✅ 使用當前配置開始訓練（階段 1）
   - ⚠️  如果 < 50%，調整 fusion 權重（階段 2）
   - ✅ 如果 > 60%，考慮微調或多任務（階段 3）

---

## 📁 生成的視覺化

```
speaker_embedding_noisy_test/
├── clean_vs_noisy_comparison.png    ← Clean vs Noisy 相似度分布
└── discrimination_comparison.png    ← 各材質的區分度對比
```

---

**報告完成**: 2025-11-01 04:10
**結論**: ✅ ECAPA-TDNN 在 Noisy Audio 上驗證通過
**調整預期**: Val Acc 50-65% (vs 原預期 65-75%)
**下一步**: 開始訓練，並根據實際結果調整策略
