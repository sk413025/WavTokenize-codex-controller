# VQ Loss 上升 + Token Accuracy 下降 問題診斷報告

## 📊 問題現象

訓練 `strong_feature_ce` 50 epochs 後:

| 指標 | 訓練前 | 訓練後 | 變化 |
|------|--------|--------|------|
| VQ Loss | 0.148 | 0.292 | +97% 🔴 |
| Token Accuracy | 100% | 24.67% | -75% 🔴 |
| Feature L2 Distance | 0 | 2.99 | - |
| Cosine Similarity | 1.0 | 0.9885 | - |

---

## 🔍 問題：為什麼 Encoder 輸出會「離開」原本的 Codebook Embedding？

### 1. 架構理解

```
audio → Encoder → emb → VQ Quantizer → quantized → Decoder
                   ↓         ↓
              (原始輸出)  (量化後 = codebook[argmin])
```

WavTokenizer 的 `feature_extractor` 返回的是：
- `quantized`: 量化後的特徵 (等於 `codebook[selected_token]`)
- `codes`: 選擇的 token indices
- `commit_loss`: VQ commitment loss

**關鍵發現**：我們的 Loss 使用的是 `quantized`，而不是 `emb`！

### 2. STE (Straight-Through Estimator) 的運作

```python
# encoder/quantization/core_vq.py
if self.training:
    quantize = x + (quantize - x).detach()
```

- **Forward**: `quantize` = `codebook[argmin(distance)]` (離散選擇)
- **Backward**: 梯度直接傳給 `x` (encoder 輸出)，跳過 `argmin`

### 3. 問題根源

#### 訓練前狀態
```
Student Encoder = Teacher Encoder (初始化相同)
         ↓
同樣的輸入 → 同樣的 emb → 同樣的 argmin 結果
         ↓
Token Accuracy = 100%
```

#### 訓練後狀態
```
LoRA 改變了 Encoder 輸出
         ↓
emb 移動到不同的 Voronoi 區域
         ↓
argmin 選擇了不同的 token
         ↓
Token Accuracy 暴跌！
```

### 4. 圖解說明

```
                    Codebook 空間 (簡化為 2D)
                    
        ●─────────────│─────────────●
      #123           │            #456
                     │
        T ↗         │         ↖ S
                     │
                     │ ← Voronoi 邊界
                     
    T = Teacher encoder 輸出 (在 #123 的區域)
    S = Student encoder 輸出 (訓練後移動)
    
    問題：S 雖然「方向」接近 T (cosine similarity 0.99)
         但它跨越了 Voronoi 邊界，落在 #456 的區域
         → argmin 選擇 #456 而不是 #123
         → Token 選錯！
```

### 5. 為什麼 Feature Loss 沒有防止這個問題？

```python
# 目前的 Feature Loss
feature_loss = MSE(student_features, teacher_features)
             = MSE(codebook[student_codes], codebook[teacher_codes])
```

**問題**：
- `student_features` = `quantized` = `codebook[student選的token]`
- `teacher_features` = `quantized` = `codebook[teacher選的token]`

這是在比較「兩個 codebook embedding 的距離」，而不是監督 encoder 的原始輸出！

如果 Student 選錯了 token：
- `student_features` = `codebook[錯誤的token]`
- Feature Loss 會讓 Student 的錯誤 token 接近 Teacher 的正確 token
- 但這**不會**讓 encoder 輸出往正確方向移動！

### 6. 為什麼 VQ Loss 上升？

```python
# VQ Loss (Commitment Loss)
commit_loss = MSE(quantize.detach(), x)
            = MSE(codebook[選的token], encoder_output)
```

- 訓練改變了 `encoder_output`
- 但 codebook 是 frozen 的
- `encoder_output` 離開了原本的 codebook embedding
- → VQ Loss 上升

---

## ✅ 解決方案

### 方案：使用 Encoder 原始輸出 (emb) 計算 Loss

**目標**：讓 Student encoder 輸出「等於」Teacher 選的 codebook embedding

```python
# 修正後的架構
student_emb = student.encoder(noisy_audio)  # encoder 原始輸出
teacher_codes = teacher.get_codes(clean_audio)
target_embedding = codebook[teacher_codes]  # Teacher 選的 codebook embedding

# 新的 Loss
loss = MSE(student_emb, target_embedding)
```

**為什麼這樣有效**：
1. 直接監督 encoder 的原始輸出
2. Target 是 Teacher 選的 codebook embedding 的「中心」
3. 當 `student_emb ≈ target_embedding` 時，`argmin` 必然選擇正確的 token
4. 不會跨越 Voronoi 邊界

### 修改清單

1. **model.py**: 新增方法獲取 encoder 原始輸出 (emb)
2. **losses.py**: 修改 Loss 使用 emb 而不是 quantized
3. **train.py**: 更新訓練循環

---

## 📈 預期結果

| 指標 | 修正前 | 修正後 (預期) |
|------|--------|---------------|
| VQ Loss | 上升 | 下降或穩定 |
| Token Accuracy | 暴跌 | 上升或穩定 |
| Feature Loss | 下降 | 下降 |

---

## 🔬 驗證數據

### Student 到各 Codebook 的距離分析

訓練後的測試結果：
```
Student 到「自己選的 token」的平均距離: 2.56
Student 到「Teacher 選的 token」的平均距離: 4.52
距離差: 1.95

選錯率: 86.7%
選錯時距離差都 > 1.0 (不是邊界模糊的問題！)
```

這證明了 Student encoder 輸出確實移動到了「錯誤」的 Voronoi 區域。

---

## 📝 實驗記錄

- **診斷日期**: 2025-12-03
- **問題發現**: exp_1201/strong_feature_ce 實驗
- **修正版本**: exp_1203 (待實施)
