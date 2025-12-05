# exp_1204 實驗分析報告

> 日期: 2024-12-04
> 作者: Claude Code 自動生成

---

## 1. 實驗概述

### 1.1 實驗目標
解決 exp_1203 的瓶頸：Token Accuracy 約 10%，距離 100% 還很遠。

### 1.2 測試方案
| 實驗 | Loss | Curriculum | Temperature | 預期效果 |
|------|------|------------|-------------|----------|
| exp_1204 | MSE + CE | ✅ | 2.0 → 0.1 | 完整方案 |
| exp11 | MSE + CE | ❌ | 1.0 (固定) | 只加 CE |
| exp12 | MSE only | ✅ | 2.0 → 0.1 | 只加 Temp Annealing |

---

## 2. 實驗結果

### 2.1 最終結果

| 實驗 | 狀態 | Val Token Acc | 備註 |
|------|------|---------------|------|
| exp_1204 | ❌ 崩潰 | NaN | Epoch 49 出現 loss=nan |
| exp11 | ✅ 完成中 | ~8-10% | 無明顯改善 |
| exp12 | ✅ 完成中 | ~8-10% | 無明顯改善 |

### 2.2 關鍵觀察

1. **exp_1204 訓練崩潰**
   - 在 Epoch 49/50 出現 `loss=nan, acc=0.1%`
   - 原因：低溫度 (τ=0.1) 導致 CE Loss 數值不穩定
   - 當 τ=0.1 時，logits = -distance/0.1 = -10x distance
   - 距離差異被放大 10 倍，softmax 產生數值溢出

2. **exp11 和 exp12 效果相似**
   - Val Token Acc 都在 8-10% 範圍波動
   - 表示 CE Loss 和 Temperature Annealing 單獨都無法解決問題
   - 根本原因不在於 Loss 設計

3. **MSE Loss 持續下降但 Token Accuracy 停滯**
   - 這是核心矛盾
   - 下方詳細分析原因

---

## 3. 核心問題分析

### 3.1 MSE Loss vs Token Accuracy 的根本矛盾

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MSE Loss vs Token Accuracy 的根本矛盾                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Codebook 空間示意圖 (2D 簡化):                                              │
│                                                                              │
│       • token_0        • token_1                                            │
│                                                                              │
│                    ★ student_emb                                            │
│                                                                              │
│       • token_2        • token_3   ← teacher_codes = token_3               │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│  MSE Loss 優化目標: min ||student_emb - codebook[token_3]||²                │
│  Token Accuracy 目標: argmin_i ||student_emb - codebook[token_i]|| == 3    │
│                                                                              │
│  問題：MSE 可以讓 student_emb 「靠近」target，但不保證它是「最近的」！       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 兩種可能的失敗模式

#### 情況 A: MSE 下降但 Acc 不變
```
┌───────────────────────────────────────────────────────────────────────────┐
│  student 從距離 5.0 移動到距離 3.0 (MSE ↓ 64%)                            │
│  但 student 到其他 token 的距離也在變化                                   │
│  可能：到 token_2 的距離從 6.0 變成 2.5                                   │
│  → Token Accuracy 預測錯誤！選到 token_2 而不是 token_3                   │
│                                                                           │
│  診斷方式：                                                               │
│  - 檢查 student 預測的 token 分布                                         │
│  - 如果預測分散在多個不同的 token → 是情況 A                             │
└───────────────────────────────────────────────────────────────────────────┘
```

#### 情況 B: Mode Collapse
```
┌───────────────────────────────────────────────────────────────────────────┐
│  所有 student_emb 都收斂到 codebook 的「中心點」                          │
│  這個中心點到所有 token 的平均距離最小 → MSE 下降                         │
│  但 argmin 選到的 token 是固定的幾個 → Token Accuracy ≈ 隨機              │
│                                                                           │
│  診斷方式：                                                               │
│  - 檢查 student 預測的 token 分布                                         │
│  - 如果預測集中在少數幾個 token → 是情況 B (Mode Collapse)               │
│                                                                           │
│  實際觀察：Token Acc ~10% 表示有些結構被學到，但遠未達標                  │
└───────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Embedding 子空間問題

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Embedding 子空間問題                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Teacher encoder + VQ 產生的 codebook 結構:                                  │
│  ──────────────────────────────────────────────────────────────────────     │
│  - Codebook 有 4096 個 token，維度 512                                      │
│  - 這些 token 是通過 VQ-VAE 訓練得到的                                      │
│  - 它們在 512 維空間中有特定的分布結構                                      │
│                                                                              │
│  Student encoder (LoRA fine-tuned) 的問題:                                   │
│  ──────────────────────────────────────────────────────────────────────     │
│  - Student 使用 LoRA，只微調了部分參數                                      │
│  - LoRA rank=128 意味著只能調整 128 個方向的參數                            │
│  - Student 輸出的 embedding 被「限制」在某個子空間中                        │
│                                                                              │
│  數學解釋:                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Teacher codebook: 在完整的 512 維空間中分布                          │  │
│  │  Student embedding: 被限制在 LoRA 能調整的子空間中                    │  │
│  │                                                                       │  │
│  │  如果 codebook token 分布需要用到 512 維中的某些方向，               │  │
│  │  而 LoRA 無法調整這些方向，                                          │  │
│  │  Student 就無法「到達」這些 token 的位置！                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  類比:                                                                       │
│  - Teacher codebook 像一個 3D 立方體的頂點                                  │
│  - Student embedding 被困在一個 2D 平面上                                   │
│  - 無論 Student 怎麼移動，都無法「到達」正確的 3D 位置                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 CE Loss 的數值穩定性問題

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CE Loss 數值問題                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CE Loss 計算方式:                                                           │
│  logits = -distances / τ                                                     │
│  loss = CrossEntropy(softmax(logits), teacher_codes)                        │
│                                                                              │
│  當 τ=0.1 (低溫度) 時:                                                       │
│  ──────────────────────────────────────────────────────────────────────     │
│  - distance ≈ 1.0 → logit = -1.0 / 0.1 = -10                                │
│  - distance ≈ 2.0 → logit = -2.0 / 0.1 = -20                                │
│  - distance ≈ 5.0 → logit = -5.0 / 0.1 = -50                                │
│                                                                              │
│  softmax([-10, -20, -50, ...]) 的問題:                                       │
│  - exp(-10) ≈ 4.5e-5                                                        │
│  - exp(-50) ≈ 1.9e-22 → 下溢為 0                                            │
│  - 導致 log(0) = -inf → NaN                                                 │
│                                                                              │
│  這解釋了為什麼 exp_1204 在 epoch 49 崩潰！                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 建議的解決方案

### 方案 A: 直接 Token 預測 (推薦)
```python
# 不經過 codebook 距離計算，直接用 Linear projection
student_emb → Linear(512, 4096) → softmax → CE with teacher_codes

# 優點：
# - 繞過距離計算的問題
# - 直接優化 Token Accuracy
# - 數值穩定
```

### 方案 B: Encoder 對齊
```python
# 對齊 encoder 輸出，而不是 codebook embedding
Loss = MSE(student_encoder_output, teacher_encoder_output)

# 優點：
# - Student 學習 teacher 的完整表示
# - 不依賴於 VQ 後的離散結果
```

### 方案 C: 對比學習
```python
# 使用 InfoNCE Loss
# 正樣本: 同一個 frame 的 teacher_code 對應的 embedding
# 負樣本: 其他 frame 的 embedding

# 優點：
# - 強制模型學習「區分」不同的 token
# - 不只是「接近」某個 target
```

### 方案 D: Full Fine-tuning
```python
# 放棄 LoRA，進行完整的 encoder fine-tuning

# 優點：
# - 消除 LoRA 的表達能力限制
# - Student 可以到達任意 embedding 位置

# 缺點：
# - 顯存需求大幅增加
# - 訓練時間更長
```

---

## 5. 診斷工具

### 5.1 驗證 Mode Collapse
```python
# 檢查 student 預測的 token 分布
# 如果集中在少數 token → Mode Collapse
# 如果分散在多個 token → 情況 A
```

### 5.2 可視化 Embedding 空間
```python
# 使用 PCA/t-SNE 降維
# 比較 student_emb 和 codebook 的分布
# 觀察是否存在「子空間」問題
```

---

## 6. 後續實驗建議

1. **首先進行診斷**
   - 運行 `diagnose_embedding_space.py` 確認問題類型

2. **根據診斷結果選擇方案**
   - 如果是 Mode Collapse → 方案 C (對比學習)
   - 如果是子空間問題 → 方案 D (Full Fine-tuning)
   - 如果兩者都有 → 方案 A (直接 Token 預測)

3. **修復 CE Loss 數值問題**
   - 設定最小溫度 τ_min = 0.5 (不要太低)
   - 使用 label smoothing
   - 或直接使用方案 A 繞過

---

## 7. 總結

exp_1204 系列實驗證明了：
1. **Curriculum Learning + CE Loss 無法解決 Token Accuracy 停滯問題**
2. **根本原因是 MSE Loss 的優化目標和 Token Accuracy 的目標不一致**
3. **LoRA 的表達能力限制可能是另一個瓶頸**
4. **需要使用直接優化 Token Accuracy 的方法或增強模型表達能力**

下一步：運行診斷腳本，確認具體問題，然後選擇合適的解決方案。

---

## 8. 診斷結果 (2024-12-05 更新)

### 8.1 診斷數據

| 指標 | 數值 | 說明 |
|------|------|------|
| Unique predictions | 1208 / 4096 | 29.5% 覆蓋率 |
| Normalized entropy | 66.5% | 不是 Mode Collapse |
| Token Accuracy | 2.21% | 極低 |
| Top-5 Accuracy | 6.46% | |
| Top-10 Accuracy | 8.61% | |
| Top-100 Accuracy | 31.34% | |
| 到正確 token 平均距離 | 3.75 | |
| 到最近 token 平均距離 | 0.45 | 差距 8x |
| 正確 token 平均排名 | 1505 / 4096 | |

### 8.2 確診結果

**✓ 確認是情況 A：MSE 下降但預測分散到錯誤的 token**

**排除情況 B (Mode Collapse)**：
- Unique predictions = 1208 (遠超 Mode Collapse 的特徵)
- Entropy = 66.5% (Mode Collapse 會接近 0%)

**排除子空間限制問題**：
- Student 有效維度 = 1.7
- Codebook 有效維度 = 2.5
- 比值 = 0.68 (相近，不是子空間問題)

### 8.3 問題根源

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   MSE Loss 優化目標不匹配                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MSE Loss: min ||student_emb - codebook[teacher_code]||²                    │
│                                                                              │
│  問題：MSE 只讓 student 「接近」target，不保證它是「最近的」！              │
│                                                                              │
│  實際情況：                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  • token_0        • token_1        • token_2                          │  │
│  │                                                                       │  │
│  │              ★ student    ← MSE 讓 student 往 target 移動            │  │
│  │                  ↓                                                    │  │
│  │              • target (token_3)                                       │  │
│  │                                                                       │  │
│  │  但 student 移動過程中，可能更靠近 token_0, token_1, token_2！        │  │
│  │  → argmin 選到錯誤的 token → Token Accuracy 不提升                   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  證據：                                                                      │
│  - 到正確 token 距離 = 3.75                                                 │
│  - 到最近 token 距離 = 0.45                                                 │
│  - Student 確實「靠近」某些 token，但不是正確的 token！                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.4 可視化證據

診斷生成了以下圖表：

1. **embedding_pca.png**: Student embedding 的 PCA 分布
   - 紅點 (錯誤預測) 分散在大範圍
   - 綠點 (正確預測) 僅 116 個，集中在右上角小區域
   - Student embedding 集中在 codebook 中央，沒有分散到邊緣

2. **distance_distribution.png**: 距離分布
   - 到最近 token 的距離很小 (~0.4)
   - 到正確 token 的距離很大 (~3.7)
   - 正確 token 的排名大多在 100-500 或 2000+

3. **pca_variance.png**: PCA 解釋變異量
   - Student 和 Codebook 都只需 2-3 維解釋 90%+ variance
   - 子空間維度相近，不是限制因素

### 8.5 建議的解決方案

#### 方案 A: 直接 Token 預測 (最推薦)

```python
# 不經過 codebook 距離計算，直接用 Linear projection
class TokenPredictionHead(nn.Module):
    def __init__(self, embed_dim=512, vocab_size=4096):
        super().__init__()
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, student_emb):
        # student_emb: (B, 512, T)
        logits = self.proj(student_emb.transpose(1, 2))  # (B, T, 4096)
        return logits

# Loss
loss = F.cross_entropy(logits.view(-1, 4096), teacher_codes.view(-1))
```

**優點**：
- 直接優化 Token Accuracy
- 繞過距離計算的問題
- 數值穩定

#### 方案 B: Margin-based Loss

```python
def margin_loss(student_emb, codebook, teacher_codes, margin=0.5):
    # student_emb: (N, 512)
    # codebook: (4096, 512)
    # teacher_codes: (N,)

    distances = torch.cdist(student_emb, codebook)  # (N, 4096)

    # 到正確 token 的距離
    correct_dist = distances.gather(1, teacher_codes.unsqueeze(1))  # (N, 1)

    # 到最近錯誤 token 的距離
    mask = torch.ones_like(distances).scatter_(1, teacher_codes.unsqueeze(1), 0)
    wrong_dist = (distances + mask * 1e9).min(dim=1, keepdim=True)[0]  # (N, 1)

    # Margin loss: 確保正確距離比錯誤距離小一個 margin
    loss = F.relu(correct_dist - wrong_dist + margin).mean()
    return loss
```

**優點**：
- 強制模型區分正確和錯誤 token
- 不只「接近」，還要「比其他更近」

#### 方案 C: Hard Negative Mining + CE

```python
def hard_negative_ce_loss(student_emb, codebook, teacher_codes, k=100, temperature=1.0):
    distances = torch.cdist(student_emb, codebook)  # (N, 4096)

    # 找到最近的 K 個 token (不包含正確 token)
    _, top_k_indices = distances.topk(k, dim=1, largest=False)

    # 確保正確 token 在候選中
    # ... (實作省略)

    # 只在這 K+1 個 token 上計算 CE
    logits = -distances[:, candidates] / temperature
    loss = F.cross_entropy(logits, local_labels)
    return loss
```

---

## 9. 診斷文件位置

```
exp_1204/
├── diagnosis_results/
│   ├── diagnosis_results.json    # 診斷數據 (JSON)
│   ├── embedding_pca.png         # PCA 降維可視化
│   ├── distance_distribution.png # 距離分布圖
│   └── pca_variance.png          # PCA 解釋變異量
├── diagnose_simple.py            # 診斷腳本
└── ANALYSIS_REPORT.md            # 本報告
```
