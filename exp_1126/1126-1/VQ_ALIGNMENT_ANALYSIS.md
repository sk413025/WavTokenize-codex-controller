# VQ Codebook 對齊分析報告

> **日期**: 2025-11-27
> **實驗**: LoRA Encoder Denoising (1126-1)
> **分支**: `feat/hdf5-training-implementation`
> **更新**: 修正 Codebook EMA 更新機制分析

---

## 1. 實驗背景

### 1.1 目標
使用 Teacher-Student 架構 + LoRA 微調 WavTokenizer 的 Encoder，實現音訊去噪。

### 1.2 架構
```
┌─────────────────────────────────────────────────────────────┐
│                   Teacher (完全凍結)                         │
│   clean_audio → [Encoder] → features → [VQ] → codes_clean   │
│                                         ↑                    │
│                                    Codebook_T (凍結)         │
└─────────────────────────────────────────────────────────────┘
                        ↓ Feature Loss (MSE)
┌─────────────────────────────────────────────────────────────┐
│                   Student (LoRA 微調)                        │
│   noisy_audio → [Encoder+LoRA] → features' → [VQ] → codes'  │
│                                               ↑              │
│                                          Codebook_S (EMA 更新!)
└─────────────────────────────────────────────────────────────┘
```

### 1.3 三個實驗配置

| 實驗 | Feature Loss | Distance Loss | VQ Loss | Epochs |
|------|-------------|---------------|---------|--------|
| `lora_encoder_1126_1` | 1.0 | **0.1** | **0.01** | 50 |
| `lora_encoder_1126_1_balanced` | 1.0 | **0.01** | 0.0 | 40 |
| `lora_encoder_1126_1_FD_v2` | 1.0 | **0.0** | 0.0 | 20 |

---

## 2. 實驗結果

### 2.1 Training Curves 對比

| 指標 | 原始 (Ep50) | 平衡 (Ep40) | FD_v2 (Ep20) |
|------|------------|------------|--------------|
| **Total Loss** | ~0.38 (震盪) | ~0.064 (穩定) | ~0.031 (最低) |
| **Feature Loss** | 0.042→0.030 | 0.042→0.030 | 0.042→0.031 |
| **Distance Loss** | 3.3→3.7 ↑ | 3.3→3.5 | 3.2→3.7 ↑ |
| **Train VQ Loss** | ~0.00035 | ~0.00038 | ~0.00035 |
| **Val VQ Loss** | ≈0 | ≈0 | ≈0 |
| **Token Accuracy** | **25%→3%** ↓ | **30%→5%** ↓ | **30%→5%** ↓ |

### 2.2 關鍵觀察

1. **Feature Loss 正常下降** - Student 確實在學習 Teacher 的 feature
2. **Distance Loss 沒有收斂** - 在所有實驗中都呈現震盪或上升
3. **Token Accuracy 崩潰** - 從 25-30% 急劇下降到 3-5%
4. **Val VQ Loss = 0** - 這是預期行為（見第 3 節）

---

## 3. Val VQ Loss 為何為零？

### 3.1 程式碼追蹤

在 `encoder/quantization/core_vq.py:301-311`：

```python
def forward(self, x):
    # ...
    loss = torch.tensor([0.0], device=device, requires_grad=self.training)

    if self.training:  # ⚠️ 只在 training=True 時計算
        if self.commitment_weight > 0:
            commit_loss = F.mse_loss(quantize.detach(), x)
            loss = loss + commit_loss * self.commitment_weight

    return quantize, embed_ind, loss
```

### 3.2 結論

**Val VQ Loss = 0 是預期行為，不是 bug。**

- `model.eval()` 使 `self.training = False`
- VQ Commitment Loss 只在 training 時計算
- Validation 時直接返回 `0.0`

---

## 4. 🔴 重大發現：Codebook EMA 更新機制

### 4.1 原本的假設（錯誤）

> "只有一本 codebook，Student 的 VQ 選出的 code 不是最近的"

### 4.2 實際情況（正確）

**Teacher 和 Student 各有獨立的 Codebook！**

```python
# 驗證結果
Teacher codebook == Student codebook: False
Max diff: 0.741366
Mean diff: 0.043706
Entries with diff > 0.01: 1846 / 4096
```

### 4.3 為什麼 Codebook 會不同？

在 `core_vq.py:217-229` 中，Codebook 通過 **EMA** 在訓練時自動更新：

```python
if self.training:
    # EMA 更新 cluster 統計
    ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
    embed_sum = x.t() @ embed_onehot
    ema_inplace(self.embed_avg, embed_sum.t(), self.decay)

    # 更新 codebook embeddings
    cluster_size = laplace_smoothing(...)
    embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
    self.embed.data.copy_(embed_normalized)  # ⚠️ Codebook 被更新！
```

**關鍵點：**
- Codebook 是 **buffer** 不是 **parameter**
- 不受 `requires_grad=False` 控制
- 在 `training=True` 時通過 EMA 自動更新

### 4.4 驗證結果

| 驗證方式 | Nearest == Selected |
|----------|---------------------|
| 用 **Teacher** codebook | **0%** ❌ |
| 用 **Student** codebook | **100%** ✅ |

**結論：Student 的 VQ 選擇在自己的 codebook 空間中是正確的！**

---

## 5. 問題根因分析（修正版）

### 5.1 核心問題：Teacher-Student Codebook 分歧

```
┌─────────────────────────────────────────────────────────────┐
│                        訓練過程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Teacher (凍結)                                            │
│   ┌────────────────┐     ┌──────────────┐                  │
│   │ Encoder (凍結) │ --> │ VQ + Codebook_T │ --> codes_T   │
│   └────────────────┘     │   (凍結)       │                │
│                          └──────────────┘                  │
│                                                             │
│   Student (訓練)                                            │
│   ┌────────────────┐     ┌──────────────┐                  │
│   │ Encoder + LoRA │ --> │ VQ + Codebook_S │ --> codes_S   │
│   └────────────────┘     │   (EMA 更新!)  │                │
│                          └──────────────┘                  │
│                                                             │
│   Feature Loss: MSE(features_S, features_T) ✅ 下降         │
│   但 Codebook_S ≠ Codebook_T → codes_S ≠ codes_T ❌        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 為什麼 Token Accuracy 下降？

1. **Encoder 被 LoRA 修改** → 輸出 feature 改變
2. **Student Codebook 通過 EMA 適應新的 feature** → Codebook_S 改變
3. **Teacher Codebook 保持不變** → Codebook_T 固定
4. **Token Accuracy 計算的是 codes_S vs codes_T** → 越來越不匹配

### 5.3 Feature Loss 降低但 Token Accuracy 下降的悖論

| 觀察 | 解釋 |
|------|------|
| Feature MSE: 0.073 → 0.047 ↓ | Student feature 接近 Teacher feature |
| Token Accuracy: 30% → 3% ↓ | 但 Codebook 分歧導致 code 不同 |

**這不是 VQ 失配的問題，而是 Codebook 分歧的問題！**

---

## 6. Distance Loss 為什麼沒有收斂？

### 6.1 Distance Loss 的定義

```python
# losses.py
distances = distance_matrix[student_codes, teacher_codes]
distance_loss = distances.mean()
```

這裡 `distance_matrix` 是基於 **Teacher codebook** 計算的：
```python
# model.py
codebook = self.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed
dist_matrix = torch.cdist(codebook, codebook, p=2)
```

### 6.2 問題

1. `student_codes` 是用 **Student codebook** 選出來的
2. `distance_matrix` 是用 **Teacher codebook** 計算的
3. 這兩個 codebook **不同**！

**結果：Distance Loss 在比較「蘋果和橘子」，無法有效優化！**

### 6.3 示意圖

```
Teacher Codebook:  [A, B, C, D, E, ...]  (用來計算 distance_matrix)
Student Codebook:  [A', B', C', D', E', ...]  (被 EMA 修改)

Student 選的 code index = 2 (對應 C')
Teacher 選的 code index = 5 (對應 E)

Distance Loss = distance_matrix[2, 5]
             = dist(C, E)  ← 用的是 Teacher codebook 的 C 和 E
             ≠ dist(C', E)  ← 但 Student 實際輸出的是 C'
```

---

## 7. 修正方案

### 7.1 方案 1: 凍結 Student Codebook（推薦優先嘗試）

**在訓練時關閉 EMA 更新：**

```python
# 方法 1: 設置 model.eval() 後再 forward
model.student.eval()  # 這會讓 VQ 不做 EMA 更新
output = model(noisy, clean)
model.student.train()  # 恢復 training mode

# 方法 2: 直接修改 VQ 的 training flag
model.student.feature_extractor.encodec.quantizer.eval()
```

**優點**：
- 簡單，不需要修改架構
- 確保 Teacher 和 Student 使用相同的 codebook
- Distance Loss 會變得有意義

**缺點**：
- 需要確認這不會影響其他訓練行為

### 7.2 方案 2: 使用 Teacher Codebook 做 Student 的 VQ

```python
# 在 forward 時，用 Teacher 的 codebook 替換 Student 的
student_vq = model.student.feature_extractor.encodec.quantizer.vq.layers[0]
teacher_codebook = model.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed

# 複製 Teacher codebook 到 Student
student_vq._codebook.embed.data.copy_(teacher_codebook.data)
```

**優點**：確保 codebook 一致
**缺點**：需要每個 forward 都做

### 7.3 方案 3: 修改 Distance Loss 使用動態計算

```python
def compute_distance_loss(student_codes, teacher_codes, student_codebook, teacher_codebook):
    """使用實際的 embedding 計算距離"""
    B, T = student_codes.shape

    # 取出實際的 embedding
    student_embed = student_codebook[student_codes.reshape(-1)]  # (B*T, 512)
    teacher_embed = teacher_codebook[teacher_codes.reshape(-1)]  # (B*T, 512)

    # 計算 L2 距離
    distances = (student_embed - teacher_embed).norm(dim=1)
    return distances.mean()
```

**優點**：正確計算兩個空間的距離
**缺點**：需要修改 loss 計算邏輯

### 7.4 方案 4: Token-Level KL Distillation

用 soft target 監督 token 選擇，避免 codebook 分歧問題：

```python
def token_distillation_loss(student_feat, teacher_feat, codebook, temperature=1.0):
    """使用同一個 codebook (Teacher's) 計算 soft targets"""
    # 用 Teacher codebook 計算 logits
    student_logits = -torch.cdist(student_feat, codebook)  # (B*T, K)
    teacher_logits = -torch.cdist(teacher_feat, codebook)

    # KL Divergence
    loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)

    return loss
```

**優點**：
- 直接監督 VQ 選擇
- 使用同一個 codebook 避免分歧
- 不需要修改 VQ 機制

---

## 8. 建議的下一步

### 8.1 優先嘗試：方案 1（凍結 Student Codebook）

```python
# 在 train.py 的 train_epoch 中
for batch_idx, batch in enumerate(pbar):
    # 凍結 quantizer 的 EMA 更新
    self.model.student.base_model.model.feature_extractor.encodec.quantizer.eval()

    # Forward pass
    with autocast(enabled=self.config.use_amp):
        output = self.model(noisy_audio, clean_audio)
        loss, loss_dict = self.criterion(output, self.distance_matrix)

    # 其餘訓練邏輯...
```

### 8.2 驗證指標

成功的標準：
1. **Token Accuracy 不下降**（維持或提升）
2. **Distance Loss 正常收斂**（不再震盪）
3. **Codebook 保持一致**：`torch.allclose(teacher_cb, student_cb) = True`

### 8.3 驗證腳本使用

```bash
# 訓練後驗證
cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1126/1126-1
python verify_vq_alignment.py --checkpoint <path_to_checkpoint> --num_batches 15
```

---

## 9. 相關檔案

| 檔案 | 說明 |
|------|------|
| `train.py` | 訓練腳本 |
| `model.py` | Teacher-Student 模型定義 |
| `losses.py` | Loss 函數 |
| `verify_vq_alignment.py` | VQ 對齊驗證腳本 |
| `encoder/quantization/core_vq.py` | VQ 實現（含 EMA 更新） |

---

## 10. 結論

### 問題總結（修正版）

| 問題 | 原因 | 證據 |
|------|------|------|
| **Val VQ Loss = 0** | VQ Commitment Loss 只在 training 時計算 | `core_vq.py:301-311` |
| **Token Accuracy 崩潰** | **Codebook EMA 更新導致 Teacher-Student 分歧** | Teacher CB ≠ Student CB |
| **Distance Loss 不收斂** | Distance matrix 基於 Teacher CB，但 codes 基於 Student CB | 比較「蘋果和橘子」 |

### 關鍵發現

1. **Codebook 是 buffer，不是 parameter**，不受 `requires_grad=False` 控制
2. **VQ 的 EMA 機制在 `training=True` 時自動更新 Codebook**
3. **Student 的 VQ 選擇在自己的 codebook 空間中是正確的（100% match）**
4. **問題不是 VQ 選錯 code，而是 Teacher-Student Codebook 分歧**

### 解決方向

1. **凍結 Student Codebook**（設置 quantizer.eval()）
2. **使用 Teacher Codebook 做 Student 的 VQ**
3. **使用 Token-Level KL Distillation** 避免 codebook 依賴
