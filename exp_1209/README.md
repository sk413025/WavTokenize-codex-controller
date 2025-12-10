# Exp_1209: Adapter + 方向性 Loss 實驗

## 實驗背景

基於 exp_1207 的分析結論：
1. **MSE 優化方向錯誤**：cosine similarity 僅 0.061（幾乎正交）
2. **LoRA 容量不足**：0.19% 參數無法做大幅修正
3. **VQ 空間幾何問題**：Noise 位移是 Voronoi cell 半徑的 16 倍

## 實驗設計

### 方案 D: Adapter + 方向性 Loss

```
Audio → Encoder(凍結) → [DenoiseAdapter(訓練)] → VQ → Decoder
```

- **DenoiseAdapter**: 輕量 MLP，專門修正 encoder 輸出
- **參數量**: ~262K (0.33%)
- **Loss**: Triplet Loss 或 Contrastive Loss

### 方案 B+A: 擴大 LoRA + 方向性 Loss

```
Audio → Encoder(LoRA 18層, rank=256) → VQ → Decoder
```

- **可訓練參數**: ~3.7M (4.4%)
- **Loss**: Triplet Loss 或 Contrastive Loss

---

## Loss 設計說明

### 為什麼需要新的 Loss？

原本的 MSE Loss 問題：
```
MSE = ||student - teacher||²
```
- 只優化「距離」，不管「方向」
- Student 可能往錯誤的 code 方向移動
- 結果：Feature Loss 下降，Token Accuracy 反而下降

### Triplet Loss vs Contrastive Loss

| 特性 | Triplet Loss | Contrastive Loss |
|------|--------------|------------------|
| 樣本組成 | (anchor, positive, negative) | (sample1, sample2, label) |
| 負樣本數 | 每次 1 個 | 可以多個 |
| 難度 | 需要 mining 策略 | 相對簡單 |
| 效果 | 精確控制 margin | batch 內對比學習 |

### 本實驗選擇：Triplet Loss + Hard Negative Mining

```python
Triplet Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)

其中：
- anchor: teacher encoder output（正確的特徵）
- positive: student encoder output（應該靠近 anchor）
- negative: 錯誤 code 的 embedding（應該遠離）
```

### Loss 組合策略

**不是替換，是組合！**

```python
Total Loss = λ₁ * Feature_Loss + λ₂ * Triplet_Loss + λ₃ * CE_Loss

建議配置：
- Feature Loss (MSE): 保持特徵結構穩定
- Triplet Loss: 確保方向正確
- CE Loss (可選): 直接優化 token 分類
```

---

## 實驗配置

### Exp19: Adapter + Triplet Loss

```yaml
model:
  type: adapter
  adapter_hidden: 256
  freeze_encoder: true

loss:
  feature_weight: 1.0
  triplet_weight: 1.0
  triplet_margin: 0.5
  ce_weight: 0.0  # 先不用 CE

training:
  lr: 1e-4
  epochs: 50
  batch_size: 8
```

### Exp20: Adapter + Triplet + CE

```yaml
loss:
  feature_weight: 1.0
  triplet_weight: 1.0
  triplet_margin: 0.5
  ce_weight: 0.1
  ce_temperature: 0.1
```

### Exp21: 擴大 LoRA + Triplet Loss

```yaml
model:
  type: lora
  lora_rank: 256
  lora_target_modules: all_encoder_conv  # 18 層

loss:
  feature_weight: 1.0
  triplet_weight: 1.0
  triplet_margin: 0.5
```

### Exp22: 擴大 LoRA + Triplet + CE

```yaml
loss:
  feature_weight: 1.0
  triplet_weight: 1.0
  triplet_margin: 0.5
  ce_weight: 0.1
```

---

## Hard Negative Mining 策略

### 為什麼需要 Hard Negative？

隨機選擇 negative 太簡單，模型不需要學習就能區分。
Hard negative 是「幾乎正確但實際錯誤」的樣本，迫使模型學習精細區分。

### 實現方式

```python
def get_hard_negatives(student_out, codebook, teacher_codes, k=5):
    """
    選擇最難區分的負樣本

    Args:
        student_out: (B, C, T) student encoder 輸出
        codebook: (num_codes, C) VQ codebook
        teacher_codes: (B, T) 正確的 token
        k: 選擇 top-k 最近的錯誤 code

    Returns:
        hard_negatives: (B, T, C) 最難區分的錯誤 code embedding
    """
    B, C, T = student_out.shape

    # 計算到所有 code 的距離
    z = student_out.permute(0, 2, 1)  # (B, T, C)
    dists = torch.cdist(z.reshape(-1, C), codebook)  # (B*T, num_codes)

    # 把正確 code 的距離設為無窮大（排除）
    teacher_flat = teacher_codes.reshape(-1)
    dists[torch.arange(len(teacher_flat)), teacher_flat] = float('inf')

    # 選擇最近的錯誤 code（最難區分的）
    hard_neg_idx = dists.argmin(dim=1)  # (B*T,)
    hard_negatives = codebook[hard_neg_idx]  # (B*T, C)

    return hard_negatives.reshape(B, T, C)
```

---

## 預期結果

| 實驗 | 預期 Token Accuracy | 說明 |
|------|---------------------|------|
| Exp19 | 10-20% | Adapter 容量有限 |
| Exp20 | 15-25% | 加入 CE 應該有幫助 |
| Exp21 | 20-35% | LoRA 容量大 |
| Exp22 | 25-40% | 最完整的配置 |

baseline (原始 noisy): ~5%

---

## 文件結構

```
exp_1209/
├── README.md                 # 本文件
├── models.py                 # DenoiseAdapter 定義
├── losses.py                 # Triplet Loss, Contrastive Loss
├── train_adapter.py          # 方案 D 訓練腳本
├── train_lora_expanded.py    # 方案 B+A 訓練腳本
├── run_exp19_adapter_triplet.sh
├── run_exp20_adapter_triplet_ce.sh
├── run_exp21_lora_triplet.sh
├── run_exp22_lora_triplet_ce.sh
└── experiments/              # 實驗結果
```

---

## 重現實驗

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1209

# 方案 D: Adapter
bash run_exp19_adapter_triplet.sh
bash run_exp20_adapter_triplet_ce.sh

# 方案 B+A: 擴大 LoRA
bash run_exp21_lora_triplet.sh
bash run_exp22_lora_triplet_ce.sh
```
