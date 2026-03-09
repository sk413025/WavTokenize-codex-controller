# Loss Components: Exp 6c vs Baseline 詳細對比

**日期**: 2026-02-06
**目的**: 詳細比較 Phase 3-2 Exp 6c 與 Baseline (exp_k_v6) 的 Loss 設計差異

---

## 目錄

1. [總覽對比](#總覽對比)
2. [Baseline Loss Components](#baseline-loss-components)
3. [Exp 6c Loss Components](#exp-6c-loss-components)
4. [關鍵差異分析](#關鍵差異分析)
5. [為什麼 Exp 6c 的 Loss 設計更有效](#為什麼-exp-6c-的-loss-設計更有效)

---

## 總覽對比

### Baseline (exp_0112_intermediate v6)

```python
Total Loss =
    feature_weight * L_feature +        # MSE(z_e, t_e) - Pre-quant
    triplet_weight * L_triplet +        # Triplet loss on codebook
    intermediate_weight * L_inter       # Intermediate supervision (L3,L4,L6)

# 權重配置
feature_weight = 1.0
triplet_weight = 0.5
intermediate_weight = 0.5 (動態衰減至 0.25)

# Quantizer
- Single VQ (K=4096)
- Frozen (eval mode, no gradient)
- No commitment loss
```

### Exp 6c (Phase 3-2 EMA+RVQ)

```python
Total Loss =
    λ_quant * L_quant +                 # MSE(z_q, t_e) - Post-quant ✅
    λ_pre * L_pre +                     # MSE(z_e, t_e) - Pre-quant ❌
    λ_inter * L_inter +                 # Intermediate supervision (L4,L8)
    β_commit * L_commit                 # Encoder commitment ✅

# 權重配置
λ_quant = 1.0
λ_pre = 0.0        # 完全禁用!
λ_inter = 0.5
β_commit = 1.0

# Quantizer
- Residual VQ (K=2048, L=4)
- EMA update (decay=0.99)
- Dead-code reset (threshold=2)
- Bidirectional commitment ✅
```

---

## Baseline Loss Components

### 架構圖

```
Clean Audio  →  Teacher Encoder  →  t_e [B,512,T]  →  (Frozen Quantizer)
                     ↓ (L3,L4,L6)
                   t_inter

Noisy Audio  →  Student Encoder  →  z_e [B,512,T]  →  (Frozen Quantizer)
                     ↓ (L3,L4,L6)
                   s_inter
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Loss Components                           │
├─────────────────────────────────────────────────────────────────┤
│ L_feature   = MSE(z_e, t_e.detach())          [Pre-quant]      │
│ L_triplet   = TripletLoss(z_e, t_codes, CB)  [VQ alignment]   │
│ L_inter     = Σ CosineLoss(s_Li, t_Li)       [Multi-scale]    │
│                                                                 │
│ Total = 1.0·L_feature + 0.5·L_triplet + 0.5·L_inter           │
└─────────────────────────────────────────────────────────────────┘
```

### 1. **L_feature** (Pre-Quantization MSE) - weight=1.0

```python
# 定義
L_feature = MaskedMSE(z_e, t_e.detach())

# 實現 (MaskedFeatureLoss)
def forward(student_features, teacher_features, lengths):
    # Create frame-level mask
    mask = create_length_mask(lengths, max_audio_len, encoder_stride=320)
    mask_expanded = mask.unsqueeze(1)  # [B, 1, T]

    # MSE with mask
    diff_sq = (student_features - teacher_features) ** 2  # [B, D, T]
    masked_diff = diff_sq * mask_expanded
    loss = masked_diff.sum() / (mask_expanded.sum() * D + 1e-8)

    return loss

# 輸入
student_features = z_e [B, 512, T]  # Student encoder output (pre-quant)
teacher_features = t_e [B, 512, T]  # Teacher encoder output (detached)

# 目的
- 對齊 student 和 teacher 的 encoder output
- 在 quantization 之前對齊

# 問題 ⚠️
- 只監督 pre-quantization features
- Quantizer 被繞過 (因為 loss 不依賴 quantized output)
- 導致 token collapse (Phase 3 失敗的根本原因)
```

### 2. **L_triplet** (Triplet Loss on Codebook) - weight=0.5

```python
# 定義
L_triplet = MaskedTripletLoss(z_e, t_codes, codebook)

# 實現
def forward(student_features, teacher_codes, codebook, lengths):
    # Compute distances to all codebook entries
    z = student_features.permute(0, 2, 1).reshape(-1, D)  # [B*T, D]
    dists = torch.cdist(z, codebook, p=2)  # [B*T, K]

    # Positive distance (to teacher's code)
    teacher_flat = teacher_codes.reshape(-1)  # [B*T]
    pos_dist = dists[batch_indices, teacher_flat]  # [B*T]

    # Negative distance (to closest non-teacher code)
    dists_for_neg = dists.clone()
    dists_for_neg[batch_indices, teacher_flat] = float('inf')
    neg_dist = dists_for_neg.min(dim=1).values  # [B*T]

    # Triplet loss: max(0, pos_dist - neg_dist + margin)
    triplet = F.relu(pos_dist - neg_dist + margin)  # margin=0.2

    # Apply mask
    masked_triplet = triplet.reshape(B, T) * mask
    loss = masked_triplet.sum() / (mask.sum() + 1e-8)

    return loss

# 輸入
student_features = z_e [B, 512, T]  # Pre-quant encoder output
teacher_codes = t_codes [B, T]      # Teacher VQ codes (from frozen quantizer)
codebook = [4096, 512]              # Frozen VQ codebook

# 目的
- 鼓勵 z_e 靠近 teacher 選擇的 codebook entry
- 間接引導 student 使用正確的 codes

# 問題 ⚠️
- 依賴 frozen codebook (無法更新)
- 依然是 pre-quant supervision (z_e, 不是 z_q)
- Teacher codes 可能本身就有 collapse 問題
```

### 3. **L_inter** (Intermediate Supervision) - weight=0.5

```python
# 定義 (IntermediateSupervisionLossV6)
L_inter = Σ w_i * CosineLoss(s_Li, t_Li)

# 監督層
layer_weights = {
    3: 0.3,   # model[3]: Downsample 1
    4: 0.5,   # model[4]: ResBlock 2
    6: 0.5,   # model[6]: Downsample 2
}

# 實現
def forward(student_features, teacher_features, layer_scale=1.0):
    total_loss = 0.0

    for idx in [3, 4, 6]:
        student_feat = student_features[idx]  # [B, C, T]
        teacher_feat = teacher_features[idx]  # [B, C, T]

        # Flatten to [B*T, C]
        student_flat = student_feat.permute(0, 2, 1).reshape(-1, C)
        teacher_flat = teacher_feat.permute(0, 2, 1).reshape(-1, C)

        # Cosine Loss: 1 - cos_sim
        student_norm = F.normalize(student_flat, dim=-1)
        teacher_norm = F.normalize(teacher_flat, dim=-1)
        cos_sim = (student_norm * teacher_norm).sum(dim=-1).mean()
        loss = 1 - cos_sim

        # Weighted sum
        weight = layer_weights[idx] * layer_scale
        total_loss += weight * loss

    return total_loss, layer_losses

# 目的
- 多尺度監督: L3 (淺層), L4 (中層), L6 (中層)
- 幫助 encoder 早期層學習
- 使用 Cosine Loss (方向對齊)

# 動態權重衰減
layer_scale = {
    epoch ≤ 100: 1.0 (base_weight=0.5)
    epoch 101-200: 線性衰減 0.5 → 0.25
    epoch > 200: 0.25
}

# 問題 ⚠️
- 可能造成 "shortcut learning" (H3)
- Encoder 走向「少量 codes 即可擬合中間層」的捷徑
```

### Baseline Total Loss

```python
# 訓練循環 (train_v6.py:259-260)
total_loss = final_loss + intermediate_weight * inter_loss

where:
    final_loss = feature_weight * L_feature + triplet_weight * L_triplet
    intermediate_weight = 0.5 (動態衰減)
    inter_loss = L_inter

# 完整展開
total_loss =
    1.0 * MSE(z_e, t_e) +
    0.5 * TripletLoss(z_e, t_codes, codebook) +
    0.5 * [0.3*Cos(s_L3,t_L3) + 0.5*Cos(s_L4,t_L4) + 0.5*Cos(s_L6,t_L6)]

# Quantizer
- Frozen (eval mode)
- No gradient, no update
- Codebook 完全固定
```

---

## Exp 6c Loss Components

### 架構圖

```
Clean Audio  →  Teacher Encoder  →  t_e [B,512,T]  →  (Frozen Quantizer)
                     ↓ (L4,L8)
                   t_inter

Noisy Audio  →  Student Encoder  →  z_e [B,512,T]
                     ↓ (L4,L8)           ↓
                   s_inter         ┌─────────────┐
                                   │     RVQ     │
                                   │  4 Layers   │
                                   │  EMA Update │
                                   └─────────────┘
                                         ↓
                                    z_q [B,512,T]
                                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Loss Components                           │
├─────────────────────────────────────────────────────────────────┤
│ L_quant  = MSE(z_q, t_e.detach())        [Post-quant] ✅       │
│ L_pre    = MSE(z_e, t_e.detach())        [Pre-quant]  ❌ (禁用)│
│ L_inter  = MSE(s_L4, t_L4) + MSE(s_L8, t_L8)  [Multi-scale]   │
│ L_commit = MSE(z_e, z_q.detach())        [Encoder commit] ✅   │
│ L_codebook = (EMA update, no grad)       [Codebook] (自動)     │
│                                                                 │
│ Total = 1.0·L_quant + 0.0·L_pre + 0.5·L_inter + 1.0·L_commit  │
└─────────────────────────────────────────────────────────────────┘
```

### 1. **L_quant** (Post-Quantization MSE) - λ=1.0 ✅

```python
# 定義
L_quant = masked_mse(z_q, t_e.detach(), lengths)

# 實現 (train_rvq_short_run.py:73-92)
def masked_mse(student, teacher, lengths):
    """Masked MSE over valid frames."""
    if lengths is None:
        return F.mse_loss(student, teacher)

    # Convert audio sample lengths to frame lengths
    hop = 320  # 24kHz / frame_rate=75
    T = student.shape[-1]
    frame_lens = (lengths + hop - 1) // hop  # ceil
    frame_lens = torch.clamp(frame_lens, min=0, max=T)

    # Create frame mask [B, T]
    frame_idx = torch.arange(T, device=student.device).unsqueeze(0)  # [1, T]
    mask = frame_idx < frame_lens.unsqueeze(1)  # [B, T]
    mask = mask.unsqueeze(1).to(student.dtype)  # [B, 1, T]

    # Apply mask
    sq = (student - teacher) ** 2
    sq = sq * mask

    denom = mask.sum() * student.shape[1]
    return sq.sum() / denom.clamp(min=1.0)

# 訓練循環調用 (train_rvq_short_run.py:1149-1154)
loss_quant = masked_mse(
    student=output['student_quantized'],  # z_q [B, 512, T]
    teacher=output['teacher_encoder_out'],  # t_e [B, 512, T]
    lengths=lengths,
)

# 目的 ✅
- 對齊 QUANTIZED output 與 teacher encoder
- 確保 quantizer 不能被繞過
- 這是 H1 的核心修復
```

### 2. **L_pre** (Pre-Quantization MSE) - λ=0.0 ❌

```python
# 定義 (完全禁用!)
L_pre = masked_mse(z_e, t_e.detach(), lengths)

# 訓練循環調用 (train_rvq_short_run.py:1155-1159)
loss_pre = masked_mse(
    student=output['student_encoder_out'],  # z_e [B, 512, T]
    teacher=output['teacher_encoder_out'],  # t_e [B, 512, T]
    lengths=lengths,
)

# 權重
λ_pre = 0.0  # 完全禁用!

# 為什麼禁用 ✅
- Phase 3 失敗的根本原因: 只優化 z_e → quantizer 被繞過
- Exp 6c 完全依賴 L_quant (post-quant alignment)
- 不再需要 pre-quant alignment
```

### 3. **L_inter** (Intermediate Supervision) - λ=0.5

```python
# 定義 (IntermediateSupervisionLossV6)
# 與 baseline 相同的實現，但監督層不同

# 監督層 (Exp 6c)
intermediate_indices = [3, 6]  # L4 (model[3]), L8 (model[6])

# Layer weights (args.lambda_inter)
layer_weights = {
    3: 0.5,   # L4 (Downsample 1)
    6: 0.5,   # L8 (Downsample 2)
}

# 訓練循環調用 (train_rvq_short_run.py:1161-1166)
loss_inter_raw, _ = inter_loss_fn(
    student_features=output['student_intermediates'],
    teacher_features=output['teacher_intermediates'],
)

# Warmup (optional)
loss_inter = loss_inter_raw if step >= inter_warmup_steps else (loss_inter_raw * 0.0)

# 權重
λ_inter = 0.5

# 目的
- 與 baseline 類似: 多尺度監督
- 但只監督 L4, L8 (vs baseline 的 L3, L4, L6)
- 降低 shortcut learning 風險
```

### 4. **L_commit** (Encoder Commitment) - β=1.0 ✅

```python
# 定義 (RVQ forward pass)
L_commit = Σ_layers MSE(r_i, q_i.detach())

# 實現 (models_rvq.py:217)
# 在 RVQ forward 中計算
loss_commit = 0

for layer_idx, codebook in enumerate(self.codebooks):
    # ... (quantization logic)

    # Encoder commitment (updates encoder; codebook is detached)
    loss_commit += F.mse_loss(residual, q.detach())

    # ...

return {
    'loss_commit': loss_commit,  # Σ over all RVQ layers
    ...
}

# 訓練循環調用 (train_rvq_short_run.py:1169)
loss_commit = output['rvq_loss_commit']

# 權重
β_commit = 1.0

# 目的 ✅
- 推動 encoder output 靠近 quantized vectors
- 雙向 commitment:
  - L_commit: encoder → quantized (更新 encoder)
  - L_codebook: quantized → encoder (EMA 更新 codebook)
- 這是 H2 的核心修復
```

### 5. **L_codebook** (Codebook Update via EMA)

```python
# 定義 (EMA mode: no gradient-based loss)
# 在 gradient mode 下: L_codebook = Σ_layers MSE(r_i.detach(), q_i)
# 在 EMA mode 下: 通過 EMA update 自動更新 codebook

# 實現 (models_rvq.py:108-147)
@torch.no_grad()
def _ema_update_layer(layer_idx, residual_flat, indices):
    # 1. Count code usage
    counts = torch.bincount(indices, minlength=K)  # [K]

    # 2. Sum embeddings
    embed_sum = torch.zeros(K, dim, device=device)
    embed_sum.index_add_(0, indices, residual_flat)

    # 3. EMA update
    ema_cluster_size[layer_idx] *= decay  # decay=0.99
    ema_cluster_size[layer_idx] += (1 - decay) * counts

    ema_embed_avg[layer_idx] *= decay
    ema_embed_avg[layer_idx] += (1 - decay) * embed_sum

    # 4. Laplace smoothing & normalization
    n = ema_cluster_size[layer_idx].sum()
    cluster_size = (ema_cluster_size + eps) / (n + K*eps) * n
    embed = ema_embed_avg / cluster_size.unsqueeze(1).clamp(min=1e-12)

    # 5. Update codebook weights
    codebook.weight.data = embed

    # 6. Dead-code reset (threshold=2)
    dead = ema_cluster_size < 2.0
    if dead.any():
        codebook[dead] = random_sample(residual_flat)
        ema_cluster_size[dead] = 1.0
        ema_embed_avg[dead] = codebook[dead]

# 調用 (models_rvq.py:226)
if self.training:
    self._ema_update_layer(layer_idx, residual_flat.detach(), indices)

# 目的 ✅
- 穩定的 codebook 更新 (vs gradient-based)
- Dead-code reset 自動恢復未使用 codes
- 不依賴 gradient, 不受 optimizer 影響
```

### Exp 6c Total Loss

```python
# 訓練循環 (train_rvq_short_run.py:1173-1179)
total_loss = (
    args.lambda_quant * loss_quant +
    args.lambda_pre * loss_pre +
    args.lambda_inter * loss_inter +
    args.beta_commit * loss_commit +
    args.lambda_codebook * loss_codebook
) / args.grad_accum

# 實際權重 (Exp 6c 最佳配置)
lambda_quant = 1.0
lambda_pre = 0.0      # 禁用!
lambda_inter = 0.5
beta_commit = 1.0
lambda_codebook = N/A  # EMA mode: no gradient

# 完整展開
total_loss =
    1.0 * MSE(z_q, t_e) +                    # Post-quant alignment
    0.0 * MSE(z_e, t_e) +                    # (禁用)
    0.5 * [0.5*MSE(s_L4,t_L4) + 0.5*MSE(s_L8,t_L8)] +  # Intermediate
    1.0 * Σ_layers MSE(r_i, q_i.detach())   # Encoder commitment

# Quantizer
- Residual VQ (4 layers, K=2048 each)
- EMA update (automatic, no gradient)
- Dead-code reset (automatic)
```

---

## 關鍵差異分析

### 對比表

| 特性 | Baseline | Exp 6c | 影響 |
|------|----------|--------|------|
| **主對齊目標** | `MSE(z_e, t_e)` (Pre-quant) | `MSE(z_q, t_e)` (Post-quant) | **防止繞過** ✅ |
| **Pre-quant Loss** | ✅ weight=1.0 | ❌ weight=0.0 (禁用) | 專注 post-quant |
| **Post-quant Loss** | ❌ 無 | ✅ weight=1.0 | H1 核心修復 |
| **Commitment** | ❌ 無 | ✅ Bidirectional (β=1.0) | H2 核心修復 |
| **Triplet Loss** | ✅ weight=0.5 | ❌ 無 | 不依賴 frozen CB |
| **Intermediate** | L3,L4,L6 (Cosine) | L4,L8 (MSE) | 降低 shortcut |
| **Inter. 權重** | 0.5 → 0.25 (衰減) | 0.5 (固定) | 更簡單 |
| **Quantizer** | Frozen VQ (K=4096) | EMA RVQ (4×2048) | 穩定更新 |
| **Dead-code** | ❌ 無 | ✅ Auto reset (th=2) | 持續多樣性 |

### Loss Term 對比

```
Baseline:
┌─────────────────────────────────────────────────────┐
│ 1.0 × MSE(z_e, t_e)         [Pre-quant]  ← 主要    │
│ 0.5 × Triplet(z_e, CB)      [VQ align]   ← 輔助    │
│ 0.5 × Σ Cos(s_Li, t_Li)     [L3,L4,L6]  ← 多尺度   │
└─────────────────────────────────────────────────────┘
  ⚠️ 問題: 只監督 z_e, quantizer 可被繞過

Exp 6c:
┌─────────────────────────────────────────────────────┐
│ 1.0 × MSE(z_q, t_e)         [Post-quant] ✅ 主要    │
│ 0.0 × MSE(z_e, t_e)         [Pre-quant]  ❌ 禁用    │
│ 0.5 × Σ MSE(s_Li, t_Li)     [L4,L8]     ← 多尺度   │
│ 1.0 × MSE(z_e, z_q.detach)[Commitment]  ✅ 雙向    │
└─────────────────────────────────────────────────────┘
  ✅ 優勢: 強制通過 quantizer, 雙向 commitment
```

### 訓練目標對比

```
Baseline Training Objective:
    Minimize: ||z_e - t_e||²

    Student encoder output (z_e) → 對齊 → Teacher encoder output (t_e)

    ⚠️ Quantizer 不在優化路徑上 → 可被繞過
    ⚠️ Codebook frozen → 無法適應 student encoder
    ⚠️ Token collapse 無法避免

Exp 6c Training Objective:
    Minimize: ||z_q - t_e||² + β||z_e - z_q||²

    Student quantized output (z_q) → 對齊 → Teacher encoder output (t_e)
    Student encoder output (z_e) → 對齊 → Student quantized (z_q)

    ✅ Quantizer 在優化路徑上 → 必須參與
    ✅ Codebook EMA 更新 → 自動適應
    ✅ Bidirectional commitment → 雙向約束
    ✅ Token diversity 自然維持
```

---

## 為什麼 Exp 6c 的 Loss 設計更有效

### 1. **Post-Quant Alignment 強制通過 Quantizer**

```
Baseline 問題:
    Loss = MSE(z_e, t_e)

    Gradient 流向:
    t_e ← z_e ← encoder
           ↓ (沒有梯度流向 quantizer!)
          z_q (quantizer 被繞過)

    結果:
    - Encoder 學會繞過 quantizer
    - z_e 接近 t_e, 但 z_q 可能很差
    - Token collapse 發生

Exp 6c 解決:
    Loss = MSE(z_q, t_e) + β·MSE(z_e, z_q.detach())

    Gradient 流向:
    t_e ← z_q ← RVQ ← z_e ← encoder
          ↑          ↑
        L_quant   L_commit

    結果:
    - Encoder 必須通過 quantizer
    - z_q 被強制接近 t_e
    - L_commit 推動 z_e 接近 z_q
    - Token diversity 自然維持
```

### 2. **Bidirectional Commitment 雙向約束**

```
Baseline:
    - 無 commitment loss
    - Encoder 和 codebook 沒有雙向約束
    - Codebook frozen → encoder 必須完全適應固定 codebook
    - 容易 collapse 到少數 codes

Exp 6c:
    - L_commit = MSE(z_e, z_q.detach()): encoder → quantized
    - L_codebook = EMA update: quantized → codebook

    雙向約束:
    ┌─────────┐                 ┌──────────┐
    │ Encoder │ ──L_commit────→ │ Codebook │
    │   z_e   │ ←──L_codebook── │   z_q    │
    └─────────┘                 └──────────┘

    - Encoder 推動 codebook 更新
    - Codebook 推動 encoder 對齊
    - 平衡狀態 → diverse codes
```

### 3. **EMA Update 穩定性**

```
Baseline:
    - Codebook frozen (eval mode)
    - 無法適應 student encoder 的變化
    - Dead codes 無法恢復

Exp 6c:
    - EMA update (decay=0.99): 穩定、平滑
    - Dead-code reset (th=2): 自動恢復
    - 不依賴 optimizer, 不受 learning rate 影響

    EMA 優勢:
    - 更穩定 (vs gradient update)
    - 自動 dead-code 處理
    - 不受 batch size / learning rate 影響
```

### 4. **RVQ 架構強制多樣性**

```
Baseline:
    - Single VQ (K=4096)
    - 一次性量化 → 容易 collapse

Exp 6c:
    - Residual VQ (4 layers, K=2048 each)
    - 多層殘差量化 → 強制分散

    Layer 0: 粗粒度 (主要特徵)
    Layer 1: 細粒度 (殘差1)
    Layer 2: 更細粒度 (殘差2)
    Layer 3: 最細粒度 (殘差3)

    - 每層負責不同 granularity
    - 總表達力: 2048^4 >> 4096
    - 自然分散到多層
```

### 5. **Loss 設計簡潔性**

```
Baseline:
    - 3 個 loss terms: Feature + Triplet + Intermediate
    - Triplet loss 依賴 frozen codebook (可能有 collapse)
    - Intermediate 動態權重衰減 (複雜)

Exp 6c:
    - 2 個主要 loss terms: Quant + Commit
    - 1 個輔助 loss: Intermediate (固定權重)
    - 更簡潔、更直接
    - 不依賴 external codebook
```

---

## 實驗驗證

### Baseline Results (exp_k_v6, epoch 300)

```
Entropy: 6.07 bits
Top-10 mass: 19.7%
Used codes: 740/4096 (18%)
Feature MSE: ~0.05 (好)

問題:
- Token diversity 不足
- Top-10 codes 占 19.7%
- 82% codebook 未使用
```

### Exp 6c Results (step 1000, K=2048×4, EMA)

```
Entropy: 9.03 bits (+49%)
Top-10 mass: 15.8% (-20%)
Used codes: 1089/2048 (53%, +35pp)
Feature MSE: 0.034 (-32%)

優勢:
- Token diversity 大幅提升
- Top-10 mass 下降
- Codebook 使用率翻倍
- Feature alignment 更好
```

### Loss 設計的直接影響

| Metric | Baseline | Exp 6c | 改善原因 |
|--------|----------|--------|----------|
| **Entropy** | 6.07 | 9.03 | Post-quant + RVQ |
| **Top-10** | 19.7% | 15.8% | Bidirectional commitment |
| **Used** | 18% | 53% | EMA + Dead-code reset |
| **MSE** | 0.05 | 0.034 | Post-quant alignment |

---

## 總結

### Baseline Loss 的根本問題

1. ❌ **Pre-quant only**: 只監督 `z_e`, quantizer 被繞過
2. ❌ **No commitment**: encoder 和 codebook 無雙向約束
3. ❌ **Frozen codebook**: 無法適應, dead codes 無法恢復
4. ❌ **Triplet on frozen CB**: 依賴可能已 collapse 的 codebook

### Exp 6c Loss 的核心優勢

1. ✅ **Post-quant alignment**: 強制通過 quantizer (`MSE(z_q, t_e)`)
2. ✅ **Bidirectional commitment**: 雙向約束 (`MSE(z_e, z_q)` + EMA)
3. ✅ **EMA update**: 穩定、自動 dead-code reset
4. ✅ **RVQ architecture**: 多層強制分散

### 關鍵設計原則

```
Baseline 設計哲學:
    "讓 encoder 學習接近 teacher encoder output"

    問題: Quantizer 不在優化路徑上

Exp 6c 設計哲學:
    "讓 quantized output 接近 teacher encoder output"

    優勢: Quantizer 必須參與, token diversity 自然維持
```

---

**創建日期**: 2026-02-06
**參考文檔**:
- [EXP6C_ARCHITECTURE.md](EXP6C_ARCHITECTURE.md)
- [SUMMARY.md](SUMMARY.md)
- [Baseline: exp_0112_intermediate/train_v6.py](../../exp_0112_intermediate/train_v6.py)
- [Exp 6c: train_rvq_short_run.py](../phase3/residual_vq/train_rvq_short_run.py)
