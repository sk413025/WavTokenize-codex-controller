# Phase 3-2 Exp 6c: EMA+RVQ 架構詳解

**日期**: 2026-02-06
**狀態**: ✅ P2 驗收通過

---

## 目錄

1. [核心問題與假設](#核心問題與假設)
2. [Exp 6c 架構詳解](#exp-6c-架構詳解)
3. [與 Baseline 差異比較](#與-baseline-差異比較)
4. [為什麼凍結 Decoder 有效](#為什麼凍結-decoder-有效)
5. [實驗結果](#實驗結果)
6. [代碼位置](#代碼位置)

---

## 核心問題與假設

### Phase 3 失敗揭示的根本問題

**訓練目標允許 quantizer 被繞過** → Token collapse 照樣發生

### 三大假設 (H1-H3)

1. **H1**: 主 loss 只對齊 pre-quantization (`z_e`) → quantizer 可被繞過 → collapse 照樣發生
2. **H2**: 原本的 commitment loss 只推動 codebook，缺少 encoder commitment 和 EMA 穩定機制
3. **H3**: Intermediate supervision 過強 → encoder 走向「少量 codes 即可擬合」的捷徑

---

## Exp 6c 架構詳解

### 架構圖

```
                          Teacher (Frozen)                    Student (LoRA)
                                 ↓                                    ↓
                          Clean Audio                           Noisy Audio
                                 ↓                                    ↓
┌────────────────────────────────────────────────────────────────────────────┐
│                             Encoder (18 Layers)                            │
│                                                                            │
│  L0-L3: Conv blocks                     L0-L3: Conv blocks + LoRA         │
│     ↓                                       ↓                              │
│  L4: Downsample  ←──────────────────────→  L4 + LoRA                      │
│     ↓          (Intermediate Loss 1)        ↓                              │
│  L5-L7: ResBlocks                       L5-L7: ResBlocks + LoRA           │
│     ↓                                       ↓                              │
│  L8: Downsample  ←──────────────────────→  L8 + LoRA                      │
│     ↓          (Intermediate Loss 2)        ↓                              │
│  L9-L17: ResBlocks                      L9-L17: ResBlocks + LoRA          │
│     ↓                                       ↓                              │
│  t_e [B,512,T]                          z_e [B,512,T]                     │
└────────────────────────────────────────────────────────────────────────────┘
                                                 ↓
                   ┌─────────────────────────────────────────────┐
                   │  Residual Vector Quantizer (RVQ)           │
                   │                                             │
                   │  Layer 0: Quantize z_e → q0, r0 = z_e - q0 │
                   │  Layer 1: Quantize r0  → q1, r1 = r0 - q1  │
                   │  Layer 2: Quantize r1  → q2, r2 = r1 - q2  │
                   │  Layer 3: Quantize r2  → q3                │
                   │                                             │
                   │  z_q = q0 + q1 + q2 + q3                   │
                   │                                             │
                   │  Update: EMA (decay=0.99)                  │
                   │  Dead-code reset: threshold=2              │
                   │  Usage penalty: optional (0.0~0.2)         │
                   └─────────────────────────────────────────────┘
                                     ↓
                                  z_q [B,512,T]
                                     ↓
                   ┌─────────────────────────────────────────────┐
                   │          Decoder (Frozen)                   │
                   │                                             │
                   │  (用於推理時重建音頻，訓練時不參與)           │
                   └─────────────────────────────────────────────┘

Loss Components:
┌────────────────────────────────────────────────────────────────────────────┐
│ L_quant      = MSE(z_q, t_e)           [Post-quant alignment]  λ=1.0      │
│ L_pre        = MSE(z_e, t_e)           [Pre-quant alignment]   λ=0.0      │
│ L_inter      = MSE(s_L4, t_L4) +       [Intermediate sup.]     λ=0.5      │
│                MSE(s_L8, t_L8)                                             │
│ L_commit     = MSE(z_e, z_q.detach())  [Encoder commitment]    β=1.0      │
│ L_codebook   = (EMA update, no grad)   [Codebook update]       N/A        │
│                                                                            │
│ Total = λ_quant·L_quant + λ_pre·L_pre + λ_inter·L_inter + β·L_commit     │
└────────────────────────────────────────────────────────────────────────────┘
```

### RVQ (Residual Vector Quantizer) 詳解

#### 原理

```python
# 多層殘差量化
residual = z_e  # [B, 512, T]
z_q = 0

for layer_idx in range(4):  # 4 layers
    # 1. 找最近的 codebook entry
    distances = ||residual - codebook||²

    # 2. (Optional) Usage penalty: 懲罰熱門 codes
    if ema_usage_penalty > 0:
        penalty = log(ema_cluster_size) * ema_usage_penalty
        distances += penalty

    # 3. 量化
    indices = argmin(distances)
    q = codebook[indices]

    # 4. Straight-through estimator
    q = residual + (q - residual).detach()

    # 5. 累積 & 計算新殘差
    z_q += q
    residual = residual - q.detach()

    # 6. EMA update (training only)
    if training:
        ema_update(layer_idx, residual, indices)
        dead_code_reset(layer_idx, threshold=2)
```

#### EMA Update (Exponential Moving Average)

```python
def _ema_update_layer(layer_idx, residual_flat, indices):
    # 1. 統計使用次數
    counts = bincount(indices, minlength=K)  # [K]

    # 2. 累積 embedding 總和
    embed_sum = zeros(K, dim)
    embed_sum.index_add_(0, indices, residual_flat)

    # 3. EMA 更新
    ema_cluster_size[layer_idx] *= decay
    ema_cluster_size[layer_idx] += (1 - decay) * counts

    ema_embed_avg[layer_idx] *= decay
    ema_embed_avg[layer_idx] += (1 - decay) * embed_sum

    # 4. Laplace smoothing & normalization
    n = ema_cluster_size[layer_idx].sum()
    cluster_size = (ema_cluster_size + eps) / (n + K*eps) * n
    embed = ema_embed_avg / cluster_size.unsqueeze(1)

    # 5. 更新 codebook weights
    codebook.weight.data = embed

    # 6. Dead-code reset (threshold=2)
    dead = ema_cluster_size < 2.0
    if dead.any():
        # 從當前 batch 隨機採樣替換
        codebook[dead] = random_sample(residual_flat)
        ema_cluster_size[dead] = 1.0
        ema_embed_avg[dead] = codebook[dead]
```

#### Usage Penalty (Optional)

```python
# 目的: 抑制 top-10 mass 漂移
# 原理: 懲罰高頻 codes，鼓勵使用低頻 codes

# 在 distance 計算時加入 penalty
penalty = log(ema_cluster_size.clamp(min=1.0)) * ema_usage_penalty
distances += penalty

# Schedule: 逐步增加 penalty
if step < start_step:
    penalty_weight = 0.0
elif step < start_step + ramp_steps:
    t = (step - start_step) / ramp_steps
    penalty_weight = max_penalty * t
else:
    penalty_weight = max_penalty
```

### Loss Components 詳解

#### 1. **L_quant** (Post-Quantization Alignment) - λ=1.0

```python
# 主要對齊目標: quantized features 對齊 teacher encoder output
L_quant = MSE(z_q, t_e)

# 為什麼重要:
# - 確保 quantizer 不能被繞過
# - 強制 quantized output 必須接近 teacher
# - 這是 H1 的核心修復
```

#### 2. **L_pre** (Pre-Quantization Alignment) - λ=0.0

```python
# Pre-quant 對齊 (Phase 3-2 中禁用)
L_pre = MSE(z_e, t_e)

# 為什麼設為 0:
# - Phase 3 失敗的原因: 只對齊 z_e → quantizer 被繞過
# - Exp 6c 完全依賴 L_quant，不使用 L_pre
```

#### 3. **L_inter** (Intermediate Supervision) - λ=0.5

```python
# 中間層監督
L_inter = 0.5 * MSE(student_L4, teacher_L4) +
          0.5 * MSE(student_L8, teacher_L8)

# 為什麼保留:
# - 幫助 encoder 早期層學習
# - λ=0.5 (vs 原本 1.0) 降低 shortcut 風險
# - 與 H3 相關
```

#### 4. **L_commit** (Encoder Commitment) - β=1.0

```python
# Encoder commitment loss
L_commit = MSE(z_e, z_q.detach())

# 為什麼重要:
# - 推動 encoder output 靠近 quantized vectors
# - 雙向 commitment (vs 原本單向 codebook loss)
# - 這是 H2 的核心修復
```

#### 5. **L_codebook** (Codebook Update)

```python
# Gradient mode (Exp 6a/6b):
L_codebook = MSE(z_e.detach(), q)  # 通過梯度更新 codebook

# EMA mode (Exp 6c): ✅
L_codebook = N/A  # 通過 EMA 更新，不用梯度
```

---

## 與 Baseline 差異比較

### Baseline (exp_0112_intermediate)

```
Architecture:
- Single VQ (K=4096, single layer)
- Gradient-based codebook update
- Codebook frozen (quantizer.eval())

Loss:
L_total = MSE(z_e, t_e)              # Pre-quant alignment only
        + λ_inter * L_inter          # Intermediate supervision
        + β * MSE(z_e.detach(), q)   # Single-direction commitment

Quantizer:
- Single layer VQ
- No EMA, no dead-code reset
- Codebook 完全凍結 (eval mode)
```

### Exp 6c (EMA+RVQ)

```
Architecture:
- Residual VQ (K=2048, 4 layers)  ← 多層殘差量化
- EMA-based codebook update       ← 穩定更新機制
- Dead-code reset (threshold=2)   ← 自動恢復未使用 codes

Loss:
L_total = MSE(z_q, t_e)              # Post-quant alignment ← 關鍵修復
        + λ_inter * L_inter          # Intermediate supervision
        + β * MSE(z_e, z_q.detach()) # Bidirectional commitment ← 雙向

Quantizer:
- 4-layer RVQ (hierarchical)
- EMA update (decay=0.99)
- Dead-code reset (automatic recovery)
- Optional usage penalty (anti-drift)
```

### 關鍵差異對比表

| 特性 | Baseline | Exp 6c | 影響 |
|------|----------|--------|------|
| **Quantizer 架構** | Single VQ (1 layer) | Residual VQ (4 layers) | 強制多樣性 |
| **Codebook 大小** | K=4096 | K=2048 (per layer) | 總表達力 2048^4 |
| **Codebook 更新** | Frozen (eval mode) | EMA update | 穩定學習 |
| **Dead-code 處理** | 無 | Auto reset (th=2) | 持續多樣性 |
| **主對齊目標** | Pre-quant (`z_e`) | Post-quant (`z_q`) | **防止繞過** ✅ |
| **Commitment** | Single direction | Bidirectional | 雙向約束 |
| **Usage penalty** | 無 | Optional (0~0.2) | 抑制漂移 |

---

## 為什麼凍結 Decoder 有效

### Decoder 在訓練中的角色

```python
# 訓練階段: Decoder 完全不參與
forward():
    teacher_encoder_out = teacher.encode(clean_audio)  # [B, 512, T]
    student_encoder_out = student.encode(noisy_audio)  # [B, 512, T]
    student_quantized = rvq(student_encoder_out)       # [B, 512, T]

    # Loss 計算: 只在 feature space
    loss = MSE(student_quantized, teacher_encoder_out)

    # Decoder 不被調用!

# 推理階段: Decoder 用於重建音頻
decode():
    audio = teacher.decode(student_quantized)  # [B, 1, T_audio]
```

### 為什麼可以凍結

#### 1. **Feature Space 對齊 ≠ Audio Space 對齊**

```
Training objective:
    Align student_quantized with teacher_encoder_out (512-dim features)

NOT:
    Align reconstructed_audio with clean_audio

Why:
    - WavTokenizer decoder 已經預訓練好
    - 只要 feature space 對齊，decoder 就能正確重建
    - 不需要重新訓練 decoder
```

#### 2. **Quantizer 是瓶頸，不是 Decoder**

```
Token collapse 的原因:
    ✅ Quantizer 輸出過於集中 (top-10 codes)
    ❌ 不是 Decoder 無法重建多樣 codes

Solution:
    ✅ 修復 Quantizer (EMA + RVQ)
    ❌ 不需要調整 Decoder
```

#### 3. **Decoder 凍結的好處**

| 優點 | 說明 |
|------|------|
| **穩定訓練** | Decoder 參數固定，減少優化空間 |
| **快速收斂** | 只需訓練 Encoder + RVQ |
| **可解釋性** | 變數只有 encoder，easier to debug |
| **記憶體效率** | 不需要存 decoder 的梯度 |
| **可重用性** | 任何 quantizer 都能配合原 decoder |

#### 4. **實驗證據**

```
Exp 6c Results (step 1000):
    feature_mse = 0.034      ← Feature space 對齊良好
    entropy = 9.03           ← Token diversity 優秀
    top10_mass = 15.8%       ← 沒有 collapse
    used_codes = 1089/2048   ← 高 codebook 使用率

Conclusion:
    在 feature space 對齊的情況下，凍結 decoder 是有效的
    Audio reconstruction quality 依賴預訓練 decoder 即可
```

### Decoder 凍結的限制

當需要調整 Decoder 時:

1. **新的音頻任務** (e.g., 不同 sample rate)
2. **顯著改變 feature space** (e.g., 換 backbone)
3. **需要 end-to-end 音質優化** (e.g., PESQ/STOI metrics)

在 **token diversity** 任務中，這些都不是問題。

---

## 實驗結果

### Exp 6c 最佳配置 (P2 通過)

**配置**: `K=2048, L=4, EMA decay=0.99, threshold=2, usage_penalty=0.1`

```
Step 1000 Metrics:
┌──────────────────────┬──────────────┬────────────┬────────────┐
│ Metric               │ Result       │ Target     │ Status     │
├──────────────────────┼──────────────┼────────────┼────────────┤
│ Layer0 Entropy       │ 9.03         │ ≥5.0       │ ✅ (+80%)  │
│ Layer0 Top-10 Mass   │ 15.8%        │ ≤50%       │ ✅ (-68%)  │
│ Layer0 Used Codes    │ 1089/2048    │ ≥205       │ ✅ (+432%) │
│ Joint Diversity      │ 99.2%        │ ≥30%       │ ✅ (+231%) │
│ Feature MSE          │ 0.034        │ ≤0.1       │ ✅ (-66%)  │
└──────────────────────┴──────────────┴────────────┴────────────┘

P2 驗收: ✅ PASS (所有指標通過)
```

### 與 Baseline 比較

```
Baseline (Single VQ, K=4096):
- Entropy: 6.07
- Top-10 mass: 19.7%
- Used codes: 740/4096 (18%)

Exp 6c (RVQ, K=2048 x 4):
- Entropy: 9.03 (+49%)
- Top-10 mass: 15.8% (-20%)
- Used codes: 1089/2048 (53%, +35 percentage points)

Improvement:
✅ 更高 entropy (更均勻分佈)
✅ 更低 top-10 mass (更少集中)
✅ 更高 usage rate (更多 codes 被使用)
✅ Feature MSE 維持 (主目標未受損)
```

---

## 代碼位置

### 核心實現

```
exp_0128/phase3/residual_vq/
├── models_rvq.py                    # RVQ + EMA 實現
│   ├── ResidualVectorQuantizer     # 多層 RVQ
│   │   ├── forward()               # 殘差量化流程
│   │   ├── _ema_update_layer()     # EMA 更新 + dead-code reset
│   │   └── get_codebook_usage()    # 使用統計
│   └── TeacherStudentRVQ            # 完整模型
│       ├── __init__()              # 替換 quantizer 為 RVQ
│       ├── forward()               # 前向傳播 (EMA mode)
│       └── decode()                # 使用 frozen decoder 重建
└── train_rvq_short_run.py           # 訓練腳本
    ├── main()                       # 訓練循環
    ├── evaluate_collapse_metrics()  # RVQ-specific metrics
    └── masked_mse()                 # Loss 計算
```

### 關鍵函數

#### 1. RVQ Forward ([models_rvq.py:148-261](models_rvq.py:148-261))

```python
def forward(self, z, frame_rate=75, bandwidth=0.075):
    """
    多層殘差量化

    Returns:
        quantized: z_q [B, dim, T]
        all_layer_codes: [n_layers, B, T]
        loss_commit: Encoder commitment
        loss_codebook: (EMA mode: no grad)
    """
```

#### 2. EMA Update ([models_rvq.py:107-147](models_rvq.py:107-147))

```python
@torch.no_grad()
def _ema_update_layer(self, layer_idx, residual_flat, indices):
    """
    EMA codebook update + dead-code reset

    1. EMA update (decay=0.99)
    2. Laplace smoothing
    3. Dead-code reset (threshold=2)
    """
```

#### 3. Training Loop ([train_rvq_short_run.py:1119-1226](train_rvq_short_run.py:1119-1226))

```python
# Loss 計算
loss_quant = masked_mse(z_q, t_e, lengths)       # λ=1.0
loss_pre = masked_mse(z_e, t_e, lengths)         # λ=0.0
loss_inter = inter_loss_fn(...)                  # λ=0.5
loss_commit = output['rvq_loss_commit']          # β=1.0

total_loss = (
    λ_quant * loss_quant +
    λ_pre * loss_pre +
    λ_inter * loss_inter +
    β * loss_commit
) / grad_accum
```

---

## 總結

### Exp 6c 成功的三大關鍵

1. **Post-Quant Alignment** (`L_quant = MSE(z_q, t_e)`)
   - 防止 quantizer 被繞過
   - 驗證 H1

2. **EMA + Dead-Code Reset**
   - 穩定的 codebook 更新
   - 自動恢復未使用 codes
   - 驗證 H2

3. **Residual VQ**
   - 多層架構強制多樣性
   - 每層負責不同 granularity

### 為什麼凍結 Decoder 有效

- ✅ Token diversity 問題在 **Quantizer**，不在 Decoder
- ✅ Feature space 對齊 → 預訓練 decoder 即可重建
- ✅ 減少訓練變數，更穩定收斂
- ✅ 記憶體效率高

### 建議後續工作

1. **Full Training** (300 epochs)
   - 驗證長期穩定性

2. **Audio Reconstruction**
   - 評估實際音質
   - 使用 PESQ/STOI metrics

3. **Usage Penalty Schedule**
   - 緩解 top-10 mass 漂移
   - 多 seed 驗證穩定性

---

**創建日期**: 2026-02-06
**參考文檔**:
- [Phase 3-2 Summary](SUMMARY.md)
- [Baseline Analysis](../baseline_token_analysis/PROGRESS.md)
