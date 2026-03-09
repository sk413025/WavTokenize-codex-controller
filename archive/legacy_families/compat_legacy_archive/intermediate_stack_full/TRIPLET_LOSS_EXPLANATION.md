# Baseline Triplet Loss 计算方式详解

## 📍 源代码位置

- 文件：[exp_1219/losses.py:136-171](../exp_1219/losses.py)
- 类名：`MaskedTripletLoss`

---

## 🎯 核心目标

**Triplet Loss 的目的**：让 student encoder 输出的特征 `z` 更接近 teacher 分配的 codebook vector（positive），远离其他 codebook vectors（negative）。

**核心公式**：
```
Loss = ReLU(d_pos - d_neg + margin)
```

其中：
- `d_pos`：student 特征到 teacher 指定 code 的距离（希望小）
- `d_neg`：student 特征到最近的其他 code 的距离（希望大）
- `margin`：安全边界（默认 0.2）

---

## 🔢 Shape 和 Dimension 详解

### 输入 Shapes

| 参数 | Shape | 说明 |
|------|-------|------|
| `student_features` | **(B, D, T)** | Student encoder 输出<br>B=batch_size, D=feature_dim (512), T=time_frames |
| `teacher_codes` | **(B, T)** 或 **(1, B, T)** | Teacher VQ 分配的 code indices |
| `codebook` | **(K, D)** | VQ Codebook 矩阵<br>K=codebook_size (4096), D=feature_dim (512) |
| `lengths` | **(B,)** | 每个样本的有效 audio samples 数 |

### 中间计算 Shapes

#### Step 1: Reshape student features

```python
z = student_features.permute(0, 2, 1).reshape(-1, D)
# (B, D, T) → (B, T, D) → (B*T, D)
```

**Shape 变化示例**（B=4, T=100, D=512）：
```
student_features: (4, 512, 100)
    ↓ permute(0, 2, 1)
(4, 100, 512)
    ↓ reshape(-1, 512)
z: (400, 512)  # B*T = 4*100 = 400
```

#### Step 2: 计算距离矩阵

```python
dists = torch.cdist(z, codebook, p=2)
# (B*T, D) × (K, D) → (B*T, K)
```

**Shape 示例**：
```
z: (400, 512)
codebook: (4096, 512)
    ↓ cdist (L2 distance)
dists: (400, 4096)
# 每个 frame 到所有 codebook entries 的距离
```

**含义**：`dists[i, j]` = frame `i` 到 codebook entry `j` 的 L2 距离

#### Step 3: 提取 Positive 距离

```python
teacher_flat = teacher_codes.reshape(-1)  # (B*T,)
batch_indices = torch.arange(len(teacher_flat))  # (B*T,)
pos_dist = dists[batch_indices, teacher_flat]  # (B*T,)
```

**含义**：对于每个 frame，取 teacher 指定的 code 对应的距离

**示例**：
```
teacher_flat = [12, 345, 67, ..., 89]  # (400,) 每个 frame 的 teacher code
batch_indices = [0, 1, 2, ..., 399]

pos_dist[0] = dists[0, 12]     # frame 0 → code 12 的距离
pos_dist[1] = dists[1, 345]    # frame 1 → code 345 的距离
pos_dist[2] = dists[2, 67]     # frame 2 → code 67 的距离
...
```

#### Step 4: 找到 Hard Negative 距离

```python
dists_for_neg = dists.clone()
dists_for_neg[batch_indices, teacher_flat] = float('inf')  # 排除 positive
neg_dist = dists_for_neg.min(dim=1).values  # (B*T,)
```

**含义**：对于每个 frame，找到除了 teacher code 之外最近的 code（hard negative）

**示例**：
```
frame 0:
  dists[0, :] = [2.3, 1.8, 3.1, ..., 1.2, ..., 2.9]
  teacher_code = 12
  dists_for_neg[0, 12] = inf
  neg_dist[0] = min(dists_for_neg[0, :]) = 1.2  # 最近的非 teacher code
```

#### Step 5: 计算 Triplet Loss

```python
pos_dist = pos_dist.reshape(B, T)  # (B, T)
neg_dist = neg_dist.reshape(B, T)  # (B, T)

triplet = F.relu(pos_dist - neg_dist + margin)  # (B, T)
```

**含义**：
- `pos_dist - neg_dist`：正样本距离 - 负样本距离
- 如果 `pos_dist + margin < neg_dist`，loss = 0（已满足 margin 约束）
- 否则 loss = `pos_dist - neg_dist + margin`（需要继续优化）

#### Step 6: 应用 Mask 并归一化

```python
mask = create_length_mask(...)  # (B, T) - 标记有效 frames
masked_triplet = triplet * mask  # (B, T)
loss = masked_triplet.sum() / (mask.sum() + 1e-8)  # scalar
```

**含义**：只计算有效 frames 的 loss，忽略 padding 部分

---

## 📊 完整计算流程图

```
Input:
  student_features: (B=4, D=512, T=100)
  teacher_codes:    (B=4, T=100)
  codebook:         (K=4096, D=512)
  lengths:          [32000, 24000, 16000, 8000]  # audio samples

┌──────────────────────────────────────────────────┐
│ Step 1: Reshape student features                │
│   (B, D, T) → (B, T, D) → (B*T, D)              │
│   (4, 512, 100) → (400, 512)                    │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Step 2: Compute distance matrix                 │
│   dists = cdist(z, codebook, p=2)               │
│   (400, 512) × (4096, 512) → (400, 4096)        │
│   每个 frame 到所有 codebook entries 的 L2 距离   │
└──────────────────────────────────────────────────┘
                    ↓
       ┌────────────┴────────────┐
       ↓                         ↓
┌──────────────────┐      ┌──────────────────┐
│ Step 3: Positive │      │ Step 4: Negative │
│                  │      │                  │
│ pos_dist[i] =    │      │ 1. Mask out      │
│  dists[i,        │      │    positive      │
│    teacher[i]]   │      │ 2. neg_dist[i] = │
│                  │      │    min(dists[i]) │
│ Shape: (400,)    │      │ Shape: (400,)    │
└──────────────────┘      └──────────────────┘
       └────────────┬────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Step 5: Compute triplet loss                    │
│   triplet = ReLU(pos_dist - neg_dist + 0.2)    │
│   Reshape to (B, T) = (4, 100)                  │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Step 6: Apply mask & normalize                  │
│   mask: (4, 100) [1=valid, 0=padding]           │
│   masked_triplet = triplet * mask               │
│   loss = sum(masked_triplet) / sum(mask)        │
│   Output: scalar                                │
└──────────────────────────────────────────────────┘
```

---

## 💡 直观理解

### 类比：停车场找车位

想象你要把车（student feature）停到离目标车位（teacher code）最近：

1. **Positive distance (`pos_dist`)**：你的车到目标车位的距离
2. **Negative distance (`neg_dist`)**：你的车到最近的其他车位的距离
3. **Triplet loss 要求**：目标车位要比其他车位近至少 `margin`

**理想情况**：
```
pos_dist = 0.5  (很近目标)
neg_dist = 2.0  (远离其他)
loss = ReLU(0.5 - 2.0 + 0.2) = ReLU(-1.3) = 0 ✅
```

**需要优化**：
```
pos_dist = 1.8  (离目标太远)
neg_dist = 1.5  (离其他太近)
loss = ReLU(1.8 - 1.5 + 0.2) = 0.5 ❌ (需要继续训练)
```

---

## 🔍 关键参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `margin` | 0.2 | 安全边界，要求 positive 比 negative 近至少 margin |
| `encoder_stride` | 320 | 用于计算 frame-level mask（320 samples = 1 frame） |
| `p` in `cdist` | 2 | L2 距离（欧式距离） |

---

## 📈 训练效果

在 baseline (exp_k v6 @ epoch 300) 中：
- **Triplet weight**: 1.0
- **Margin**: 0.2
- **效果**：帮助 student encoder 学习到与 teacher codebook 对齐的特征空间

---

## 🔗 相关 Loss

在 `MaskedCombinedLossV2` 中，Triplet Loss 与其他 loss 组合：

```python
total_loss = (
    feature_weight * feature_loss +      # MSE on encoder features
    cosine_weight * cosine_loss +        # Cosine similarity
    triplet_weight * triplet_loss +      # Triplet (this one!)
    ce_weight * ce_loss                  # Cross-entropy on logits
)
```

**Baseline 配置**：
- `feature_weight = 1.0`
- `triplet_weight = 1.0`
- `cosine_weight = 0` (不使用)
- `ce_weight = 0` (不使用)

---

## 📝 代码摘要

```python
class MaskedTripletLoss(nn.Module):
    def __init__(self, margin=0.2, encoder_stride=320):
        self.margin = margin
        self.encoder_stride = encoder_stride

    def forward(self, student_features, teacher_codes, codebook, lengths):
        B, D, T = student_features.shape

        # 1. Reshape: (B, D, T) → (B*T, D)
        z = student_features.permute(0, 2, 1).reshape(-1, D)

        # 2. Distance matrix: (B*T, D) × (K, D) → (B*T, K)
        dists = torch.cdist(z, codebook, p=2)

        # 3. Positive distance: (B*T,)
        teacher_flat = teacher_codes.reshape(-1)
        batch_indices = torch.arange(len(teacher_flat))
        pos_dist = dists[batch_indices, teacher_flat]

        # 4. Hard negative distance: (B*T,)
        dists_for_neg = dists.clone()
        dists_for_neg[batch_indices, teacher_flat] = float('inf')
        neg_dist = dists_for_neg.min(dim=1).values

        # 5. Triplet loss: (B, T)
        pos_dist = pos_dist.reshape(B, T)
        neg_dist = neg_dist.reshape(B, T)
        triplet = F.relu(pos_dist - neg_dist + self.margin)

        # 6. Masked average
        mask = create_length_mask(lengths, ...)
        masked_triplet = triplet * mask
        loss = masked_triplet.sum() / (mask.sum() + 1e-8)

        return loss
```

---

**文档创建时间**: 2026-02-04
**Baseline 版本**: exp_k v6 @ epoch 300
**源代码**: [exp_1219/losses.py](../exp_1219/losses.py)
