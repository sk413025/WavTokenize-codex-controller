# Baseline Triplet Loss 计算方式 - 简明版

## 🎯 核心目标

让 student encoder 输出的特征 **靠近** teacher 指定的 codebook vector，**远离** 其他 codebook vectors。

**公式**：`Loss = ReLU(d_pos - d_neg + margin)`

---

## 📊 Shape 流程图

```
输入:
  student_features: (B=4, D=512, T=100)  ← Student encoder 输出
  teacher_codes:    (B=4, T=100)         ← Teacher 分配的 code indices
  codebook:         (K=4096, D=512)      ← VQ Codebook 矩阵

Step 1: Reshape
  student_features: (4, 512, 100) → permute → (4, 100, 512) → reshape → z: (400, 512)
  ↓
Step 2: 计算距离矩阵
  z: (400, 512) × codebook: (4096, 512) → dists: (400, 4096)
  【含义】每个 frame (400 个) 到所有 codebook entries (4096 个) 的 L2 距离
  ↓
Step 3: 提取 Positive 距离
  pos_dist[i] = dists[i, teacher_codes[i]]  → (400,)
  【含义】每个 frame 到 teacher 指定 code 的距离
  ↓
Step 4: 找 Hard Negative 距离
  排除 positive code → neg_dist[i] = min(dists[i, :])  → (400,)
  【含义】每个 frame 到最近的其他 code 的距离
  ↓
Step 5: 计算 Triplet Loss
  triplet = ReLU(pos_dist - neg_dist + 0.2)  → (4, 100)
  【含义】如果 pos_dist + margin < neg_dist，loss=0（满足约束）
         否则需要继续优化
  ↓
Step 6: 应用 Mask 并归一化
  mask: (4, 100)  ← 标记有效 frames（排除 padding）
  loss = sum(triplet * mask) / sum(mask)  → scalar
```

---

## 💡 直观类比：停车场找车位

想象你要把车（student feature）停到目标车位（teacher code）：

| 变量 | 含义 |
|------|------|
| `pos_dist` | 你的车到**目标车位**的距离（希望**小**） |
| `neg_dist` | 你的车到**最近的其他车位**的距离（希望**大**） |
| `margin` | 安全边界（0.2），目标车位要比其他车位近至少这么多 |

**理想情况**：
```
pos_dist = 0.5  ← 很近目标
neg_dist = 2.0  ← 远离其他
loss = ReLU(0.5 - 2.0 + 0.2) = 0  ✅ 已满足约束
```

**需要优化**：
```
pos_dist = 1.8  ← 离目标太远
neg_dist = 1.5  ← 离其他太近
loss = ReLU(1.8 - 1.5 + 0.2) = 0.5  ❌ 需要继续训练
```

---

## 🔢 具体例子（真实数值）

**输入**：
- Frame 0: `teacher_code = 12`
- `dists[0, :]` = 所有 4096 个 codes 的距离

**计算**：
```python
# Step 3: Positive distance
pos_dist[0] = dists[0, 12] = 1.8  # Frame 0 到 code 12 的距离

# Step 4: Hard negative distance
dists_for_neg[0, 12] = inf  # 排除 positive code
neg_dist[0] = min(dists_for_neg[0, :]) = 1.5  # 最近的其他 code

# Step 5: Triplet loss
triplet[0] = ReLU(1.8 - 1.5 + 0.2) = ReLU(0.5) = 0.5
```

**结果**：Loss = 0.5，需要继续优化让 student feature 更靠近 code 12。

---

## 📐 维度表格总结

| 变量 | Shape | 说明 |
|------|-------|------|
| `student_features` | `(B, D, T)` | B=batch_size (4), D=feature_dim (512), T=time_frames (100) |
| `z` | `(B*T, D)` | Reshape 后：400 个 frames，每个 512 维 |
| `dists` | `(B*T, K)` | 距离矩阵：400 frames × 4096 codes |
| `pos_dist` | `(B*T,)` → `(B, T)` | 每个 frame 到 teacher code 的距离 |
| `neg_dist` | `(B*T,)` → `(B, T)` | 每个 frame 到最近其他 code 的距离 |
| `triplet` | `(B, T)` | Triplet loss (经过 ReLU) |
| `loss` | `scalar` | 最终 loss（masked average） |

---

## 🖼️ 可视化图表

运行以下命令生成详细的可视化图表：

```bash
python exp_0112_intermediate/visualize_triplet_loss.py
```

生成的图表包含：
1. **Shape Transformation**：维度变化流程
2. **Distance Matrix**：距离矩阵热图（标记 positive codes）
3. **Positive vs Negative**：每个 frame 的 pos/neg 距离对比
4. **Triplet Loss**：每个 frame 的 loss 值
5. **Formula Breakdown**：具体例子的计算过程
6. **Geometric Interpretation**：2D 投影几何示意图

---

## 🔑 关键参数

| 参数 | Baseline 值 | 作用 |
|------|-------------|------|
| `margin` | 0.2 | 安全边界 |
| `triplet_weight` | 1.0 | Loss 权重 |
| `encoder_stride` | 320 | 用于计算 mask (320 samples = 1 frame) |

---

## 📝 核心代码（简化版）

```python
# 1. Reshape: (B, D, T) → (B*T, D)
z = student_features.permute(0, 2, 1).reshape(-1, D)

# 2. 距离矩阵: (B*T, D) × (K, D) → (B*T, K)
dists = torch.cdist(z, codebook, p=2)

# 3. Positive 距离
pos_dist = dists[range(B*T), teacher_codes.reshape(-1)]

# 4. Hard Negative 距离
dists_for_neg = dists.clone()
dists_for_neg[range(B*T), teacher_codes.reshape(-1)] = float('inf')
neg_dist = dists_for_neg.min(dim=1).values

# 5. Triplet Loss
triplet = torch.relu(pos_dist - neg_dist + margin).reshape(B, T)

# 6. Masked Average
loss = (triplet * mask).sum() / mask.sum()
```

---

## 📚 完整文档

- **详细解释**：[TRIPLET_LOSS_EXPLANATION.md](TRIPLET_LOSS_EXPLANATION.md)
- **可视化脚本**：[visualize_triplet_loss.py](visualize_triplet_loss.py)
- **源代码**：[exp_1219/losses.py:136-171](../exp_1219/losses.py)

---

**创建时间**: 2026-02-04
**Baseline**: exp_k v6 @ epoch 300
