# Exp 6c: EMA + Dead-Code Reset 详细流程

## 📍 源代码位置

- 文件：[exp_0128/phase3/residual_vq/models_rvq.py:107-147](../phase3/residual_vq/models_rvq.py)
- 方法：`ResidualVectorQuantizer._ema_update_layer()`

---

## 🎯 核心目标

**EMA (Exponential Moving Average) Codebook Update**：
- 用**移动平均**的方式更新 codebook，比梯度下降更稳定
- 防止 codebook collapse（所有 features 都映射到少数几个 codes）

**Dead-Code Reset**：
- 检测并重新初始化"死亡 codes"（很少或从未被使用的 codebook entries）
- 保持 codebook 的多样性和利用率

---

## 🔢 完整计算流程

### Step 0: 初始化 EMA Buffers

```python
# 在 __init__ 中注册 persistent buffers
self.register_buffer(
    "ema_cluster_size",        # 每个 code 的累积使用次数
    torch.zeros(n_layers, codebook_size),
    persistent=True,
)
self.register_buffer(
    "ema_embed_avg",           # 每个 code 的累积向量和
    torch.zeros(n_layers, codebook_size, dim),
    persistent=True,
)
```

**Shape 示例**（4 layers, K=2048, D=128）：
```
ema_cluster_size: (4, 2048)      # 每层每个 code 的累积使用计数
ema_embed_avg:    (4, 2048, 128) # 每层每个 code 的累积向量和
```

---

### Step 1: 统计当前 Batch 的 Code 使用情况

```python
# 输入
residual_flat: (N, D)  # N = B*T, 所有 frames 的残差
indices: (N,)          # 每个 frame 分配到的 code index

# 统计每个 code 被使用的次数
counts = torch.bincount(indices, minlength=K)  # (K,)
```

**Shape 示例**（B=4, T=100, K=2048）：
```
residual_flat: (400, 128)  # 400 个 frames，每个 128 维
indices:       (400,)       # 每个 frame 对应的 code index [0~2047]
counts:        (2048,)      # 每个 code 在这个 batch 被使用的次数
```

**counts 示例**：
```python
counts[0] = 5    # code 0 被使用了 5 次
counts[1] = 0    # code 1 未被使用
counts[12] = 3   # code 12 被使用了 3 次
...
counts[234] = 8  # code 234 被使用了 8 次
...
sum(counts) = 400  # 总共 400 个 frames
```

---

### Step 2: 计算当前 Batch 的 Embedding Sum

```python
# 对每个 code，累加分配给它的所有 residuals
embed_sum = torch.zeros(K, D, device=device)        # (K, D)
embed_sum.index_add_(0, indices, residual_flat)     # (K, D)
```

**含义**：
```python
# 对于每个 code k，embed_sum[k] = 所有分配给 code k 的 residuals 的总和
embed_sum[0] = sum(residual_flat[i] for i in range(N) if indices[i] == 0)
embed_sum[1] = sum(residual_flat[i] for i in range(N) if indices[i] == 1)
...
```

**Shape 示例**：
```
embed_sum: (2048, 128)  # 每个 code 对应的 residuals 总和
```

---

### Step 3: EMA 更新 Cluster Size

```python
# 更新公式: ema_new = decay * ema_old + (1 - decay) * current
self.ema_cluster_size[layer_idx].mul_(self.ema_decay).add_(
    counts, alpha=(1.0 - self.ema_decay)
)
```

**数学表达式**：
```
ema_cluster_size[k] = decay * ema_cluster_size[k] + (1 - decay) * counts[k]
```

**参数**：`decay = 0.99` (Exp 6c 配置)

**更新示例**：
```python
# 假设 decay = 0.99
# code 234 在上一步的 ema_cluster_size = 100.0
# 当前 batch counts[234] = 8

ema_cluster_size[234] = 0.99 * 100.0 + (1 - 0.99) * 8
                      = 99.0 + 0.08
                      = 99.08
```

**特点**：
- 新值占比小 (1%)，历史值占比大 (99%)
- 平滑累积，不会被单个 batch 的波动影响太大

---

### Step 4: EMA 更新 Embedding Average

```python
# 同样使用 EMA 更新向量和
self.ema_embed_avg[layer_idx].mul_(self.ema_decay).add_(
    embed_sum, alpha=(1.0 - self.ema_decay)
)
```

**数学表达式**：
```
ema_embed_avg[k] = decay * ema_embed_avg[k] + (1 - decay) * embed_sum[k]
```

**Shape 更新**：
```
ema_embed_avg[layer_idx]: (2048, 128)
embed_sum:                 (2048, 128)
→ element-wise EMA update
```

---

### Step 5: Laplace Smoothing + 计算新 Codebook

```python
# Laplace smoothing (防止除零)
n = self.ema_cluster_size[layer_idx].sum()  # 总累积次数
cluster_size = (self.ema_cluster_size[layer_idx] + eps) / (n + K * eps) * n

# 计算新 codebook: embed_avg / cluster_size
embed = self.ema_embed_avg[layer_idx] / cluster_size.unsqueeze(1).clamp(min=1e-12)

# 更新 codebook weights
self.codebooks[layer_idx].weight.data.copy_(embed)
```

**Laplace Smoothing 目的**：
- 防止 `cluster_size[k] = 0` 导致除零错误
- `eps = 1e-5` (很小的平滑因子)

**新 Codebook 计算**：
```
codebook[k] = ema_embed_avg[k] / ema_cluster_size[k]
```

**含义**：每个 code 的新向量 = 历史累积向量和 / 历史累积次数（即平均）

**Shape**：
```
cluster_size: (2048,)
ema_embed_avg: (2048, 128)
cluster_size.unsqueeze(1): (2048, 1)  ← broadcast to (2048, 128)
embed: (2048, 128)  ← 新 codebook
```

---

### Step 6: Dead-Code Reset

```python
# 检测 dead codes (使用次数 < threshold)
dead = self.ema_cluster_size[layer_idx] < float(self.ema_dead_code_threshold)

if dead.any() and residual_flat.numel() > 0:
    dead_idx = dead.nonzero(as_tuple=False).squeeze(1)  # 找到 dead codes 的 indices
    num_dead = int(dead_idx.numel())

    # 从当前 batch 随机采样 residuals 来替换 dead codes
    rand = torch.randint(0, residual_flat.shape[0], (num_dead,), device=device)
    sampled = residual_flat[rand]

    # 重新初始化 dead codes
    self.codebooks[layer_idx].weight.data[dead_idx] = sampled

    # 重置 EMA buffers 以保持一致性
    self.ema_cluster_size[layer_idx, dead_idx] = 1.0
    self.ema_embed_avg[layer_idx, dead_idx] = sampled
```

**Dead-Code Threshold**：`threshold = 2` (Exp 6c 最佳配置)

**检测示例**：
```python
# 假设 threshold = 2
ema_cluster_size[layer_idx] = [100.5, 0.8, 50.2, 1.5, 200.0, 0.1, ...]
                              #   ✓    ❌    ✓    ❌     ✓     ❌

dead = [False, True, False, True, False, True, ...]
dead_idx = [1, 3, 5, ...]  # 需要重置的 code indices
```

**重新初始化**：
```python
# 从当前 batch 的 residual_flat 随机采样
rand = [42, 156, 89, ...]  # 随机 frame indices
sampled = residual_flat[rand]  # (num_dead, 128)

# 替换 dead codes
codebook.weight.data[1] = residual_flat[42]   # 用 frame 42 的特征初始化
codebook.weight.data[3] = residual_flat[156]  # 用 frame 156 的特征初始化
codebook.weight.data[5] = residual_flat[89]   # 用 frame 89 的特征初始化
```

---

## 📊 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ 输入 (每个 training step)                                        │
│   residual_flat: (N=400, D=128)  # B*T frames                   │
│   indices:       (N=400,)         # 分配的 code indices          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 统计当前 Batch 使用情况                                  │
│   counts = bincount(indices)  → (K=2048,)                       │
│   例: [5, 0, 3, 0, 2, ..., 8, ...]                              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: 计算当前 Batch Embedding Sum                            │
│   embed_sum = zeros(K, D)                                       │
│   embed_sum.index_add_(0, indices, residual_flat)               │
│   → (K=2048, D=128)                                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: EMA 更新 Cluster Size                                   │
│   ema_cluster_size[k] = 0.99 * ema_cluster_size[k] + 0.01 * counts[k] │
│   → (K=2048,)                                                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: EMA 更新 Embedding Average                              │
│   ema_embed_avg[k] = 0.99 * ema_embed_avg[k] + 0.01 * embed_sum[k] │
│   → (K=2048, D=128)                                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: 计算新 Codebook (Laplace Smoothing)                     │
│   cluster_size = (ema_cluster_size + 1e-5) / (n + K*1e-5) * n  │
│   codebook[k] = ema_embed_avg[k] / cluster_size[k]             │
│   → 更新 codebook.weight.data                                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 6: Dead-Code Reset (threshold=2)                          │
│   dead = (ema_cluster_size < 2)                                 │
│   if dead.any():                                                │
│     dead_idx = dead.nonzero()                                   │
│     sampled = random_sample_from(residual_flat)                 │
│     codebook[dead_idx] = sampled                                │
│     ema_cluster_size[dead_idx] = 1.0                            │
│     ema_embed_avg[dead_idx] = sampled                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                ┌───────────────────────┐
                │ 输出: 更新后的 Codebook │
                └───────────────────────┘
```

---

## 💡 为什么 EMA 比梯度更新有效？

### Exp 6a/6b (梯度更新) ❌ 全部失败

**方法**：
```python
# Codebook loss (梯度更新)
loss_codebook = F.mse_loss(residual.detach(), q)
# 通过 optimizer 更新 codebook weights
```

**问题**：
1. **不稳定**：梯度更新容易被少数高频 codes 主导
2. **无平滑机制**：每个 batch 的梯度波动大
3. **缺少死亡检测**：Dead codes 无法自动恢复

**结果**：
```
Exp 6a (β=0.25): top10=1.0, used=9/1024   ← collapse at step 200
Exp 6b (β=0.5):  top10=1.0, used=10/1024  ← collapse at step 200
Exp 6b (β=1.0):  top10=0.9996, used=11/1024  ← collapse
Exp 6b (β=2.0):  top10=1.0, used=9/1024   ← collapse at step 200
```

---

### Exp 6c (EMA + Dead-Code Reset) ✅ 成功

**方法**：
```python
# EMA 更新 (平滑 + 稳定)
ema_cluster_size = 0.99 * ema_cluster_size + 0.01 * counts
ema_embed_avg = 0.99 * ema_embed_avg + 0.01 * embed_sum
codebook = ema_embed_avg / ema_cluster_size

# Dead-code reset (保持多样性)
if ema_cluster_size < threshold:
    codebook[dead] = random_sample(residuals)
```

**优势**：
1. **平滑更新**：99% 历史 + 1% 当前，不受单个 batch 波动影响
2. **自动平衡**：高频 codes 的 ema_cluster_size 大，embed 被归一化
3. **死亡恢复**：Dead codes 被实时检测并重新初始化

**结果**：
```
Exp 6c-long-up0.1-K2048 @ step 1000:
  entropy:        9.03    (baseline 6.07, +49%)  ✅
  top10_mass:     0.158   (target <0.5)          ✅
  used_codes:     1089/2048 (53%)                ✅
  joint_diversity: 0.992   (接近完美)            ✅
  feature_mse:    0.034   (主目标未受损)         ✅
```

---

## 🔍 关键参数

| 参数 | Exp 6c 最佳值 | 作用 |
|------|--------------|------|
| `ema_decay` | 0.99 | EMA 衰减率（历史权重）|
| `ema_eps` | 1e-5 | Laplace smoothing 因子 |
| `ema_dead_code_threshold` | 2 | Dead-code 检测阈值 |
| `ema_usage_penalty` | 0.1 | 抑制高频 codes 的惩罚系数 |
| `codebook_size` | 2048 | 每层 codebook 大小 |
| `n_layers` | 4 | RVQ 层数 |

---

## 📈 实验结果对比

### Top-10 Mass 漂移现象

**发现**：即使使用 EMA，top10_mass 在训练后期仍有上升趋势

| Config | step 200 | step 400 | step 600 | step 800 | step 1000 |
|--------|----------|----------|----------|----------|-----------|
| K=1024, th=2 | 0.175 | 0.311 | 0.219 | 0.234 | 0.234 |
| K=2048, th=2 | 0.129 | - | - | - | 0.231 |
| K=2048, up=0.1 | **0.135** | 0.245 | 0.180 | 0.217 | **0.158** |

**缓解方法**：
- 增加 `codebook_size`：K=1024 → 2048
- 添加 `usage_penalty=0.1`：惩罚高频 codes

---

## 🔬 Per-Layer Usage 分析

**最佳配置** (`6c-long-up0.1-K2048` @ step 1000)：

| Layer | Used Codes | Entropy | Usage % |
|-------|------------|---------|---------|
| L0 | 639 / 2048 | 6.07 | 31.2% |
| L1 | 891 / 2048 | 6.62 | 43.5% |
| L2 | 841 / 2048 | 6.51 | 41.1% |
| L3 | 757 / 2048 | 6.36 | 37.0% |

**Joint Diversity**: 0.992 (接近完美，说明各层使用不同的 codes)

---

## 📝 核心代码（精简版）

```python
@torch.no_grad()
def _ema_update_layer(self, layer_idx, residual_flat, indices):
    # Step 1-2: 统计当前 batch
    K = self.codebook_size
    counts = torch.bincount(indices, minlength=K).float()
    embed_sum = torch.zeros(K, self.dim, device=device)
    embed_sum.index_add_(0, indices, residual_flat.float())

    # Step 3-4: EMA 更新
    self.ema_cluster_size[layer_idx].mul_(0.99).add_(counts, alpha=0.01)
    self.ema_embed_avg[layer_idx].mul_(0.99).add_(embed_sum, alpha=0.01)

    # Step 5: 计算新 codebook
    n = self.ema_cluster_size[layer_idx].sum()
    cluster_size = (self.ema_cluster_size[layer_idx] + 1e-5) / (n + K*1e-5) * n
    embed = self.ema_embed_avg[layer_idx] / cluster_size.unsqueeze(1).clamp(min=1e-12)
    self.codebooks[layer_idx].weight.data.copy_(embed)

    # Step 6: Dead-code reset
    if self.ema_dead_code_threshold > 0:
        dead = self.ema_cluster_size[layer_idx] < float(self.ema_dead_code_threshold)
        if dead.any():
            dead_idx = dead.nonzero().squeeze(1)
            num_dead = int(dead_idx.numel())
            rand = torch.randint(0, residual_flat.shape[0], (num_dead,), device=device)
            sampled = residual_flat[rand]

            self.codebooks[layer_idx].weight.data[dead_idx] = sampled
            self.ema_cluster_size[layer_idx, dead_idx] = 1.0
            self.ema_embed_avg[layer_idx, dead_idx] = sampled
```

---

## 🎓 关键理解

### 1. EMA 的平滑效果

**无 EMA**（梯度更新）：
```
batch 1: code 12 梯度 = +0.5
batch 2: code 12 梯度 = -0.3
batch 3: code 12 梯度 = +0.8
→ 波动大，不稳定
```

**有 EMA**：
```
step 0: ema_cluster_size[12] = 0
step 1: ema_cluster_size[12] = 0.99*0 + 0.01*5 = 0.05
step 2: ema_cluster_size[12] = 0.99*0.05 + 0.01*3 = 0.0795
step 3: ema_cluster_size[12] = 0.99*0.0795 + 0.01*8 = 0.1587
→ 平滑增长，稳定
```

### 2. Dead-Code Reset 的必要性

**无 Reset**：
```
某些 codes 从未被使用 → ema_cluster_size = 0 → codebook 永远不更新 → 浪费空间
```

**有 Reset**：
```
检测到 ema_cluster_size < 2 → 重新初始化为当前 residuals → 给 code 新生机会
```

---

**文档创建时间**: 2026-02-04
**实验来源**: Phase 3-2 Exp 6c
**最佳配置**: K=2048, layers=4, EMA th=2, β=1.0, up=0.1
**验收状态**: ✅ P2 PASS (建议继续 RVQ 实验)
