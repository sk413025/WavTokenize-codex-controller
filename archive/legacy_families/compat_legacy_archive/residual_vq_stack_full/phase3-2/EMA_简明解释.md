# EMA + Dead-Code Reset - 简明版

## 🎯 核心概念

**EMA (Exponential Moving Average)**：用移动平均更新 codebook，比梯度下降更稳定。

**Dead-Code Reset**：重新初始化很少使用的 codebook entries，保持多样性。

---

## 📊 6 步流程

```
每个 training step:

Step 1: 统计当前 batch
  counts[k] = code k 被使用的次数

Step 2: 计算 embedding sum
  embed_sum[k] = 所有分配给 code k 的 residuals 的总和

Step 3: EMA 更新 cluster size
  ema_cluster_size[k] = 0.99 × ema_old[k] + 0.01 × counts[k]
  ↳ 99% 历史 + 1% 当前 → 平滑稳定

Step 4: EMA 更新 embedding average
  ema_embed_avg[k] = 0.99 × ema_old[k] + 0.01 × embed_sum[k]

Step 5: 计算新 codebook
  codebook[k] = ema_embed_avg[k] / ema_cluster_size[k]
  ↳ 累积向量和 / 累积次数 = 平均向量

Step 6: Dead-code reset (threshold=2)
  if ema_cluster_size[k] < 2:
    codebook[k] = random_sample(residuals)
    ema_cluster_size[k] = 1.0
  ↳ 重新初始化未使用的 codes
```

---

## 💡 为什么 EMA 有效？

### ❌ 梯度更新 (Exp 6a/6b) 失败

**问题**：
- 不稳定：每个 batch 梯度波动大
- 无平滑：容易被高频 codes 主导
- 无恢复：Dead codes 永远死亡

**结果**：
```
Exp 6a (β=0.25): top10=1.0, used=9/1024   ← collapse
Exp 6b (β=0.5):  top10=1.0, used=10/1024  ← collapse
Exp 6b (β=1.0):  top10=1.0, used=11/1024  ← collapse
Exp 6b (β=2.0):  top10=1.0, used=9/1024   ← collapse
```

### ✅ EMA + Dead-Code Reset (Exp 6c) 成功

**优势**：
1. **平滑更新**：99% 历史 + 1% 当前，不受单个 batch 影响
2. **自动平衡**：高频 codes 的 ema_cluster_size 大，归一化后不会过大
3. **死亡恢复**：Dead codes 被实时检测并重新初始化

**结果** (6c-long-up0.1-K2048 @ step 1000)：
```
entropy:         9.03    (baseline 6.07, +49%)  ✅
top10_mass:      0.158   (target <0.5)          ✅
used_codes:      1089/2048 (53%)                ✅
joint_diversity: 0.992   (接近完美)             ✅
feature_mse:     0.034   (主目标未受损)         ✅
```

---

## 🔢 具体数值例子

**假设 decay=0.99, threshold=2**

### Code 234 的 EMA 演化

```python
# 初始状态
ema_cluster_size[234] = 0

# Step 1: 当前 batch counts[234] = 5
ema_cluster_size[234] = 0.99 * 0 + 0.01 * 5 = 0.05

# Step 2: 当前 batch counts[234] = 3
ema_cluster_size[234] = 0.99 * 0.05 + 0.01 * 3 = 0.0795

# Step 3: 当前 batch counts[234] = 8
ema_cluster_size[234] = 0.99 * 0.0795 + 0.01 * 8 = 0.1587

# ... 继续累积 ...

# Step 100: ema_cluster_size[234] ≈ 100.0 (稳定)
# Step 101: 当前 batch counts[234] = 8
ema_cluster_size[234] = 0.99 * 100.0 + 0.01 * 8 = 99.08
                       ↑ 主要贡献    ↑ 微调
```

### Dead-Code Reset 例子

```python
# Code 567 很少被使用
ema_cluster_size[567] = 1.5  < threshold (2)  ← Dead!

# 重新初始化
rand_frame = random.choice(0, 399)  # 假设 = 156
codebook[567] = residual_flat[156]  # 用当前 batch 的 frame 156 初始化
ema_cluster_size[567] = 1.0         # 重置计数
ema_embed_avg[567] = residual_flat[156]  # 重置向量和
```

---

## 📐 Shape 表格

| 变量 | Shape | 说明 |
|------|-------|------|
| `residual_flat` | `(N=400, D=128)` | B*T frames 的残差 |
| `indices` | `(N=400,)` | 每个 frame 分配的 code index |
| `counts` | `(K=2048,)` | 当前 batch 每个 code 使用次数 |
| `embed_sum` | `(K=2048, D=128)` | 每个 code 的 residuals 总和 |
| `ema_cluster_size` | `(K=2048,)` | 每个 code 的累积使用次数 (EMA) |
| `ema_embed_avg` | `(K=2048, D=128)` | 每个 code 的累积向量和 (EMA) |
| `codebook` | `(K=2048, D=128)` | 最终 codebook weights |

---

## 🔑 关键参数 (Exp 6c 最佳配置)

| 参数 | 值 | 作用 |
|------|-----|------|
| `ema_decay` | 0.99 | 历史权重（99% 历史 + 1% 当前） |
| `ema_dead_code_threshold` | 2 | Dead-code 检测阈值 |
| `codebook_size` | 2048 | 每层 codebook 大小 |
| `n_layers` | 4 | RVQ 层数 |
| `ema_usage_penalty` | 0.1 | 抑制高频 codes 的惩罚 |

---

## 📈 实验结果

### Exp 6a/6b vs Exp 6c

| Exp | Method | top10 @ 200 | used @ 200 | top10 @ 1000 | used @ 1000 | 结果 |
|-----|--------|-------------|------------|--------------|-------------|------|
| 6a (β=0.25) | 梯度 | 1.0000 | 9/1024 | - | - | ❌ Collapse |
| 6b (β=0.5) | 梯度 | 1.0000 | 10/1024 | - | - | ❌ Collapse |
| 6b (β=1.0) | 梯度 | 0.9996 | 11/1024 | - | - | ❌ Collapse |
| 6b (β=2.0) | 梯度 | 1.0000 | 9/1024 | - | - | ❌ Collapse |
| **6c (EMA, th=2)** | **EMA** | **0.175** | **671/1024** | **0.234** | **671/1024** | ✅ **P2 PASS** |
| **6c (K=2048, up=0.1)** | **EMA** | **0.135** | **1075/2048** | **0.158** | **1089/2048** | ✅ **最佳** |

---

## 🎓 核心理解

### EMA 的平滑效果

**梯度更新**（不稳定）：
```
batch 1: code 12 更新 +0.5
batch 2: code 12 更新 -0.3
batch 3: code 12 更新 +0.8
→ 波动大，容易 collapse
```

**EMA 更新**（平滑）：
```
step 1: ema[12] = 0.99*0 + 0.01*5 = 0.05
step 2: ema[12] = 0.99*0.05 + 0.01*3 = 0.0795
step 3: ema[12] = 0.99*0.0795 + 0.01*8 = 0.1587
→ 平滑增长，稳定
```

### Dead-Code Reset 的必要性

```
无 Reset: 某些 codes 永远不被使用 → 浪费空间
有 Reset: 检测到 dead → 重新初始化 → 给新机会
```

---

## 🖼️ 可视化

运行以下命令生成详细可视化：

```bash
python exp_0128/phase3-2/visualize_ema_process.py
```

生成的图表包含：
1. **EMA Cluster Size Evolution**：各 code 的 EMA 演化
2. **Codebook Usage Over Time**：Used codes 和 dead codes 变化
3. **Top-5 Mass**：集中度变化（越低越好）
4. **EMA Update Mechanism**：更新公式图解
5. **Heatmap**：EMA cluster size 热图
6. **Single Code Example**：高频 vs 低频 code 对比

---

## 📚 详细文档

- **完整流程**：[EMA_DEAD_CODE_RESET_详解.md](EMA_DEAD_CODE_RESET_详解.md)
- **源代码**：[exp_0128/phase3/residual_vq/models_rvq.py:107-147](../phase3/residual_vq/models_rvq.py)
- **实验记录**：[PROGRESS.md](PROGRESS.md)
- **总结报告**：[SUMMARY.md](SUMMARY.md)

---

**创建时间**: 2026-02-04
**实验来源**: Phase 3-2 Exp 6c
**最佳配置**: K=2048, layers=4, EMA th=2, β=1.0, up=0.1
**验收状态**: ✅ P2 PASS - 建议继续 RVQ 实验
