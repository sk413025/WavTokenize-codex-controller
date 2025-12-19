# Exp_1219: Cosine Similarity Loss 實驗

基於 Exp48 分析結果的改進實驗。

**建立日期**: 2025-12-19

---

## 🎯 實驗目的

解決 Exp48 發現的 **特徵方向不對齊問題**：
- Exp48 最佳配置 (feature_weight=1.0, triplet_weight=1.0) 達到 15.82% Token Accuracy
- 但 **Cosine Similarity 僅 0.21** (應該接近 0.9 以上才正常)
- MSE Loss 同時優化方向和尺度，但方向優化效果不足

### 問題診斷

| 指標 | 數值 | 評估 |
|------|------|------|
| Norm Ratio (Stu/Tea) | 1.43 | ✓ 可接受 |
| **Cosine Similarity** | **0.21** | ⚠️ 太低! |
| Student Range | [-3.85, 5.10] | - |
| Teacher Range | [-0.99, 0.98] | - |

### 解決方案

新增 **Cosine Similarity Loss**，專門優化特徵方向：

```python
cosine_loss = 1 - F.cosine_similarity(z_stu, z_tea, dim=-1).mean()
```

---

## 📊 實驗設計

### Exp49: Cosine Loss 基準

在 Exp48 配置上加入 Cosine Loss：

```bash
--feature_weight 1.0 \
--cosine_weight 0.5 \
--triplet_weight 1.0 \
--triplet_margin 0.2
```

**假設**: Cosine Loss 能有效提升 cos_sim 到 0.7+

### Exp50: Triplet Margin 增大

基於 codebook 距離分析，55% code 有 NN=0，有效 code NN mean=1.27：

```bash
--feature_weight 1.0 \
--triplet_weight 1.0 \
--triplet_margin 0.5  # 從 0.2 增加到 0.5
```

**假設**: 更大的 margin 能增強 token 區分度

### Exp51: 組合改進

```bash
--feature_weight 1.0 \
--cosine_weight 0.5 \
--triplet_weight 1.0 \
--triplet_margin 0.5
```

---

## 📁 文件結構

```
exp_1219/
├── README.md                    # 本文件
├── ANALYSIS_REPORT.md           # Exp48 配置分析報告
├── losses.py                    # 新增 MaskedCosineLoss
├── train.py                     # 訓練腳本 (支援 cosine_weight)
├── run_exp49_cosine.sh          # Exp49 啟動腳本
├── run_exp50_margin.sh          # Exp50 啟動腳本
├── run_exp51_combined.sh        # Exp51 啟動腳本
├── verify_feature_scale.py      # 特徵尺度驗證工具
└── runs/                        # 實驗結果目錄
    ├── exp49_cosine/
    ├── exp50_margin/
    └── exp51_combined/
```

---

## 🔧 核心改動

### 1. MaskedCosineLoss

新增的 Cosine Similarity Loss：

```python
class MaskedCosineLoss(nn.Module):
    """專門優化特徵方向對齊"""

    def forward(self, student_features, teacher_features, lengths):
        # student_features: (B, D, T)
        # Reshape to (B*T, D)
        stu = student_features.permute(0, 2, 1).reshape(-1, D)
        tea = teacher_features.permute(0, 2, 1).reshape(-1, D)

        # Cosine similarity: 1 = 完全對齊, 0 = 正交, -1 = 相反
        cos_sim = F.cosine_similarity(stu, tea, dim=1)

        # Loss = 1 - cosine (越接近 0 越好)
        loss = (1 - cos_sim) * mask_flat
        return loss.sum() / (mask_flat.sum() + 1e-8)
```

### 2. 新的 Loss 組合

```python
MaskedCombinedLossV2(
    feature_weight=1.0,    # MSE Loss
    cosine_weight=0.5,     # 新增: Cosine Loss
    triplet_weight=1.0,    # Triplet Loss
    triplet_margin=0.2,    # or 0.5
)
```

---

## 📈 預期指標

| 指標 | Exp48 (baseline) | Exp49 (目標) | Exp50 (目標) | Exp51 (目標) |
|------|------------------|--------------|--------------|--------------|
| Token Accuracy | 15.82% | 18%+ | 17%+ | 20%+ |
| Cosine Sim | 0.21 | 0.6+ | 0.25+ | 0.7+ |
| Triplet Loss | 0.76 | 0.70 | 0.60 | 0.55 |

---

## 🚀 執行方式

```bash
# 確保使用 GPU 1 (避免 OOM)
export CUDA_VISIBLE_DEVICES=1

# Exp49: Cosine Loss
bash exp_1219/run_exp49_cosine.sh

# Exp50: Margin 0.5
bash exp_1219/run_exp50_margin.sh

# Exp51: Combined
bash exp_1219/run_exp51_combined.sh
```

---

## 📝 參考

- [ANALYSIS_REPORT.md](./ANALYSIS_REPORT.md) - 完整的 Exp48 配置分析
- [exp_1217/runs/exp48_best_config](../exp_1217/runs/exp48_best_config) - 基準實驗結果
