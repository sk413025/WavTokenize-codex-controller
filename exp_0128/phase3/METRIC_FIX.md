# Phase 3 Metric Fix: 正確的 RVQ 評估指標

## 問題發現

**原始問題**：`train_rvq_short_run.py` 使用 strict_accuracy 比較 teacher codes 與 student codes，但：

1. **Codebook space 不同**：
   - Teacher: Single VQ, 4096 codes
   - Student (Exp 5a): RVQ Layer 0, 2048 codes
   - Student (Exp 5b): RVQ Layer 0, 1024 codes
   - Student (Exp 5c): RVQ Layer 0, 512 codes

2. **語義不對等**：
   - RVQ Layer 0 只是粗略逼近
   - 不應與 teacher 單層 VQ 直接比較

3. **成功判準失效**：
   - `strict_acc >= 0.82%` 永遠無法達成
   - 幾乎是 random chance (~1024/4096 ≈ 25% 或更低)

## 修正方案

### 新的評估指標

#### 1. Layer 0 多樣性（與 baseline 比較）

```python
'layer0_entropy': float          # Layer 0 entropy (bits)
'layer0_top10_mass': float       # Layer 0 top-10 mass
'layer0_used_codes': int         # Layer 0 unique codes used
'layer0_usage_pct': float        # Layer 0 usage percentage
```

**意義**：Layer 0 是 RVQ 的粗略逼近層，可與 baseline 單層 VQ 比較

#### 2. Joint Diversity（RVQ 特有）

```python
'joint_unique_codes': int        # Unique (layer0, layer1, ...) tuples
'joint_total_codes': int         # Total code combinations
'joint_diversity': float         # unique / total (higher = better)
```

**意義**：所有層組合的多樣性，理論上遠高於單層 VQ

#### 3. Feature Space Alignment（主要訓練目標）

```python
'feature_mse': float             # MSE(student_quantized, teacher_encoder_out)
```

**意義**：Student quantized 與 Teacher encoder output 的對齊程度（訓練的真正目標）

### 新的成功判準

```python
success = (
    metrics['layer0_entropy'] > 6.5 and          # vs baseline 6.07
    metrics['layer0_top10_mass'] < 0.15 and     # vs baseline 19.7%
    metrics['joint_diversity'] > 0.7 and         # RVQ unique: >70%
    metrics['feature_mse'] < 0.1                 # Feature alignment good
)
```

## 修改的檔案

### 1. train_rvq_short_run.py

**修改函數**：
- `evaluate_collapse_metrics()`: 重寫為 RVQ-specific 指標
- Success criteria: 更新為新的判準
- `plot_loss_curves()`: 改為 3×3 grid，包含新指標

**新增圖表**：
1. Total Training Loss
2. Loss Components (main, intermediate, RVQ)
3. RVQ Commitment Loss
4. Layer 0 Entropy (vs baseline)
5. Layer 0 Top-10 Mass (vs baseline)
6. Joint Diversity (RVQ-specific)
7. Feature MSE (primary objective)
8. Layer 0 Codebook Usage
9. Per-Layer Entropy (if available)

### 2. PLAN.md

**更新章節**：
- **成功判準**：改為 RVQ-specific 指標
- 移除不適用的 strict_accuracy
- 添加詳細說明

## 基準比較

| 指標 | Baseline (exp_k v6) | RVQ Target |
|------|---------------------|------------|
| **Layer 0 Entropy** | 6.07 | > 6.5 |
| **Layer 0 Top-10 Mass** | 19.7% | < 15% |
| **Joint Diversity** | N/A (single VQ) | > 70% |
| **Feature MSE** | N/A | < 0.1 |

## 為什麼這些指標有意義？

### Layer 0 Metrics

- Layer 0 是 RVQ 的第一層（粗略逼近）
- 使用與 baseline 相同的評估方式（entropy, top-k）
- **可以公平比較**，判斷 RVQ 是否改善了 collapse

### Joint Diversity

- 衡量所有層組合的多樣性
- 理論上：1024^4 種可能組合
- 高 joint diversity 表示 RVQ 真正利用多層結構

### Feature MSE

- 直接衡量訓練目標的達成度
- Student quantized 應該接近 teacher encoder output
- **這才是實際要優化的東西**

## 移除的指標

### ❌ Strict Accuracy (teacher codes vs student codes)

**原因**：
```
Teacher codes:  [0, 4095]     # 4096 space
Student codes:  [0, 1023]     # 1024 space (exp5b)

比較這兩個毫無意義！
```

**替代方案**：使用 Feature MSE 直接衡量 quantized vectors 的對齊

## 實驗建議

運行 Exp 5a/5b/5c 時，關注：

1. **Layer 0 是否優於 baseline**（entropy > 6.5, top-10 < 15%）
2. **Joint diversity 是否高**（> 70%，表示多層真正有效）
3. **Feature MSE 是否低**（< 0.1，表示訓練目標達成）
4. **Per-layer entropy 分布**（每層是否都有多樣性，還是只有某層有用）

## 參考

- 修改時間：2026-02-03
- 原因：發現 strict_accuracy 比較 codebook space 不同的問題
- 感謝：User 指出這個關鍵問題
