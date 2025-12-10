# exp17 & exp18 實驗說明

## 實驗概述

**日期**: 2025-12-08  
**目的**: 解決 Token Accuracy 下降問題

---

## exp17: Margin-based Contrastive Loss

### 核心概念

**替換 MSE Loss 為 Margin Loss**，直接優化 Voronoi boundary。

```python
# 原本: MSE + CE
loss = mse_loss + ce_loss

# exp17: Margin + CE  
loss = margin_loss + ce_loss
```

### Margin Loss 公式

```python
def margin_loss(student_emb, teacher_codes, codebook, margin=0.5):
    """
    確保 student embedding 在正確的 Voronoi cell 內部
    
    Args:
        student_emb: (B, T, C) Student embedding (VQ 前)
        teacher_codes: (B, T) Teacher token indices
        codebook: (num_codes, C) Codebook vectors
        margin: 安全邊距
        
    Returns:
        loss: Margin contrastive loss
    """
    B, T, C = student_emb.shape
    num_codes = codebook.shape[0]
    
    # 1. 計算到所有 codebook 的距離
    # student_emb: (B, T, C) -> (B*T, C, 1)
    # codebook: (num_codes, C) -> (1, C, num_codes)
    z = student_emb.reshape(-1, C, 1)  # (B*T, C, 1)
    c = codebook.t().unsqueeze(0)      # (1, C, num_codes)
    
    # distances: (B*T, num_codes)
    distances = torch.norm(z - c, dim=1)  # (B*T, num_codes)
    
    # 2. 獲取到正確 token 的距離
    correct_codes = teacher_codes.reshape(-1)  # (B*T,)
    correct_dist = distances.gather(1, correct_codes.unsqueeze(1)).squeeze(1)  # (B*T,)
    
    # 3. 獲取到最近錯誤 token 的距離
    # Mask out correct token
    mask = torch.ones_like(distances)  # (B*T, num_codes)
    mask.scatter_(1, correct_codes.unsqueeze(1), 0)  # Set correct token to 0
    
    masked_distances = distances + (1 - mask) * 1e10  # 將正確 token 設為很大
    nearest_wrong_dist = masked_distances.min(dim=1)[0]  # (B*T,)
    
    # 4. Margin Loss: 希望 correct_dist < nearest_wrong_dist - margin
    # loss = max(0, correct_dist - nearest_wrong_dist + margin)
    margin_loss = torch.clamp(correct_dist - nearest_wrong_dist + margin, min=0).mean()
    
    return margin_loss
```

### 優點

1. **直接優化 Voronoi boundary** - 不關心絕對距離，只關心相對位置
2. **避免 MSE 與 CE 矛盾** - Margin Loss 已包含「靠近正確 token」的目標
3. **更強的幾何約束** - 強制 embedding 在正確 cell 的內部

### 實驗設定

```bash
# exp17
Margin: 0.5
CE Weight: 1.0
Total Loss = margin_loss + ce_loss
```

---

## exp18: 反向 Curriculum Learning

### 核心概念

**先用 CE 定位 cell，再用 MSE 精細化**，避免 MSE 把 embedding 推到錯誤 cell。

### 三階段訓練

```python
# Stage 1 (epoch 1-10): CE only - 快速定位到正確 Voronoi cell
if epoch <= 10:
    loss = 1.0 * ce_loss + 0.0 * mse_loss
    
# Stage 2 (epoch 11-30): CE + MSE - 在 cell 內部精細調整  
elif epoch <= 30:
    loss = 0.5 * ce_loss + 1.0 * mse_loss
    
# Stage 3 (epoch 31-50): MSE dominant - 穩定 embedding
else:
    loss = 0.1 * ce_loss + 1.0 * mse_loss
```

### 邏輯

1. **Stage 1**: CE 強力監督，快速學會「選對 token」
2. **Stage 2**: MSE 開始介入，在正確 cell 內部優化
3. **Stage 3**: MSE 主導，CE 只做微調，穩定 embedding space

### 為什麼是「反向」？

對比 exp_1204 的 Curriculum:

```python
# exp_1204 (錯誤順序)
Stage 1: MSE only  → 學到錯誤分佈
Stage 2: MSE + CE  → 嘗試修正，但衝突
Stage 3: CE dominant → 推翻之前學的，catastrophic forgetting

# exp18 (正確順序)  
Stage 1: CE only  → 直接學正確分類
Stage 2: CE + MSE → 精細化，不會跨界
Stage 3: MSE dominant → 穩定化
```

### 實驗設定

```bash
# exp18
Total epochs: 50
Stage 1 (1-10):   CE=1.0, MSE=0.0
Stage 2 (11-30):  CE=0.5, MSE=1.0
Stage 3 (31-50):  CE=0.1, MSE=1.0
```

---

## 實驗執行

### 1. 梯度衝突分析（已啟動）

```bash
# 檢查執行狀態
tail -f exp_1207/gradient_analysis.log

# 完成後查看結果
ls exp_1207/gradient_analysis/
```

**預期輸出**:
- `gradient_analysis_YYYYMMDD_HHMMSS.json`: 數值結果
- `gradient_analysis_YYYYMMDD_HHMMSS.png`: 可視化圖表

**解讀指標**:
- `mean_cosine > 0.5`: 梯度對齊，協同學習 ✅
- `mean_cosine ≈ 0`: 梯度正交，互不影響 ⚠️
- `mean_cosine < 0`: 梯度衝突，互相抵消 ❌
- `conflict_ratio > 50%`: 超過一半樣本存在衝突 ❌

---

### 2. exp17 (Margin Loss) - 需要實現訓練程式

**TODO**: 創建 `train_margin_loss.py`

核心修改：
1. 複製 `train_with_ce.py`
2. 移除 `feature_loss` (MSE)
3. 新增 `margin_loss` 函式
4. 更新 `compute_losses`:
   ```python
   total_loss = margin_loss + ce_weight * ce_loss
   ```

**執行**:
```bash
cd exp_1207
./run_exp17_margin.sh
```

---

### 3. exp18 (Curriculum) - 需要實現訓練程式

**TODO**: 創建 `train_curriculum.py`

核心修改：
1. 複製 `train_with_ce.py`
2. 新增 `get_stage_weights(epoch)` 函式:
   ```python
   def get_stage_weights(epoch, args):
       if epoch <= args.stage1_epochs:
           return args.stage1_ce_weight, args.stage1_mse_weight
       elif epoch <= args.stage1_epochs + args.stage2_epochs:
           return args.stage2_ce_weight, args.stage2_mse_weight
       else:
           return args.stage3_ce_weight, args.stage3_mse_weight
   ```
3. 在訓練迴圈中動態更新權重

**執行**:
```bash
cd exp_1207  
./run_exp18_curriculum.sh
```

---

## 預期結果

### exp17 (Margin Loss)

**如果成功**:
- Token Accuracy 應該**不再下降**
- 訓練過程更穩定
- `mean_correct_distance` < `mean_min_distance` (在正確 cell 內)

**如果失敗**:
- Margin 太大 → 過度約束，難以優化
- Margin 太小 → 約束不足，仍會選錯

**下一步**: 調整 margin (0.3, 0.5, 1.0)

---

### exp18 (Curriculum)

**如果成功**:
- Token Accuracy 在 Stage 1 快速上升
- Stage 2, 3 持續優化，不會崩潰
- 最終 Token Accuracy > exp16

**如果失敗**:
- Stage 1 太短 → CE 沒學好就進入 Stage 2
- Stage 轉換太突然 → Loss 跳躍

**下一步**: 調整 stage epochs 或使用平滑過渡

---

## 診斷檢查清單

完成實驗後，檢查：

1. ✅ Token Accuracy 是否上升？
2. ✅ Loss 是否穩定下降？
3. ✅ `mean_correct_distance` vs `mean_min_distance`?
4. ✅ 音質是否改善？
5. ✅ t-SNE embedding space 是否分離？

---

## 後續實驗

如果 exp17/18 成功：
- **exp19**: Margin + Curriculum (結合兩者)
- **exp20**: 增加 LoRA rank (64→256)
- **exp21**: Hard Negative Mining

如果失敗：
- 重新檢查梯度衝突分析結果
- 考慮 Full Fine-tuning (移除 LoRA 限制)
