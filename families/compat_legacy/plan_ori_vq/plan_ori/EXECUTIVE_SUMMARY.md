# Plan Original 音質提升：執行摘要

**日期**: 2026-02-15
**當前性能**: Feature MSE 0.0418 (比 V2 高 +13.9%)

---

## 🎯 核心問題

Plan Original 使用**單層 VQ 量化**，無法像 V2 的 4 層殘差 VQ 那樣建模細節。

```
Plan Original (1-Layer):
  encoder → VQ₀ → quantized
  └─ 一次性量化，損失細節

V2 (4-Layer Residual):
  encoder → VQ₀ → q₀
       ↓ res₁ → VQ₁ → q₁
       ↓ res₂ → VQ₂ → q₂
       ↓ res₃ → VQ₃ → q₃
  final = q₀ + q₁ + q₂ + q₃
  └─ 逐層細化，捕獲細節
```

---

## ✅ 推薦解決方案

### **2-Layer Residual VQ** (最佳平衡點)

#### 為什麼選 2 層？

| 層數 | MSE | 提升 % | 訓練時間 | 性價比 | 推薦度 |
|------|-----|--------|----------|--------|--------|
| 1 (當前) | 0.0418 | - | 1.0 天 | - | ⭐⭐⭐ |
| **2 (推薦)** | **0.039** | **-6.7%** | **1.5 天** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** |
| 3 | 0.038 | -9.1% | 2.0 天 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 4 (V2) | 0.0367 | -12.2% | 3.0 天 | ⭐⭐⭐ | ⭐⭐⭐ |

**2 層是最優選擇**：
- ✅ 音質提升明顯（MSE -6.7%）
- ✅ 訓練時間可接受（+50%，仍比 V2 快 2 倍）
- ✅ 保持較高 token diversity（Entropy ~9.2-9.3）
- ✅ 實施簡單，風險低

---

## 📊 預期性能

### 當前 vs 改進後

| 指標 | 當前 (1-Layer) | 改進後 (2-Layer) | V2 (4-Layer) |
|------|----------------|------------------|--------------|
| **Feature MSE** | 0.0418 | **0.039** ⬇️ -6.7% | 0.0367 |
| **Entropy** | 9.45 | 9.2-9.3 | 9.01 |
| **Top-10 mass** | 12.5% | 13-15% | 18.9% |
| **訓練時間** | 1.0 天 | 1.5 天 | 3.0 天 |
| **Token Diversity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**結論**: 2-Layer 方案在音質、效率、diversity 間達到最佳平衡。

---

## 🚀 實施計劃

### Week 1: 實施與驗證（3 天）

**Day 1: 代碼實施**
```bash
# 1. 在 models_single_vq_ema.py 添加
class TwoLayerResidualVQ(nn.Module):
    """2 層殘差 VQ"""
    def __init__(self, ...):
        self.layers = nn.ModuleList([
            SingleVQWithEMA(...) for _ in range(2)
        ])

    def forward(self, z):
        residual = z
        all_quantized = []

        for layer in self.layers:
            out = layer(residual)
            all_quantized.append(out['quantized'])
            residual = residual - out['quantized'].detach()

        return {'quantized': sum(all_quantized), ...}

# 2. 創建訓練腳本
cp families/compat_legacy/plan_ori_vq/plan_ori/train_single_vq_ema.py \
   families/compat_legacy/plan_ori_vq/plan_ori/train_two_layer_vq_ema.py
```

**Day 2: Short-run 驗證**
```bash
# 1000 steps 快速驗證
python families/compat_legacy/plan_ori_vq/plan_ori/train_two_layer_vq_ema.py \
    --mode step \
    --steps 1000 \
    --output_dir families/compat_legacy/plan_ori_vq/runs/plan_ori_2layer_short \
    --eval_interval 200

# 驗收標準 (step 1000):
# ✓ entropy ≥ 5.0
# ✓ top10 ≤ 50%
# ✓ feature_mse ≤ 0.1
```

**Day 3: 啟動 Long-run**
```bash
# 300 epochs 完整訓練
python families/compat_legacy/plan_ori_vq/plan_ori/train_two_layer_vq_ema.py \
    --mode epoch \
    --epochs 300 \
    --output_dir families/compat_legacy/plan_ori_vq/runs/plan_ori_2layer_long \
    --save_checkpoint_every 10 \
    --save_audio_interval 50

# 預期完成時間: 1.5 天
```

### Week 2: 評估與優化（4 天）

**Day 4-5: 結果分析**
- ✅ Long-run 完成（預計 Day 4 晚上）
- ✅ 計算 final metrics
- ✅ 三模式音質評估 (PESQ/STOI)
- ✅ 與 V2 對比分析

**Day 6-7: 可選優化**
- ✅ 如果 MSE > 0.040：調整 loss 權重
- ✅ 如果 MSE ≤ 0.040：嘗試預訓練 Layer 0 進一步提升
- ✅ 最終決策：Plan Ori 2-Layer vs V2

---

## 📝 核心代碼片段

### TwoLayerResidualVQ 類 (簡化版)

```python
class TwoLayerResidualVQ(nn.Module):
    def __init__(self, codebook_size=4096, dim=512, ...):
        super().__init__()
        self.layers = nn.ModuleList([
            SingleVQWithEMA(codebook_size, dim, ...) for _ in range(2)
        ])

    def forward(self, z):
        """
        Args:
            z: [B, 512, T] encoder output

        Returns:
            quantized: [B, 512, T]
            codes: [2, B, 1, T]
            loss_commit: scalar
        """
        residual = z
        all_quantized = []
        all_codes = []
        total_commit_loss = 0.0

        for layer_idx, layer in enumerate(self.layers):
            # 量化當前殘差
            out = layer(residual)
            all_quantized.append(out['quantized'])
            all_codes.append(out['codes'])
            total_commit_loss += out['loss_commit']

            # 計算新殘差 (detach to stop gradient)
            if layer_idx < len(self.layers) - 1:
                residual = residual - out['quantized'].detach()

        # 累加所有層的輸出
        quantized = torch.stack(all_quantized, dim=0).sum(dim=0)
        codes = torch.cat(all_codes, dim=0)  # [2, B, 1, T]

        return {
            'quantized': quantized,
            'codes': codes,
            'loss_commit': total_commit_loss,
        }
```

---

## 🎯 成功標準

### Minimum Viable (必須達成)
- ✅ Feature MSE ≤ 0.042
- ✅ Entropy ≥ 9.0
- ✅ P2 Gate: PASS
- ✅ 訓練時間 ≤ 2 天

### Target (目標)
- ✅ Feature MSE ≤ 0.040
- ✅ Entropy ≥ 9.2
- ✅ PESQ improvement > +1% vs Noisy VQ
- ✅ 訓練時間 ≤ 1.8 天

### Stretch (理想)
- ✅ Feature MSE ≤ 0.038 (接近 V2)
- ✅ Entropy ≥ 9.3 (保持優勢)
- ✅ PESQ improvement > +2% vs Noisy VQ

---

## 💡 關鍵洞察

### 為什麼不是 3 層或 4 層？

**收益遞減定律**:
- 1→2 層: MSE -6.7%, 時間 +50% ⭐⭐⭐⭐⭐
- 2→3 層: MSE -2.4%, 時間 +33% ⭐⭐⭐⭐
- 3→4 層: MSE -3.1%, 時間 +50% ⭐⭐⭐

**2 層已捕獲大部分增益**，3/4 層的額外提升不值得時間成本。

### Plan Original 的獨特優勢

即使增加到 2 層，Plan Original 仍保有：
- ✅ **更好的 token diversity** (Entropy 9.2-9.3 vs V2 9.01)
- ✅ **更快的訓練速度** (1.5 天 vs V2 3 天)
- ✅ **更簡潔的架構** (2 層 vs V2 4 層)

---

## 📊 視覺化結果

已生成以下分析圖表：
1. [Improvement_Strategy.png](Improvement_Strategy.png) - 綜合改進策略
2. [Layer_Analysis.png](Layer_Analysis.png) - 層數選擇分析

---

## ✅ 行動清單

### 立即執行（今天）
- [ ] 閱讀 [QUICK_IMPROVEMENT_GUIDE.md](QUICK_IMPROVEMENT_GUIDE.md)
- [ ] 閱讀 [IMPROVEMENT_STRATEGIES.md](IMPROVEMENT_STRATEGIES.md)
- [ ] 備份 models_single_vq_ema.py
- [ ] 添加 TwoLayerResidualVQ 類
- [ ] 創建 train_two_layer_vq_ema.py

### 本週完成（3 天）
- [ ] Day 1: 代碼實施完成
- [ ] Day 2: Short-run 驗證通過
- [ ] Day 3: 啟動 Long-run 訓練

### 下週完成（4 天）
- [ ] Day 4-5: 評估結果
- [ ] Day 6-7: 優化調整（如需要）
- [ ] 最終決策文檔

---

## 🎯 預期結果

### 保守估計
```
Feature MSE: 0.039-0.040 (比當前降低 5-7%)
Entropy: 9.2-9.3 (保持優勢)
訓練時間: 1.5 天
與 V2 差距: 僅 6% (可接受)
```

### 樂觀估計
```
Feature MSE: 0.037-0.038 (接近 V2)
Entropy: 9.3-9.4 (優於 V2)
訓練時間: 1.5 天
與 V2 差距: <3% (可忽略)
→ Plan Ori 2-Layer 成為主要方案
```

---

## 🚀 最終建議

**立即實施 2-Layer Residual VQ**

**理由**：
1. ✅ 效果明顯（MSE -6.7%）
2. ✅ 成本可控（時間 +50%）
3. ✅ 風險低（基於已驗證的 EMA 機制）
4. ✅ 保持 Plan Original 的核心優勢（diversity + efficiency）

**預期**：
- 音質提升至接近 V2 水平
- 保持訓練效率優勢（仍快 2 倍）
- 成為生產環境的首選方案

---

**創建**: 2026-02-15
**狀態**: 🟢 Ready to Execute
**優先級**: P0 (立即實施)
