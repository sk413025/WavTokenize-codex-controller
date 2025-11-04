# Zero-Shot Speaker Denoising 實驗總結

## 📊 已完成實驗

### 實驗 1: Simple Addition + Layer=4 ✅
**配置:**
- num_layers: 4
- fusion: Simple Addition
- dropout: 0.2
- batch_size: 28

**結果:**
- 最佳 Val Acc: **39.29%**
- 相比 Baseline (38.19%): **+1.10%**
- 泛化差距: ~16%
- 訓練時間: 2.5 小時

**評價:** 超越 baseline，但提升有限

---

### 實驗 2: Simple Addition + Layer=3 ❌
**配置:**
- num_layers: 3 (從 4 降低)
- fusion: Simple Addition
- dropout: 0.2
- batch_size: 28

**結果:**
- 最佳 Val Acc: **38.69%**
- 相比 Baseline (38.19%): **+0.50%**
- 相比 Layer=4 (39.29%): **-0.60%** (變差)
- 訓練時間: 1.8 小時

**評價:** 模型容量不是問題，不應該縮小

**關鍵洞察:**
- ❌ 降低模型容量（num_layers=3）反而變差
- ✅ 說明 num_layers=4 是合適的
- ✅ 問題不在模型容量，而在 fusion 策略

---

## 🚀 準備執行實驗

### 實驗 3: Gated Fusion + Layer=4 ⭐ (準備執行)

**配置:**
- num_layers: 4 (恢復到最佳)
- fusion: **Gated Fusion** (核心改進)
- dropout: 0.2
- batch_size: 28
- learning_rate: 1e-4

**改進點:**
- ✅ 使用動態門控機制融合 token 和 speaker
- ✅ 自動學習每個位置的融合權重
- ✅ 只增加 0.53M 參數 (3.6%)
- ✅ 訓練時間基本不變 (~2.5 小時)

**預期效果:**
- **保守估計**: Val Acc 40.0-40.5% (+0.7-1.2%)
- **樂觀估計**: Val Acc 40.5-41.5% (+1.2-2.2%)

**執行命令:**
```bash
cd /home/sbplab/ruizi/c_code/done/exp
bash run_zeroshot_gated.sh
```

**預計完成時間**: 今天 18:00

---

## 📈 實驗對比表

| 實驗 | Fusion | Layers | Val Acc | vs Baseline | vs Best | 狀態 |
|------|--------|--------|---------|-------------|---------|------|
| Baseline | - | - | 38.19% | - | -1.10% | 基準 |
| **Simple Add (L4)** | Addition | 4 | **39.29%** | +1.10% | - | ✅ 當前最佳 |
| Simple Add (L3) | Addition | 3 | 38.69% | +0.50% | -0.60% | ❌ 變差 |
| **Gated (L4)** | Gated | 4 | ? | ? | ? | ⏳ 準備中 |

---

## 🎯 Gated Fusion 詳解

### 核心概念

**當前方法 (Simple Addition):**
```python
combined = token_emb + speaker_emb  # 固定 50:50 權重
```

**改進方法 (Gated Fusion):**
```python
gate = sigmoid(MLP(concat(token, speaker)))  # 學習權重 [0,1]
combined = gate * token_emb + (1-gate) * speaker_emb  # 動態融合
```

### 為什麼更好？

想像降噪一句話：
- **靜音段**: speaker 信息更重要 → gate ≈ 0.2 (20% token + 80% speaker)
- **語音段**: token 內容更重要 → gate ≈ 0.8 (80% token + 20% speaker)

**Gated Fusion 自動學習**每個位置該用多少比例！

### 技術細節

```python
class GatedFusion(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # 1024 → 512
            nn.Dropout(dropout),
            nn.Sigmoid()  # [0, 1]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, token_emb, speaker_emb):
        concat = torch.cat([token_emb, speaker_emb], dim=-1)
        gate = self.gate_network(concat)
        fused = gate * token_emb + (1 - gate) * speaker_emb
        return self.layer_norm(fused)
```

**參數量:**
- Gated Fusion: 525K 參數 (0.53M)
- 相比原版 14.8M: 增加 3.6%
- 非常輕量級！

---

## 🔍 為什麼 num_layers=3 失敗了？

### 原始假設
- 16K 樣本 vs 14.8M 參數 → 參數/樣本比太高
- 降低模型容量應該減少過擬合
- 預期提升泛化能力

### 實際結果
- Val Acc 從 39.29% 降到 38.69% (❌ 變差)
- 說明模型容量**不是問題**

### 可能原因

1. **任務複雜度高**
   - Token denoising 是 4096-way 分類
   - 每個 token 需要從 4096 個選項中選一個
   - 需要足夠的模型容量

2. **Transformer 層數需求**
   - 4 層 Transformer 對於這個任務可能是最小需求
   - 3 層不足以學習複雜的 token 轉換

3. **Speaker conditioning 需要容量**
   - 模型需要同時處理 token 信息和 speaker 信息
   - 縮小容量可能削弱了這種能力

### 結論
- ✅ num_layers=4 是合適的
- ❌ 不應該縮小模型容量
- ✅ 應該改進 fusion 策略 (Gated Fusion)

---

## 📝 實驗記錄

### 2025-11-03 10:48 - 13:22
**實驗**: Simple Addition + Layer=4
**結果**: Val Acc 39.29%
**結論**: 超越 baseline，但提升有限

### 2025-11-03 13:26 - 15:26
**實驗**: Simple Addition + Layer=3
**結果**: Val Acc 38.69%
**結論**: 變差，模型容量不是問題

### 2025-11-03 15:40 - ?
**實驗**: Gated Fusion + Layer=4
**結果**: 待執行
**預期**: Val Acc 40.5-41.5%

---

## 🎓 學到的經驗

### 1. 模型容量不是越小越好
- 盲目縮小模型不一定改善泛化
- 需要根據任務複雜度選擇合適的容量

### 2. 改進方向應該基於分析
- num_layers=4 → 3 失敗
- 說明問題不在模型大小
- 而在於 fusion 機制不夠好

### 3. 輕量級改進可能更有效
- Gated Fusion 只增加 3.6% 參數
- 但可能帶來更大的提升
- 「更聰明」比「更小」更重要

---

## 🔜 後續實驗規劃

### 如果 Gated Fusion 成功 (Val Acc ≥ 40.5%)
繼續疊加改進：
1. **Label Smoothing** (label_smoothing=0.1)
   - 預期提升 0.5-1%
2. **Token Augmentation** (aug_prob=0.15)
   - 預期提升 1-2%
3. **Speaker Adapter** (bottleneck_dim=64)
   - 預期提升 0.5-1%

**最終目標**: Val Acc ≥ 43% (+4.81% vs Baseline)

### 如果 Gated Fusion 部分成功 (Val Acc 39.8-40.5%)
分析並調整：
1. 分析 gate 值分布
2. 嘗試不同的 gate network 架構
3. 考慮 Cross-Attention Fusion

### 如果 Gated Fusion 失敗 (Val Acc < 39.8%)
嘗試其他 fusion 策略：
1. **Cross-Attention Fusion**
2. **FiLM (Feature-wise Linear Modulation)**
3. **Concatenation + MLP**

---

## 📚 相關文件

- [model_zeroshot_gated.py](model_zeroshot_gated.py) - Gated Fusion 模型實現
- [train_zeroshot_gated_cached.py](train_zeroshot_gated_cached.py) - 訓練腳本
- [run_zeroshot_gated.sh](run_zeroshot_gated.sh) - 執行腳本
- [GATED_FUSION_EXPLAINED.md](GATED_FUSION_EXPLAINED.md) - 詳細解釋
- [FUSION_AND_ECAPA_ANALYSIS.md](FUSION_AND_ECAPA_ANALYSIS.md) - 改進方案分析
- [NEXT_EXPERIMENT_PLAN.md](NEXT_EXPERIMENT_PLAN.md) - 實驗規劃

---

## ✅ 準備開始實驗

所有準備工作已完成：
- ✅ Gated Fusion 模型已實現並測試
- ✅ 訓練腳本已創建
- ✅ 執行腳本已準備
- ✅ 緩存數據已就緒

**立即開始:**
```bash
cd /home/sbplab/ruizi/c_code/done/exp
bash run_zeroshot_gated.sh
```

預計 2.5 小時後（~18:00）可以看到結果！

---

生成時間: 2025-11-03 15:45
當前狀態: 準備執行 Gated Fusion 實驗
預期完成: 2025-11-03 18:00
