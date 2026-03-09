# 實驗現狀總結

**更新時間**: 2025-11-04 14:59 PM

## 📊 最新結果：Gated Fusion 實驗完成

### ❌ Gated Fusion 實驗結果（失敗）
- **狀態**: 已完成
- **輸出目錄**: `results/zeroshot_gated_20251103_231609/`
- **最終結果**:
  - **Val Acc: 39.08%** (vs Simple Addition 39.29%)
  - **結論**: Gated Fusion **反而變差** -0.21%
  - **原因分析**:
    1. 動態 gate 可能過度複雜化簡單的 fusion 任務
    2. 訓練數據不足以學習有效的 gate 值分佈
    3. Simple Addition 已經是這個數據集上較優的融合方式

### ⚠️ 分析功能的已知問題

**問題**: Speaker Visualization 無法正確區分語者
- **原因**: 緩存數據中的 `content_id` 只是數字（"100", "099"），不包含 speaker 信息
- **影響**: t-SNE 可視化無法正確標註語者編號
- **實際情況**: Zero-Shot 分割本身是正確的（val_speakers: girl9, girl10, boy7, boy8）
- **狀態**: 需要修改預處理腳本在緩存中保存完整的 speaker 信息

**訓練本身不受影響**，只是分析功能無法正確可視化語者分布。

## ✅ 已完成的工作

### 1. GPU 效率優化 ✅
- **問題**: GPU 利用率僅 22-52%
- **解決方案**: 預處理 + 緩存機制
- **成果**: 達成 **23x 加速** (原 115 小時 → 2.5 小時)
- **驗證準確率**: 39.29% (vs Baseline 38.19%, +1.10%)

### 2. 模型容量調整實驗 ✅
- **嘗試**: num_layers=3 (減少模型容量)
- **結果**: Val Acc 38.69% (❌ 變差 -0.6%)
- **結論**: 模型容量不是問題，恢復為 num_layers=4

### 3. 融合策略改進：Gated Fusion ❌
- **目標**: 改善 speaker conditioning 效果
- **核心改進**:
  ```python
  # 舊: Simple Addition
  combined = token_emb + speaker_emb

  # 新: Gated Fusion
  gate = sigmoid(Linear(concat(token, speaker)))
  combined = gate * token_emb + (1 - gate) * speaker_emb
  ```
- **預期效果**: Val Acc 40.5-41.5% (+1.2-2.2%)
- **實際結果**: Val Acc 39.08% (-0.21% vs Simple Addition) ❌
- **結論**: Gated Fusion 在此任務上無效，Simple Addition 是更好的選擇

### 4. 分析功能增強 ✅
已實現以下分析功能（在 `train_zeroshot_full_cached_analysis.py`）:

#### a) Token Prediction Analysis
- **目的**: 檢測模型是否總是預測相同 token
- **指標**:
  - Unique tokens count (應接近 4096)
  - Prediction entropy (越高越好)
  - Top-1 token ratio (應 <50%)
- **警告**: 若 >50% 預測為同一 token，發出警告

#### b) Speaker Embedding Collection
- **功能**:
  - 收集所有 18 位語者的 embeddings
  - 每位語者採樣 20 個樣本
  - 保存為 `.npy` 文件供後續分析

#### c) Speaker Embedding Visualization
- **功能**:
  - 使用 t-SNE 將 256-dim embeddings 降維至 2D
  - 藍色點 = 訓練語者 (14 位)
  - 紅色點 = 驗證語者 (4 位)
  - **每個語者都有編號標註** (G1, B2, S3, ...)
  - 白色框 = train speakers，黃色框 = val speakers
  - 粗體字 = val speakers

- **預期發現**:
  - 若 train/val 語者完全分離 → 零樣本效果好
  - 若有部分重疊 → 模型已學到通用特徵

## 📊 實驗歷史總結

| 實驗配置 | Val Acc | 對比 Baseline | 狀態 |
|---------|---------|--------------|------|
| **Baseline** (Simple Addition, layer=4, no cache) | 38.19% | - | ✅ 完成 |
| **GPU Optimized** (Simple Addition, layer=4, cached) | **39.29%** | +1.10% | ✅ 完成 ⭐ |
| **Reduced Capacity** (Simple Addition, layer=3, cached) | 38.69% | +0.50% | ❌ 變差 |
| **Gated Fusion** (Gated Fusion, layer=4, cached) | 39.08% | +0.89% | ❌ 變差 (-0.21% vs GPU Opt) |

## 🎯 下一步計劃

### 基於實驗結果的建議:

**核心發現**:
- Simple Addition (39.29%) 是目前最好的配置 ⭐
- Gated Fusion (39.08%) 和 num_layers=3 (38.69%) 都變差
- **結論**: 問題不在 fusion 策略或模型容量

**建議方向**:
1. ✅ **使用 Simple Addition + 分析工具運行完整訓練**
   - 啟用 Token Analysis 和 Speaker Visualization
   - 了解模型預測分布和語者分離情況

2. **數據層面改進** (優先):
   - Token Augmentation (aug_prob=0.15)
   - 增加訓練數據多樣性

3. **正則化改進** (次優先):
   - Label Smoothing (label_smoothing=0.1)
   - Increase Dropout to 0.25

4. **暫不考慮**:
   - ❌ 更複雜的 fusion (已證明無效)
   - ❌ 減少模型容量 (已證明無效)
   - ❌ Fine-tune ECAPA (可能破壞預訓練特徵)

## 📁 相關文件

### 代碼
- [train_zeroshot_full_cached_analysis.py](train_zeroshot_full_cached_analysis.py) - 帶分析功能的訓練腳本
- [model_zeroshot_gated.py](model_zeroshot_gated.py) - Gated Fusion 模型
- [train_zeroshot_gated_cached.py](train_zeroshot_gated_cached.py) - Gated Fusion 訓練腳本

### 執行腳本
- [run_zeroshot_full_cached.sh](run_zeroshot_full_cached.sh) - 標準訓練（Simple Addition）
- [run_zeroshot_gated.sh](run_zeroshot_gated.sh) - Gated Fusion 訓練

### 文檔
- [ANALYSIS_FEATURES_EXPLAINED.md](ANALYSIS_FEATURES_EXPLAINED.md) - 分析功能詳細說明
- [GATED_FUSION_EXPLAINED.md](GATED_FUSION_EXPLAINED.md) - Gated Fusion 詳細解釋
- [GENERALIZATION_IMPROVEMENT_RECOMMENDATIONS.md](GENERALIZATION_IMPROVEMENT_RECOMMENDATIONS.md) - 泛化性改進建議
- [FUSION_AND_ECAPA_ANALYSIS.md](FUSION_AND_ECAPA_ANALYSIS.md) - Fusion 策略和 ECAPA Fine-tuning 分析
- [GPU_OPTIMIZATION_COMPLETE.md](GPU_OPTIMIZATION_COMPLETE.md) - GPU 優化完整記錄

### 實驗結果
- `results/zeroshot_full_cached_*/` - 標準訓練結果
- `results/zeroshot_gated_20251103_231609/` - Gated Fusion 訓練結果（運行中）

## 🔬 技術細節

### 模型架構
```
Noisy Tokens (B, T) + Speaker Embedding (B, 256)
→ Token Embedding (B, T, 512) [Frozen Codebook]
→ Speaker Projection (B, 512)
→ Gated Fusion (B, T, 512) ⭐ 核心改進
   gate = sigmoid(Linear(concat(token, speaker)))
   output = gate * token + (1-gate) * speaker
→ Positional Encoding
→ Transformer Encoder (4 layers)
→ Output Projection (B, T, 4096)
```

### 訓練配置
- **Batch Size**: 28
- **Learning Rate**: 1e-4
- **Dropout**: 0.2
- **Optimizer**: AdamW
- **Epochs**: 100
- **數據集**: 16,128 樣本 (78% train, 22% val)
- **語者分佈**: 14 訓練語者, 4 驗證語者 (zero-shot)

### 數據流水線
1. **預處理**: 提取 WavTokenizer tokens + ECAPA embeddings
2. **緩存**: 保存為 `.pt` 文件 (train_cache.pt, val_cache.pt)
3. **訓練**: 從緩存加載，23x 加速

---

**最後更新**: Gated Fusion 訓練運行中 (Epoch 71/100)
