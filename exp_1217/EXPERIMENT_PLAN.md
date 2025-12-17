# Exp1217 實驗計劃

## 背景分析

### 之前實驗結果對比

| 實驗 | Loss 配置 | Val Masked Acc | Val Distance | Epochs | 音質 |
|------|-----------|----------------|--------------|--------|------|
| Exp35 | Feature=1.0, Triplet=1.0, CE=0.0 | **15.70%** | **2.6406** | 989 | 較好 |
| Exp36 | Feature=1.0, Triplet=0.0, CE=0.0 | 10.29% | 2.7293 | 100 | 較好 |
| Exp37 | Feature=0.0, Triplet=0.0, CE=1.0 | 13.23% | 3.1231 | 100 | 較差 |

### 觀察

1. **Exp35 (Feature+Triplet)**: 最高準確率 15.70%，最低 distance 2.64
2. **Exp37 (純 CE)**: 準確率 13.23%（100 epochs），但 distance 較高 3.12，音質較差
3. **Exp36 (純 Feature)**: 準確率最低 10.29%，但音質較好

### 假設

1. **CE Loss** 可以直接優化 token 預測，但可能犧牲連續性（音質）
2. **Feature Loss** 保持特徵空間的連續性，有利於音質
3. **Triplet Loss** 似乎有幫助（Exp35 > Exp36）
4. **LoRA rank=128, 18層** 可能限制了模型容量

---

## 實驗設計

### 軸 1: Loss 組合

| ID | Feature | Triplet | CE | 說明 |
|----|---------|---------|-----|------|
| A | 1.0 | 0.0 | 0.5 | Feature 主導 + 輕 CE |
| B | 0.5 | 0.0 | 1.0 | CE 主導 + 輕 Feature |
| C | 1.0 | 0.5 | 0.5 | 三者平衡 |
| D | 0.0 | 0.5 | 1.0 | CE + Triplet (無 Feature) |

### 軸 2: LoRA 配置

目前: 18 層全部 LoRA, rank=128, ~6.8M 參數

| ID | Layers | Rank | 預估參數 | 說明 |
|----|--------|------|----------|------|
| 1 | 18層全部 | 128 | 6.8M | 基準（現狀） |
| 2 | 18層全部 | 256 | 13.6M | 更高容量 |
| 3 | 關鍵8層 | 256 | 6.0M | 選擇性高rank |
| 4 | 關鍵8層 | 512 | 12.1M | 最高容量 |

### 關鍵層選擇 (8層)

```python
CRITICAL_ENCODER_CONV_MODULES = [
    # 輸入投影
    "feature_extractor.encodec.encoder.model.0.conv.conv",
    # 各 downsampling block 的主卷積
    "feature_extractor.encodec.encoder.model.3.conv.conv",
    "feature_extractor.encodec.encoder.model.6.conv.conv",
    "feature_extractor.encodec.encoder.model.9.conv.conv",
    "feature_extractor.encodec.encoder.model.12.conv.conv",
    # Residual block 關鍵層
    "feature_extractor.encodec.encoder.model.1.block.3.conv.conv",
    "feature_extractor.encodec.encoder.model.4.block.3.conv.conv",
    # 輸出投影
    "feature_extractor.encodec.encoder.model.15.conv.conv",
]
```

---

## 實驗矩陣 (Phase 1)

先測試 Loss 組合，使用 LoRA 配置 1（基準）：

| 實驗 | Loss 配置 | LoRA | 說明 |
|------|-----------|------|------|
| Exp40 | F=1.0, T=0.0, CE=0.5 | 18層/r128 | Feature 主導 + 輕 CE |
| Exp41 | F=0.5, T=0.0, CE=1.0 | 18層/r128 | CE 主導 + 輕 Feature |
| Exp42 | F=1.0, T=0.5, CE=0.5 | 18層/r128 | 三者平衡 |
| Exp43 | F=0.0, T=0.5, CE=1.0 | 18層/r128 | CE + Triplet |

## 實驗矩陣 (Phase 2)

基於 Phase 1 最佳 Loss 組合，測試 LoRA 配置：

| 實驗 | Loss 配置 | LoRA | 說明 |
|------|-----------|------|------|
| Exp44 | 最佳組合 | 18層/r256 | 更高容量 |
| Exp45 | 最佳組合 | 8層/r256 | 選擇性高rank |
| Exp46 | 最佳組合 | 8層/r512 | 最高容量 |

---

## 訓練配置

```yaml
# 固定參數
lora_alpha: lora_rank * 2
lora_dropout: 0.2
weight_decay: 0.05
lr: 1e-4
batch_size: 16
num_epochs: 100
warmup_epochs: 10
grad_clip: 1.0
use_amp: true
```

---

## GPU 分配

- GPU 0 (1080Ti): Exp40
- GPU 1 (2080Ti): Exp41
- GPU 2 (2080Ti): Exp42

Exp43 等待空閒 GPU。

---

## 預期結果

1. **Loss 組合**: 預期 Feature+CE 組合能兼顧準確率和音質
2. **LoRA 配置**: 預期更高 rank 能提升性能上限
3. **選擇性 LoRA**: 預期 8 層 high-rank 可能與 18 層 low-rank 效果相當

---

## 成功指標

- Val Masked Accuracy > 15%
- Val Distance < 2.7
- 聽覺音質可接受
