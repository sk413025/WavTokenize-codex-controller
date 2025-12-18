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

## Phase 1 實驗結果

### 結果總覽

| 實驗 | Loss 配置 | Val Masked Acc | Val Distance | Epochs | 特徵 |
|------|-----------|----------------|--------------|--------|------|
| **Exp35** | F=1.0, T=1.0, CE=0.0 | **15.70%** | **2.6406** | 989 | 基準 (無 CE) |
| Exp40 | F=1.0, T=0.0, CE=0.5 | 10.78% | 2.6701 | 100 | Feature 主導 |
| Exp41 | F=0.5, T=0.0, CE=1.0 | 10.70% | 2.6584 | 100 | CE 主導 |
| Exp42 | F=1.0, T=0.5, CE=0.5 | 11.05% | **2.6397** | 100 | 三者平衡 |
| Exp43 | F=0.0, T=0.5, CE=1.0 | **13.25%** | 3.0803 | 100 | CE + Triplet |
| Exp47 | F=0.0, T=1.0, CE=0.0 | 0.07% | 7.08 | 13 | 純 Triplet (失敗) |

### 詳細分析

#### 1. Exp40 (Feature 主導 + 輕 CE: F=1.0, T=0.0, CE=0.5)

- **Final Val Masked Acc**: 10.78%
- **Final Val Distance**: 2.6701
- **觀察**:
  - Feature Loss 從 0.035 下降到 0.053，訓練穩定
  - Distance 收斂良好 (2.67)
  - 準確率提升緩慢，100 epoch 只達 10.78%
- **結論**: Feature 主導但無 Triplet，準確率受限

#### 2. Exp41 (CE 主導 + 輕 Feature: F=0.5, T=0.0, CE=1.0)

- **Final Val Masked Acc**: 10.70%
- **Final Val Distance**: 2.6584
- **觀察**:
  - CE 主導但 Feature Loss 仍在下降，兩者能共存
  - Distance 收斂與 Exp40 相近
  - 準確率與 Exp40 幾乎相同
- **結論**: CE 權重高但無 Triplet，效果有限

#### 3. Exp42 (三者平衡: F=1.0, T=0.5, CE=0.5)

- **Final Val Masked Acc**: 11.05%
- **Final Val Distance**: 2.6397 (最佳)
- **觀察**:
  - 三種 Loss 協同工作，訓練最穩定
  - Distance 達到最低 2.64，與 Exp35 持平
  - 準確率略高於 Exp40/41
- **結論**: 平衡配置保持最佳特徵品質，但準確率仍受限

#### 4. Exp43 (CE + Triplet: F=0.0, T=0.5, CE=1.0)

- **Final Val Masked Acc**: 13.25% (Phase 1 最高)
- **Final Val Distance**: 3.0803 (最差)
- **觀察**:
  - 準確率明顯高於其他 Phase 1 實驗
  - 但 Distance 大幅上升 (3.08 vs 2.64)
  - 無 Feature Loss 導致特徵空間不連續
- **結論**: CE + Triplet 能提升準確率，但犧牲特徵品質

#### 5. Exp47 (純 Triplet: F=0.0, T=1.0, CE=0.0) ❌ 失敗

- **Final Val Masked Acc**: 0.07% (幾乎為零)
- **Final Val Distance**: 7.08 (極差)
- **觀察**:
  - 訓練出現 NaN，僅跑 13 epochs 就停止
  - Accuracy 從 0.4% 急速下降到 0.07%，模型 collapse
  - Distance 暴漲到 7-8，特徵空間完全崩潰
  - **純 Triplet Loss 無法穩定訓練**
- **結論**: Triplet Loss 單獨使用會導致 model collapse，必須配合 Feature 或 CE Loss

### 與 Exp35 比較

#### 公平比較 (同樣 100 epochs)

| 實驗 | Loss 配置 | Val Masked Acc | Val Distance | 說明 |
|------|-----------|----------------|--------------|------|
| **Exp35 @100ep** | F=1.0, T=1.0, CE=0.0 | **13.20%** | **2.6070** | Feature + Triplet |
| Exp42 | F=1.0, T=0.5, CE=0.5 | 11.05% | 2.6397 | 三者平衡 |
| Exp43 | F=0.0, T=0.5, CE=1.0 | **13.25%** | 3.0803 | CE + Triplet |

**發現**: 在 100 epochs 下，Exp43 (13.25%) 略高於 Exp35 (13.20%)，但 Exp35 的 Distance (2.61) 遠優於 Exp43 (3.08)。

#### 最終結果比較

| 指標 | Exp35 (989ep) | Phase 1 最佳 (100ep) | 差距 | 說明 |
|------|---------------|---------------------|------|------|
| Val Masked Acc | 15.70% | 13.25% (Exp43) | -2.45% | 長訓練效果顯著 |
| Val Distance | 2.6406 | 2.6397 (Exp42) | +0.0009 | 持平 |

### 關鍵發現

1. **Triplet Loss 是關鍵但不能單獨使用**:
   - Exp40/41 無 Triplet: ~10.7% acc
   - Exp42/43 有 Triplet: 11.05%/13.25% acc
   - Exp35 Triplet=1.0 + Feature=1.0: 15.70% acc
   - **Exp47 純 Triplet: 0.07% acc (collapse!)**
   - **Triplet Loss 對準確率貢獻最大，但必須配合其他 Loss 穩定訓練**

2. **Feature Loss 保持品質並穩定訓練**:
   - 有 Feature (Exp40-42): Distance ~2.64-2.67
   - 無 Feature (Exp43): Distance 3.08
   - 無 Feature 且無 CE (Exp47): Distance 7.08 + NaN
   - **Feature Loss 對特徵連續性和訓練穩定性至關重要**

3. **CE Loss 效果有限**:
   - 增加 CE 未顯著提升準確率
   - 反而可能干擾 Feature/Triplet 優化
   - 但 CE 可防止 Triplet Loss 的 collapse (Exp43 vs Exp47)

4. **Epoch 數量重要**:
   - Exp35 跑 989 epochs，Phase 1 只跑 100
   - 更長訓練時間可能進一步提升

### Phase 1 結論

- **Exp35 仍是最佳配置**: F=1.0, T=1.0, CE=0.0
  - 100 epochs: 13.20% acc + 2.61 distance（準確率與品質皆最佳）
  - 989 epochs: 15.70% acc + 2.64 distance
- **CE Loss 無明顯幫助**:
  - Exp43 準確率 13.25% 略高，但 Distance 3.08 大幅惡化
  - Exp42 加入 CE 後，準確率反而下降 (11.05% vs 13.20%)
  - 加入 CE 並未帶來預期的提升
- **Phase 2 配置選擇**: F=1.0, T=0.5, CE=0.5 (Exp42)
  - 注意: Exp35 配置更優，但 Phase 2 目的是測試 LoRA 容量
  - 使用 Exp42 配置保持與 Phase 1 的可比性
  - 若 Phase 2 效果不佳，可改用 Exp35 配置重測

---

## Phase 2 實驗設計 (更新)

基於 Phase 1 結論，Phase 2 使用平衡配置 (F=1.0, T=0.5, CE=0.5)：

| 實驗 | Loss 配置 | LoRA | 參數量 | 狀態 |
|------|-----------|------|--------|------|
| Exp44 | F=1.0, T=0.5, CE=0.5 | 18層/r256 | ~13.6M | 運行中 |
| Exp45 | F=1.0, T=0.5, CE=0.5 | 8層/r256 | ~6.0M | 運行中 |
| Exp46 | F=1.0, T=0.5, CE=0.5 | 8層/r512 | ~12.1M | 待定 |

### Phase 2 目標

1. 測試更高 LoRA rank 是否能突破 15% 準確率
2. 比較 18 層低 rank vs 8 層高 rank 的效果
3. 驗證關鍵層選擇策略的有效性

---

## 預期結果

1. **Loss 組合**: ✅ 已驗證 Feature+Triplet 組合最有效
2. **LoRA 配置**: 預期更高 rank 能提升性能上限
3. **選擇性 LoRA**: 預期 8 層 high-rank 可能與 18 層 low-rank 效果相當

---

## 成功指標

- Val Masked Accuracy > 15%
- Val Distance < 2.7
- 聽覺音質可接受

---

## 更新日誌

- **2024-12-18**: Phase 1 完成，添加結果分析，啟動 Phase 2 (Exp44, Exp45)
