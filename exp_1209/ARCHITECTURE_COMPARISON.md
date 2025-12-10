# WavTokenizer LoRA 實驗架構比較

## 目錄
1. [WavTokenizer 原始架構](#1-wavtokenizer-原始架構)
2. [exp_1207 最初版本](#2-exp_1207-最初版本)
3. [exp_1209 現在版本](#3-exp_1209-現在版本)
4. [完整對比總表](#4-完整對比總表)
5. [結論](#5-結論)

---

## 1. WavTokenizer 原始架構

### 1.1 整體架構
```
Audio (24kHz) → Encoder → VQ Quantizer → Tokens
                  ↓
            SEANet Encoder
            (8.8M params)
```

### 1.2 Encoder 詳細結構 (SEANet)

WavTokenizer 使用 SEANet (Sound Event-Aware Network) 作為 encoder，包含 16 個模組：

| Index | Layer | In Ch | Out Ch | Kernel | Stride | Params |
|-------|-------|-------|--------|--------|--------|--------|
| 0 | SConv1d (Initial) | 1 | 32 | 7 | 1 | 288 |
| 1 | SEANetResnetBlock | 32 | 32 | - | - | 3,232 |
| 2 | ELU | - | - | - | - | - |
| 3 | **SConv1d (↓2x)** | 32 | 64 | 4 | **2** | 8,320 |
| 4 | SEANetResnetBlock | 64 | 64 | - | - | 12,608 |
| 5 | ELU | - | - | - | - | - |
| 6 | **SConv1d (↓4x)** | 64 | 128 | 8 | **4** | 65,792 |
| 7 | SEANetResnetBlock | 128 | 128 | - | - | 49,792 |
| 8 | ELU | - | - | - | - | - |
| 9 | **SConv1d (↓5x)** | 128 | 256 | 10 | **5** | 328,192 |
| 10 | SEANetResnetBlock | 256 | 256 | - | - | 197,888 |
| 11 | ELU | - | - | - | - | - |
| 12 | **SConv1d (↓8x)** | 256 | 512 | 16 | **8** | 2,098,176 |
| 13 | SLSTM | 512 | 512 | - | - | 4,202,496 |
| 14 | ELU | - | - | - | - | - |
| 15 | SConv1d (Final) | 512 | 512 | 7 | 1 | 1,836,032 |

**總下採樣率**: 2 × 4 × 5 × 8 = **320x**
**Encoder 總參數**: **8,802,816** (~8.8M)
**模型總參數**: **80,552,420** (~80.5M)

### 1.3 SEANetResnetBlock 內部結構

每個 ResnetBlock 包含：
```
Input → ELU → Conv(k=3) → ELU → Conv(k=1) → + → Output
  ↓                                         ↑
  └──────────── Shortcut Conv(k=1) ─────────┘
```

對應的 Conv 層：
- `block.1.conv.conv`: 主分支第一個 Conv (壓縮通道)
- `block.3.conv.conv`: 主分支第二個 Conv (恢復通道)
- `shortcut.conv.conv`: 捷徑 Conv

---

## 2. exp_1207 最初版本

### 2.1 LoRA 設定

| 參數 | 值 |
|------|-----|
| **LoRA 層數** | 4 層 |
| **LoRA Rank** | 64 |
| **LoRA Alpha** | 128 |
| **LoRA Dropout** | 0.1 |
| **可訓練參數** | ~200K |

### 2.2 Target Modules

**僅針對 4 個下採樣 Conv 層**：

```python
lora_target_modules = [
    "feature_extractor.encodec.encoder.model.0.conv.conv",   # Initial Conv
    "feature_extractor.encodec.encoder.model.3.conv.conv",   # ↓2x
    "feature_extractor.encodec.encoder.model.6.conv.conv",   # ↓4x
    "feature_extractor.encodec.encoder.model.9.conv.conv",   # ↓5x
]
```

### 2.3 架構圖

```
                    LoRA Applied (4 layers)
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Encoder                                                     │
│  ┌──────┐   ┌─────┐   ┌──────┐   ┌─────┐   ┌──────┐        │
│  │Conv0 │→→→│Res1 │→→→│Conv3 │→→→│Res4 │→→→│Conv6 │→→→...  │
│  │LoRA  │   │     │   │LoRA  │   │     │   │LoRA  │        │
│  └──────┘   └─────┘   └──────┘   └─────┘   └──────┘        │
└─────────────────────────────────────────────────────────────┘

覆蓋率: 4/18 = 22%
```

### 2.4 問題

- **模型容量不足**: 200K 參數無法學習複雜的去噪映射
- **覆蓋範圍有限**: 僅修改下採樣層，忽略 ResBlock 的特徵提取
- **結果**: Token Accuracy 停滯，Feature Loss 無明顯改善

---

## 3. exp_1209 現在版本

### 3.1 方案 A: DenoiseAdapter (exp19, exp20)

#### 設定

| 參數 | 值 |
|------|-----|
| **架構** | MLP + 殘差連接 |
| **Input Dim** | 512 |
| **Hidden Dim** | 256 |
| **Layers** | 2 |
| **Dropout** | 0.1 |
| **可訓練參數** | **263,168** |

#### 架構圖

```
┌─────────────────────────────────────────────────────────────┐
│  Student                                                     │
│  ┌──────────────────────┐   ┌──────────┐   ┌────┐           │
│  │  Encoder (凍結)       │ → │ Adapter  │ → │ VQ │ → tokens  │
│  │  (8.8M, frozen)      │   │ (263K)   │   │    │           │
│  └──────────────────────┘   └──────────┘   └────┘           │
└─────────────────────────────────────────────────────────────┘

Adapter 內部:
  Input → Linear(512→256) → LayerNorm → GELU → Dropout
        → Linear(256→512) → + Input (殘差) → Output
```

#### 結果
- Feature Loss: 0.0384 → 0.0378 (**-1.6%**, 近乎停滯)
- 結論: 模型容量仍然不足

---

### 3.2 方案 B: Expanded LoRA (exp21, exp22)

#### 設定

| 參數 | 值 |
|------|-----|
| **LoRA 層數** | **18 層** |
| **LoRA Rank** | **256** |
| **LoRA Alpha** | **512** |
| **LoRA Dropout** | 0.1 |
| **可訓練參數** | **~3,700,000 (4.4%)** |

#### Target Modules (完整 18 層)

```python
ALL_ENCODER_CONV_MODULES = [
    # Initial Conv
    "feature_extractor.encodec.encoder.model.0.conv.conv",

    # ResBlock 1
    "feature_extractor.encodec.encoder.model.1.block.1.conv.conv",
    "feature_extractor.encodec.encoder.model.1.block.3.conv.conv",
    "feature_extractor.encodec.encoder.model.1.shortcut.conv.conv",

    # Downsample 2x
    "feature_extractor.encodec.encoder.model.3.conv.conv",

    # ResBlock 2
    "feature_extractor.encodec.encoder.model.4.block.1.conv.conv",
    "feature_extractor.encodec.encoder.model.4.block.3.conv.conv",
    "feature_extractor.encodec.encoder.model.4.shortcut.conv.conv",

    # Downsample 4x
    "feature_extractor.encodec.encoder.model.6.conv.conv",

    # ResBlock 3
    "feature_extractor.encodec.encoder.model.7.block.1.conv.conv",
    "feature_extractor.encodec.encoder.model.7.block.3.conv.conv",
    "feature_extractor.encodec.encoder.model.7.shortcut.conv.conv",

    # Downsample 5x
    "feature_extractor.encodec.encoder.model.9.conv.conv",

    # ResBlock 4
    "feature_extractor.encodec.encoder.model.10.block.1.conv.conv",
    "feature_extractor.encodec.encoder.model.10.block.3.conv.conv",
    "feature_extractor.encodec.encoder.model.10.shortcut.conv.conv",

    # Downsample 8x
    "feature_extractor.encodec.encoder.model.12.conv.conv",

    # Final Conv
    "feature_extractor.encodec.encoder.model.15.conv.conv",
]
```

#### 架構圖

```
                    LoRA Applied (ALL 18 Conv layers)
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Encoder                                                                 │
│  ┌──────┐   ┌─────────────┐   ┌──────┐   ┌─────────────┐               │
│  │Conv0 │ → │  ResBlock1  │ → │Conv3 │ → │  ResBlock2  │ → ...         │
│  │LoRA  │   │LoRA×3       │   │LoRA  │   │LoRA×3       │               │
│  └──────┘   └─────────────┘   └──────┘   └─────────────┘               │
│                                                                          │
│  ... → ┌──────┐ → ┌─────┐ → ┌──────┐                                   │
│        │Conv12│   │LSTM │   │Conv15│                                    │
│        │LoRA  │   │(凍結)│   │LoRA  │                                    │
│        └──────┘   └─────┘   └──────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘

覆蓋率: 18/18 Conv = 100% (僅 LSTM 未修改)
```

#### 結果
- Feature Loss: 0.0288 → 0.0185 (**-35.8%**, 持續下降)
- 結論: **模型容量足夠，能夠學習去噪映射**

---

## 4. 完整對比總表

### 4.1 LoRA 設定對比

| 版本 | 層數 | Rank | Alpha | 可訓練參數 | 佔比 |
|------|------|------|-------|-----------|------|
| **exp_1207** | 4 | 64 | 128 | ~200K | 0.25% |
| **exp_1209 Adapter** | - | - | - | 263K | 0.33% |
| **exp_1209 LoRA** | **18** | **256** | **512** | **3.7M** | **4.4%** |

### 4.2 Target Modules 對比

| 層類型 | exp_1207 | exp_1209 LoRA |
|--------|----------|---------------|
| Initial Conv (model.0) | ✅ | ✅ |
| ResBlock 1 (model.1) | ❌ | ✅ ×3 |
| Downsample 2x (model.3) | ✅ | ✅ |
| ResBlock 2 (model.4) | ❌ | ✅ ×3 |
| Downsample 4x (model.6) | ✅ | ✅ |
| ResBlock 3 (model.7) | ❌ | ✅ ×3 |
| Downsample 5x (model.9) | ✅ | ✅ |
| ResBlock 4 (model.10) | ❌ | ✅ ×3 |
| Downsample 8x (model.12) | ❌ | ✅ |
| LSTM (model.13) | ❌ | ❌ |
| Final Conv (model.15) | ❌ | ✅ |
| **總計** | **4/18** | **18/18** |

### 4.3 實驗結果對比

| 實驗 | 方法 | 參數量 | Feature Loss 改善 | 狀態 |
|------|------|--------|------------------|------|
| exp_1207 | LoRA (4層) | 200K | 停滯 | ❌ 失敗 |
| exp19 | Adapter | 263K | -1.6% | ❌ 停滯 |
| exp21 | LoRA (18層) | 3.7M | **-35.8%** | ✅ 成功 |
| exp22 | LoRA (18層) + CE | 3.7M | 進行中 | 🔄 |

---

## 5. 結論

### 5.1 關鍵發現

1. **模型容量是決定性因素**
   - 200K-263K 參數：無法學習去噪映射
   - 3.7M 參數：能夠持續降低 Feature Loss

2. **覆蓋範圍很重要**
   - 僅修改下採樣層不夠
   - ResBlock 中的特徵提取同樣需要適應

3. **LoRA Rank 的影響**
   - rank=64 太小，無法捕捉去噪的複雜模式
   - rank=256 提供足夠的表達能力

### 5.2 最佳配置

```python
# 推薦的 LoRA 配置
LoraConfig(
    r=256,                              # 高 rank
    lora_alpha=512,                     # alpha = 2 * rank
    target_modules=ALL_18_CONV_LAYERS,  # 覆蓋全部 Conv
    lora_dropout=0.1,
    bias="none",
)
```

### 5.3 下一步

- exp21: 繼續訓練，觀察 Feature Loss 是否持續下降
- exp22: 加入 CE Loss，測試是否能提升 Token Accuracy
- 考慮: 是否需要對 LSTM 層也加入 LoRA

---

*文檔生成日期: 2024-12-09*
