# Steering 研究分析報告

> **核心結論**：純 Steering（VQ 前/後）都缺乏泛化性，但 Token 分析可以引導 LoRA 訓練

## 1. 研究背景

### 1.1 目標
探索在 WavTokenizer 框架下，是否能透過 **Steering** 技術改善 noisy 音訊的 token 品質，進而提升降噪效果。

### 1.2 WavTokenizer 架構
```
Audio → Encoder (16層) → VQ (4096 codebook) → Decoder → Reconstructed Audio
```

---

## 2. Steering 方法一：VQ 前 Activation Steering

### 2.1 原理

在 Encoder 的中間層加入 steering vector：
```
h' = h + α × v
```
其中 `v = E[h_clean - h_noisy]` 是 clean/noisy activation 差異的平均方向。

### 2.2 實驗配置

| 參數 | 說明 |
|------|------|
| Steering 層 | L3, L6 (Downsample) 或 L3, L4, L6 |
| 方法 | Global Mean / SNR-binned / Linear PCA |
| alpha_by_layer | 各層權重，如 `3:+0.1, 6:-0.05` |

### 2.3 實驗結果

#### 小樣本 (n=200~256)
| 配置 | SI-SDR 增益 |
|------|------------|
| L36 s=1.5 | +0.14 dB |
| L356 s=1 | +0.14 dB |

#### 大樣本驗證 (n=1000)
| 配置 | SI-SDR 增益 | STOI |
|------|------------|------|
| Baseline (alpha=0) | -14.223 dB | 0.5008 |
| L356 s=1 | **+0.045 dB** | 0.5012 |
| L36 s=1.5 | **+0.010 dB** | 0.5014 |

### 2.4 結論

**效果極為有限**：
- 大樣本驗證僅 +0.05 dB
- 早期 +0.14 dB 包含採樣變異
- Activation alignment 分析顯示 steering 後與 teacher(clean) 的 alignment 幾乎沒改善

### 2.5 效果有限的原因分析

| 原因 | 說明 |
|------|------|
| **非線性交互** | 噪音與信號的交互是非線性的，固定向量加法無法處理 |
| **全局平均** | 忽略 input-dependent 變異（不同頻率、時間段、SNR）|
| **早期資訊丟失** | 淺層已丟失的資訊無法在深層恢復 |
| **VQ 敏感度** | 離散化放大微小差異，decision boundary 問題 |
| **層數不足** | 只在 2-3 層 steering，覆蓋範圍不夠 |

---

## 3. 與 LoRA 方法的比較

### 3.1 LoRA 實驗發現

| 配置 | 結果 |
|------|------|
| 淺層 LoRA (L0-L4, 5層) | ❌ 效果差，增加參數量幾乎無改善 |
| 18 層全層 LoRA | ✅ Feature Loss -40%，但仍有問題 |
| LoRA exp48 | Token proxy 改善，但 SI-SDR 崩潰 -9.71 dB |

### 3.2 關鍵差異

| 特性 | Linear Steering | LoRA |
|------|----------------|------|
| 覆蓋層數 | 2-3 層 (~15%) | 全部 18 層 |
| 修改方式 | 固定向量加法 | 學習低秩權重更新 |
| 適應性 | 無 (固定 v) | 有 (input-dependent) |
| 需要訓練 | ❌ 不需要 | ✅ 需要 |

### 3.3 核心問題

> **Token Matching ≠ Audio Quality**
> **Activation Alignment ≠ Audio Quality**

兩種方法都沒有直接優化 audio quality，間接指標與最終目標不對齊。

---

## 4. Steering 方法二：VQ 後 Token Steering

### 4.1 動機

VQ 前 steering 受限於：
1. Decision boundary 敏感度
2. 連續空間微調被離散化抵消

**新思路**：直接在 token 空間操作，分析 `noisy_token → clean_token` 的映射規律。

### 4.2 Token 轉換分析方法

```python
# 對每個 (noisy_audio, clean_audio) pair
token_noisy = VQ(Encoder(noisy_audio))  # [T]
token_clean = VQ(Encoder(clean_audio))  # [T]

# Position-wise 比較
for pos in range(T):
    if token_noisy[pos] != token_clean[pos]:
        記錄轉換: (token_noisy[pos] → token_clean[pos])
```

### 4.3 數據集資訊

```
val_cache: 2592 樣本
- Speakers: boy7, boy8, girl9
- Sentence IDs: 288 個不同句子
- Noise Types: box, papercup, plastic, clean
- 每個 (speaker, sentence_id) 有 2-4 個樣本 (不同噪音類型)
```

### 4.4 實驗結果

#### 4.4.1 基本統計

| 指標 | 數值 |
|------|------|
| 總 Token 數 | 914,688 |
| Match Rate | 45.95% |
| Unique 轉換數 | 233,253 |

#### 4.4.2 按噪音類型分析

| Noise Type | Match Rate | 說明 |
|------------|------------|------|
| **clean** | **100.0%** | 基準（無噪音時完全一致）|
| papercup | 21.4% | |
| box | 20.2% | |
| plastic | 14.0% | 最嚴重 |

#### 4.4.3 Clean Token 一致性驗證

**關鍵發現**：同一人說同一句話，Clean Token **100% 一致**

```
boy7 說句子 '100' (4 個樣本，不同噪音類型)
Clean token 一致率: 50/50 (100.0%)
```

這證明 clean token 是穩定的目標，問題在於 noisy token 的偏移。

#### 4.4.4 最細粒度分析：(Speaker, Sentence_ID, Noise_Type)

| 指標 | 數值 |
|------|------|
| 組合數 | 2,592 |
| Match Rate Mean | 46.6% |
| Match Rate Std | 39.0% |
| **Top-1 可修正率 Mean** | **61.4%** |
| Top-1 可修正率 Max | 85.6% |

#### 4.4.5 高可修正率範例

| 組合 | Top-1 可修正率 | Mismatch 數 |
|------|---------------|-------------|
| (girl9, 026, papercup) | 85.6% | 187 |
| (girl9, 026, box) | 84.6% | 188 |
| (girl9, 288, papercup) | 84.4% | 167 |
| (boy8, 224, box) | 84.2% | 203 |

### 4.5 Top-1 可修正率的意義

```
定義：在所有 mismatch token 中，
      如果用「該 noisy token 最常對應的 clean token」來替換，
      能正確修正的比例。

結果：
- 平均 61.4% 的錯誤 token 可以用簡單查表修正
- 某些組合高達 85%
```

---

## 5. 結論與建議

### 5.1 VQ 前 Steering

| 結論 | 說明 |
|------|------|
| ❌ 效果有限 | 大樣本驗證僅 +0.05 dB |
| ❌ 根本限制 | 固定向量無法適應 input-dependent 變異 |
| ❌ 不建議繼續 | 除非改為可訓練的 steering |

### 5.2 VQ 後 Token Steering

| 結論 | 說明 |
|------|------|
| ✅ 有潛力 | Top-1 可修正率達 61.4% |
| ✅ 規律性存在 | 在特定條件下轉換模式相對穩定 |
| ⚠️ 需要條件資訊 | 需要知道 (speaker, sentence, noise_type) |

### 5.3 未來方向

#### 方向 A：Transition Matrix 方法
```python
# 學習 (noise_type) → Transition Matrix
P[token_clean | token_noisy, noise_type]

# 推理時
token_corrected = argmax P[:, token_noisy]
```

優點：不需要訓練神經網路，純統計方法

#### 方向 B：Context-aware Token Correction
```python
# 考慮前後 token 的 context
token_clean = SmallTransformer(token_noisy_sequence, noise_type)
```

優點：可以學習更複雜的修正模式

#### 方向 C：結合 LoRA + Token Steering
```python
# 先用 LoRA 改善 encoder 表現
# 再用 token steering 做後處理
```

---

## 6. 技術細節

### 6.1 Steering 實現方式

**不是** 使用 PyTorch `register_forward_hook`，而是 **手動遍歷 encoder layers**：

```python
@torch.no_grad()
def encode_with_intermediates(wavtok, audio, layers, steering_vectors, alpha):
    encoder = wavtok.feature_extractor.encodec.encoder
    x = audio

    for i, layer in enumerate(encoder.model):
        x = layer(x)  # 逐層 forward

        # Steering: 加入方向向量
        if alpha != 0.0 and i in steering_vectors:
            v = steering_vectors[i]
            x = x + alpha * v

    return x
```

### 6.2 Encoder 結構

```
model[0]:  SConv1d      (Input Conv)
model[1]:  ResBlock1
model[2]:  ELU
model[3]:  SConv1d      (Downsample 1) ← Steering 點
model[4]:  ResBlock2    ← Steering 點 (噪音敏感度最高 0.80)
model[5]:  ELU
model[6]:  SConv1d      (Downsample 2) ← Steering 點
...
model[15]: SConv1d      (Output Conv)
```

### 6.3 診斷腳本

- `diagnostic_token_transition.py`: 全局 token 轉換分析
- `diagnostic_token_transition_v2.py`: 按語者/句子分層分析（使用錯誤的欄位）
- `diagnostic_token_transition_v3.py`: 從 filename 解析正確資訊的完整分析

---

## 7. 相關 Commits (2026-01-19)

| Commit | 說明 |
|--------|------|
| `fd9328a` | n=1000 大樣本驗證 |
| `dd3b154` | Metrics 相關性分析 |
| `2b1523c` | L356 Activation Alignment 評估 |
| `808d33b` | LoRA vs PCA Steering 比較 |
| `7a0a4e6` | Token-audit LoRA exp48 分析 |
| `a900e75` | alpha_by_layer L356 優化 |

---

---

## 8. 泛化性深度分析

### 8.1 Steering vs LoRA 的本質差異

| 特性 | Steering (VQ 前/後) | LoRA |
|------|---------------------|------|
| **本質** | Inference-time 干預 | Fine-tuning |
| **是否訓練** | ❌ 不訓練模型 | ✅ 訓練低秩矩陣 |
| **修正方式** | 固定向量/查表（對所有 input 相同）| Input-dependent（不同 input 不同修正）|
| **泛化性** | ❌ 無 | ✅ 有（依賴訓練資料多樣性）|

### 8.2 VQ 前 Steering 的泛化性

```python
# VQ 前 Steering
v = mean(h_clean - h_noisy)  # 跨所有樣本計算
h_new = h + α * v            # 對所有 input 加同樣的 v
```

**理論上可泛化**：因為 v 是跨樣本計算的平均值
**實際上沒用**：效果只有 +0.05 dB，泛化了也沒意義

### 8.3 VQ 後 Token Steering 的泛化性

| 統計粒度 | Top-1 可修正率 | 泛化性 |
|---------|---------------|--------|
| (speaker, sentence, noise_type) | **61.4%** | ❌ 完全無法泛化 |
| 只按 noise_type | **3.0%** | ❌ 理論上可泛化，但效果太差 |

**原因**：Token 轉換高度依賴具體內容（誰說什麼），沒有通用規律。

### 8.4 LoRA 為什麼可以泛化？

```python
# LoRA
W_new = W + ΔW  # ΔW = B @ A (低秩矩陣)
h_new = W_new @ x  # 不同 x 產生不同 h_new
```

**關鍵**：LoRA 學習的是「如何根據 input 做修正」，不是死記「特定 input 對應什麼輸出」。

---

## 9. Token 分析 + LoRA：結合兩者優勢

### 9.1 核心思路

```
Token 分析的價值：找出「問題在哪」
  → 哪些 token 容易錯？
  → 哪種噪音最難處理？
  → 哪些層最敏感？

LoRA 的價值：有泛化性地「解決問題」
  → 學習 input-dependent 的修正
  → 可以處理沒見過的樣本

結合：用 Token 分析引導 LoRA 訓練
  → 針對易錯 token 設計加權 loss
  → 針對敏感層集中 LoRA 參數
  → 針對難噪音類型調整訓練策略
```

### 9.2 實驗設計 (exp_0121_token_guided)

#### Exp A: Token-Weighted Loss
```python
# 對易錯 token 給更高的 loss 權重
token_error_rate = load("token_error_rates.pt")
weight = 2.0 if error_rate > 0.7 else 1.0
loss = weighted_mse(pred, target, weights=weight)
```

#### Exp B: Noise-Type Aware Training
```python
# 對難噪音類型給更高的訓練權重
noise_weights = {
    'plastic': 2.0,   # 最難 (error_rate=0.86)
    'box': 1.5,
    'papercup': 1.2,
}
```

#### Exp C: Layer-Selective LoRA
```python
# 只在敏感層加 LoRA
# model[4] ResBlock2: 噪音敏感度 0.80
# model[6] Downsample2: 噪音敏感度 0.79
sensitive_layers = ["4.block.1", "4.block.3", "6.conv"]
```

### 9.3 與之前 LoRA 實驗的差異

| 實驗 | 方法 | 設計依據 |
|------|------|---------|
| exp_0112 | 全層 LoRA | 直覺 |
| exp_0112_intermediate | 中間層監督 | 架構分析 |
| **exp_0121** | Token-Guided LoRA | **數據驅動** |

---

## 10. 總結與建議

### 10.1 方法比較

| 方法 | 需要訓練 | 泛化性 | 效果 |
|------|---------|--------|------|
| VQ 前 Steering | ❌ | ⚠️ 有限 | ❌ 很弱 (+0.05 dB) |
| VQ 後 Token Steering | ❌ | ❌ 無 | ⚠️ 有限 (61%) |
| 純 LoRA | ✅ | ✅ 有 | ⚠️ Token↑ Audio↓ |
| **Token-Guided LoRA** | ✅ | ✅ 有 | 📊 待驗證 |

### 10.2 關鍵洞察

1. **純 Steering 的限制是根本性的**：固定向量/查表無法適應不同 input
2. **Token 分析的價值是診斷性的**：告訴我們「問題在哪」，但不能直接解決
3. **LoRA 是必要的**：要泛化就必須訓練，學習 input-dependent 的修正
4. **Token 分析 + LoRA 是有希望的方向**：用診斷結果引導訓練

### 10.3 下一步

1. 執行 `exp_0121_token_guided` 的三個實驗
2. 比較 Token-Guided LoRA vs 普通 LoRA
3. 檢驗是否能同時改善 Token accuracy 和 Audio quality

---

## 附錄 A：檔案命名格式

```
nor_girl9_box_LDV_100.wav

nor     - 無意義前綴
girl9   - speaker_id
box     - noise_type (box/papercup/plastic/clean)
LDV     - 無意義
100     - sentence_id
```

## 附錄 B：實驗檔案結構

```
exp_0121_token_guided/
├── README.md                    # 實驗說明
├── analyze_error_tokens.py      # Token 分析腳本
├── losses.py                    # Token-Guided Loss
├── run_exp_a.sh                 # Token-Weighted Loss
├── run_exp_b.sh                 # Noise-Type Aware
├── run_exp_c.sh                 # Layer-Selective LoRA
└── analysis_outputs/            # 分析結果
    ├── token_error_rates.pt
    ├── error_token_analysis.json
    └── noise_type_difficulty.json
```
