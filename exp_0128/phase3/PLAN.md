# exp_0128 Phase 3: Residual Vector Quantization (RVQ)

## 背景

Phase 1 & 2 失敗總結：
- **Phase 1 (Sampling)**: 資料分布調整 ❌ - 問題不在資料
- **Phase 2 (Loss & Codebook)**: 表面修復 ❌ - 治標不治本
  - Exp 3 (Entropy Regularization): 所有實驗 entropy 下降
  - Exp 4 (Codebook Refresh): 所有實驗 top-10 mass 上升

**根本問題**: Encoder 輸出空間坍縮 → 只映射到 ~740/4096 codes

**關鍵發現**: 初始模型比訓練後更好
- Step 0: Entropy 6.26, Top-10 11.8%
- Step 1000: Entropy 5.27-5.70, Top-10 25-33%

## Phase 3 策略: Residual Vector Quantization (RVQ)

### 什麼是 RVQ？

**Residual Vector Quantization (殘差向量量化)** 是一種多層量化技術，透過逐層量化殘差來提高表達能力和多樣性。

### 原理對比

#### 傳統 VQ (單層量化)

```
┌─────────────┐
│  Encoder    │
│   output    │
└──────┬──────┘
       │ z = [3.2, 1.5, -0.8, ...]
       │
       v
┌─────────────────────────────────┐
│  找最近的 codebook entry         │
│                                 │
│  Codebook (4096 entries):      │
│  code_0:   [3.1, 1.6, -0.7]    │ ← 最接近
│  code_1:   [1.2, 0.5, 2.3]     │
│  code_2:   [-2.1, 3.4, 1.1]    │
│  ...                            │
└─────────────┬───────────────────┘
              │
              v
        z_q = code_0

❌ 問題：模型可以只用少數 codes (740/4096)
```

#### RVQ (多層殘差量化)

```
┌─────────────┐
│  Encoder    │
│   output    │
└──────┬──────┘
       │ z = [3.2, 1.5, -0.8, ...]
       │
       v
┌─────────────────────────────────┐
│  Layer 0: 粗略逼近               │
│  找最近的 code                   │
│  q0 = [3.0, 1.0, -1.0]          │ ← Codebook 0 (1024 entries)
└─────────────┬───────────────────┘
              │
              v 計算殘差
        residual_0 = z - q0
                   = [0.2, 0.5, 0.2]
              │
              v
┌─────────────────────────────────┐
│  Layer 1: 修正殘差               │
│  q1 = [0.2, 0.4, 0.3]           │ ← Codebook 1 (1024 entries)
└─────────────┬───────────────────┘
              │
              v 計算新殘差
        residual_1 = residual_0 - q1
                   = [0.0, 0.1, -0.1]
              │
              v
┌─────────────────────────────────┐
│  Layer 2: 精細修正               │
│  q2 = [0.0, 0.1, -0.1]          │ ← Codebook 2 (1024 entries)
└─────────────┬───────────────────┘
              │
              v 計算新殘差
        residual_2 ≈ [0, 0, 0]
              │
              v
┌─────────────────────────────────┐
│  Layer 3: 最終微調               │
│  q3 = [0.0, 0.0, 0.0]           │ ← Codebook 3 (1024 entries)
└─────────────┬───────────────────┘
              │
              v
    最終量化結果:
    z_q = q0 + q1 + q2 + q3
        = [3.0, 1.0, -1.0] + [0.2, 0.4, 0.3] + [0.0, 0.1, -0.1] + [0.0, 0.0, 0.0]
        = [3.2, 1.5, -0.8]  ✅ 精確逼近原始 z

✅ 優勢：
1. 漸進式逼近，更精確
2. 每層獨立選擇，強制多樣性
3. 表達能力: 1024^4 = 1.1 trillion 組合
```

### 為什麼 RVQ 能解決 Token Collapse？

#### 問題 1: 單層 VQ 可以坍縮

```
傳統 VQ:
┌────────────────────────────────────┐
│  Encoder 學會輸出相似的向量         │
│                                    │
│  所有輸入 → [接近 code_5 的區域]   │
│           → [接近 code_17 的區域]  │
│           → [接近 code_42 的區域]  │
│           ...                      │
│           (只用 740 個固定區域)     │
└────────────────────────────────────┘
              ↓
    Quantizer 只會選到這 740 個 codes
    ❌ Collapse!
```

#### 解法: RVQ 強制多樣性

```
RVQ 的數學約束:

假設 Layer 0 collapse (只用 100/1024 codes):
  → residual_0 會有很大的變化範圍
  → Layer 1 必須處理各種不同的 residual_0
  → Layer 1 被迫使用更多不同的 codes

如果 Layer 1 也 collapse:
  → residual_1 更大
  → Layer 2 被迫使用更多 codes

結論: 越往後層，越難 collapse
```

**具體例子**：

```
情境：Layer 0 只用了 10 個 codes (collapse)

Layer 0 可能輸出:
  q0 ∈ {code_0, code_1, ..., code_9}  (只有 10 種)

但原始輸入 z 有無限種可能：
  z 可能是任何 512 維向量

所以 residual_0 = z - q0 的範圍非常大:
  - 如果 z 離 code_0 很遠 → residual_0 很大
  - 如果 z 離 code_5 中等距離 → residual_0 中等
  - 不同的 z 配上不同的 q0 → residual_0 變化很大

Layer 1 面對各種 residual_0:
  → 無法只用少數 codes 來表達
  → 被迫使用更多不同的 codes ✅
```

### 架構設計

#### 原始架構 (單層 VQ)

```
Teacher-Student Intermediate Supervision:

┌─────────────────────────────────────────────────┐
│  Teacher (Clean Audio) - 全部凍結 ❄️             │
├─────────────────────────────────────────────────┤
│  Audio → Encoder ❄️ → [Single VQ] ❄️ → codes   │
│           ↓ L3,L6       ↓                       │
│        中間層        quantized (512-dim)         │
│                         ↓                       │
│                    Decoder ❄️ → audio           │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  Student (Noisy Audio + LoRA)                   │
├─────────────────────────────────────────────────┤
│  Audio → Encoder 🔥 → [Single VQ] ❄️ → codes   │
│           ↓ L3,L6       ↓                       │
│        中間層        quantized                   │
│         (可訓練)     (不使用)                     │
└─────────────────────────────────────────────────┘

Loss = MSE(student_encoder_out, teacher_encoder_out)
     + MSE(student_L3, teacher_L3)
     + MSE(student_L6, teacher_L6)

說明：
- 🔥 = 可訓練 (LoRA)
- ❄️ = 凍結
- L3, L6 = encoder.model[3], encoder.model[6] (中間層)
```

#### RVQ 架構 (多層 VQ)

```
Teacher-Student Intermediate Supervision + RVQ:

┌─────────────────────────────────────────────────┐
│  Teacher (Clean Audio) - 全部凍結 ❄️             │
├─────────────────────────────────────────────────┤
│  Audio → Encoder ❄️ → [Single VQ] ❄️ → codes   │
│           ↓ L3,L6       ↓                       │
│        中間層        quantized (512-dim)         │
│                         ↓                       │
│                    Decoder ❄️ → audio           │
└─────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  Student (Noisy Audio + LoRA)                        │
├──────────────────────────────────────────────────────┤
│  Audio → Encoder 🔥 → [RVQ Quantizer 🔥]            │
│           ↓ L3,L6       ↓                            │
│        中間層        quantized                        │
│         (可訓練)         ↓                            │
│                    Teacher Decoder ❄️ → audio        │
│                                                      │
│  RVQ Quantizer 內部 (新的，可訓練 🔥):                │
│  ┌────────────────────────────────────────┐         │
│  │ z (from Encoder)                       │         │
│  │   ↓                                    │         │
│  │ Layer 0 → q0 (Codebook 0: 1024) 🔥     │         │
│  │   ↓                                    │         │
│  │ residual_0 = z - q0                    │         │
│  │   ↓                                    │         │
│  │ Layer 1 → q1 (Codebook 1: 1024) 🔥     │         │
│  │   ↓                                    │         │
│  │ residual_1 = residual_0 - q1           │         │
│  │   ↓                                    │         │
│  │ Layer 2 → q2 (Codebook 2: 1024) 🔥     │         │
│  │   ↓                                    │         │
│  │ residual_2 = residual_1 - q2           │         │
│  │   ↓                                    │         │
│  │ Layer 3 → q3 (Codebook 3: 1024) 🔥     │         │
│  │   ↓                                    │         │
│  │ quantized = q0 + q1 + q2 + q3          │         │
│  │   → [batch, 512, time]                 │         │
│  └────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────┘

Loss = MSE(student_quantized, teacher_quantized)  # Encoder output 對齊
     + MSE(student_L3, teacher_L3)                # 中間層對齊
     + MSE(student_L6, teacher_L6)                # 中間層對齊
     + RVQ_commitment_loss                        # RVQ 穩定訓練

說明：
- 🔥 = 可訓練 (LoRA for Encoder, 新建 for RVQ)
- ❄️ = 凍結
- L3, L6 = encoder.model[3], encoder.model[6] (中間層)
- Teacher Decoder: 使用 teacher.decode(quantized, bandwidth_id=0)
- 關鍵：Decoder 處理 quantized vectors，不需要知道來自哪個 codebook
```

### RVQ 實作細節

#### Forward Pass 算法

```python
def forward(z):
    """
    Args:
        z: [batch, dim, time] - Encoder 輸出

    Returns:
        z_q: [batch, dim, time] - 量化後的向量
        codes: [n_layers, batch, time] - 所有層的 code indices
    """
    z_q = 0
    residual = z
    all_codes = []

    for layer in range(n_layers):
        # 1. 找最近的 codebook entry
        distances = compute_distance(residual, codebook[layer])
        indices = argmin(distances)  # [batch, time]

        # 2. 獲取對應的向量
        q = codebook[layer][indices]  # [batch, dim, time]

        # 3. Straight-through estimator (for gradient)
        q = residual + (q - residual).detach()

        # 4. 累積量化結果
        z_q += q

        # 5. 計算新的殘差
        residual = residual - q.detach()

        # 6. 保存 codes
        all_codes.append(indices)

    return z_q, all_codes
```

#### 關鍵技術

1. **Straight-Through Estimator**
   ```python
   q = residual + (q - residual).detach()
   ```
   - 前向: 用 quantized value q
   - 反向: 梯度直接傳回 residual (繞過 argmin)

2. **Commitment Loss**
   ```python
   commitment_loss = MSE(residual.detach(), q)
   ```
   - 鼓勵 encoder 輸出靠近 codebook
   - 穩定訓練

3. **Residual Detach**
   ```python
   residual = residual - q.detach()
   ```
   - 每層獨立訓練
   - 避免梯度爆炸

### 為什麼 RVQ 比 Phase 2 方法更好？

| 方法 | 層級 | 問題 | 為什麼失敗/成功 |
|------|------|------|----------------|
| **Entropy Reg (Exp 3)** | Loss-level | 只改 loss，不改架構 | ❌ Encoder 仍學會坍縮，正則化無法阻止 |
| **Codebook Refresh (Exp 4)** | Codebook-level | 只改 codebook，不改 encoder | ❌ 刷新的 codes 在不同空間，encoder 選不到 |
| **RVQ (Exp 5)** | Architecture-level | 改變量化架構 | ✅ **強制約束**，每層必須獨立選擇，無法繞過 |

**核心差異**：

```
Phase 2 方法:
  Encoder → [Single VQ] → Loss/Refresh
                ↑
        只在這裡動手腳
        Encoder 行為不變

Phase 3 (RVQ):
  Encoder → [Multi-layer VQ with residuals]
            ↑
        架構層面改變
        強制 encoder 適應多層結構
```

### 理論保證

**數學證明** (簡化版):

假設：
- 每層 codebook size = K
- n 層 RVQ

單層 VQ 表達能力:
```
P(collapse) = 可以只用 M 個 codes (M << K)
```

n 層 RVQ 表達能力:
```
要全部 collapse，需要每層都只用 M 個 codes
P(all collapse) = P(layer_0 collapse) ×
                   P(layer_1 collapse | layer_0 collapse) × ...

由於 residual 的不可預測性:
P(layer_i collapse | previous layers collapse) << P(layer_0 collapse)

因此: P(all collapse) << P(single layer collapse) ✅
```

### 實驗驗證

已在以下 SOTA 模型中驗證有效：
- **SoundStream (Google, 2021)**: 使用 RVQ，成功壓縮音頻
- **EnCodec (Meta, 2022)**: 使用 RVQ，高質量音頻編碼
- **DAC (Descript, 2023)**: 使用 RVQ，優於單層 VQ

我們的 baseline (exp_k v6) 使用單層 VQ → 出現 collapse
改用 RVQ → 理論上應該解決問題 ✅

### 實驗設計

| 實驗 | 層數 | 每層 Codebook | 總表達能力 | 策略 |
|------|------|---------------|-----------|------|
| 5a   | 2    | 2048          | 2048²     | 溫和 (驗證概念) |
| 5b   | 4    | 1024          | 1024⁴     | 中等 (推薦) |
| 5c   | 8    | 512           | 512⁸      | 激進 (最大多樣性) |

### Baseline (exp_k v6 @ epoch 300)
- Single VQ Entropy: 6.07
- Single VQ Top-10 Mass: 19.7%
- Used codes: ~740/4096

### 成功判準（RVQ-specific，已修正）

**重要**：RVQ 與 baseline 使用不同 codebook space，不能直接比較 codes。
改用以下指標：

#### 1. Layer 0 多樣性（與 baseline 單層 VQ 比較）
- **Layer0 Entropy > 6.5** (baseline: 6.07)
- **Layer0 Top-10 Mass < 15%** (baseline: 19.7%)
- 說明：Layer 0 是 RVQ 的粗略逼近層，可與 baseline 單層 VQ 比較

#### 2. Joint Diversity（RVQ 特有優勢）
- **Joint Diversity > 70%**
- 說明：所有層組合的多樣性，理論上遠高於單層 VQ

#### 3. Feature Space Alignment（主要訓練目標）
- **Feature MSE < 0.1**
- 說明：Student quantized 與 Teacher encoder output 的對齊程度

#### ❌ 不再使用的指標
- ~~Strict Accuracy (teacher codes vs student codes)~~
- 原因：Codebook space 不同（Teacher 4096 vs Student 1024/2048/512）
- 此指標在 RVQ 架構下無意義

## 常見問題解答 (Q&A)

### Q1: RVQ 是重複使用同一個 codebook 嗎？

**A: 不是！RVQ 使用多個獨立的 codebook。**

誤解：
```
residual → codebook → residual → 同一個 codebook → ...
         ↑________________________________↑
              重複使用同一本
```

實際：
```
residual_0 → Codebook 0 (1024 entries) → q0
residual_1 → Codebook 1 (1024 entries) → q1  ← 不同的 codebook
residual_2 → Codebook 2 (1024 entries) → q2  ← 不同的 codebook
residual_3 → Codebook 3 (1024 entries) → q3  ← 不同的 codebook
```

**類比**：就像畫畫
- Codebook 0: 粗筆刷 (畫輪廓)
- Codebook 1: 細筆刷 (畫細節)
- Codebook 2: 更細的筆刷 (畫陰影)
- Codebook 3: 最細的筆刷 (最終修飾)

每層有不同的"工具箱"，各司其職。

### Q2: 這些 codebook 是如何定義的？

**A: RVQ codebooks 是新建立的可訓練 nn.Embedding，與原始 WavTokenizer 無關。**

原始 WavTokenizer:
```python
# 單層 VQ，4096 個 codes
quantizer = EncodecQuantizer(
    codebook_size=4096,
    dim=512
)
# 預訓練好的，凍結
```

RVQ 新建立:
```python
# 4 層獨立 codebook，每層 1024 個 codes
self.codebooks = nn.ModuleList([
    nn.Embedding(1024, 512)  # Codebook 0
    nn.Embedding(1024, 512)  # Codebook 1
    nn.Embedding(1024, 512)  # Codebook 2
    nn.Embedding(1024, 512)  # Codebook 3
])

# 隨機初始化
for codebook in self.codebooks:
    codebook.weight.data.uniform_(-1.0 / 1024, 1.0 / 1024)
```

**關鍵差異**：
- 原始 codebook: 1 個，4096 entries，預訓練，凍結 ❄️
- RVQ codebooks: 4 個，各 1024 entries，隨機初始化，可訓練 🔥

### Q3: 原始 WavTokenizer 的 codebook 還是凍結的嗎？

**A: Teacher 的所有組件都是凍結的，但 Student 用的是新的 RVQ codebooks。**

完整架構：

```
Teacher (Clean Audio):
┌─────────────────────────────────────────┐
│ Encoder (凍結) ❄️                        │
│   ↓                                     │
│ Quantizer (凍結) ❄️                      │
│   - Original codebook (4096)           │
│   ↓                                     │
│ Decoder (凍結) ❄️                        │
└─────────────────────────────────────────┘

Student (Noisy Audio):
┌─────────────────────────────────────────┐
│ Encoder (LoRA 可訓練) 🔥                 │
│   ↓                                     │
│ [原始 Quantizer 不用了]                 │
│ ↓                                       │
│ RVQ Quantizer (新的，可訓練) 🔥          │
│   - 4 個新 codebooks (各 1024)          │
│   ↓                                     │
│ [Decoder 用 Teacher 的，凍結] ❄️         │
└─────────────────────────────────────────┘
```

**Trainable vs Frozen**:

| 組件 | Teacher | Student |
|------|---------|---------|
| Encoder | ❄️ 凍結 | 🔥 LoRA (可訓練) |
| Original Quantizer | ❄️ 凍結 | ❌ 不使用 |
| RVQ Quantizer | ❌ 沒有 | 🔥 可訓練 (新的) |
| Decoder | ❄️ 凍結 | ❄️ 使用 Teacher (凍結) |

### Q4: Decoder 如何看懂 RVQ 的輸出？

**A: Decoder 不看 codebook，只看 quantized vectors！**

**關鍵理解**: Decoder 是向量級操作，不是 code 級操作。

錯誤理解：
```
codes → (lookup codebook) → quantized → decoder
  ↑                            ↑
decoder 需要知道            decoder 需要知道
是哪個 codebook            來自哪個 codebook?
```

實際流程：
```
Original WavTokenizer:
  audio → encoder → z → quantizer → quantized
                              ↓
                          codes (只是副產物)

  quantized → decoder → audio
      ↑
  decoder 只看這個！
  [batch, 512, time] 的連續向量
```

**RVQ 同樣輸出連續向量**：
```
RVQ:
  z → Layer 0 → q0 (向量)
    → Layer 1 → q1 (向量)
    → Layer 2 → q2 (向量)
    → Layer 3 → q3 (向量)

  quantized = q0 + q1 + q2 + q3

  quantized: [batch, 512, time]  ✅ 形狀正確

  decoder(quantized) → audio  ✅ 直接使用
```

**Decoder 只關心**：
- 形狀是否正確: [batch, 512, time] ✅
- 數值範圍是否合理: 在訓練分布內 ✅
- 來自哪個 codebook: **不關心** ✅

### Q5: RVQ 需要映射回原始 codebook 嗎？

**A: 不需要！映射反而會損失精度。**

誤解：
```
RVQ codebooks (trainable) → [mapping] → Original codebook (frozen)
                                              ↓
                                          decoder
```

這樣做的問題：
1. **損失精度**: RVQ 的 q0+q1+q2+q3 已經是精確逼近，映射到單層 codebook 會損失細節
2. **不必要**: Decoder 本來就接受連續向量，不需要是原始 codebook 的
3. **破壞多樣性**: 映射會再次限制到 4096 種可能，失去 1024^4 的表達能力

**正確做法**：
```
RVQ:
  z → RVQ quantizer → quantized (q0+q1+q2+q3)
                         ↓
                  [batch, 512, time]
                  包含所有層資訊的連續向量
                         ↓
                  teacher.decoder → audio ✅

不需要任何 mapping！
```

**為什麼這樣可行**：
- Decoder 是用預訓練的權重處理 512 維向量
- 只要向量在合理的數值範圍內，decoder 就能工作
- RVQ 的 quantized 向量本來就是逼近原始 encoder output
- Decoder 見過各種 encoder output，不侷限於某個 codebook

**數學上**：
```
Original:
  z → argmin(distance(z, codebook)) → z_q
  z_q = codebook[index]  (離散選擇)

RVQ:
  z → q0+q1+q2+q3 → z_q
  z_q = sum of vectors (連續組合)

Both produce: [batch, 512, time] continuous vectors
Decoder sees no difference! ✅
```

### Q6: 為什麼 RVQ 音質不會下降？

**A: RVQ 理論上音質更好，因為逼近更精確。**

單層 VQ:
```
z → 找最近的 1 個 code → z_q
逼近誤差 = ||z - z_q||²

最多只有 4096 種可能的 z_q
```

4 層 RVQ:
```
z → q0 (粗略)
  → q1 (修正殘差)
  → q2 (進一步修正)
  → q3 (最終微調)

z_q = q0 + q1 + q2 + q3

逼近誤差 < 單層 VQ (因為多次修正)
有 1024^4 種可能的組合
```

**實際影響**：
- 更精確的量化 → 重建音質更好
- 更多樣化的表達 → 捕捉更多細節
- 漸進式逼近 → 減少量化噪音

## 實作計畫

### Step 1: RVQ 模組實作 ✅
建立 `models_rvq.py`:
- `ResidualVectorQuantizer`: 多層 VQ 模組
- 整合到現有 `TeacherStudentIntermediate` 架構

### Step 2: 測試腳本 ✅
建立 `test_rvq.py`:
- 驗證 RVQ forward pass ✅
- 檢查 codebook usage ✅
- 確認梯度流動 ✅
- 所有配置 (2/4/8層) 測試通過 ✅

### Step 3: 訓練腳本 ✅
建立 `train_rvq_short_run.py`:
- 基於 Phase 2 的 `train_short_run.py` ✅
- 整合 RVQ quantizer ✅
- 1000 steps 快速驗證 ✅
- 完整的 metrics 評估 ✅
- RVQ 每層使用情況分析 ✅

### Step 4: 實驗執行 🟢 準備就緒
建立啟動腳本:
- Exp 5a: 2 層 RVQ (`run_exp5a.sh`) ✅
- Exp 5b: 4 層 RVQ (`run_exp5b.sh`) ✅ (最推薦)
- Exp 5c: 8 層 RVQ (`run_exp5c.sh`) ✅

## 風險評估

### 可能失敗的原因
1. **訓練不穩定**: 多層量化可能導致梯度問題
   - 緩解: 使用 gradient clipping, 降低學習率
2. **記憶體問題**: 多層 codebook 增加記憶體使用
   - 緩解: 減少每層 codebook size
3. **Encoder 仍然坍縮**: 如果 encoder 本身問題太深
   - 下一步: 考慮 encoder 架構改進 (方案 3)

### 如果 RVQ 也失敗
說明問題在更深層：
1. 考慮 Encoder 正則化 (Layer norm, Spectral norm)
2. 考慮訓練策略 (Curriculum, 分階段訓練)
3. 考慮是否 VQ-VAE 根本不適合這個任務

## 時間表

- ✅ **Day 1** (2026-02-03): 實作 RVQ 模組 + 測試 + 訓練腳本
  - models_rvq.py: ResidualVectorQuantizer + TeacherStudentRVQ
  - test_rvq.py: 所有測試通過
  - train_rvq_short_run.py: 完整訓練流程
  - run_exp5{a,b,c}.sh: 啟動腳本

- ⏳ **Day 2**: 運行 Exp 5a 驗證概念
- ⏳ **Day 3**: 運行 Exp 5b, 5c 完整測試
- ⏳ **Day 4**: 結果分析 + 報告

---

**建立時間**: 2026-02-03
**完成時間**: 2026-02-03 02:07 (訓練腳本)
**狀態**: 🟢 準備就緒 - 可以開始實驗！
