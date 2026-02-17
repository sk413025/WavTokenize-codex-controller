# 碩論架構整理：基於 WavTokenizer 的語音去噪離散表示學習

> 本文件整理研究系統的完整架構、原理、損失函數設計。
> 可直接做為碩論第三章（方法論）的骨架。

---

## 目錄

1. [系統總覽](#1-系統總覽)
2. [WavTokenizer 預訓練模型](#2-wavtokenizer-預訓練模型)
3. [Teacher-Student 架構](#3-teacher-student-架構)
4. [LoRA 微調策略](#4-lora-微調策略)
5. [Single VQ + EMA 量化器](#5-single-vq--ema-量化器)
6. [中間層監督機制 (Intermediate Supervision)](#6-中間層監督機制)
7. [損失函數設計](#7-損失函數設計)
8. [課程式學習 (Curriculum Learning)](#8-課程式學習)
9. [資料增強策略](#9-資料增強策略)
10. [推論流程](#10-推論流程)
11. [建議碩論章節對照](#11-建議碩論章節對照)

---

## 1. 系統總覽

### 1.1 研究問題

在語音去噪場景中，如何讓一個離散語音編碼器（WavTokenizer）從含噪語音中產生
與乾淨語音相同品質的離散 token？

### 1.2 核心思路

> 「不直接在波形域做去噪，而是在 **離散表示 (token) 空間** 實現去噪」

### 1.3 系統全貌 ASCII 架構圖

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE (整體訓練流程)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐                     ┌──────────────┐                 │
│   │  Clean Audio  │                     │  Noisy Audio  │                │
│   │ (乾淨語音 x_c) │                    │ (含噪語音 x_n) │               │
│   └──────┬───────┘                     └──────┬───────┘                 │
│          │                                     │                         │
│          ▼                                     ▼                         │
│   ┌──────────────┐                     ┌──────────────┐                 │
│   │   Teacher     │                     │   Student     │                │
│   │   Encoder     │                     │   Encoder     │                │
│   │  (Frozen)     │                     │  (LoRA)       │                │
│   │              │                     │              │                 │
│   │  L0 ─────────╥───── L_inter ──────╥── L0+LoRA    │                 │
│   │  L1          ║     (cosine)       ║   L1+LoRA    │                 │
│   │  L2          ║                    ║   L2+LoRA    │                 │
│   │  L3 ◄────────╫── Supervision ─────╫── L3+LoRA    │                 │
│   │  L4 ◄────────╫── Supervision ─────╫── L4+LoRA    │                 │
│   │  ...         ║                    ║   ...        │                 │
│   │  L6 ◄────────╫── Supervision ─────╫── L6+LoRA    │                 │
│   │  ...         ║                    ║   ...        │                 │
│   │  L17         ║                    ║   L17+LoRA   │                 │
│   └──────┬───────┘                     └──────┬───────┘                 │
│          │                                     │                         │
│          ▼                                     ▼                         │
│    t_e [B,128,T]                         z_e [B,128,T]                  │
│   (teacher output)                     (student output)                 │
│          │                                     │                         │
│          │                              ┌──────┴───────┐                │
│          │                              │  Single VQ   │                │
│          │                              │  + EMA       │                │
│          │                              │  K=4096      │                │
│          │                              │  dim=128     │                │
│          │                              └──────┬───────┘                │
│          │                                     │                         │
│          │                               z_q [B,128,T]                  │
│          │                              (quantized out)                  │
│          │                                     │                         │
│          └──────────── L_quant ────────────────┘                        │
│                    MSE(z_q, t_e)                                        │
│                                                                         │
│   Total Loss = λ_q·L_quant + w_inter·L_inter + β·L_commit              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 推論時架構

```
┌──────────────────────────────────────────────────┐
│                INFERENCE PIPELINE                 │
├──────────────────────────────────────────────────┤
│                                                   │
│   Noisy Audio (含噪語音)                          │
│        │                                          │
│        ▼                                          │
│   ┌──────────────┐                                │
│   │   Student     │                                │
│   │   Encoder     │  ← 已訓練的 LoRA 參數          │
│   │  (LoRA)       │                                │
│   └──────┬───────┘                                │
│          │                                        │
│          ▼                                        │
│    z_e [B,128,T]                                  │
│          │                                        │
│   ┌──────┴───────┐                                │
│   │  Single VQ   │  ← 已訓練的 EMA codebook       │
│   │  K=4096      │                                │
│   └──────┬───────┘                                │
│          │                                        │
│    z_q [B,128,T]   (denoised tokens)              │
│          │                                        │
│   ┌──────┴───────┐                                │
│   │   Teacher     │                                │
│   │   Decoder     │  ← 凍結的原始 decoder          │
│   └──────┬───────┘                                │
│          │                                        │
│          ▼                                        │
│   Denoised Audio (去噪語音)                       │
│                                                   │
└──────────────────────────────────────────────────┘
```

---

## 2. WavTokenizer 預訓練模型

### 2.1 簡介

WavTokenizer 是一個端到端的離散語音編碼器，將連續波形壓縮為離散 token 序列。

```
Audio (24kHz)  →  Encoder  →  VQ  →  Decoder  →  Reconstructed Audio
[B, 1, T]         [B,128,T']   tokens    [B, 1, T]
                   75 fps      K=4096
```

### 2.2 Encoder 結構

WavTokenizer 的 encoder 是一個 **18 層** 的卷積網路，包含：
- **4 個 Downsample 階段**（每階段 stride=2~5），逐步降低時間解析度
- **4 個 ResBlock**（每個包含 2 層卷積 + shortcut）
- **1 個輸入卷積** + **1 個輸出卷積**

```
Encoder 層級結構 (18 Layers):
═══════════════════════════════════════════════════════════
Layer   Module                              Type
───────────────────────────────────────────────────────────
L0      model[0].conv.conv                  Input Conv
───────────────────────────────────────────────────────────
L1      model[1].block[1].conv.conv         ResBlock 1 - Conv1
L2      model[1].block[3].conv.conv         ResBlock 1 - Conv2
L3      model[1].shortcut.conv.conv         ResBlock 1 - Shortcut
───────────────────────────────────────────────────────────
L4      model[3].conv.conv                  ★ Downsample 1 (stride)
───────────────────────────────────────────────────────────
L5      model[4].block[1].conv.conv         ResBlock 2 - Conv1
L6      model[4].block[3].conv.conv         ResBlock 2 - Conv2
L7      model[4].shortcut.conv.conv         ResBlock 2 - Shortcut
───────────────────────────────────────────────────────────
L8      model[6].conv.conv                  ★ Downsample 2 (stride)
───────────────────────────────────────────────────────────
L9      model[7].block[1].conv.conv         ResBlock 3 - Conv1
L10     model[7].block[3].conv.conv         ResBlock 3 - Conv2
L11     model[7].shortcut.conv.conv         ResBlock 3 - Shortcut
───────────────────────────────────────────────────────────
L12     model[9].conv.conv                  ★ Downsample 3 (stride)
───────────────────────────────────────────────────────────
L13     model[10].block[1].conv.conv        ResBlock 4 - Conv1
L14     model[10].block[3].conv.conv        ResBlock 4 - Conv2
L15     model[10].shortcut.conv.conv        ResBlock 4 - Shortcut
───────────────────────────────────────────────────────────
L16     model[12].conv.conv                 ★ Downsample 4 (stride)
───────────────────────────────────────────────────────────
L17     model[15].conv.conv                 Output Conv → [B,128,T']
═══════════════════════════════════════════════════════════
```

### 2.3 原始 VQ（Residual VQ）

原始 WavTokenizer 使用 RVQ（Residual Vector Quantization），但本研究替換為
Single VQ + EMA（見第 5 節）。

---

## 3. Teacher-Student 架構

### 3.1 設計動機

直接修改 WavTokenizer 的 encoder 會「遺忘」預訓練知識。
Teacher-Student 架構的核心思想是：

> **Teacher**（凍結）以乾淨音訊為目標，提供「這是正確答案」；
> **Student**（LoRA 微調）從含噪音訊學到「如何產生和 Teacher 一樣的輸出」。

### 3.2 Teacher-Student 流程圖

```
              Teacher (Frozen)              Student (LoRA)
              ┌─────────────┐              ┌─────────────┐
              │ Clean Audio  │              │ Noisy Audio  │
              │  x_c [B,1,T] │              │  x_n [B,1,T] │
              └──────┬──────┘              └──────┬──────┘
                     │                            │
                     ▼                            ▼
        ┌────────────────────┐      ┌────────────────────┐
        │  Encoder (Frozen)  │      │  Encoder (LoRA)    │
        │  θ_T (不可訓練)     │      │  θ_T + Δθ (可訓練)  │
        │                    │      │                    │
        │  ┌──── L3 ────┐   │      │  ┌──── L3 ────┐   │
        │  │ t_3 (clean) │   │      │  │ s_3 (noisy) │   │
        │  └─────────────┘   │      │  └─────────────┘   │
        │        ↓           │      │        ↓           │
        │  ┌──── L4 ────┐   │      │  ┌──── L4 ────┐   │
        │  │ t_4 (clean) │   │      │  │ s_4 (noisy) │   │
        │  └─────────────┘   │      │  └─────────────┘   │
        │        ↓           │      │        ↓           │
        │  ┌──── L6 ────┐   │      │  ┌──── L6 ────┐   │
        │  │ t_6 (clean) │   │      │  │ s_6 (noisy) │   │
        │  └─────────────┘   │      │  └─────────────┘   │
        │        ↓           │      │        ↓           │
        └────────┬───────────┘      └────────┬───────────┘
                 │                           │
                 ▼                           ▼
           t_e [B,128,T]              z_e [B,128,T]
                 │                           │
                 │                    ┌──────┴──────┐
                 │                    │  Single VQ  │
                 │                    │  + EMA      │
                 │                    └──────┬──────┘
                 │                           │
                 │                     z_q [B,128,T]
                 │                           │
                 └─────── MSE Loss ──────────┘
```

### 3.3 關鍵設計決策

| 元件 | Teacher | Student | 理由 |
|------|---------|---------|------|
| Encoder 權重 | 凍結 θ_T | θ_T + LoRA Δθ | 保留預訓練知識 |
| Quantizer | 原始 RVQ (凍結) | SingleVQ + EMA | 防止 codebook collapse |
| 輸入 | 乾淨語音 x_c | 含噪語音 x_n | Teacher 提供乾淨目標 |
| 梯度 | 無 (torch.no_grad) | 有 (LoRA 可訓練) | 只有 Student 學習 |

---

## 4. LoRA 微調策略

### 4.1 LoRA 原理

LoRA (Low-Rank Adaptation) 凍結原始權重 $W_0$，僅訓練低秩分解 $\Delta W = BA$：

$$W = W_0 + \frac{\alpha}{r} \cdot BA$$

其中：
- $W_0 \in \mathbb{R}^{d \times d}$：原始凍結權重
- $B \in \mathbb{R}^{d \times r}$：低秩矩陣 B
- $A \in \mathbb{R}^{r \times d}$：低秩矩陣 A
- $r$：秩 (rank)，遠小於 $d$
- $\alpha$：縮放因子

### 4.2 本研究的 LoRA 設定

```
LoRA 設定:
┌────────────────────────────────────────────┐
│  Target Modules: ALL_18_LAYERS (全部 18 層) │
│                                            │
│  Plan Ori (exp_0206):                      │
│    rank = 256, alpha = 512                 │
│    dropout = 0.2                           │
│    Trainable: 4,718,592 (2.9%)             │
│                                            │
│  Plan Aug (exp_0216):                      │
│    rank = 64,  alpha = 128                 │
│    dropout = 0.2                           │
│    weight_decay = 0.02                     │
│    Trainable: 926,144 (0.6%)               │
│                                            │
│  降低 rank 可減少過擬合風險                   │
└────────────────────────────────────────────┘
```

### 4.3 為何選擇全 18 層

- 噪音會影響 encoder 的**所有階段**
- 淺層（L0-L3）：噪音會直接通過卷積傳播
- Downsample 層（L4, L8, L12, L16）：降採樣會放大噪音
- 深層（L13-L17）：最終表示需要完全去噪

因此 LoRA 施加在全部 18 層。

---

## 5. Single VQ + EMA 量化器

### 5.1 為何不用原始 RVQ

原始 WavTokenizer 的 RVQ 在訓練過程中被**凍結**（quantizer 不更新），
但 Teacher 和 Student 的 encoder 輸出分佈不同，凍結的 codebook 會導致
量化品質下降（codebook collapse）。

Single VQ + EMA 的優勢：
1. **Warm Start**: 從 WavTokenizer 預訓練的 codebook 初始化
2. **EMA Update**: 不用梯度，用指數移動平均更新 codebook
3. **Dead-code Reset**: 自動重置不使用的 code，維持 codebook 多樣性

### 5.2 VQ 量化流程

```
z_e ∈ ℝ^[B, 128, T]   (Student encoder 輸出)
        │
        ▼  transpose
z ∈ ℝ^[B, T, 128]
        │
        ▼  flatten
z_flat ∈ ℝ^[BT, 128]
        │
        ▼  L2 距離計算
  ┌─────────────────────────────────────────┐
  │                                         │
  │  dist(z, e_k) = ‖z‖² + ‖e_k‖² - 2z·eₖ │
  │                                         │
  │  k* = argmin_k  dist(z, e_k)           │
  │                                         │
  └─────────────────────────────────────────┘
        │
        ▼
indices ∈ ℤ^[BT]    (最近鄰索引)
        │
        ▼  codebook lookup
q ∈ ℝ^[BT, 128]     (量化向量)
        │
        ▼  Straight-Through Estimator (STE)
z_q = z + (q - z).detach()
        │
        ▼  reshape + transpose
z_q ∈ ℝ^[B, 128, T]
```

### 5.3 Straight-Through Estimator (STE)

VQ 的 `argmin` 操作不可微分，STE 技巧解決這個問題：

$$z_q = z_e + \text{sg}[e_{k^*} - z_e]$$

- 前向傳播：$z_q = e_{k^*}$（實際的量化向量）
- 反向傳播：$\nabla_{z_e} z_q = \nabla_{z_e} z_e = I$（梯度直通）

這讓梯度可以從 $z_q$ 直接傳回 $z_e$，繞過不可微的量化步驟。

### 5.4 EMA Codebook 更新

訓練時，codebook 不使用梯度下降，而是用 EMA (Exponential Moving Average)：

$$N_k^{(t)} = \gamma \cdot N_k^{(t-1)} + (1 - \gamma) \cdot n_k$$

$$m_k^{(t)} = \gamma \cdot m_k^{(t-1)} + (1 - \gamma) \cdot \sum_{j \in S_k} z_j$$

$$e_k = \frac{m_k}{N_k}$$

其中：
- $\gamma = 0.99$：EMA 衰減率
- $n_k$：當前 batch 中被 assign 到 code $k$ 的向量數
- $S_k$：assign 到 code $k$ 的向量集合
- $N_k$：累積使用計數（Laplace smoothing）
- $m_k$：累積嵌入向量總和

**Laplace Smoothing**（避免除以零）：

$$\hat{N}_k = \frac{N_k + \epsilon}{(\sum_j N_j) + K \cdot \epsilon} \cdot \left(\sum_j N_j\right)$$

### 5.5 Dead-Code Reset

當某些 code 的累積使用次數 $N_k < \text{threshold}$（預設 2 次）時，
表示該 code 已成為「死碼」。重置機制：

```
Dead-Code Reset 機制:
─────────────────────────────────────────
IF ema_cluster_size[k] < threshold:
    從當前 batch 隨機挑一個向量 z_rand
    codebook[k] = z_rand          ← 重置嵌入
    ema_cluster_size[k] = 1.0     ← 重置計數
    ema_embed_avg[k] = z_rand     ← 重置累積
─────────────────────────────────────────
```

---

## 6. 中間層監督機制

### 6.1 為什麼需要中間層監督

```
問題：梯度消失 / 間接監督
═══════════════════════════════════════════
只有最終輸出 Loss:
  L17 ← L16 ← ... ← L4 ← L3 ← L0
  │                              ▲
  └── 梯度要傳 17 層才到淺層 ──────┘
                ↑
           梯度太弱！

加入中間層監督:
  L17 ← L16 ← ... ← L6 ← L4 ← L3 ← L0
  │                   │     │    │
  └─ L_final          │     │    │
                      │     │    │
               L_inter_6  L_inter_4  L_inter_3
                      │     │    │
                      ▼     ▼    ▼
                   直接監督！梯度短、準、強
═══════════════════════════════════════════
```

### 6.2 IntermediateExtractor（中間層提取器）

```python
# 在 encoder 的 forward 過程中，hook 住指定層的輸出
for i, layer in enumerate(encoder.model):
    x = layer(x)
    if i in extract_indices:  # e.g., [3, 4, 6]
        intermediates[i] = x  # 儲存中間層輸出
```

### 6.3 監督層的選擇

```
監督層選擇及原因:
┌─────┬──────────────┬──────────╥───────────────────────────────┐
│ 索引 │ 對應層        │ 權重 w_i ║ 選擇原因                       │
├─────┼──────────────┼──────────╫───────────────────────────────┤
│  3  │ model[3]     │  0.3     ║ Downsample 1: 第一次降採樣，    │
│     │ Downsample 1 │          ║ 噪音在此處首次被壓縮             │
├─────┼──────────────┼──────────╫───────────────────────────────┤
│  4  │ model[4]     │  0.5     ║ ResBlock 2: 降採樣後的第一個     │
│     │ ResBlock 2   │          ║ 非線性處理，噪音表示在此定型       │
├─────┼──────────────┼──────────╫───────────────────────────────┤
│  6  │ model[6]     │  0.5     ║ Downsample 2: 第二次降採樣，    │
│     │ Downsample 2 │          ║ 中層特徵的關鍵轉換點             │
└─────┴──────────────┴──────────╨───────────────────────────────┘
```

---

## 7. 損失函數設計

### 7.1 總損失公式

$$\mathcal{L}_{\text{total}} = \lambda_q \cdot \mathcal{L}_{\text{quant}} + w_{\text{inter}} \cdot \mathcal{L}_{\text{inter}} + \beta \cdot \mathcal{L}_{\text{commit}}$$

其中預設超參數：
- $\lambda_q = 1.0$：量化對齊損失權重
- $w_{\text{inter}} = 1.0$：中間層監督權重（可動態衰減）
- $\beta = 0.25$：commitment loss 權重

### 7.2 量化對齊損失 $\mathcal{L}_{\text{quant}}$（Post-Quantization Alignment）

**目的**: 讓 Student 量化後的輸出 $z_q$ 逼近 Teacher 的 encoder 輸出 $t_e$

$$\mathcal{L}_{\text{quant}} = \frac{1}{N} \sum_{b,t} \| z_q^{(b,t)} - t_e^{(b,t)} \|^2$$

```
直覺:
  Teacher encoder (clean) → t_e  ←── 「正確答案」
                              ↕ MSE
  Student encoder (noisy) → VQ → z_q ←── 「Student 的猜測」

  → 讓 Student 的量化結果越接近「乾淨特徵」越好
```

支援 **Masked MSE**：padding 區域不參與 loss 計算。

### 7.3 中間層監督損失 $\mathcal{L}_{\text{inter}}$（Cosine Similarity Loss）

**公式** (IntermediateSupervisionLossV6):

$$\mathcal{L}_{\text{inter}} = \sum_{i \in \{3, 4, 6\}} w_i \cdot (1 - \text{cos\_sim}(\hat{s}_i, \hat{t}_i))$$

其中：

$$\hat{s}_i = \frac{s_i}{\|s_i\|_2 + \epsilon}, \quad \hat{t}_i = \frac{t_i}{\|t_i\|_2 + \epsilon}$$

$$\text{cos\_sim}(\hat{s}_i, \hat{t}_i) = \frac{1}{B \cdot T'} \sum_{b,t} \hat{s}_i^{(b,t)} \cdot \hat{t}_i^{(b,t)}$$

```
直覺:
  使用 Cosine Similarity 而非 MSE 的理由:
  ─────────────────────────────────────────────
  • MSE 受「尺度 (scale)」影響
    → 不同層深度特徵量級差異大

  • Cosine Similarity 只關注「方向 (direction)」
    → 不受量級影響
    → 只要「方向對了」就好，更穩定

  cos_sim = 1 → 完全對齊（loss = 0）
  cos_sim = 0 → 正交   （loss = 1）
  cos_sim = -1 → 反向  （loss = 2）
```

### 7.4 Commitment Loss $\mathcal{L}_{\text{commit}}$

**目的**: 鼓勵 encoder 輸出 $z_e$ 靠近量化後的 $z_q$，減少 "codebook 跳動"

$$\mathcal{L}_{\text{commit}} = \| z_e - \text{sg}[z_q] \|^2$$

- $\text{sg}[\cdot]$ 表示 stop-gradient
- 只有 encoder 收到梯度（codebook 用 EMA 單獨更新）

```
Commitment Loss 示意:
─────────────────────────────────────────────
     z_e (encoder 輸出) ──── L_commit ──── sg[z_q]
        ↑ 梯度                               │
        │                                  不傳梯度
    「encoder 要主動靠近 codebook」
─────────────────────────────────────────────
```

### 7.5 損失流程總圖

```
┌─────────── Loss Computation Flow ─────────────┐
│                                                │
│  Teacher           Student                     │
│  ┌─────┐           ┌─────┐                     │
│  │ Enc  │           │ Enc  │                     │
│  │(frz) │           │(LoRA)│                     │
│  └──┬──┘           └──┬──┘                     │
│     │                  │                        │
│  t_inter           s_inter                     │
│  {L3,L4,L6}       {L3,L4,L6}                  │
│     │                  │                        │
│     └─── L_inter ──────┘                       │
│     (cosine, w={0.3,0.5,0.5})                  │
│                                                │
│     │                  │                        │
│  t_e [B,128,T]     z_e [B,128,T]               │
│                        │                        │
│                    ┌───┴───┐                    │
│                    │  VQ   │                    │
│                    │ (EMA) │                    │
│                    └───┬───┘                    │
│                        │                        │
│                    z_q [B,128,T]                │
│     │                  │                        │
│     └─── L_quant ──────┘  MSE(z_q, t_e)        │
│                                                │
│                    z_e ──── L_commit ──── sg[z_q]│
│                                                │
│  L_total = λ_q·L_quant + w·L_inter + β·L_commit│
│                                                │
└────────────────────────────────────────────────┘
```

---

## 8. 課程式學習 (Curriculum Learning)

### 8.1 原理

不要一開始就給模型最困難的樣本。先從「簡單」（高 SNR，噪音少）的資料學起，
逐漸增加「困難」（低 SNR，噪音多）的資料比例。

### 8.2 SNR 驅動的課程排序

```
Curriculum Learning 流程:
═══════════════════════════════════════════════════════════
Phase 0.3 (初期)      Phase 0.6 (中期)      Phase 0.85 (後期)
───────────────      ───────────────      ────────────────
取 SNR 最高的        取 SNR 前 60%         取 SNR 前 85%
  前 30% 資料        的資料               的資料

  "簡單"             "中等"               "困難（幾乎全部）"
  高 SNR 優先         中低 SNR 加入         低 SNR 也加入

      ←──────── 逐 epoch 遞增 ──────────→
═══════════════════════════════════════════════════════════
```

### 8.3 CurriculumSampler

- **初始 phase**: 0.3（只用 30% 最簡單的樣本）
- **最終 phase**: 0.85（用 85% 的樣本）
- **遞增方式**: 線性遞增，跨越約 200 個 epoch
- 按 SNR 由高到低排序，取前 `phase × N` 筆

---

## 9. 資料增強策略

### 9.1 為何需要增強

實驗發現兩個模型都在 epoch ~70 出現 overfitting（train loss↓ 但 val loss↑），
V_MSE 上限約 0.037。在沒有更多原始資料的情況下，資料增強是最有效的解法。

### 9.2 四種增強方式

```
資料增強策略 (exp_0216):
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  1. SNR Remix (p=0.5) ★ 殺手級                             │
│  ─────────────────────                                     │
│  noise = noisy - clean                                     │
│  new_snr ~ Uniform(0, 20) dB                              │
│  noisy_new = clean + noise × 10^(-new_snr/20)             │
│  → 一對 (noisy, clean) 可生成無限種 SNR 變體               │
│                                                            │
│  2. Random Gain (p=0.3)                                    │
│  ─────────────────────                                     │
│  gain ~ Uniform(-3, +3) dB                                │
│  noisy *= 10^(gain/20)                                     │
│  clean *= 10^(gain/20)                                     │
│  → 模擬不同音量級別                                        │
│                                                            │
│  3. Random Crop (p=0.3)                                    │
│  ─────────────────────                                     │
│  ratio ~ Uniform(0.7, 1.0)                                │
│  length = ratio × original_length                         │
│  start ~ Uniform(0, L - length)                           │
│  → 隨機子段截取，增加位置多樣性                              │
│                                                            │
│  4. Time Stretch (p=0.2)                                   │
│  ─────────────────────                                     │
│  rate ~ Uniform(0.95, 1.05)                               │
│  使用 interpolate (nearest) 實現微幅時間伸縮                 │
│  → 模擬不同語速                                            │
│                                                            │
│  ⚠️ 僅 train 啟用，val 不增強                               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 10. 推論流程

### 10.1 端到端推論

```python
# 推論時只需 Student Encoder + VQ + Teacher Decoder
noisy_audio = load_audio("noisy.wav")          # [1, 1, T]
z_e = student_encoder(noisy_audio)             # [1, 128, T']
z_q = single_vq(z_e)['quantized']             # [1, 128, T']
denoised = teacher_decoder(z_q, bandwidth_id)  # [1, 1, T]
```

### 10.2 離散 Token 的應用

VQ 量化後的 codes（離散 token）可用於：
- 語音壓縮傳輸（僅傳 token index，大幅降低 bitrate）
- 下游 NLP 任務（Token-based 語音理解）
- 離散語音生成（Token → Decoder → Audio）

---

## 11. 建議碩論章節對照

```
碩論建議章節結構:
═══════════════════════════════════════════════════════════════
第一章 緒論
  1.1 研究動機        → 噪音環境下語音編碼品質下降
  1.2 研究目的        → 在 token 空間實現去噪
  1.3 論文架構        → 各章概述

第二章 相關研究
  2.1 Neural Audio Codec  → WavTokenizer, SoundStream, EnCodec
  2.2 Vector Quantization → VQ-VAE, RVQ, EMA-VQ
  2.3 Knowledge Distillation → Teacher-Student 框架
  2.4 Parameter-Efficient Fine-Tuning → LoRA, Adapter
  2.5 Speech Enhancement  → 傳統 vs 深度學習去噪

第三章 方法論                         ← 本文件主要對應
  3.1 系統總覽        → §1.3 架構圖
  3.2 WavTokenizer 基礎 → §2 Encoder 結構
  3.3 Teacher-Student 架構 → §3 設計動機與流程
  3.4 LoRA 微調策略     → §4 全 18 層 LoRA
  3.5 Single VQ + EMA   → §5 量化器設計
      3.5.1 STE         → §5.3
      3.5.2 EMA 更新     → §5.4
      3.5.3 Dead-Code Reset → §5.5
  3.6 多層損失設計      → §7 三項損失
      3.6.1 L_quant      → §7.2
      3.6.2 L_inter      → §7.3
      3.6.3 L_commit     → §7.4
  3.7 課程式學習       → §8 Curriculum Learning
  3.8 資料增強         → §9 四種策略

第四章 實驗設計
  4.1 資料集與前處理    → train/val split
  4.2 模型設定         → 超參數表
  4.3 評估指標         → MSE, Entropy, Codebook Usage
  4.4 實驗一: Plan Ori  → baseline (rank=256, 無增強)
  4.5 實驗二: Plan Aug  → 增強 + rank=64
  4.6 消融實驗         → 各組件貢獻

第五章 實驗結果與分析
  5.1 訓練曲線分析      → loss curve, overfitting 觀察
  5.2 定量結果比較      → val MSE 表格
  5.3 定性結果         → 音訊聽感, spectrogram
  5.4 Codebook 使用分析 → entropy, usage distribution
  5.5 增強效果分析      → 有/無增強對照

第六章 結論與未來工作
  6.1 研究結論
  6.2 研究限制
  6.3 未來展望         → 2-layer RVQ, Adversarial Training
═══════════════════════════════════════════════════════════════
```

---

## 附錄：超參數速查表

| 超參數 | Plan Ori (exp_0206) | Plan Aug (exp_0216) | 說明 |
|--------|-------------------|-------------------|------|
| LoRA rank | 256 | 64 | 低秩分解的秩 |
| LoRA alpha | 512 | 128 | 縮放因子 |
| LoRA dropout | 0.2 | 0.2 | |
| Weight decay | 0.0 | 0.02 | L2 正則化 |
| VQ codebook size | 4096 | 4096 | |
| VQ dim | 128 | 128 | |
| EMA decay | 0.99 | 0.99 | |
| Dead-code threshold | 2 | 2 | |
| λ_quant | 1.0 | 1.0 | 量化對齊權重 |
| β_commit | 0.25 | 0.25 | Commitment 權重 |
| w_inter | 1.0 | 1.0 | 中間層監督權重 |
| Batch size | 8 | 8 | |
| Learning rate | 1e-4 | 1e-4 | AdamW |
| Curriculum start | 0.3 | 0.3 | |
| Curriculum end | 0.85 | 0.85 | |
| Augmentation | ✗ | ✓ | SNR Remix etc. |
| Trainable params | 4,718,592 (2.9%) | 926,144 (0.6%) | |
| Total params | ~164M | ~164M | |
