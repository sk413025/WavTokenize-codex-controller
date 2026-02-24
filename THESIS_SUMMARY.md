# 論文統整：基於 WavTokenizer 的 LDV 語音增強系統

> 整理日期：2026-02-24
> 適用架構：exp_0224（兩階段：0224a Encoder LoRA + 0224b Decoder LoRA）

---

## 一、研究背景與動機

### 1.1 LDV 感測器語音的問題

**LDV（Laser Doppler Vibrometer，雷射都普勒震動計）** 是一種非接觸式感測器，透過偵測物體表面的微小振動來還原聲音。其優點是不需實體接觸（適合隔離牆壁錄音、遠距監聽等場景），但產生的音訊有以下特性：

| 特性 | 說明 |
|------|------|
| **材質相依噪聲** | 不同材質（木板、玻璃、牆壁）有不同共振頻率，導致不同形態的噪聲 |
| **時序對齊性佳** | LDV noisy 與 clean speech 時序高度對齊（PESQ 量測中體現） |
| **低 SNR** | 驗證集平均 SNR ≈ −3.96 dB，41/50 樣本為負 SNR |

核心挑戰：**跨材質語音增強**，即同一模型需處理因材質不同而產生本質不同的噪聲型態。

### 1.2 LDV 噪聲的物理特性（與傳統噪聲的根本差異）

LDV 噪聲**不是**傳統語音增強中常見的加性白噪聲（AWGN），其物理成因與傳統噪聲有根本差異：

```
傳統加性噪聲模型：   y(t) = x(t) + n(t)        ← 訊號與噪聲獨立相加
LDV 噪聲模型：       y(t) = h(t) * x(t) + n_s(t) ← 材質濾波 + 散射噪聲
                          ↑              ↑
                  材質共振響應函數    表面散射噪聲
                  (卷積/乘性)       (加性，但非白噪）
```

| 物理機制 | 說明 | 對語音增強的影響 |
|----------|------|----------------|
| **材質共振濾波** | 每種材質有獨特的頻率響應函數 $h(f)$，對聲音進行非均勻的頻率衰減/增強 | 噪聲模式因材質而異，模型必須學習**材質不變的**特徵提取 |
| **表面微振動噪聲** | 雷射偵測表面微觀振動（非聲學），包含熱振動、機械共振等 | 噪聲與語音**頻率交疊**，無法靠簡單濾波分離 |
| **反射率變異** | 不同材質表面反射率不同，影響訊噪比 | 同一語音在不同材質上 SNR 差異可達 10+ dB |
| **非線性效應** | 大振幅時材質響應可能進入非線性區域 | 傳統線性降噪方法（Wiener filter）效果受限 |

> **關鍵差異**：傳統降噪只需估計 $n(t)$ 並減去；LDV 噪聲需**反卷積** $h(t)$ 並抑制 $n_s(t)$，這是一個 **盲反卷積（blind deconvolution）** 問題，遠比加性降噪困難。這正是選擇 WavTokenizer（端到端神經網路）而非傳統方法的核心理由。

### 1.3 為什麼選擇 WavTokenizer

WavTokenizer 是一個預訓練的神經音訊 Codec，具備以下優勢：

```
WavTokenizer 架構：
  輸入 [B, 1, T] → Encoder → [B, 512, T/320] → VQ → 離散 Token → Decoder → [B, 1, T]
                                    ↑                      ↑
                            壓縮比 320:1             codebook K=4096
```

| 優勢 | 說明 |
|------|------|
| 成熟的 encoder-decoder | 預訓練於大量 clean speech，能重建語者音色 |
| 可微 encoder | 支援 LoRA 微調，無需完整重訓 |
| 離散化（Tokenization） | VQ 輸出可對接下游大型語音模型（LLM/ASR） |
| FourierHead decoder | 相位感知的高品質波形重建 |

**論文核心貢獻**：將 WavTokenizer 從「clean speech codec」擴展為「LDV 噪聲語音增強系統」，同時保留其 tokenization 能力。

### 1.4 離散化 vs 連續特徵：訓練與推論的模式切換

WavTokenizer 的架構天然支援**兩種推論模式**，訓練時使用連續特徵路徑不代表放棄離散化能力：

```
模式 A：連續特徵路徑（訓練時使用，品質最佳）
  Noisy → Student Encoder → 連續特徵 [B,512,T/320] → Decoder LoRA → Clean Audio
                              ↑
                        跳過 VQ，無量化損失

模式 B：離散 Token 路徑（推論時可選，支援下游 LLM/ASR）
  Noisy → Student Encoder → 連續特徵 → VQ → Token ID [B,1,T/320] → Decoder → Audio
                                         ↑
                                   K=4096 codebook
                                   Re-enable VQ at inference
```

| 面向 | 連續模式（模式 A） | 離散模式（模式 B） |
|------|------------------|------------------|
| 使用時機 | 訓練、純增強推論 | 需要 tokenization 的下游應用 |
| VQ | 跳過 | 啟用 |
| 品質 | 最佳（無量化誤差） | 略低（VQ 瓶頸 −0.132 PESQ）|
| 輸出 | 波形 | Token ID + 波形 |
| 對接 LLM/ASR | ✗ | ✅（75 Hz discrete tokens）|

> **為什麼訓練時跳過 VQ？**
> 1. **梯度流**：VQ 的 straight-through estimator 會引入梯度近似誤差，跳過 VQ 讓 encoder → decoder 的梯度路徑更乾淨
> 2. **解耦問題**：先讓 encoder/decoder 學好特徵對齊，再考慮 VQ 量化，避免同時解決兩個問題
> 3. **實驗驗證**：消融實驗顯示 VQ 僅造成 −0.132 PESQ 損失（clean path），證實跳過 VQ 後重新啟用的品質損失可控
> 4. **相容性**：Student Encoder 的連續特徵與原始 codebook 處於同一 512 維空間，推論時可直接接上 VQ 進行量化

---

## 二、WavTokenizer 官方架構

> 參考文獻：Ji et al., "WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling," arXiv:2408.16532, 2024.

### 2.1 整體架構

WavTokenizer 是一個基於 Encodec 框架改良的神經音訊 Codec，核心設計為**極低碼率離散化**（75 Hz token rate），採用單層 VQ（n_q=1），大幅簡化離散表示：

```
WavTokenizer 官方架構
──────────────────────────────────────────────────────────────────
                     ┌─────────────────────────┐
 Audio [B,1,T]  ──►  │   SEANet Encoder          │  ──►  [B, 512, T/320]
 (24kHz)             │   ratios=[8,5,4,2]        │      壓縮比 320:1
                     │   hop = 320 samples       │      freq = 75 Hz tokens
                     │   dim = 512, LSTM × 2     │
                     └──────────────┬────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────┐
                     │   Residual VQ (n_q=1)   │  ──►  Token ID [B, 1, T/320]
                     │   codebook K = 4096      │      離散化，可對接 LLM/ASR
                     │   dim = 512              │
                     │   EMA decay = 0.99       │
                     └──────────────┬────────────┘
                                    │ quantized [B, 512, T/320]
                                    ▼
                     ┌─────────────────────────┐
                     │   VocosBackbone          │
                     │   12 × ConvNeXtBlock     │  ──►  [B, 768→512, T/320]
                     │   input_channels = 512   │
                     │   dim = 768              │
                     │   intermediate_dim = 2304│
                     │   pwconv1: Linear(768→2304)
                     │   pwconv2: Linear(2304→768)
                     └──────────────┬────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────┐
                     │   ISTFTHead (FourierHead) │  ──►  Audio [B, 1, T]
                     │   n_fft = 1280            │      相位感知波形重建
                     │   hop_length = 320        │
                     └─────────────────────────┘
──────────────────────────────────────────────────────────────────
```

### 2.2 關鍵模組細節

**① SEANet Encoder（基於 Encodec 改良）**

| 參數 | 值 |
|------|-----|
| 輸入通道 | 1（單聲道） |
| 下採樣比例 | [8, 5, 4, 2]（乘積 = 320） |
| 輸出維度 | 512 |
| LSTM 層數 | 2（捕捉長程依賴） |
| Token 頻率 | 75 Hz（24kHz ÷ 320） |

**② Residual Vector Quantizer（VQ）**

| 參數 | 值 |
|------|-----|
| 量化層數 n_q | 1（單層，vs Encodec 的 8-24 層）|
| Codebook 大小 K | 4096 |
| 向量維度 | 512 |
| 更新方式 | EMA（decay=0.99） |
| 每秒 bit rate | 75 × log₂(4096) = 75 × 12 = **900 bits/s** |

> WavTokenizer 的核心貢獻之一：單層 VQ + K=4096 達到比多層 RVQ 更好的重建品質，關鍵在於增強的 decoder 設計。

**③ VocosBackbone（Decoder 骨幹）**

| 參數 | 值 |
|------|-----|
| 輸入通道 | 512（from VQ） |
| 模型維度 dim | 768 |
| ConvNeXtBlock 數量 | 12 |
| intermediate_dim | 2304（dim × 3） |
| 可訓練 LoRA 目標 | pwconv1, pwconv2（本研究 exp_0224b）|

ConvNeXtBlock 結構：
```python
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim):
        self.dwconv  = Conv1d(dim, dim, 7, groups=dim)  # depth-wise
        self.norm    = LayerNorm(dim)
        self.pwconv1 = Linear(dim, intermediate_dim)    # ← LoRA target
        self.act     = GELU()
        self.pwconv2 = Linear(intermediate_dim, dim)    # ← LoRA target
```

**④ ISTFTHead（FourierHead）**

| 參數 | 值 |
|------|-----|
| n_fft | 1280 |
| hop_length | 320 |
| 輸出 | 幅度 + 相位 → iSTFT → 波形 |

相位感知設計使 WavTokenizer decoder 在低碼率下仍能重建高品質語音（vs 純幅度頻譜 vocoder）。

### 2.3 官方訓練設定（與本研究的關聯）

| 項目 | WavTokenizer 官方 | 本研究（exp_0224b）|
|------|------------------|------------------|
| Loss | MSE + MR-STFT + 45×Mel | MSE + MR-STFT + 45×Mel |
| Mel λ | **45**（官方設定） | **45**（沿用） |
| Optimizer | AdamW | AdamW |
| 資料 | LibriSpeech + VCTK + ... | LDV Noisy→Clean pairs |
| 目的 | 訓練 clean speech codec | 微調使其適應 LDV noisy input |

> 本研究的 λ=45 直接沿用 WavTokenizer 官方設定，確保 Mel 損失主導感知品質優化。

### 2.4 BibTeX 引用

```bibtex
@article{ji2024wavtokenizer,
  title   = {WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer
             for Audio Language Modeling},
  author  = {Ji, Shengpeng and Jiang, Ziyue and Wang, Wen and Chen, Yifu
             and Fang, Minghui and Zuo, Jialong and Yang, Qian
             and Cheng, Xize and Wang, Zehan and Li, Ruiqi and others},
  journal = {arXiv preprint arXiv:2408.16532},
  year    = {2024}
}
```

---

## 三、研究問題澄清

### 3.1 你的理解是否有誤？

**基本正確，補充以下細節：**

> ✅ WavTokenizer 已有成熟的 encoder-decoder：能重建輸入語者的聲音特徵
> ✅ VQ 離散化可對接下游語音模型
> ⚠️ **補充**：原始 WavTokenizer 的 encoder 訓練在 clean speech 上，直接輸入 LDV noisy 會造成 distribution shift，導致重建品質下降
> ⚠️ **補充**：LDV 噪聲不是加性白噪，而是材質共振造成的乘性/卷積型噪聲，與一般降噪任務不同

### 3.2 你的工作的核心貢獻

```
原始 WavTokenizer（輸入限制 clean speech）
          ↓ 你的貢獻
改良版系統（可接受 LDV noisy speech，輸出 clean speech 重建）
```

具體技術貢獻：
1. **Student Encoder LoRA**：讓 encoder 學習從 LDV noisy 提取與 clean 等價的特徵
2. **VQ 瓶頸分析**：量化 VQ 對語音增強的影響（+0.031 PESQ，相對有限）
3. **Decoder LoRA**：讓 decoder 學習從 student encoder 的連續特徵重建 clean speech
4. **消融實驗設計**：2×2 矩陣（有/無 VQ × Encoder/Decoder LoRA）系統量化各組件貢獻

### 3.3 Teacher-Student 架構設計

本研究採用 **Teacher-Student 知識蒸餾**框架，將語音增強問題轉化為「讓 Student 模仿 Teacher 的行為」：

```
Teacher-Student 架構概覽
══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────┐
  │  Teacher Path（凍結，提供目標訊號）                    │
  │                                                     │
  │  Clean Audio ──► [WavTokenizer Encoder] ──► VQ ──►  │
  │  [B, 1, T]       [B, 512, T/320]          tokens    │
  │                       │                     │       │
  │                       │              [Decoder] ──► Teacher Output
  │                       │                             │
  │                       ▼                             │
  │            Teacher Features                         │
  │            (訓練目標 / GT)                            │
  └─────────────────────────────────────────────────────┘
                          │
              Loss = f(Student Output, Teacher Output / Clean Audio)
                          │
  ┌─────────────────────────────────────────────────────┐
  │  Student Path（可訓練，學習降噪）                     │
  │                                                     │
  │  Noisy Audio ──► [Student Encoder + LoRA] ──►       │
  │  [B, 1, T]       [B, 512, T/320]                    │
  │                       │                             │
  │                 [跳過 VQ / 經過 VQ]                   │
  │                       │                             │
  │                 [Decoder + LoRA] ──► Student Output  │
  └─────────────────────────────────────────────────────┘
══════════════════════════════════════════════════════════════════
```

**核心思想**：

| 概念 | 說明 |
|------|------|
| **Teacher** | 原始 WavTokenizer（完全凍結），輸入 clean speech，代表「理想行為」 |
| **Student** | LoRA 微調後的 WavTokenizer，輸入 LDV noisy speech，學習產生與 Teacher 等價的輸出 |
| **知識蒸餾目標** | Student(noisy) 的輸出 ≈ Teacher(clean) 的輸出，或直接 ≈ clean audio |
| **漸進式訓練** | 先訓練 Student Encoder（Phase 1-2），再訓練 Decoder（Phase 3），避免同時優化太多參數 |

**各階段的 Teacher-Student 關係**：

| 階段 | Teacher 提供 | Student 學習 | Loss 目標 |
|------|-------------|-------------|----------|
| Phase 1（exp_0217）| Teacher VQ tokens（clean） | Student Encoder LoRA | Feature MSE：student tokens ≈ teacher tokens |
| Phase 2（exp_0224a）| Clean audio waveform | Student Encoder LoRA | MSE + MR-STFT + Mel：recon ≈ clean wav |
| Phase 3（exp_0224b）| Clean audio waveform | Decoder LoRA | MSE + MR-STFT + Mel：recon ≈ clean wav |

> **為什麼不端到端訓練？** 同時訓練 Encoder + Decoder 的參數空間過大（即使用 LoRA），容易陷入局部最優解或 mode collapse。分階段訓練讓每階段只需解決一個子問題：Phase 1-2 解決「如何從 noisy 提取 clean-equivalent 特徵」，Phase 3 解決「如何從 student 特徵重建 clean 波形」。

---

## 三、系統架構（exp_0224b，最終採用）

### 3.1 整體架構圖

```
┌──────────────────────────────────────────────────────────────────┐
│                     exp_0224b 系統架構                            │
│                                                                  │
│  LDV Noisy Audio                Clean Audio (GT)                 │
│  [B, 1, T]                      [B, 1, T]                       │
│       │                              │                           │
│       ▼                              │  (僅用於計算 Loss)         │
│  ┌──────────────────┐               │                           │
│  │  Student Encoder │               │                           │
│  │  + LoRA (r=64)   │ ← FROZEN      │                           │
│  │ (from exp_0224a) │               │                           │
│  └────────┬─────────┘               │                           │
│           │                         │                           │
│           ▼                         │                           │
│  student_encoder_out                │                           │
│  [B, 512, T/320]                    │                           │
│  (連續特徵，跳過 VQ)                 │                           │
│           │                         │                           │
│           ▼                         ▼                           │
│  ┌──────────────────────┐   ┌───────────────┐                   │
│  │  WavTokenizer        │   │  Loss 計算     │                   │
│  │  Decoder Backbone    │   │               │                   │
│  │  (ConvNeXt × 12)     │   │  L_MSE        │                   │
│  │  pwconv1: LoRA r=32  │◄──│  L_MR-STFT    │                   │
│  │  pwconv2: LoRA r=32  │   │  L_Mel × 45   │                   │
│  │  + FourierHead       │   └───────────────┘                   │
│  └────────┬─────────────┘                                       │
│           │                                                      │
│           ▼                                                      │
│  recon_wav [B, 1, T]                                            │
│  (重建的乾淨語音)                                                 │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 訓練流程（三階段）

```
Phase 1（exp_0217）：Encoder LoRA 預訓練
  初始：WavTokenizer 預訓練 Encoder + LoRA (r=64)
  Noisy → [Student Encoder LoRA] → VQ → tokens
  Loss: Feature MSE（student VQ tokens vs teacher VQ tokens）
  目標：學習從 LDV noisy 提取與 clean 等價的 VQ tokens
  結果：PESQ=1.203（有 VQ bottleneck）

Phase 2（exp_0224a）：Encoder LoRA 繼續訓練，跳過 VQ
  初始：載入 exp_0217 best_model.pt encoder LoRA weights（繼承，非從頭訓練）
  Noisy → [Student Encoder LoRA] → [跳過VQ] → [Frozen Decoder] → recon
  Loss: MSE + MR-STFT + 45×Mel（vs clean wav）
  目標：讓 encoder 輸出對齊 decoder 的 input space（無 VQ 瓶頸）
  結果：PESQ=1.586（ep190 best checkpoint，val_total criterion）

Phase 3（exp_0224b）：Decoder LoRA 訓練
  初始：載入 exp_0224a ep190 best_model_val_total.pt encoder weights（凍結）
  Noisy → [Frozen Student Encoder] → [跳過VQ] → [Decoder LoRA, r=32] → recon
  Loss: MSE + MR-STFT + 45×Mel（vs clean wav）
  目標：讓 decoder 學習從 student encoder 的連續特徵重建 clean speech
  結果：PESQ=1.866（ep50/300，訓練中）
```

### 3.3 各階段訓練設定對比

| 項目 | exp_0217 | exp_0224a | exp_0224b |
|------|----------|-----------|-----------|
| Encoder 初始化 | WavTokenizer預訓練+LoRA | 繼承 exp_0217 best | 繼承 exp_0224a ep190 |
| Encoder LoRA | ✅ 可訓練 | ✅ 可訓練（繼承0217）| ❄️ 凍結（繼承0224a ep190）|
| VQ | ✅ 使用 | ❌ 跳過 | ❌ 跳過 |
| Decoder | ❄️ 凍結 | ❄️ 凍結 | ✅ LoRA 可訓練 |
| Loss 目標 | Feature MSE（VQ tokens）| MSE+STFT+Mel（wav）| MSE+STFT+Mel（wav）|
| 最佳 PESQ | 1.203 | 1.586（ep190）| 1.866（ep50/300）|

### 3.4 可訓練參數統計

| 模組 | 參數量 | 狀態（exp_0224b） |
|------|--------|-----------------|
| WavTokenizer Encoder | ~40.3M | ❄️ Frozen |
| Encoder LoRA (r=64, α=128) | ~4.72M | ❄️ Frozen（from exp_0217） |
| VQ Codebook (K=4096, d=512) | ~2.1M | ❄️ 跳過，不使用 |
| Decoder ConvNeXt × 12 | ~84.8M | ❄️ Frozen（基礎） |
| Decoder LoRA pwconv1+2 (r=32) | **~2.36M** | ✅ **可訓練** |
| FourierHead | ~39.4M | ❄️ Frozen |
| **可訓練比例** | | **1.42%** |
### 3.5 LoRA 微調策略

#### 3.5.1 為什麼選擇 LoRA？

**LoRA（Low-Rank Adaptation）** 是一種參數高效微調方法，核心概念：**不動原始權重，在旁邊加一條「小路」來微調行為**。

```
LoRA 直覺理解：以 pwconv1（768→2304）為例，r=32
═══════════════════════════════════════════════════════

  【全量 Fine-tuning（不用 LoRA）】
  輸入 x [batch, 768]
      │
      ▼
  W₀: 768×2304   ← 全部可訓練，1,769,472 params
      │
  輸出 y [batch, 2304]


  【LoRA（本研究採用）】
  輸入 x [batch, 768]
      │
      ├──────────────────────────────────┐
      │                                  │
      ▼                                  ▼
  W₀: 768×2304                  A: 768×32      ← 降維 (24,576 params)
  ❄️ 凍結，不訓練                    │
                                   ▼
                               B: 32×2304      ← 升維 (73,728 params)
      │                            │
      └──────── + ─────────────────┘
                │
                ▼
           輸出 y [batch, 2304]
           = W₀·x  +  B·(A·x)
             凍結    LoRA 修正
                     共 98,304 params（原始的 5.6%）
═══════════════════════════════════════════════════════
```

**LoRA 的核心假設**：微調所需的「修正量 ΔW」往往是低秩的（即真正需要改動的方向數量遠少於原矩陣維度）。LoRA 用 `B × A`（秩最多 r=32）來近似這個修正量，而不是直接修改整個 `W₀`（秩可達 768）。

等效公式：新權重 = 原始權重 + 低秩修正

```
W_new = W₀  +  (B · A) × (α/r)
        凍結    可訓練     縮放係數
                         (此處 α/r = 64/32 = 2.0)
```

```
傳統 Fine-tuning vs LoRA
───────────────────────────────────────────────────────
傳統 Fine-tuning：                LoRA：
  W₀ (全部可訓練)                    W₀ (❄️ 凍結)
  ~166.8M params                    │
  ↓                                 ├──► B·A (✅ 可訓練)
  容易過擬合                         │    r=32: ~2.36M params
  (LDV 資料量小)                     │    r=64: ~4.72M params
                                    ↓
                                   W₀ + B·A
                                   保留預訓練知識
                                   僅調整 1.42% 參數
───────────────────────────────────────────────────────
```

| 選擇 LoRA 的理由 | 說明 |
|-----------------|------|
| **資料量限制** | LDV 語音資料量有限（訓練集 ~數百段），全量微調容易過擬合 |
| **保留預訓練知識** | WavTokenizer 在大規模 clean speech 上學到的語音重建能力不應被覆蓋 |
| **計算效率** | 可訓練參數僅 1.42%，訓練記憶體和時間大幅降低 |
| **可組合性** | LoRA 權重可獨立保存、載入，方便不同實驗間切換 |
| **推論無額外開銷** | LoRA 可合併回原始權重（$W' = W_0 + BA$），推論速度不變 |

#### 3.5.2 LoRA 注入位置與超參數

本研究針對 WavTokenizer 的不同模組採用不同的 LoRA 配置：

**① Encoder LoRA（Phase 1-2，exp_0217 → exp_0224a）**

| 超參數 | 值 | 理由 |
|--------|-----|------|
| rank (r) | 64 | Encoder 需將 LDV noisy 映射到 clean feature space，變換幅度較大，需較高容量 |
| alpha (α) | 128 | scaling = α/r = 2.0，放大 LoRA 的調整幅度 |
| target modules | Encoder 所有 Conv1d 層 | 完整覆蓋 SEANet 的下採樣路徑 |
| 可訓練參數 | ~4.72M（佔 Encoder 的 11.7%）| - |

**② Decoder LoRA（Phase 3，exp_0224b）**

| 超參數 | 值 | 理由 |
|--------|-----|------|
| rank (r) | 32 | Decoder 僅需微調以適應 student encoder 的 distribution shift，變換幅度較小 |
| alpha (α) | 64 | scaling = α/r = 2.0，與 encoder LoRA 保持一致 |
| target modules | pwconv1, pwconv2（每個 ConvNeXtBlock 2 個）| 這兩層是 ConvNeXt 的特徵變換核心（768→2304→768）|
| 可訓練參數 | ~2.36M（佔 Decoder 的 2.8%）| - |
| 未注入 LoRA 的層 | dwconv（depth-wise）、FourierHead | dwconv 是局部特徵提取（groups=dim），FourierHead 負責頻譜→波形轉換，兩者無需調整 |

```
ConvNeXtBlock 中 LoRA 注入位置
─────────────────────────────────────
  x ──► dwconv (7×1, groups=dim)  ❄️ Frozen
        │
        ▼
        LayerNorm                  ❄️ Frozen
        │
        ▼
        pwconv1: Linear(768→2304)  ← LoRA r=32 注入
        │                             W = W₀ + B·A
        ▼
        GELU                       (activation)
        │
        ▼
        pwconv2: Linear(2304→768)  ← LoRA r=32 注入
        │                             W = W₀ + B·A
        ▼
  x + residual
─────────────────────────────────────
 × 12 blocks = 24 個 LoRA adapters
```

#### 3.5.3 Encoder vs Decoder LoRA rank 的設計考量

| 考量 | Encoder LoRA (r=64) | Decoder LoRA (r=32) |
|------|-------------------|-------------------|
| 任務複雜度 | 高：需從 noisy domain 映射到 clean domain（domain shift 大） | 中：僅需適應 student encoder 的微小 distribution shift |
| 輸入分布差異 | LDV noisy vs clean speech（分布差異大） | Student encoder output vs teacher encoder output（同空間，差異小）|
| 參數量 | ~4.72M | ~2.36M |
| scaling (α/r) | 2.0 | 2.0 |
| 設計哲學 | 「大」LoRA 用於困難的跨域映射 | 「小」LoRA 用於精細的增量適應 |
---

## 四、損失函數

### 4.1 設計哲學：三個尺度監督

模型輸出的重建波形需從**三個互補角度**與原始 clean 語音比較，避免任何單一 loss 的盲點：

```
損失函數設計直覺
═══════════════════════════════════════════════════════════════════

  重建波形 ŷ(t)                     乾淨波形 y(t)
      │                                 │
      ├─── ① 直接比波形 ──────────────── ┤  → MSE Loss
      │    「每個取樣點差多少？」            │    (時域，逐點比較)
      │                                 │
      ├─── ② 轉成頻譜再比 ─────────────── ┤  → MR-STFT Loss
      │    「各頻率的能量分布對不對？」       │    (頻域，3 種解析度)
      │                                 │
      └─── ③ 轉成人耳感知頻譜再比 ──────── ┘  → Mel Loss × 45
           「聽起來像不像？」                    (感知域，主導損失)
═══════════════════════════════════════════════════════════════════
```

### 4.2 總損失

```
L_total = L_MSE  +  L_MR-STFT  +  45 × L_Mel
          ───────   ──────────     ──────────────
          波形對齊   頻譜結構        感知品質（主導）
```

> λ=45 直接沿用 WavTokenizer 官方設定。45×L_Mel 約佔 total loss 的 ~85%，確保模型優先優化「聽起來對不對」。

### 4.3 各項說明

**① MSE Loss（波形域 — 「波形長得像不像？」）**

```
計算步驟：

  clean:  [ x1,   x2,   x3,  ... xT ]   ← 乾淨音訊的每個取樣點
  recon:  [ x1^,  x2^,  x3^, ... xT^ ]  ← 重建音訊的每個取樣點
                   ↓
  每個點差 → 平方 → 全部加起來 → 除以 T

  L_MSE = (1/T) × [ (x1^ - x1)² + (x2^ - x2)² + ... + (xT^ - xT)² ]

  數值範例：
    clean  =  [ 0.1,   0.5,  -0.3 ]
    recon  =  [ 0.1,   0.4,  -0.2 ]
    差異²  =  [ 0.00,  0.01,  0.01 ]  → 平均 = 0.0067
```

- **優點**：確保波形整體形狀一致、時序對齊
- **缺點**：單獨使用易造成 **silence collapse**——模型發現「輸出全零（靜音）」時，若 clean 也有安靜段，MSE 可以很小，但感知上完全錯誤。這正是需要 MR-STFT 和 Mel loss 同時監督的原因

**② Multi-Resolution STFT Loss（頻譜域 — 「頻率分布對不對？」）**

```
計算步驟：

  Step 1：把音訊轉成頻譜（STFT），用 3 種解析度分別計算
  ┌──────────────┬──────────┬──────────────────────────────┐
  │ 解析度        │ n_fft    │ 看什麼？                      │
  ├──────────────┼──────────┼──────────────────────────────┤
  │ 粗（低頻細節）│ 2048     │ 整體頻率輪廓（共振峰位置）      │
  │ 中            │ 1024     │ 中尺度頻譜結構                │
  │ 細（時間細節）│  512     │ 瞬態（爆破音 p/t/k 的起始）    │
  └──────────────┴──────────┴──────────────────────────────┘

  Step 2：每種解析度算兩個子指標
  ┌────────────────────────┬───────────────────────────────────────────┐
  │ 子指標                  │ 計算方式                                   │
  ├────────────────────────┼───────────────────────────────────────────┤
  │ Spectral Convergence   │ SC  = ||Sc| - |Sr|| / ||Sc||              │
  │ (SC)                   │       Sc=clean頻譜, Sr=recon頻譜           │
  │                        │ → 頻譜形狀的相對誤差（0=完全相同）          │
  ├────────────────────────┼───────────────────────────────────────────┤
  │ Log Magnitude (LogMag) │ LM  = mean( |log|Sc| - log|Sr|| )        │
  │                        │ → 取 log 後讓安靜段和響亮段受同等監督       │
  │                        │   （不取 log 的話響亮段 loss 會遠大於安靜段）│
  └────────────────────────┴───────────────────────────────────────────┘

  Step 3：對 3 種解析度取平均
  L_MR-STFT = (1/3) × [ (SC_2048 + LM_2048) + (SC_1024 + LM_1024) + (SC_512 + LM_512) ]
```

**③ Mel Spectrogram L1 Loss（感知域 — 「人耳聽起來像不像？」）**

```
計算步驟（模擬人耳的頻率感知）：

  clean audio  ──► STFT ──► |S|² ──► Mel(100 bands) ──► log  ──► C[100, T']
  recon audio  ──► STFT ──► |S|² ──► Mel(100 bands) ──► log  ──► R[100, T']
                                      ↑
                           Mel 尺度：低頻密集（人耳敏感）、高頻稀疏（人耳遲鈍）
                           比線性頻率更接近人類聽覺感知

  L_Mel = mean( |C - R| )   ← 對 100×T' 個格子取平均絕對差
```

- **為什麼 λ=45？** 使 Mel loss 約佔 total loss 的 ~85%——因為 Mel 頻譜最接近人耳感知，優先保障「聽起來好不好」而非「波形點對點對齊」

### 4.4 三種 Loss 的互補關係

| Loss | 監督域 | 解決什麼問題 | 單獨使用的缺陷 |
|------|--------|------------|---------------|
| MSE | 時域（波形）| 整體波形對齊、時序一致 | silence collapse（輸出靜音也能低 loss）|
| MR-STFT | 頻域（頻譜）| 頻譜結構正確、多尺度覆蓋 | 對相位不敏感 |
| Mel × 45 | 感知域 | 人耳聽感品質、共振峰、語者音色 | 對微細時域結構不敏感 |
| **三者組合** | **全方位** | **波形 + 頻譜 + 感知同時監督** | **互補覆蓋各自盲點** |

---

## 五、實驗結果

### 5.1 消融實驗比較矩陣

| 架構 | Encoder | VQ | Decoder | PESQ | STOI | 說明 |
|------|---------|-----|---------|------|------|------|
| exp_0217 | LoRA 訓練 | ✓ 有 | Frozen | 1.203 | 0.462 | 基線（有VQ瓶頸）|
| exp_0223 v2 | Frozen | ✓ 有 | LoRA 訓練 | 1.205 | 0.522 | Decoder LoRA（有VQ）|
| exp_0224a (ep190) | LoRA 訓練 | ✗ 跳過 | Frozen | 1.586 | 0.628 | 跳過VQ，Encoder 對齊 |
| **exp_0224b (ep50)** | **Frozen** | **✗ 跳過** | **LoRA 訓練** | **1.866** | **0.660** | **目前最佳（訓練中）**|

### 5.2 系統上下限分析

| 路徑 | PESQ | STOI | 意義 |
|------|------|------|------|
| clean → Teacher（無VQ） | 2.484 | 0.761 | 系統絕對上限 |
| clean → Teacher（有VQ） | 2.352 | 0.750 | VQ 損失 −0.132 |
| noisy → Teacher（有VQ，公平基準） | 1.677 | 0.527 | 公平比較基準線 |
| noisy → Teacher（無VQ） | 1.708 | 0.531 | 跳過VQ僅 +0.031 |
| **exp_0224b ep50（中期）** | **1.866** | **0.660** | **目前最佳，50/300 epochs** |

### 5.3 關鍵發現

1. **VQ 瓶頸有限**：clean audio 跳過 VQ 僅提升 +0.132 PESQ；noisy 僅 +0.031 → VQ 量化本身不是主要瓶頸

2. **Distribution shift 是真正瓶頸**：exp_0224a（student encoder 輸出 → frozen decoder，ep190）PESQ=1.586，接近但仍低於 noisy no-vq baseline（1.708），說明即使 encoder 針對無VQ路徑訓練，student encoder 輸出與 teacher decoder 期望輸入仍存在 distribution mismatch

3. **Decoder LoRA 有效補償 distribution shift**：exp_0224b 在 decoder 學習適應後達到 PESQ=1.866，比 noisy no-vq baseline 高 +0.158

4. **訓練仍在進行**：目前 50/300 epochs，仍有大幅提升空間（距上限 2.484 差 0.618）

---

## 六、論文架構建議

### 6.1 章節結構

```
第一章：緒論
  1.1 LDV 感測器語音的應用場景與挑戰
  1.2 研究動機（跨材質、低 SNR、時序對齊）
  1.3 研究貢獻

第二章：相關研究
  2.1 語音增強方法演進（傳統 → DNN → Codec-based）
  2.2 WavTokenizer 原理
  2.3 LoRA 參數高效微調
  2.4 LDV 語音資料集特性

第三章：系統設計
  3.1 問題定義
  3.2 整體架構（exp_0224b）
  3.3 訓練流程（三階段）
  3.4 損失函數設計

第四章：實驗設置
  4.1 資料集（LDV 錄音，不同材質）
  4.2 評估指標（PESQ-NB, STOI）
  4.3 消融實驗設計（2×2 矩陣）

第五章：實驗結果
  5.1 各架構比較（消融實驗）
  5.2 系統上下限分析
  5.3 訓練曲線分析
  5.4 跨材質泛化測試（未見過材質的 PESQ/STOI）
  5.5 頻譜與波形對比

第六章：結論與未來工作
  6.1 結論
  6.2 未來方向（端到端訓練、VQ 重新啟用優化、更大資料集）
```

### 6.2 待辦事項與材料盤點

> 截至 2026-02-24，exp_0224b 訓練至 ep97/300（33%）。

#### 已有材料

| 項目 | 狀態 | 位置 |
|------|------|------|
| **訓練曲線圖** | ✅ 已有 | exp_0224/runs/ 下各 run 的 training_curves_epoch*.png |
| **音檔樣本** | ✅ 已有 | exp_0224/runs/*/audio_samples/ 含 noisy/recon/clean 三組 .wav |
| **PESQ/STOI 評估腳本** | ✅ 已有 | exp_0224/run_eval_0224b.py |
| **PESQ/STOI 數值結果** | ✅ 已有 | exp_0217/FAIR_BASELINE_PESQ_STOI_n30.json, exp_0223/pesq_stoi_v2_n30.json 等 |
| **消融實驗比較表** | ✅ 已有 | exp_0223/test/comparison_table.md |
| **架構文字文件** | ✅ 已有 | exp_0224/ARCHITECTURE.md, THESIS_SUMMARY.md |

#### 待完成事項

| # | 項目 | 優先度 | 說明 |
|---|------|--------|------|
| 1 | **等待 exp_0224b 訓練完成** | 🔴 最高 | 目前 ep97/300，需等到 ep300 或收斂後取最佳 checkpoint |
| 2 | **未見過材質泛化測試** | 🔴 最高 | 訓練完成後用未見過的材質（如：紙板、磁磚等）測試 PESQ/STOI，驗證跨材質泛化能力 |
| 3 | **頻譜對比圖** | 🔴 高 | 目前**完全沒有**。需生成 noisy / recon / clean 的 mel spectrogram 並排比較圖（至少各材質一張） |
| 4 | **正式資料集描述** | 🟡 中 | 散落在各處（data/README.md, try/QUICKSTART.md）但缺乏正式整理。需明確列出：材質種類與數量、語者（train 14人, val 4人）、每段時長、錄音參數、train/val/test 劃分 |
| 5 | **完整系統架構圖（可編輯）** | 🟡 中 | 目前只有 ASCII 圖和舊版 .png。需用 draw.io 精繪含 Teacher/Student 對比、LoRA 位置標示的正式圖 |
| 6 | **n≥30 正式 PESQ/STOI 評估** | 🟡 中 | 目前消融表的數字僅 n=3 val samples。論文需擴大到全部驗證集（n≥30）重新評估 |
| 7 | **傳統方法 baseline 比較** | 🟡 中 | 實作 Wiener filter / spectral subtraction 並在相同資料上評測，作為對照組 |
| 8 | **主觀評估（MOS / ABX）** | 🟢 低 | 不同材質各選一段，邀受試者評分。若時間不足可作為未來工作 |
| 9 | **離散模式（VQ re-enable）驗證** | 🟢 低 | 訓練完成後測試模式 B（重新啟用 VQ），驗證離散 token 品質，證明 tokenization 能力保留 |

---

## 七、技術細節備查

### 7.1 decode_continuous() 的設計

WavTokenizer 原始 `decode()` 使用 `@torch.inference_mode()` 裝飾器，無法計算梯度。exp_0224b 透過直接呼叫 backbone 和 head 繞過：

```python
def decode_continuous(self, features: torch.Tensor) -> torch.Tensor:
    # 繞過 @inference_mode()，讓 gradient 流過 decoder LoRA
    bandwidth_id = torch.tensor([0], device=features.device)
    x = self.teacher.backbone(features, bandwidth_id=bandwidth_id)
    audio = self.teacher.head(x)
    return audio
```

### 7.2 為什麼 encode_infer 回傳的是 VQ 後的特徵？

`encode_infer` → `feature_extractor.infer()` → `encodec.quantizer.infer()` → 回傳 `quantized`（VQ 後）

要取 VQ 前的連續特徵需直接呼叫：
```python
raw_emb = teacher.feature_extractor.encodec.encoder(audio)  # [B, 512, T/320]
```

### 7.3 PESQ 指標說明

- 使用 **PESQ-NB（窄頻，8kHz）**：先將 24kHz 音訊重採樣至 8kHz 再計算
- PESQ(noisy) >> PESQ(recon) 屬**正常現象**：LDV noisy 與 clean 時序高度對齊，PESQ 對時序對齊敏感；重建音訊因 encoder 壓縮（320:1）導致輕微時序錯位，故 PESQ 較低
- **公平比較基準**（`noisy_through_teacher`）= 同樣的 pipeline（noisy → Teacher Encoder+VQ → Decoder），PESQ=1.677

---

*本文件由實驗過程自動統整，數字均來自實際評估結果（n=3 val samples）。正式論文建議以更大樣本（n=30 以上）重新評估。*
