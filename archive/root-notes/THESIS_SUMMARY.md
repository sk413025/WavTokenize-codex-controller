# 論文統整：基於 WavTokenizer 的 LDV 語音增強系統

> 初版日期：2026-02-24 ｜ 更新日期：2026-03-02
> 主推架構：**Encoder-only LoRA + Frozen Decoder**（無機械音、保留 token 相容性）
> 消融對照：Decoder LoRA（exp_0224b）/ E2E LoRA（exp_0226）作為 ablation study
> ⚠️ **損失函數仍在迭代中**（exp_0225~0229 系列），架構已確定但最佳 loss 配置待定

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
| **高頻衰減** | 雷射反射效率隨頻率下降，4 kHz 以上能量大幅損失 | 高頻資訊在感測器端**物理性消失**，模型須從低頻上下文「幻想」（hallucinate）高頻 |
| **相位畸變（phase jitter）** | 反射角度不穩定、表面微觀抖動造成 phase 隨機跳動 | 傳統時頻域方法難以處理；端到端模型需隱式學習 phase 修正 |
| **頻率相依 SNR** | 低頻 SNR 尚可，高頻 SNR 極低甚至被噪聲完全掩埋 | 不同頻段需要**不同程度**的增強策略，均勻去噪反而有害 |
| **表面微振動噪聲** | 雷射偵測表面微觀振動（非聲學），包含熱振動、機械共振等 | 噪聲與語音**頻率交疊**，無法靠簡單濾波分離 |
| **反射率變異** | 不同材質表面反射率不同，影響訊噪比 | 同一語音在不同材質上 SNR 差異可達 10+ dB |
| **非線性效應** | 大振幅時材質響應可能進入非線性區域 | 傳統線性降噪方法（Wiener filter）效果受限 |

> **關鍵差異**：傳統降噪只需估計 $n(t)$ 並減去；LDV 噪聲需**反卷積** $h(t)$ 並抑制 $n_s(t)$，這是一個 **盲反卷積（blind deconvolution）** 問題，遠比加性降噪困難。這正是選擇 WavTokenizer（端到端神經網路）而非傳統方法的核心理由。

> **本質是超解析度問題**：由於高頻在感測器端已物理性消失（4 kHz 以上衰減嚴重），LDV 語音增強**不僅僅是降噪**，更本質上是一個**帶寬擴展（bandwidth extension）/ 超解析度（super-resolution）**問題——模型必須從有限的低頻上下文推斷並生成缺失的高頻成分。這解釋了為何傳統降噪方法（假設 clean 頻譜完整保留）在 LDV 場景效果有限，也是本論文選擇具備生成能力的 WavTokenizer 作為骨幹的關鍵理由。

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
  Noisy → Student Encoder → 連續特徵 [B,512,T/320] → Frozen Decoder → Clean Audio
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

### 2.3 官方訓練方式：GAN 對抗訓練（判別器非 HiFi-GAN）

WavTokenizer 的訓練**範式**與 HiFi-GAN 相似——都是 **Generator-Discriminator 的 GAN 對抗訓練**——但**判別器組合不同**：HiFi-GAN 使用 MPD + MSD（Multi-Scale Discriminator），WavTokenizer 則使用 MPD + MRD（Multi-Resolution Discriminator，源自 UnivNet）+ DAC Discriminator，**整個系統不含任何 MSD**。WavTokenizer 在 Generator 前面多了 Encoder + VQ：

```
WavTokenizer 官方訓練架構
──────────────────────────────────────────────────────────────────

  Generator（全參數可訓）：
  ┌─────────┐    ┌────┐    ┌──────────────┐    ┌───────────┐
  │ Encoder │ →  │ VQ │ →  │ VocosBackbone│ →  │ ISTFTHead │ → recon_wav
  └─────────┘    └────┘    │ (12×ConvNeXt)│    │(mag+phase) │
                           └──────────────┘    └───────────┘

  Discriminator（3 個，全參數可訓，交替更新）：
  ┌───────────────────────────────────────────────────────────┐
  │  MPD (Multi-Period Discriminator)     — 時域週期結構       │
  │  MRD (Multi-Resolution Discriminator) — 頻域多尺度結構     │
  │  DAC (DAC Discriminator)              — 補充覆蓋          │
  └───────────────────────────────────────────────────────────┘
          ↕                    ↕
    real_wav (clean)     recon_wav (Generator 輸出)
    → Disc 判定哪個是真的 → Generator 學會騙過 Disc → 音質提升
```

**這就是 WavTokenizer 音質好的核心原因——不是 Encoder/Decoder 本身特別厲害，而是 GAN 對抗訓練讓 Decoder 學會了生成聽感自然的波形（包括正確的 phase）。**

> **⚠️ 對本研究的關鍵啟示：Decoder 的自然波形生成能力是 GAN 對抗訓練的產物。**
> GAN 的梯度方向是「讓 Discriminator 無法區分 real/fake」→ 引導 Decoder 學習正確的 phase pattern。
> 若使用 MSE/Mel loss（而非 GAN）去微調 Decoder，梯度方向從「自然度」轉為「逐點精度」→
> Decoder 原本由 GAN 建立的精細 phase generation 能力被破壞 → 產生機械音。
> **這是本研究選擇凍結 Decoder、僅微調 Encoder 的根本理論依據（見 6.3 節消融分析）。**

### 2.4 三個 Discriminator 的互補角色

| Discriminator | 分析域 | 原理 | 捕捉的特徵 | 參數量 |
|---------------|--------|------|-----------|-------|
| **MPD** (Multi-Period) | 時域 | 把 waveform 以不同週期 p=[2,3,5,7,11] reshape 成 2D → Conv2d 判別 | 基頻 (F0)、週期性結構、語音的振動模式 | ~15M |
| **MRD** (Multi-Resolution) | 頻域 | 在 3 個 STFT 解析度 [(1024,256,1024), (2048,512,2048), (512,128,512)] 上取 magnitude spectrogram → Conv2d 判別 | 諧波結構、共振峰、頻譜包絡、高頻細節 | ~26M |
| **DAC** (DAC Discriminator) | 混合 | 來自 Descript Audio Codec，含自己的 MPD + MRD 變體（`rates=[]`，**無 MSD**） | 上述兩者的互補覆蓋，增加判別多樣性 | ~42M |

> **三者合計約 84M 參數**，佔官方 checkpoint 1091 個 state_dict key 中的 800 個（73%）。
>
> ⚠️ **DAC Discriminator 的 `rates=[]`**：儘管 DAC 原始設計（Descript Audio Codec）含 MPD + MSD + MRD，WavTokenizer 設定 `rates=[]` 使 MSD 不被啟用，實際只使用 DAC 內部的 MPD（periods=[2,3,5,7,11]）+ MRD（fft_sizes=[2048,1024,512]）。加上主判別器的 MPD + MRD，**整個系統不含任何 MSD（Multi-Scale Discriminator）**。

**Generator Loss 組成（官方 `experiment.py`）：**

$$L_{gen} = \underbrace{L_{adv}^{MPD} + \lambda_{mrd} \cdot L_{adv}^{MRD}}_{\text{GAN 對抗 loss}} + \underbrace{L_{FM}^{MPD} + \lambda_{mrd} \cdot L_{FM}^{MRD}}_{\text{Feature Matching loss}} + \underbrace{45 \cdot L_{mel}}_{\text{Mel 重建}} + 1000 \cdot L_{commit} + \underbrace{L_{adv}^{DAC} + L_{FM}^{DAC}}_{\text{DAC 對抗 + FM}}$$

其中：
- **GAN adversarial loss**：逼迫 Generator 輸出能騙過 Discriminator 的波形
- **Feature Matching loss**：L1(Disc 中間層 feature map of recon, Disc 中間層 feature map of real)，穩定 GAN 訓練
- **Mel loss (λ=45)**：保證頻譜整體結構正確
- **Commit loss (×1000)**：VQ codebook 對齊
- **DAC loss**：分為 `loss_dac_1`（DAC adversarial）+ `loss_dac_2`（DAC feature matching），由 `DACGANLoss.generator_loss()` 分別回傳

> **Feature Matching（FM）在官方訓練中已經存在**——它是 GAN 訓練的穩定劑。本研究 exp_0225d / exp_0227 的做法是：在不能做 GAN 對抗訓練的情況下，**單獨提取 FM loss 作為感知回饋**。

### 2.5 為什麼本研究不能使用 GAN 對抗訓練

| 條件 | 官方 WavTokenizer 訓練 | 本研究 |
|------|----------------------|--------|
| Generator 可訓參數 | 166M（**全部**） | 2.4M（**LoRA 僅 1.4%**） |
| 訓練資料 | 數百小時 clean speech | LDV paired data（**極少量**）|
| GPU 資源 | 多卡 | 單張 11GB RTX 2080 Ti |

核心問題是**能力不對等**：GAN 對抗訓練需要 Generator 和 Discriminator 勢均力敵，但 LoRA（2.4M）的表達能力遠不及 Discriminator（84M）。Discriminator 幾十個 epoch 就能學會辨識 LoRA 的 artifact pattern → Generator（LoRA）無力反抗 → 訓練崩潰。

**替代方案**：凍結 Discriminator、僅取中間層 feature map 做 L1 距離（Feature Matching Loss），既利用了 Discriminator 學到的「好的語音長什麼樣」的感知知識，又繞過了對抗訓練的不穩定性。

### 2.6 官方訓練設定對比

| 項目 | WavTokenizer 官方 | 本研究（exp_0224b）|
|------|------------------|------------------|
| Loss | GAN + FM + 45×Mel + Commit | MSE + MR-STFT + 45×Mel |
| Discriminator | MPD + MRD + DAC（對抗訓練） | **無**（不能用 GAN）|
| Mel λ | **45**（官方設定） | **45**（沿用） |
| Optimizer | AdamW × 2（Gen + Disc） | AdamW × 1（LoRA only）|
| 資料 | LibriSpeech + VCTK + ... | LDV Noisy→Clean pairs |
| 目的 | 訓練 clean speech codec | 微調使其適應 LDV noisy input |

> 本研究的 λ=45 直接沿用 WavTokenizer 官方設定，確保 Mel 損失主導感知品質優化。
>
> **GAN 在本研究中的角色：不是使用的技術，而是「不動 Decoder」的理論依據。**
> 本研究不使用 GAN 對抗訓練（資源與能力不對等，見 2.5 節），但理解 GAN 如何塑造 Decoder 能力至關重要——
> 它解釋了為什麼用 MSE/Mel loss 微調 Decoder 會產生機械音（梯度方向不同，破壞 phase generation），
> 從而論證凍結 Decoder 的必要性。

### 2.7 BibTeX 引用

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

## 三、研究問題與核心貢獻

### 3.1 問題澄清

**基本定位**：

> ✅ WavTokenizer 已有成熟的 encoder-decoder：能重建輸入語者的聲音特徵
> ✅ VQ 離散化可對接下游語音模型
> ⚠️ **關鍵問題**：原始 WavTokenizer 的 encoder 在 clean speech 上訓練，直接輸入 LDV noisy 會造成 distribution shift → 重建品質下降
> ⚠️ **噪聲特殊性**：LDV 噪聲不是加性白噪，而是材質共振造成的乘性/卷積型噪聲，需特殊處理

### 3.2 核心貢獻

```
原始 WavTokenizer（輸入限制 clean speech）
          ↓ 本研究貢獻
改良版系統（可接受 LDV noisy speech，輸出 clean speech 重建）
```

具體技術貢獻：
1. **Student Encoder LoRA**：僅微調 encoder，讓其從 LDV noisy 提取與 clean 等價的特徵，同時保留 decoder 完整性
2. **VQ 瓶頸量化分析**：量化 VQ 對語音增強的影響，建立跳過 VQ 的訓練策略
3. **Encoder-only vs Decoder LoRA vs E2E 消融分析**：系統性比較三種微調策略的 PESQ/STOI/聽感差異，論證 encoder-only 為最佳實用方案
4. **PESQ/STOI 悖論分析**：解釋 encoder-only PESQ 微降但 STOI 顯著上升的 distribution mismatch 機制
5. **保留 Tokenization 能力**：不動 decoder → 推論時可重新啟用 VQ → 離散 token 可對接下游 LLM/ASR

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
  │                 [Frozen Decoder / Decoder+LoRA*] ──► Student Output  │
  └─────────────────────────────────────────────────────┘
══════════════════════════════════════════════════════════════════
```

> \* Decoder LoRA 僅用於消融實驗（exp_0224b / 0225b-d），**主推方案為 Frozen Decoder**。

**核心思想**：

| 概念 | 說明 |
|------|------|
| **Teacher** | 原始 WavTokenizer（完全凍結），輸入 clean speech，代表「理想行為」 |
| **Student** | LoRA 微調後的 WavTokenizer，輸入 LDV noisy speech，學習產生與 Teacher 等價的輸出 |
| **知識蒸餾目標** | Student(noisy) 的輸出 ≈ Teacher(clean) 的輸出，或直接 ≈ clean audio |
| **訓練策略** | 僅訓練 Student Encoder LoRA，Decoder 完全凍結（主推方案）|

**各階段的 Teacher-Student 關係**：

| 階段 | Teacher 提供 | Student 學習 | Loss 目標 |
|------|-------------|-------------|----------|
| Phase 1（exp_0217）| Teacher encoder output（clean, pre-VQ 連續特徵） | Student Encoder LoRA + VQ | Masked MSE(student quantized vs teacher encoder out) + IntermediateSupervision(L3,L4,L6) + VQ commit |
| Phase 2（exp_0224a）| Clean audio waveform | Student Encoder LoRA | MSE + MR-STFT + Mel：recon ≈ clean wav |
| Phase 3（exp_0224b）| Clean audio waveform | Decoder LoRA | MSE + MR-STFT + Mel：recon ≈ clean wav |

> **為什麼不端到端訓練？** 同時訓練 Encoder + Decoder 的參數空間過大（即使用 LoRA），容易陷入局部最優解或 mode collapse。分階段訓練讓每階段只需解決一個子問題：Phase 1-2 解決「如何從 noisy 提取 clean-equivalent 特徵」，Phase 3 解決「如何從 student 特徵重建 clean 波形」。

---

## 四、系統架構

> ⚠️ 以下描述基礎架構。最終採用 **Encoder-only LoRA + Frozen Decoder**（見第六章分析）。
> Decoder LoRA 版本（exp_0224b）及 E2E 版本（exp_0226）僅作為消融對照。

### 4.1 整體架構圖

**A. Encoder-only 版本（主推方案，exp_0224a / 0225a / 0226a / 0227）**

```
┌──────────────────────────────────────────────────────────────────┐
│              Encoder-only LoRA 系統架構（主推）                    │
│                                                                  │
│  LDV Noisy Audio                Clean Audio (GT)                 │
│  [B, 1, T]                      [B, 1, T]                       │
│       │                              │                           │
│       ▼                              │  (僅用於計算 Loss)         │
│  ┌──────────────────┐               │                           │
│  │  Student Encoder │               │                           │
│  │  + LoRA (r=64)   │ ← ✅ 可訓練    │                           │
│  └────────┬─────────┘               │                           │
│           │                         │                           │
│           ▼                         │                           │
│  student_encoder_out                │                           │
│  [B, 512, T/320]                    │                           │
│  (連續特徵，跳過 VQ)                 │                           │
│           │                         │                           │
│           ▼                         ▼                           │
│  ┌──────────────────────┐   ┌────────────────────┐              │
│  │  WavTokenizer        │   │  Loss 計算          │              │
│  │  Decoder Backbone    │   │  (各實驗不同，      │              │
│  │  (ConvNeXt × 12)     │   │   見第五章)         │              │
│  │  ❄️ 完全凍結          │◄──│                    │              │
│  │  + FourierHead ❄️    │   └────────────────────┘              │
│  └────────┬─────────────┘                                       │
│           │                                                      │
│           ▼                                                      │
│  recon_wav [B, 1, T]                                            │
│  (重建的乾淨語音)                                                 │
└──────────────────────────────────────────────────────────────────┘
```

**B. Decoder LoRA 版本（消融對照，exp_0224b / 0225b~d）**

```
┌──────────────────────────────────────────────────────────────────┐
│              Decoder LoRA 系統架構（消融對照）                     │
│                                                                  │
│  LDV Noisy Audio → [Student Encoder + LoRA ❄️ 凍結]              │
│                     → [跳過 VQ]                                   │
│                     → [Decoder + LoRA r=32 ✅ 可訓練]             │
│                     → recon_wav                                  │
│                                                                  │
│  ⚠️ PESQ 較高但產生機械音（phase artifact），見第六章分析           │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 訓練流程（漸進式）

```
Phase 1（exp_0217）：Encoder LoRA + VQ 預訓練
  初始：WavTokenizer 預訓練 Encoder + LoRA (r=64) + VQ (EMA)
  Noisy → [Student Encoder LoRA] → VQ → quantized
  Loss: λ_quant × Masked MSE（student quantized vs teacher encoder output）
        + 0.03 × IntermediateSupervision（L3, L4, L6 中間層對齊）
        + β_commit × VQ commit loss
  ⚠️ Teacher 提供的是 pre-VQ 連續特徵（teacher_encoder_out），非 VQ tokens
  目標：學習從 LDV noisy 提取 encoder 特徵，經 VQ 後對齊 teacher encoder 連續輸出
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

> **⚠️ 兩條獨立的訓練線（Training Lineage）**
>
> 上述 Phase 1→2→3 僅描述 **Lineage A（exp_0217 → 0224a → 0224b）**。
> 後續實驗發現從頭訓練 Encoder LoRA 也可行，形成第二條線：
>
> | Lineage | 路徑 | 說明 |
> |---------|------|------|
> | **A** | exp_0217 → 0224a → 0224b | Phase 1 VQ 預訓練 → Phase 2 跳過 VQ → Phase 3 Decoder LoRA |
> | **B** | exp_0225a → 0225b-d / 0226 / 0226a / 0226b | 直接跳過 VQ 從頭訓練 Encoder LoRA，不經 Phase 1 |
>
> Lineage B 的 exp_0225a 直接從 WavTokenizer pretrained 起始（不繼承 exp_0217），
> 後續 exp_0226a（EncOnly + FeatAlign）和 exp_0226（E2E）均初始化自 exp_0225a。
> **第六章的消融分析涵蓋兩條線的實驗結果。**

### 4.3 各階段訓練設定對比

| 項目 | exp_0217 | exp_0224a | exp_0224b |
|------|----------|-----------|-----------|
| Encoder 初始化 | WavTokenizer預訓練+LoRA | 繼承 exp_0217 best | 繼承 exp_0224a ep190 |
| Encoder LoRA | ✅ 可訓練 | ✅ 可訓練（繼承0217）| ❄️ 凍結（繼承0224a ep190）|
| VQ | ✅ 使用 | ❌ 跳過 | ❌ 跳過 |
| Decoder | ❄️ 凍結 | ❄️ 凍結 | ✅ LoRA 可訓練 |
| Loss 目標 | Masked MSE + IntermediateSupervision + VQ commit | MSE+STFT+Mel（wav）| MSE+STFT+Mel（wav）|
| 最佳 PESQ | 1.203 | 1.586（ep190）| 1.866（ep50/300）|

### 4.4 可訓練參數統計

| 模組 | 參數量 | 狀態（exp_0224b） |
|------|--------|-----------------|
| WavTokenizer Encoder | ~40.3M | ❄️ Frozen |
| Encoder LoRA (r=64, α=128) | ~4.72M | ❄️ Frozen（from exp_0217） |
| VQ Codebook (K=4096, d=512) | ~2.1M | ❄️ 跳過，不使用 |
| Decoder ConvNeXt × 12 | ~84.8M | ❄️ Frozen（基礎） |
| Decoder LoRA pwconv1+2 (r=32) | **~2.36M** | ✅ **可訓練** |
| FourierHead | ~39.4M | ❄️ Frozen |
| **可訓練比例** | | **1.42%** |
### 4.5 LoRA 微調策略

#### 4.5.1 為什麼選擇 LoRA？

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

#### 4.5.2 LoRA 注入位置與超參數

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

#### 4.5.3 Encoder vs Decoder LoRA rank 的設計考量

| 考量 | Encoder LoRA (r=64) | Decoder LoRA (r=32) |
|------|-------------------|-------------------|
| 任務複雜度 | 高：需從 noisy domain 映射到 clean domain（domain shift 大） | 中：僅需適應 student encoder 的微小 distribution shift |
| 輸入分布差異 | LDV noisy vs clean speech（分布差異大） | Student encoder output vs teacher encoder output（同空間，差異小）|
| 參數量 | ~4.72M | ~2.36M |
| scaling (α/r) | 2.0 | 2.0 |
| 設計哲學 | 「大」LoRA 用於困難的跨域映射 | 「小」LoRA 用於精細的增量適應 |
---

## 五、損失函數

> ⚠️ 本章描述基礎損失函數（MSE + MR-STFT + Mel），適用於 exp_0224a/b。
> exp_0225～0229 系列在此基礎上新增/調整 loss 組合，見第六章損失函數迭代歷程。

### 5.1 設計哲學：三個尺度監督

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

### 5.2 總損失

```
L_total = L_MSE  +  L_MR-STFT  +  45 × L_Mel
          ───────   ──────────     ──────────────
          波形對齊   頻譜結構        感知品質（主導）
```

> λ=45 直接沿用 WavTokenizer 官方設定。45×L_Mel 約佔 total loss 的 ~85%，確保模型優先優化「聽起來對不對」。

### 5.3 各項說明

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

### 5.4 三種 Loss 的互補關係

| Loss | 監督域 | 解決什麼問題 | 單獨使用的缺陷 |
|------|--------|------------|---------------|
| MSE | 時域（波形）| 整體波形對齊、時序一致 | silence collapse（輸出靜音也能低 loss）|
| MR-STFT | 頻域（頻譜）| 頻譜結構正確、多尺度覆蓋 | 對相位不敏感 |
| Mel × 45 | 感知域 | 人耳聽感品質、共振峰、語者音色 | 對微細時域結構不敏感 |
| **三者組合** | **全方位** | **波形 + 頻譜 + 感知同時監督** | **互補覆蓋各自盲點** |

---

## 六、實驗結果與分析

### 6.1 系統上下限分析

| 路徑 | PESQ | STOI | 意義 |
|------|------|------|------|
| clean → Teacher（無VQ） | 2.484 | 0.761 | 系統絕對上限（理想 codec 重建） |
| clean → Teacher（有VQ） | 2.352 | 0.750 | 有 VQ 的上限 |
| **noisy → Teacher（有VQ）** | **1.677** | **0.527** | **公平比較基準線**（noisy_through_teacher） |
| noisy → Teacher（無VQ） | 1.708 | 0.531 | 去除 VQ → 僅 +0.031 PESQ |

### 6.2 VQ 量化瓶頸分析

> 本節量化說明 VQ 量化誤差的影響，justify 訓練時跳過 VQ 的決定。

| 比較 | PESQ 差異 | STOI 差異 | 說明 |
|------|----------|----------|------|
| clean: 有VQ vs 無VQ | 2.352 vs 2.484 (**−0.132**) | 0.750 vs 0.761 (−0.011) | VQ 量化誤差：clean 路徑損失 |
| noisy: 有VQ vs 無VQ | 1.677 vs 1.708 (−0.031) | 0.527 vs 0.531 (−0.005) | VQ 量化誤差在 noisy 影響更小 |
| exp_0217（有VQ） vs exp_0224a（無VQ） | 1.203 vs 1.586 (**+0.383**) | 0.462 vs 0.628 (**+0.166**) | 跳過 VQ 的巨大提升 |

**解讀**：
- VQ 在 clean path 造成 −0.132 PESQ，在 noisy path 僅 −0.031：說明 noisy encoder 輸出本身已偏離 codebook 分佈，VQ 強制量化反而是二次傷害
- exp_0217→0224a 的 +0.383 PESQ 提升中，VQ 去除只貢獻 ~0.031，其餘 ~0.352 來自 loss 從「Feature MSE（VQ空間）」改為「全路徑波形 loss」
- **結論**：跳過 VQ 一方面消除量化瓶頸，更重要的是讓 loss 直接優化最終波形品質

### 6.3 Encoder-only vs Decoder LoRA vs E2E：為什麼選 Encoder-only

> 本節是論文核心論述。

**完整實驗比較表**（所有 Δ 相對 `noisy_through_teacher` 基準 PESQ=1.677, STOI=0.527）：

| 實驗 | 微調位置 | Loss 配置 | PESQ | STOI | ΔPESQ | ΔSTOI | 備註 |
|------|---------|----------|------|------|-------|-------|------|
| noisy_through_teacher | — | — | 1.677 | 0.527 | 0 | 0 | 公平基準 |
| exp_0224a | Enc LoRA | MSE+STFT+Mel | 1.586 | 0.628 | −0.091 | **+0.101** | Encoder-only 基礎版 |
| exp_0225a | Enc LoRA (scratch) | MSE+STFT+Mel | 1.554 | 0.631 | −0.123 | **+0.104** | 從頭訓練（非繼承） |
| exp_0226a | Enc LoRA+FeatAlign | MSE+STFT+Mel+FeatAlign | 1.535 | 0.627 | −0.141 | **+0.100** | +特徵對齊 loss |
| exp_0226b | Enc LoRA+FeatAlign+HF-Mel | +HF-Mel(bin40+) | 1.535 | 0.610 | −0.141 | +0.084 | +高頻 mel 強調 |
| exp_0227 | Enc LoRA+FeatAlign+MRD-FM | +FrozenMRD FM | 1.571 | 0.620 | −0.105 | **+0.094** | +MRD Feature Matching |
| **exp_0224b** | **Dec LoRA** | MSE+STFT+Mel | **1.868** | **0.667** | **+0.192** | **+0.140** | PESQ 最高，但有機械音 ⚠️ |
| exp_0225b | Dec LoRA (from 0225a) | MSE+STFT+Mel | 1.731 | 0.654 | +0.055 | +0.127 | |
| exp_0225c | Dec LoRA+Phase | +Phase loss | 1.648 | 0.646 | −0.029 | +0.119 | Phase 修正效果有限 |
| exp_0225d | Dec LoRA+FM | +FrozenMRD FM | 1.712 | 0.652 | +0.036 | +0.126 | |
| **exp_0226** | **E2E LoRA** | MSE+STFT+Mel | **1.816** | **0.687** | **+0.140** | **+0.160** | 最高 STOI，但仍有機械音 ⚠️ |

**三種策略對比分析**：

| 面向 | Encoder-only | Decoder LoRA | E2E LoRA |
|------|-------------|-------------|----------|
| PESQ vs 基準 | −0.09~−0.14 ↘ | **+0.19** ↗ | +0.14 ↗ |
| STOI vs 基準 | **+0.094~+0.101** ↗ | +0.140 ↗ | **+0.160** ↗ |
| 機械音 | ❌ 無 | ⚠️ 有 | ⚠️ 有 |
| Token 相容性 | ✅ 保留 | ❌ Decoder 被修改 | ❌ 兩端都被修改 |
| 展示適用性 | ✅ 適合實體展示 | ❌ 不適合（聽感差） | ❌ 不適合（聽感差） |

#### 6.3.1 為什麼 Decoder LoRA PESQ 上升但產生機械音？

**PESQ 上升的原因**：Decoder LoRA 讓解碼器**學習適應** student encoder 的輸出分佈。原本 student encoder 輸出 ≈ 但 ≠ teacher encoder 輸出，凍結的 decoder 面對這個微小的分佈偏移會產生重建偏差。Decoder LoRA 補償了這個 **distribution mismatch** → 波形重建精度提高 → PESQ 上升。

**但代價嚴重——Magnitude 與 Phase 的骨幹纏結**：

WavTokenizer Decoder 的波形生成流程為：

```
ConvNeXt Backbone（共享參數 W）
         │
         ▼
    shared features [B, 768, T/320]
    ╱            ╲
Magnitude branch    Phase branch
  (sigmoid)         (sin/cos)
    ╲            ╱
      iSTFT(mag, phase) → waveform
```

Backbone 沒有「magnitude 專用神經元」和「phase 專用神經元」——它們**共享同一組參數 W**（12 層 ConvNeXtBlock）。任何對 W 的修改，**同時影響 magnitude 和 phase 的生成**。

**MSE/Mel 梯度幾乎完全由 magnitude 主導**：

$$\frac{\partial L_{total}}{\partial W} \approx \underbrace{\frac{\partial L}{\partial \text{mag}} \cdot \frac{\partial \text{mag}}{\partial W}}_{\text{大（magnitude 梯度主導）}} + \underbrace{\frac{\partial L}{\partial \text{phase}} \cdot \frac{\partial \text{phase}}{\partial W}}_{\text{小（MSE/Mel 對 phase 不直接敏感）}}$$

- **MSE**：波形 = iSTFT(mag, phase)，magnitude 對波形振幅的貢獻遠大於相位
- **Mel loss（佔 total 的 ~85%）**：比的是 log-magnitude spectrogram，**完全不含 phase 資訊**
- **MR-STFT**：比的也是 magnitude spectrogram

結果：backbone W 的更新方向 **幾乎完全由 magnitude 梯度決定** → phase 分支只是「被附帶修改」→ 原本 GAN 精心調好的 phase pattern 被覆蓋 → **機械音**。

```
GAN 訓練 Decoder 時（官方）：
  Discriminator（84M）提供「自然度」梯度信號
  → MPD 檢查週期結構 → 約束 phase 的諧波一致性
  → MRD 檢查多解析度頻譜 → 約束 magnitude+phase 的整體自然度
  → magnitude 和 phase 被**同時**、**平衡地**優化

MSE/Mel 微調 Decoder 時（exp_0224b）：
  無 Discriminator → 無 phase 自然度約束
  → W 被 magnitude 梯度主導更新
  → phase 生成能力被「附帶」破壞
  → magnitude 改善（PESQ↑）但 phase 劣化 → 機械音
```

**Phase loss 能解決嗎？（exp_0225c 已驗證：不能）**

exp_0225c 嘗試加入 phase loss，PESQ 反而從 1.731 降至 1.648，機械音仍在。原因：

| Phase loss 的局限 | 說明 |
|-------------------|------|
| **Phase wrapping** | 相位值在 $[-\pi, \pi]$ 範圍存在不連續跳躍（$\pi$ 和 $-\pi$ 差 $2\pi$ 但實際接近），L1/L2 距離不可靠 |
| **Point-wise ≠ Distribution-level** | Phase loss 是逐 time-frequency bin 的約束；GAN 提供的是**整體分佈級**約束（「整段波形聽起來自然嗎？」）|
| **知識不對等** | GAN 的 phase 知識來自 84M Discriminator 在大規模資料上學到的「自然相位模式」；一個 phase angle L1 loss **無法編碼這種結構性知識** |
| **骨幹仍共享** | 加了 phase loss 後，magnitude 梯度和 phase 梯度**在共享 backbone 上互相干擾**，反而造成兩者都退化 |

**重新訓 GAN 能解決嗎？（不能）**

GAN 對抗訓練需要 Generator 和 Discriminator 能力對等。本研究的 Decoder LoRA 僅 2.36M 參數，面對 84M 的 Discriminator 完全無力對抗（見 2.5 節）。

**完整因果鏈**：

```
① 為什麼需要波形 loss？
   → exp_0217 只能在 latent space 做 Feature MSE（@inference_mode 擋住梯度）
   → 跳過 VQ + decode_continuous() 打通梯度路徑 → 波形 loss 可用

② 波形 loss 更新 Decoder 時出了什麼問題？
   → MSE/Mel 梯度由 magnitude 主導
   → 共享 backbone 的 phase 生成能力被附帶破壞 → 機械音

③ Phase loss 能修嗎？→ 不能（phase wrapping + point-wise ≠ distribution-level）
④ 重新訓 GAN 嗎？  → 不能（LoRA 2.4M vs Disc 84M，能力不對等）

⑤ 解法：不動 Decoder
   波形 loss 仍然使用，但梯度只更新 Encoder LoRA
   Decoder 凍結 → ∂Loss/∂W_decoder = 0 → phase generation 完整保留
   享受波形 loss 的好處（直接優化最終品質），
   躲過波形 loss 的壞處（破壞 decoder phase）
```

> **核心結論**：問題不在 loss 函數設計不夠好，而在 **Decoder backbone 的 magnitude/phase 纏結結構**使得任何非 GAN 的 loss 都無法平衡優化兩者。唯一安全的做法是凍結 Decoder。

#### 6.3.2 為什麼 Encoder-only PESQ 微降？

Student encoder 的輸出落在 teacher decoder 預期輸入分佈的**邊緣**（而非正中心）。凍結的 decoder 針對這些略微偏移的 latent 仍能正確解碼語音內容（STOI↑），但波形重建精度不如處理完全 in-distribution 的 teacher latent（PESQ 微降）。

$$\text{PESQ 微降原因} = \underbrace{f_{dec}(z_{student})}_{\text{decoder 在邊緣分佈的輸出}} \neq \underbrace{f_{dec}(z_{teacher})}_{\text{decoder 在訓練分佈中心的輸出}}$$

**但 STOI 大幅上升**說明：encoder 確實學會了從 noisy 中提取 clean 語音內容。PESQ 對波形逐點精度敏感（分佈偏移造成微小時域差異），但 STOI 衡量的是語音可懂度（語音內容正確）。

#### 6.3.3 最終選擇 Encoder-only 的理由

1. **聽感優先於分數**：PESQ 微降 0.09~0.14（人耳幾乎感知不到），但機械音在 Decoder LoRA 方案中**立刻可辨**
2. **STOI 才是 LDV 核心指標**：LDV 語音最大問題是「聽不懂」。STOI +0.094~+0.101 代表可懂度從 ~53% → ~62%，有實質意義
3. **保持 token 相容性**：不動 decoder → 推論時可接上 VQ 產生離散 token → 支援下游 LLM/ASR pipeline
4. **E2E 不解決問題**：exp_0226（E2E LoRA）STOI 最高但仍有機械音 → 問題確定出在 decoder 被修改，不是 encoder 能力不足

> **論文定位**：Encoder-only LoRA 是本研究的主推方案，Decoder LoRA 和 E2E LoRA 作為 ablation study，用於說明「為什麼不微調 decoder」。

### 6.4 損失函數迭代歷程

> ⚠️ 本節仍在實驗中，損失函數配置尚未最終確定。

| 實驗 | 基礎 Loss | 新增 Loss | 設計動機 | 結果 |
|------|----------|----------|---------|------|
| exp_0224a | MSE+STFT+Mel | — | 基礎配置 | STOI +0.101 |
| exp_0225a | MSE+STFT+Mel | 從頭訓練（非繼承 0217） | 測試是否需要 VQ 預訓練 | STOI +0.104（差異不大） |
| exp_0226a | MSE+STFT+Mel | +Feature Alignment | 特徵空間直接對齊 student/teacher encoder 輸出 | STOI +0.100 |
| exp_0226b | MSE+STFT+Mel+FA | +HF-Mel (bin 40+, ~1.6kHz) | 強調高頻 mel bin 重建 | STOI +0.084（反而下降）|
| exp_0227 | MSE+STFT+Mel+FA | +Frozen MRD FM | 用預訓練 MRD 辨別器特徵圖做 Feature Matching | STOI +0.094, PESQ 最高 encoder-only |
| exp_0228 | MSE+STFT+Mel+FA | +HuBERT FM | 用 HuBERT 特徵做語義層級對齊 | 🔄 訓練中 |
| exp_0229b | LatentBWE | MSE+STFT+Mel | Latent 空間帶寬擴展 | BWE Δ 持續負（效果差）|
| exp_0229c | LatentBWE v2 | +HF-emphasis STFT | 強調高頻 STFT 損失 | 🔄 待訓練 |

**發現**：
- 基礎 MSE+STFT+Mel 組合已經是有效的基線（exp_0224a STOI +0.101）
- Feature Alignment 略提升 PESQ 但未顯著改善 STOI
- HF-Mel 過度強調高頻反而傷害整體表現 → 高頻在 latent space 的修正可能比 loss 層面更有效
- MRD Feature Matching（exp_0227）在不使用 GAN 的情況下是目前 encoder-only 中 PESQ 最高的（1.571）
- ⏳ exp_0228（HuBERT FM）和 exp_0229c（Latent BWE + HF-emphasis）仍在探索中

### 6.5 關鍵發現總結

1. **VQ 量化瓶頸可控**：clean path VQ 損失 −0.132 PESQ；noisy 僅 −0.031 → VQ 不是主要瓶頸
2. **跳過 VQ 的真正收益不在去量化，而在 loss 改良**：全路徑波形 loss 比 VQ 空間的 Feature MSE 有效得多
3. **Distribution mismatch 是 encoder-only 的核心限制**：student encoder 輸出與 teacher decoder 預期分佈的偏移導致 PESQ 微降
4. **微調 decoder 提升 PESQ 但引入機械音**：GAN 訓練建立的相位生成能力被非 GAN 的 LoRA 微調破壞
5. **STOI 比 PESQ 更反映 LDV 增強效果**：所有 encoder-only 方案 STOI 從 0.527 提升至 0.610~0.631（+16~20%）
6. **Encoder-only 是最佳實用方案**：無機械音 + 保留 token 相容性 + STOI 顯著提升

---

## 七、論文章節結構

> ✅ = 已有足夠材料，可直接撰寫
> 🔄 = 架構確定但數據還在迭代（loss 配置待定）
> ⏳ = 需待實驗完成

```
第一章：緒論                                                    ✅
  1.1 LDV 感測器語音的應用場景與挑戰
  1.2 研究動機（跨材質、低 SNR、帶寬擴展）
  1.3 研究貢獻

第二章：相關研究                                                ✅
  2.1 語音增強方法演進（傳統 → DNN → Codec-based）
  2.2 WavTokenizer / Neural Audio Codec 原理
  2.3 LoRA 參數高效微調
  2.4 LDV 語音感測技術與資料集

第三章：系統設計                                                ✅
  3.1 問題定義（LDV noisy → clean + tokenization）
  3.2 Teacher-Student 知識蒸餾框架
  3.3 Encoder-only LoRA 架構（主推方案）
  3.4 VQ 跳過策略與推論時重新啟用
  3.5 LoRA 注入位置與超參數設計

第四章：損失函數設計                                            🔄
  4.1 基礎損失（MSE + MR-STFT + Mel）                          ✅
  4.2 進階損失探索                                             🔄
      - Feature Alignment (exp_0226a)
      - MRD Feature Matching (exp_0227)
      - HuBERT Feature Matching (exp_0228)                     🔄
      - Latent BWE (exp_0229 系列)                              🔄

第五章：實驗設置與結果                                          🔄
  5.1 資料集描述（LDV 錄音，14 train / 3 val speakers）         ✅
  5.2 評估指標（PESQ-NB, STOI）                                ✅
  5.3 VQ 量化瓶頸分析                                          ✅
  5.4 Encoder-only vs Decoder LoRA vs E2E 消融比較              ✅
  5.5 損失函數消融實驗                                         🔄
  5.6 PESQ / STOI 悖論分析（為何 PESQ↓ 但 STOI↑）             ✅
  5.7 頻譜與波形對比圖                                         ⏳
  5.8 跨材質泛化測試                                           ⏳

第六章：結論與未來工作
  6.1 結論                                                     🔄
  6.2 未來方向                                                  ✅
      - 更好的 loss 組合（HuBERT FM, Latent BWE）
      - VQ 重新啟用優化
      - 更大 LDV 資料集
      - 主觀評估（MOS/ABX）
```

### 7.1 已確定 vs 待定清單

| 類別 | 項目 | 狀態 |
|------|------|------|
| ✅ 已確定 | 整體架構（Encoder-only LoRA + Frozen Decoder） | 不會改變 |
| ✅ 已確定 | 訓練策略（No-VQ, Teacher-Student, LoRA r=64） | 不會改變 |
| ✅ 已確定 | Decoder LoRA / E2E 機械音問題 → ablation 論述 | 不會改變 |
| ✅ 已確定 | VQ 瓶頸量化數據 | 不會改變 |
| ✅ 已確定 | 公平基準（noisy_through_teacher）建立 | 不會改變 |
| ✅ 已確定 | 資料集分割（14 train / 3 val speakers, 0 overlap） | 不會改變 |
| 🔄 待定 | 最佳損失函數組合 | exp_0228, 0229c 結果待定 |
| 🔄 待定 | 最終呈報的最佳 checkpoint | 取決於 loss 迭代結果 |
| ⏳ 待做 | 頻譜對比圖（mel spectrogram） | 需最終模型確定後製作 |
| ⏳ 待做 | n≥30 完整驗證集評估 | 需最終模型確定後重跑 |
| ⏳ 待做 | 跨材質泛化測試 | 需最終模型確定後測試 |
| ⏳ 待做 | VQ re-enable 驗證（token 品質） | 需最終模型確定後測試 |

---

## 八、技術細節備查

### 8.1 decode_continuous() 的設計

WavTokenizer 原始 `decode()` 使用 `@torch.inference_mode()` 裝飾器，無法計算梯度。exp_0224b 透過直接呼叫 backbone 和 head 繞過：

```python
def decode_continuous(self, features: torch.Tensor) -> torch.Tensor:
    # 繞過 @inference_mode()，讓 gradient 流過 decoder LoRA
    bandwidth_id = torch.tensor([0], device=features.device)
    x = self.teacher.backbone(features, bandwidth_id=bandwidth_id)
    audio = self.teacher.head(x)
    return audio
```

### 8.2 為什麼 encode_infer 回傳的是 VQ 後的特徵？

`encode_infer` → `feature_extractor.infer()` → `encodec.quantizer.infer()` → 回傳 `quantized`（VQ 後）

要取 VQ 前的連續特徵需直接呼叫：
```python
raw_emb = teacher.feature_extractor.encodec.encoder(audio)  # [B, 512, T/320]
```

### 8.3 PESQ 指標說明

- 使用 **PESQ-NB（窄頻，8kHz）**：先將 24kHz 音訊重採樣至 8kHz 再計算
- PESQ(noisy) >> PESQ(recon) 屬**正常現象**：LDV noisy 與 clean 時序高度對齊，PESQ 對時序對齊敏感；重建音訊因 encoder 壓縮（320:1）導致輕微時序錯位，故 PESQ 較低
- **公平比較基準**（`noisy_through_teacher`）= 同樣的 pipeline（noisy → Teacher Encoder+VQ → Decoder），PESQ=1.677

---

*本文件由實驗過程自動統整，數字均來自實際評估結果（n=3 val samples）。正式論文建議以更大樣本（n=30 以上）重新評估。*
*最後更新：2026-03-02，調整為 Encoder-only 主推架構，新增 VQ 瓶頸分析、Enc/Dec/E2E 比較、損失函數迭代歷程。*
