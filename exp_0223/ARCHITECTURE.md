# exp\_0223: Decoder LoRA Fine-tune — 原理、架構與實驗設計

> 實驗日期：2026-02-23
> 作者：自動產生（exp_0223）

---

## 一、實驗動機與背景

### 1.1 問題根源

從 exp\_0206 → exp\_0217 的實驗歷程中，我們嘗試了多種 **encoder-side** 改進（LoRA rank、T453 加權、augmentation 等），但 PESQ/STOI 始終無法超過 frozen decoder 的天花板：

| 基線 | PESQ | STOI |
|---|---|---|
| Clean → TeacherVQ → Decode（天花板） | 1.790 | 0.744 |
| Noisy → StudentVQ → Decode（exp\_0217 best） | ~1.147 | ~0.511 |
| Noisy raw（不處理） | 1.535 | 0.586 |

根因分析得出兩個關鍵假設：

- **H1**：Student encoder 的 feature-space MSE 訓練目標與 PESQ/STOI 感知指標脫鉤
- **H2**：Frozen decoder 的 waveform 合成能力受限於 VQ 量化失真，無法從 student tokens 重建高品質音訊

exp\_0223 轉向 **decoder-side fine-tune**，用 LoRA 微調 decoder backbone 的 ConvNeXt 層。

### 1.2 為什麼選 Decoder LoRA 而非全量微調？

| 方案 | 可訓練參數 | 風險 |
|---|---|---|
| 全量解凍 decoder | 84.8M (100%) | 過擬合、catastrophic forgetting |
| Decoder LoRA (rank=32) | 2.36M (1.42%) | 低風險、保留預訓練 decoder 能力 |

LoRA 在 decoder backbone 的 12 個 ConvNeXt Block 中的 `pwconv1` (768→2304) 和 `pwconv2` (2304→768) 各加一組低秩適配矩陣，總共 24 個 Linear 層。PEFT 原生支援 nn.Linear，不需要自定義 patch。

---

## 二、系統架構

### 2.1 整體管線

```
Noisy Audio ──→ [Student Encoder + LoRA] ──→ [Student VQ] ──→ student_quantized
                     (FROZEN)                  (FROZEN)              │
                                                                     ▼
                                                          [Decoder Backbone]
                                                          (ConvNeXt × 12)
                                                          pwconv1: LoRA ✓
                                                          pwconv2: LoRA ✓    ← 可訓練
                                                                     │
                                                                     ▼
                                                             [Fourier Head]
                                                              (FROZEN)
                                                                     │
                                                                     ▼
                                                              recon_wav
                                                                     │
                                        Loss(recon_wav, clean_wav) ──┘
```

### 2.2 凍結策略

| 模組 | 來源 | 狀態 |
|---|---|---|
| Student Encoder (+ LoRA) | exp\_0217 epoch 175 best | ❄️ FROZEN |
| Student VQ (EMA codebook) | exp\_0217 epoch 175 best | ❄️ FROZEN |
| Teacher Encoder | WavTokenizer 預訓練 | ❄️ FROZEN |
| Decoder Backbone (ConvNeXt) | WavTokenizer 預訓練 + LoRA | 🔥 LoRA 可訓練 |
| Decoder Head (FourierHead) | WavTokenizer 預訓練 | ❄️ FROZEN |

### 2.3 Decoder 內部架構（VocosBackbone）

```
VocosBackbone
├── embed: Conv1d(512 → 768)           ❄️ frozen
├── norm: LayerNorm(768)               ❄️ frozen
├── convnext[0..11]: ConvNeXtBlock     🔥 LoRA on pwconv1/pwconv2
│   ├── dwconv: Conv1d(768, 768, k=7)  ❄️ frozen
│   ├── norm: LayerNorm(768)           ❄️ frozen
│   ├── pwconv1: Linear(768 → 2304)    🔥 LoRA(rank=32, α=64)
│   ├── act: GELU                      ─
│   ├── pwconv2: Linear(2304 → 768)    🔥 LoRA(rank=32, α=64)
│   ├── grn: GRN(2304)                ❄️ frozen
│   └── gamma: Parameter(768)          ❄️ frozen
├── final_layer_norm: LayerNorm(768)   ❄️ frozen
└── apply_weight_norm                  ❄️ frozen
```

每個 LoRA adapter 的計算：

$$
h = W_{\text{pretrained}} x + \frac{\alpha}{r} \cdot B A x
$$

其中 $A \in \mathbb{R}^{r \times d_{in}}$, $B \in \mathbb{R}^{d_{out} \times r}$，初始化時 $A$ 為 Kaiming, $B = 0$。
因此訓練開始時 LoRA 輸出為 0，等價於 frozen decoder。

### 2.4 `decode()` 覆寫

WavTokenizer 原始的 `decode()` 方法有 `@torch.inference_mode()` 裝飾器，
在 inference mode 下所有運算都不追蹤梯度。我們在 `TeacherStudentDecoderLoRA` 中覆寫此方法：

```python
def decode(self, quantized):
    bandwidth_id = torch.tensor([0], device=quantized.device)
    x = self.teacher.backbone(quantized, bandwidth_id=bandwidth_id)
    audio = self.teacher.head(x)
    return audio
```

直接呼叫 `backbone` → `head`，讓梯度可以流過 decoder LoRA 參數。

---

## 三、損失函數設計

### 3.1 WavTokenizer 原始訓練的 Loss（decoder/experiment.py）

WavTokenizer 原始使用 **GAN + Mel** 的複合損失訓練整個 codec：

$$
\mathcal{L}_{\text{WavTok}} = \underbrace{\mathcal{L}_{\text{GAN-MPD}} + \lambda_{\text{MRD}} \cdot \mathcal{L}_{\text{GAN-MRD}} + \mathcal{L}_{\text{FM-MPD}} + \lambda_{\text{MRD}} \cdot \mathcal{L}_{\text{FM-MRD}}}_{\text{GAN losses（需 Discriminator）}} + \underbrace{45 \cdot \mathcal{L}_{\text{MelSpec}}}_{\text{感知重建}} + 1000 \cdot \mathcal{L}_{\text{commit}} + \mathcal{L}_{\text{DAC}}
$$

其中：
- **MultiPeriodDiscriminator (MPD)**：將 waveform 以 [2,3,5,7,11] 的週期 reshape 為 2D，用多尺度 Conv2d 判別
- **MultiResolutionDiscriminator (MRD)**：在不同 STFT 解析度上判別
- **DAC Discriminator**：額外的 neural audio codec 判別器
- **Feature Matching (FM)**：取 D 中間層特徵的 L1 距離
- **MelSpecReconstructionLoss**：log-mel spectrogram 的 L1，`n_fft=1024, hop=256, n_mels=100`，**係數 45**
- **Commit Loss**：VQ codebook commitment（VQ 已凍結，不需要）

### 3.2 v1 的 Loss 設計（失敗）

```
L_v1 = MSE(recon_wav, clean_wav)
```

**失敗模式：Silence Collapse**

- MSE 對所有時間點平均 → 模型發現「縮小輸出幅度」是降低 MSE 的安全策略
- val recon RMS 從 clean 的 -20 dB 降到 **-38 ~ -53 dB**（近乎靜音）
- val\_wav\_mse 從 epoch 5 起停在 ~0.0154，接近「全零輸出」的 MSE (0.0115)
- v1 在 77 epoch 時手動終止

**為什麼 MSE 會 collapse？**
- Waveform 的相位（phase）非常難精確預測
- 給定正確的頻譜包絡，相位差 180° 的兩個波形 MSE 是 clean 能量的 4 倍
- 模型的最優策略：輸出幅度極小的「安全」波形，避免相位猜錯的懲罰

### 3.3 v2 的 Loss 設計（改進）

$$
\mathcal{L}_{\text{v2}} = \underbrace{\lambda_{\text{wav}} \cdot \text{MSE}(w_{\text{recon}}, w_{\text{clean}})}_{\text{波形對齊}} + \underbrace{\lambda_{\text{stft}} \cdot \mathcal{L}_{\text{MR-STFT}}}_{\text{頻譜結構}} + \underbrace{\lambda_{\text{mel}} \cdot \mathcal{L}_{\text{Mel}}}_{\text{感知能量}}
$$

預設權重：$\lambda_{\text{wav}} = 1.0$、$\lambda_{\text{stft}} = 1.0$、$\lambda_{\text{mel}} = 45.0$

#### 3.3.1 Multi-Resolution STFT Loss

$$
\mathcal{L}_{\text{MR-STFT}} = \frac{1}{M} \sum_{m=1}^{M} \left[ \underbrace{\frac{\| |S_m| - |\hat{S}_m| \|_F}{\| |S_m| \|_F}}_{\text{Spectral Convergence}} + \underbrace{\| \log |S_m| - \log |\hat{S}_m| \|_1}_{\text{Log Magnitude}} \right]
$$

三組解析度設定：

| 解析度 | n\_fft | hop | win | 特化 |
|---|---|---|---|---|
| 低頻 | 2048 | 512 | 2048 | 基頻、語調 |
| 中頻 | 1024 | 256 | 1024 | 共振峰、母音 |
| 高頻 | 512 | 128 | 512 | 摩擦音、瞬態 |

**為什麼能防止 silence collapse？**
- Spectral Convergence 是**比值**（Frobenius norm ratio），靜音時分子分母都小 → 比值不會為零
- Log Magnitude 在 log domain 計算 → 靜音的 log magnitude 趨近 $-\infty$，與 clean 的差異巨大

#### 3.3.2 Mel-Spectrogram L1 Loss

$$
\mathcal{L}_{\text{Mel}} = \| \log \text{MelSpec}(w_{\text{recon}}) - \log \text{MelSpec}(w_{\text{clean}}) \|_1
$$

參數完全沿用 WavTokenizer 原始設定：`sr=24000, n_fft=1024, hop=256, n_mels=100, power=1`。
係數 **45** 也直接沿用（`decoder/experiment.py` line 34: `mel_loss_coeff=45`）。

**設計理由**：Mel 頻譜在 log 域的 L1 距離與人耳感知高度相關（MEL 頻率尺度模擬耳蝸響應）。
45 這個係數是 WavTokenizer 團隊調優後的結果，我們直接繼承以保持一致性。

### 3.4 為什麼不用 GAN Loss？

WavTokenizer 預訓練 checkpoint 中**確實包含** discriminator 權重：

| Discriminator | 參數量 |
|---|---|
| MultiPeriodDiscriminator | 95 tensors |
| MultiResolutionDiscriminator | 57 tensors |
| DAC Discriminator | 648 tensors |
| **合計** | **126.8M params** |

不使用 GAN 的原因：

1. **記憶體不足**：D 本身 126.8M ~ 484 MB + 梯度 + optimizer ~ **2-3 GB**，GPU:1 (11 GB) 空間不夠
2. **G/D 不對稱**：G 只有 2.36M LoRA 參數 vs D 有 126.8M 全量參數，GAN 訓練極不穩定
3. **D 的 domain mismatch**：預訓練 D 是為「clean audio → VQ → decode」訓練的，而我們的輸入是 student VQ tokens（帶噪音失真），D 會錯誤判斷
4. **MR-STFT 是輕量替代**：不需要可訓練模型，純數學公式計算，效果接近 GAN 的頻譜監督

### 3.5 Loss 設計總覽對照

| 元素 | WavTokenizer 原始 | v1 | v2 |
|---|---|---|---|
| Wav MSE | ✗ | ✓ (唯一) | ✓ (λ=1.0) |
| Mel L1 (log) | ✓ (coeff=45) | ✗ | ✓ (λ=45.0) |
| MR-STFT | ✗ | ✗ | ✓ (λ=1.0) |
| GAN (MPD+MRD+DAC) | ✓ | ✗ | ✗ |
| Feature Matching | ✓ | ✗ | ✗ |
| Commit Loss | ✓ (×1000) | ✗ | ✗ (VQ 凍結) |

v2 = 「WavTokenizer 的非 GAN 重建損失」 + 「MR-STFT 替代 GAN 的頻譜監督」 + 「wav MSE 穩定波形」

---

## 四、訓練配置

### 4.1 超參數

| 參數 | v1 | v2 |
|---|---|---|
| Epochs | 150 (77 時手動終止) | 150 |
| Batch size | 8 | 8 |
| Gradient accumulation | 2 | 2 |
| Effective batch size | 16 | 16 |
| Learning rate | 1e-4 | 1e-4 |
| Min LR (cosine) | 1e-6 | 1e-6 |
| Warmup epochs | 5 | 5 |
| Weight decay | 0.01 | 0.01 |
| Grad clip | 1.0 | 1.0 |
| AMP (fp16) | ✓ | ✓ |
| Decoder LoRA rank | 32 | 32 |
| Decoder LoRA alpha | 64 | 64 |
| Decoder LoRA dropout | 0.1 | 0.1 |
| Loss | wav MSE only | wav MSE + MR-STFT + Mel |
| Resume from | - | 從零開始（不繼承 v1） |

### 4.2 資料增強

| 增強 | 預設 |
|---|---|
| SNR remix | prob=0.5, range=[-5, 25] dB |
| Random gain | prob=0.3, ±3 dB |
| Random crop | prob=0.3, min\_ratio=0.7 |
| Time stretch | prob=0.2, range=[0.95, 1.05] |

### 4.3 資料集

- 訓練集：`data/train_cache_filtered.pt`（10,368 samples, batch\_size=8 → 1,296 steps/epoch）
- 驗證集：`data/val_cache_filtered.pt`（1,728 samples, eval\_max\_batches=30）
- 音訊格式：24 kHz, mono, 3 秒

---

## 五、實驗結果

### 5.1 v1 結果（MSE-only — 失敗）

| 指標 | Epoch 1 | Epoch 77 (最終) | 趨勢 |
|---|---|---|---|
| train\_loss | 0.018039 | 0.011804 | 持續下降 |
| val\_wav\_mse | 0.016290 | 0.015358 | Epoch 5 後幾乎不動 |
| val\_noisy\_mse | 0.041547 | 0.041547 | 固定基線 |
| improvement | — | +63.0% | 看似不錯但是假象 |

**Silence Collapse 診斷**：

| 音訊 | RMS (dB) | 說明 |
|---|---|---|
| Clean（原始） | -19.8 ~ -19.1 dB | 正常音量 |
| Noisy（輸入） | -17.0 ~ -20.1 dB | 正常音量 |
| Recon (val, epoch 50) | **-38.1 ~ -53.3 dB** | 近乎靜音 |
| Recon (train, epoch 50) | -20.0 ~ -25.8 dB | 偏暗但有聲 |

- 全零輸出的 MSE = 0.0115
- v1 的 val\_wav\_mse = 0.0154 → 模型輸出接近零
- **val 比 train 嚴重**：train 時 augmentation noise 較多，模型學會「抑制噪音」≈「抑制能量」

### 5.2 v2 初步結果（進行中）

| 指標 | Epoch 1 | Epoch 2 | 趨勢 |
|---|---|---|---|
| train\_total\_loss | 88.08 | 27.78 | 快速下降 |
| train\_wav\_mse | 0.0208 | 0.0206 | 穩定 |
| train\_stft\_sc | 0.7454 | 0.5664 | 下降 |
| train\_stft\_mag | 2.0118 | 0.8578 | 快速下降 |
| train\_mel | 1.8955 | 0.5852 | 快速下降 |
| val\_wav\_mse | 0.0201 | 0.0201 | 穩定 |
| val\_mel\_loss | 0.9313 | 0.8401 | 下降中 |
| val\_noisy\_mel | 1.2082 | 1.2082 | 固定基線 |

v2 的 Mel Loss 已低於 noisy 基線（0.84 < 1.21），表示 decoder 正在改善頻譜品質。
wav\_mse 暫時略高於 v1 best（0.020 vs 0.015），因為 v2 不再允許靜音策略。

---

## 六、程式碼結構

```
exp_0223/
├── models_decoder_lora.py        # TeacherStudentDecoderLoRA 模型定義
│   ├── __init__()                # 凍結繼承參數 + 對 backbone 加 LoRA
│   ├── decode()                  # 覆寫，繞過 @inference_mode
│   ├── forward_wav()             # encoder+VQ (no_grad) → decode (grad)
│   ├── get_decoder_lora_state_dict()  # 提取 LoRA 參數
│   └── load_encoder_vq_checkpoint()   # 載入 exp_0217 weights
│
├── train_decoder_lora.py         # v1 訓練腳本（MSE-only, 已停止）
│   ├── train_epoch()             # wav MSE 訓練循環
│   ├── evaluate_decoder()        # val MSE 評估
│   └── _save_audio_samples()     # 定期儲存 noisy/clean/recon wav
│
├── train_decoder_lora_v2.py      # v2 訓練腳本（MR-STFT + Mel）
│   ├── STFTLoss                  # 單一解析度 STFT (SC + LogMag)
│   ├── MultiResolutionSTFTLoss   # 3 解析度 MR-STFT
│   ├── MelReconstructionLoss     # log-Mel L1
│   ├── train_epoch_v2()          # 複合損失訓練循環
│   ├── evaluate_decoder_v2()     # 多指標評估
│   └── _save_audio_samples()     # 音檔儲存
│
├── monitor_v1_launch_v2.sh       # v1→v2 自動監控腳本
│
└── runs/
    ├── decoder_lora_epoch_20260223_010247/   # v1 run（已停止）
    │   ├── best_model.pt
    │   ├── config.json
    │   ├── metrics_history.json
    │   ├── train.log
    │   └── audio_samples/{train,val}/epoch_{025,050}/
    │
    └── decoder_lora_v2_epoch_20260223_042124/  # v2 run（進行中）
        ├── best_model.pt
        ├── config.json
        ├── metrics_history.json
        ├── train.log
        └── audio_samples/
```

---

## 七、關鍵設計決策摘要

| 決策 | 選擇 | 理由 |
|---|---|---|
| Fine-tune 對象 | Decoder backbone (pwconv1/pwconv2) | Encoder-side 已到極限，轉向 decoder |
| 微調方式 | LoRA (rank=32) | 低風險、少參數 (1.42%)、保留預訓練 |
| loss 設計 | wav MSE + MR-STFT + Mel | 沿用 WavTokenizer 的 Mel Loss 防止 collapse |
| 不用 GAN | MR-STFT 替代 | 記憶體不足、G/D 不對稱、穩定性 |
| 不用預訓練 D | — | Domain mismatch、可用 MR-STFT 替代 |
| v2 不 resume v1 | 從零開始 | v1 的 LoRA 已學會壓制能量，是壞的 local minimum |
| Mel coeff = 45 | 沿用 WavTokenizer 原始 | 已驗證的權重，保持一致性 |

---

## 八、未來方向（v3 考量）

如果 v2 效果不足，可考慮：

1. **凍結 D 的 Feature Matching Loss**：載入預訓練 D 僅作為特徵提取器，不做 GAN 對抗，增加 ~500 MB 但完全穩定
2. **調整 Loss 權重**：$\lambda_{\text{mel}}$ 從 45 降到 10-20，觀察 wav MSE 是否跟著改善
3. **LoRA rank 提升**：rank 32 → 64，增加 decoder 的適配能力
4. **多階段訓練**：先 Mel-only warmup → 再加入 wav MSE
