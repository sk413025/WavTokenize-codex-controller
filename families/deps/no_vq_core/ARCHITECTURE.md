# exp\_0224: No-VQ Encoder LoRA / Decoder LoRA — 原理、架構與實驗設計

> 實驗日期：2026-02-23 ～ 2026-02-24
> 作者：自動產生（exp\_0224）

---

## 一、實驗動機與背景

### 1.1 核心研究問題

> **VQ 量化瓶頸是否為制約語音增強品質的主因？**

從 exp\_0206 → exp\_0217 → exp\_0223 的實驗歷程，我們發現兩條改進路線都遇到難以突破的天花板：

| 實驗 | 方法 | 瓶頸 |
|---|---|---|
| exp\_0217 | Encoder LoRA（有 VQ 量化） | Feature MSE 下降但 PESQ 不升 |
| exp\_0223 v1 | Decoder LoRA + MSE only | Silence collapse（方向性錯誤） |
| exp\_0223 v2 | Decoder LoRA + MR-STFT + Mel | val\_mel 0.84 < noisy 1.21（有改善） |

**共同瓶頸假設**：VQ 量化（K=4096）在 encoder output → decoder input 之間引入不可逆的離散化誤差，使得：

1. Encoder 端：feature MSE 與 VQ commitment loss 耦合，梯度信號被稀釋
2. Decoder 端：即使 decoder 學習適應 student VQ tokens，token 本身已丟失連續分佈的精細信息

exp\_0224 提出 **跳過 VQ**，直接讓 encoder 的連續輸出 `[B, 512, T]` 餵入 decoder，以此量化 VQ 瓶頸帶來的損失。

### 1.2 實驗對比矩陣

```
         │  Decoder Frozen      │  Decoder LoRA (rank=32)
─────────┼──────────────────────┼───────────────────────────
有 VQ    │  exp_0217            │  exp_0223 (v1/v2)
         │  Encoder LoRA        │  Encoder+VQ frozen
         │  → feature MSE 訓練  │  → 波形重建 loss 訓練
─────────┼──────────────────────┼───────────────────────────
無 VQ    │  exp_0224a  ← 本實驗  │  exp_0224b ← 本實驗
         │  Encoder LoRA        │  Encoder frozen (from 0217)
         │  → 波形重建 loss 訓練 │  → Decoder LoRA 波形重建
```

這形成一個 2×2 的完整消融實驗，可以獨立量化 VQ 瓶頸和 Decoder LoRA 各自的貢獻。

---

## 二、系統架構

### 2.1 exp\_0224a：Encoder LoRA + No-VQ + Decoder Frozen

```
                         ┌─────────── Loss ───────────┐
                         │  wav MSE + MR-STFT + Mel   │
                         └────────┬──────────┬────────┘
                                  │          │
  Noisy Audio [B,1,T]            ▼          ▼
       │                    recon_wav   clean_wav
       ▼
  ┌─────────────────────┐
  │   Student Encoder   │
  │   + LoRA (r=64)     │  ← 可訓練（繼承 exp_0217 weights）
  │   α=128             │
  └────────┬────────────┘
           │
           ▼
  student_encoder_out [B, 512, T_frame]
           │
           │  ← 連續特徵，跳過 VQ！
           │
           ▼
  ┌─────────────────────┐
  │   WavTokenizer      │
  │   Decoder Backbone  │  ← 完全 Frozen
  │   (ConvNeXt × 12)   │
  │   + FourierHead     │
  └────────┬────────────┘
           │
           ▼
      recon_wav [B, 1, T_wav]
```

**關鍵特性**：
- VQ 層（K=4096, dim=512）完全跳過，不執行量化
- Encoder LoRA 從 exp\_0217 best checkpoint 初始化
- Decoder 使用原始 WavTokenizer pretrained weights
- `decode_continuous()` 繞過 `@torch.inference_mode`，讓梯度可以流回 encoder

### 2.2 exp\_0224b：Encoder Frozen + No-VQ + Decoder LoRA

```
                         ┌─────────── Loss ───────────┐
                         │  wav MSE + MR-STFT + Mel   │
                         └────────┬──────────┬────────┘
                                  │          │
  Noisy Audio [B,1,T]            ▼          ▼
       │                    recon_wav   clean_wav
       ▼
  ┌─────────────────────┐
  │   Student Encoder   │
  │   + LoRA (r=64)     │  ← 完全 Frozen（from exp_0224a best）
  │   α=128             │
  └────────┬────────────┘
           │
           ▼
  student_encoder_out [B, 512, T_frame]
           │
           │  ← 連續特徵，跳過 VQ！
           │     .detach() 截斷梯度
           ▼
  ┌─────────────────────┐
  │   WavTokenizer      │
  │   Decoder Backbone  │
  │   (ConvNeXt × 12)   │
  │   pwconv1: + LoRA   │  ← 可訓練 (r=32, α=64)
  │   pwconv2: + LoRA   │  ← 可訓練
  │   + FourierHead     │  ← Frozen
  └────────┬────────────┘
           │
           ▼
      recon_wav [B, 1, T_wav]
```

**關鍵特性**：
- Encoder LoRA 完全 frozen，使用 exp\_0224a 訓練好的 weights
- 梯度只流過 Decoder LoRA 參數
- `student_encoder_out.detach()` 截斷反向傳播
- 實驗依賴順序：必須先完成 exp\_0224a

### 2.3 模型參數統計

| 模組 | 參數量 | exp\_0224a | exp\_0224b |
|---|---|---|---|
| WavTokenizer Encoder | ~40.3M | LoRA 可訓練 | Frozen |
| Encoder LoRA (r=64) | ~4.72M | ✅ 可訓練 | ❄️ Frozen |
| VQ Codebook (K=4096) | ~2.1M | ❄️ Frozen (unused) | ❄️ Frozen (unused) |
| Decoder Backbone (ConvNeXt×12) | ~84.8M | ❄️ Frozen | LoRA 可訓練 |
| Decoder LoRA (r=32) | ~2.36M | — | ✅ 可訓練 |
| FourierHead | ~39.4M | ❄️ Frozen | ❄️ Frozen |
| **可訓練合計** | | ~4.72M | ~2.36M |

---

## 三、Loss 設計

### 3.1 複合損失函數

exp\_0224 的 loss 設計與 exp\_0223 v2 完全一致，由三個互補的損失項組成：

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{wav}} \cdot \mathcal{L}_{\text{MSE}} + \lambda_{\text{stft}} \cdot \mathcal{L}_{\text{MR-STFT}} + \lambda_{\text{mel}} \cdot \mathcal{L}_{\text{Mel}}
$$

其中 $\lambda_{\text{wav}} = 1.0$、$\lambda_{\text{stft}} = 1.0$、$\lambda_{\text{mel}} = 45.0$。

### 3.2 MSE Loss（波形域）

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{T} \sum_{t=1}^{T} \left( \hat{x}(t) - x(t) \right)^2
$$

- **作用**：逐樣本點對齊，提供全域形狀約束
- **問題**：單獨使用會導致 silence collapse（exp\_0223 v1 的教訓）
- **加權**：$\lambda = 1.0$（輔助角色，非主導）

### 3.3 Multi-Resolution STFT Loss（頻譜域）

用三種不同的「視窗大小」同時觀察頻譜，各自有不同側重：

```
解析度 1:  n_fft=2048, hop=512,  win=2048  → 頻率解析度高，適合辨識基頻/諧波
解析度 2:  n_fft=1024, hop=256,  win=1024  → 頻率與時間兼顧
解析度 3:  n_fft=512,  hop=128,  win=512   → 時間解析度高，適合捕捉快速瞬態
```

每個解析度各算兩個子損失，再加總：

**① Spectral Convergence (SC)**：量化「頻譜形狀」整體差距
```
         ‖ |S_clean| - |S_recon| ‖_F          ← 各頻率的幅度差（Frobenius norm）
L_SC = ─────────────────────────────
              ‖ |S_clean| ‖_F                  ← 除以 clean 的幅度作正規化
```
→ 值越小代表 recon 的頻譜「輪廓」越像 clean；對頻譜整體能量分布敏感。

**② Log Magnitude (LogMag)**：在對數尺度比較幅度
```
L_LogMag = mean( | log|S_clean| - log|S_recon| | )
```
→ 對數尺度讓低能量頻段（如安靜段）和高能量頻段（如母音）受同等重視，
  避免大能量段主導梯度。

**三個解析度平均**：
```
L_MR-STFT = (1/3) × Σ_r [ L_SC(r) + L_LogMag(r) ]
```

- **作用**：補足 MSE 無法偵測的頻譜塌縮、諧波失真、時序錯位
- **加權**：λ = 1.0

### 3.4 Mel Spectrogram L1 Loss（感知域）

先把波形轉換成 Mel 頻譜，再在**對數 Mel 尺度**上做 L1 比較：

```
步驟 1：audio → STFT（n_fft=1024, hop=256）→ 幅度頻譜 |S|
步驟 2：|S| × Mel filterbank（100 個 mel 頻帶）→ Mel 幅度
步驟 3：取對數 → log-Mel（模擬人耳的對數感知）
步驟 4：計算 recon 與 clean 的 log-Mel 差距

L_Mel = mean( | log-Mel(recon) - log-Mel(clean) | )
        ↑ 對 100 個 mel 頻帶 × 所有時間幀取平均
```

**為什麼用 Mel 而不直接用 STFT？**
- 線性頻率的 STFT 中，高頻佔了大多數 bin，但人耳對低頻更敏感
- Mel 濾波器組模擬人耳的非線性頻率感知（低頻密集、高頻稀疏）
- 結果：語音基頻和共振峰（決定音色/清晰度的頻段）得到更多監督

**為什麼 λ=45？**
- 直接沿用 WavTokenizer 原始訓練設定的 `mel_loss_coeff=45`
- 使 Mel loss 在數值上主導 total loss（~85%），確保感知品質優先
- 副作用：val_total 的最佳 epoch 由 Mel loss 決定（不是 val_mse）

### 3.5 為什麼不用 GAN Loss？

| 項目 | GAN (MPD+MRD+DAC) | MR-STFT + Mel |
|---|---|---|
| 參數量 | 126.8M 判別器 | 0（純函數計算） |
| GPU 記憶體 | +4-6 GB | +~0 |
| 訓練穩定性 | G/D 平衡困難 | 穩定收斂 |
| D 預訓練適用性 | 域不匹配（clean domain → noisy student tokens） | 無此問題 |

結論：MR-STFT + Mel 是 GAN perceptual loss 的**輕量穩定替代品**。

---

## 四、訓練設定

### 4.1 超參數

| 參數 | 值 | 說明 |
|---|---|---|
| Epochs | 300 | exp\_0224a 設定 |
| Batch Size | 8 | 有效 batch=16（grad\_accum=2） |
| Learning Rate | 1e-4 → 1e-6 | Cosine Annealing with Warmup |
| Warmup Epochs | 5 | 線性 warmup |
| Weight Decay | 0.01 | AdamW |
| Gradient Clip | 1.0 | Max norm |
| AMP | ✅ | Mixed precision (FP16) |
| Seed | 42 | 可重現 |

### 4.2 資料增強

| 增強方式 | 機率 | 參數 |
|---|---|---|
| SNR Remix | 50% | range: [-5, 25] dB |
| Random Gain | 30% | ±3 dB |
| Random Crop | 30% | min ratio: 0.7 |
| Time Stretch | 20% | range: [0.95, 1.05] |

### 4.3 Encoder LoRA 配置（exp\_0224a）

```
LoRA rank:   64
LoRA alpha:  128   (alpha/rank = 2.0)
dropout:     預設（繼承 exp_0217）
target:      Encoder 的指定 Linear 層
init ckpt:   families/deps/t453_weighted_baseline/runs/t453_weighted_epoch_20260217_104843/best_model.pt
```

### 4.4 Decoder LoRA 配置（exp\_0224b）

```
LoRA rank:   32
LoRA alpha:  64    (alpha/rank = 2.0)
dropout:     0.1
target:      backbone.ConvNeXtBlock.pwconv1 (768→2304) × 12
             backbone.ConvNeXtBlock.pwconv2 (2304→768) × 12
             共 24 個 Linear 層
init ckpt:   exp_0224a best_model.pt（encoder 部分）
```

### 4.5 執行流程

```
Step 1: exp_0224a（Encoder LoRA 訓練）
    python quarantine/python/families/deps/no_vq_core/train_no_vq.py --mode epoch --epochs 300 --device cuda:0
    → 產出 best_model.pt（encoder LoRA 參數）

Step 2: exp_0224b（Decoder LoRA 訓練，依賴 Step 1）
    python quarantine/python/families/deps/no_vq_core/train_no_vq_decoder_lora.py \
        --mode epoch --epochs 300 --device cuda:1 \
        --encoder_ckpt families/deps/no_vq_core/runs/no_vq_epoch_YYYYMMDD_HHMMSS/best_model.pt
```

---

## 五、WavTokenizer 原始架構參考

### 5.1 Decoder 架構

```
WavTokenizer Decoder
├── VQ Codebook Lookup (K=4096, dim=512)  ← exp_0224 跳過此步驟
├── Backbone: VocosBackbone
│   ├── embed (Conv1d: 512 → 768)
│   ├── adanorm_num_embeddings = 1 (bandwidth_id)
│   └── ConvNeXtBlock × 12
│       ├── dwconv    (Conv1d: 768→768, k=7, groups=768)
│       ├── adanorm   (AdaLayerNorm)
│       ├── pwconv1   (Linear: 768 → 2304)    ← LoRA target
│       ├── GELU activation
│       └── pwconv2   (Linear: 2304 → 768)    ← LoRA target
├── Head: FourierHead
│   ├── out (Linear: 768 → 4098)
│   ├── istft (n_fft=2048, hop=300, win=2048)
│   └── 產生 24kHz 波形
```

### 5.2 Encoder 架構

```
WavTokenizer Encoder (EncodecEncoder variant)
├── 多層 Conv1d downsampling
├── LSTM layers
├── 最終輸出: [B, 512, T_frame]
│   T_frame = T_wav / (hop_product)
└── LoRA 施加於指定 Linear 層
```

### 5.3 信號流對比

```
【有 VQ（exp_0217, exp_0223）】
   Audio → Encoder → z_e [B,512,T] → VQ → z_q [B,512,T] → Decoder → Audio
                                      ↑
                               離散化！K=4096
                               量化誤差不可避免

【無 VQ（exp_0224）】
   Audio → Encoder → z_e [B,512,T] ──────────────────→ Decoder → Audio
                                    ↑
                             連續特徵！保留完整信息
                             梯度可以直接回傳
```

---

## 六、實驗進展與初步結果

### 6.1 exp\_0224a 訓練狀態

| 指標 | Epoch 1 | Best (Epoch 179) | Latest (Epoch 287) |
|---|---|---|---|
| val\_mel\_loss | 0.8134 | **0.7442** | 0.7505 |
| val\_wav\_mse | 0.02392 | 0.02776 | 0.02950 |
| val\_stft\_sc | — | 0.6608 | 0.6658 |
| noisy baseline (mel) | 1.2013 | 1.2013 | 1.2013 |

**觀察**：
- val\_mel 從 0.81 降至 0.74，遠低於 noisy baseline 1.20（**改善 38%**）
- 但 val\_wav\_mse 在後期略上升（0.024 → 0.030），可能出現輕微過擬合
- Best epoch 在 179，之後開始 plateau / 微幅退化
- LR 已降至 1e-6（cosine schedule 末期）

### 6.2 與 exp\_0223 v2 對比

| 指標 | exp\_0223 v2 (Decoder LoRA, 有 VQ) | exp\_0224a (Encoder LoRA, 無 VQ) |
|---|---|---|
| val\_mel\_loss (early) | ~0.84 (epoch 2) | 0.81 (epoch 1) |
| val\_mel\_loss (best) | 待完成 | **0.7442** |
| 可訓練參數 | 2.36M (decoder) | 4.72M (encoder) |
| VQ 瓶頸 | 有 | **無** |

### 6.3 exp\_0224b 狀態

尚未啟動。等待 exp\_0224a 完成 300 epochs 後，取 best\_model.pt 作為 encoder 初始化。

---

## 七、科學假設與預期

### 7.1 核心假設

- **H1**：跳過 VQ 能讓 Encoder LoRA 產生更接近 clean decoder input 的連續特徵，降低 mel loss
- **H2**：exp\_0224b（No-VQ + Decoder LoRA）應優於 exp\_0223 v2（有 VQ + Decoder LoRA），因為 decoder 接收的是更高品質的連續 feature
- **H3**：如果 exp\_0224a 比 exp\_0217（有 VQ + Encoder LoRA + feature MSE）的 PESQ 更高，則證明 VQ 確實是主要瓶頸

### 7.2 預期結果排序

```
PESQ 排序預期（高到低）：
  exp_0224b (No-VQ + Decoder LoRA)
  > exp_0224a (No-VQ + Encoder LoRA)
  > exp_0223 v2 (VQ + Decoder LoRA)
  > exp_0217 (VQ + Encoder LoRA + Feature MSE)
  > Noisy baseline
```

### 7.3 失敗模式分析

| 風險 | 症狀 | 對策 |
|---|---|---|
| 連續特徵域漂移 | Decoder 接收 OOD 輸入，輸出雜訊 | 加 feature norm / 約束 encoder output range |
| Encoder 過擬合 | val loss 上升 (已觀察到輕微跡象) | Early stopping / 降低 LR / 加 regularization |
| Decoder LoRA 不穩定 | exp\_0224b loss 震盪 | 降低 decoder LR / warmup 更長 |

---

## 八、檔案結構

```
families/deps/no_vq_core/
├── ARCHITECTURE.md           ← 本文件
├── models_no_vq.py           ← exp_0224a 模型（TeacherStudentNoVQ）
├── models_no_vq_decoder_lora.py  ← exp_0224b 模型（TeacherStudentNoVQDecoderLoRA）
├── quarantine/python/families/deps/no_vq_core/train_no_vq.py
│                           ← exp_0224a 歷史訓練腳本 (854 lines)
├── quarantine/python/families/deps/no_vq_core/train_no_vq_decoder_lora.py
│                           ← exp_0224b 歷史訓練腳本 (819 lines)
└── runs/
    └── no_vq_epoch_20260223_055458/  ← exp_0224a 訓練產出
        ├── config.json           ← 超參數快照
        ├── history.json          ← 逐 epoch metrics
        ├── best_model.pt         ← val_mel_loss 最佳 checkpoint
        ├── checkpoint_epoch*.pt  ← 每 10 epoch 備份
        ├── training_curves_*.png ← 每 25 epoch 曲線圖
        ├── audio_samples/        ← 評估音訊樣本
        └── train.log             ← 完整訓練日誌
```

---

## 九、如何重現

```bash
# 環境（conda env "test"）
conda activate test

# Step 1: exp_0224a — Encoder LoRA + No-VQ
python quarantine/python/families/deps/no_vq_core/train_no_vq.py \
    --mode epoch \
    --epochs 300 \
    --device cuda:0 \
    --encoder_ckpt families/deps/t453_weighted_baseline/runs/t453_weighted_epoch_20260217_104843/best_model.pt \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lambda_wav 1.0 \
    --lambda_stft 1.0 \
    --lambda_mel 45.0 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr 1e-4

# Step 2: exp_0224b — Decoder LoRA + No-VQ（需先完成 Step 1）
python quarantine/python/families/deps/no_vq_core/train_no_vq_decoder_lora.py \
    --mode epoch \
    --epochs 300 \
    --device cuda:1 \
    --encoder_ckpt families/deps/no_vq_core/runs/no_vq_epoch_20260223_055458/best_model.pt \
    --decoder_lora_rank 32 \
    --decoder_lora_alpha 64
```
