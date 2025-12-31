# exp_1231: 優化方向測試實驗

**日期**：2024-12-31
**基準**：Exp67 (Best Val Acc = 1.06%)

---

## 實驗概覽

| 實驗 | 方向 | 方法 | 目標 |
|------|------|------|------|
| Exp71 | 1-A | Soft Token Loss (KLD) | 用分布監督取代 hard token 監督 |
| Exp72 | 1-B | Contrastive Token Loss | 正負樣本對比學習 |
| Exp73 | 2-A | Denoising Adapter | 加入專門的去噪模組 |
| Exp74 | 2-B | Multi-Scale Feature Loss | 監督多層中間特徵 |
| Exp75 | 3-A | Progressive Loss Schedule | 漸進式 loss 難度 |
| Exp76 | 3-B | Two-Stage Training | 先去噪再匹配 tokens |

---

## Exp71: Soft Token Loss (KL Divergence)

### 核心想法

當前的 Token Accuracy loss 是 hard decision（只看 argmax），
改用 KL Divergence 監督整個 logits 分布，讓梯度更平滑。

### 改動

```python
# 原本：只監督 argmax token
loss = CrossEntropy(student_codes, teacher_codes)

# 新：監督整個分布
student_logits = -distance(student_features, codebook) / temperature
teacher_logits = -distance(teacher_features, codebook) / temperature

student_probs = log_softmax(student_logits)
teacher_probs = softmax(teacher_logits).detach()  # teacher 不需要梯度

loss = KL_Divergence(student_probs, teacher_probs)
```

### 預期效果

- 梯度更平滑，學習更穩定
- 即使沒選到相同的 token，也會鼓勵靠近正確區域
- 預期 Token Accuracy 提升 1-5%

### 配置

```
soft_token_weight: 1.0
soft_token_temperature: 1.0  # 可調整
feature_weight: 1.0
triplet_weight: 1.0
```

---

## Exp72: Contrastive Token Loss

### 核心想法

用對比學習的方式，讓 student feature 靠近正確的 codebook entry，遠離錯誤的。

### 改動

```python
# 正樣本：Teacher 選擇的 code
positive = codebook[teacher_codes]  # (B, T, D)

# 負樣本：隨機採樣其他 codes
negative = sample_hard_negatives(codebook, teacher_codes, k=16)  # (B, T, k, D)

# InfoNCE Loss
pos_sim = cosine_sim(student_features, positive)
neg_sim = cosine_sim(student_features, negative)

loss = -log(exp(pos_sim / tau) / (exp(pos_sim / tau) + sum(exp(neg_sim / tau))))
```

### Hard Negative Mining

選擇「距離 student feature 近但不是正確答案」的 codes 作為負樣本：

```python
def sample_hard_negatives(student_features, codebook, teacher_codes, k=16):
    # 計算 student 到所有 codes 的距離
    distances = cdist(student_features, codebook)

    # 排除正確答案
    distances[teacher_codes] = float('inf')

    # 選擇最近的 k 個作為 hard negatives
    hard_neg_indices = distances.topk(k, largest=False)
    return codebook[hard_neg_indices]
```

### 配置

```
contrastive_weight: 0.5
num_negatives: 16
temperature: 0.1
hard_negative_mining: True
```

---

## Exp73: Denoising Adapter

### 核心想法

LoRA 調整的是 attention weights，可能不足以學習去噪。
加入專門的 Adapter 模組來學習 noisy → clean 的殘差映射。

### 架構

```python
class DenoisingAdapter(nn.Module):
    def __init__(self, dim=512, expansion=4, dropout=0.1):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden, dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + x

# 在 encoder 最後一層輸出後加入
class StudentEncoderWithAdapter(nn.Module):
    def __init__(self, base_encoder, adapter):
        self.encoder = base_encoder
        self.adapter = adapter

    def forward(self, x):
        features = self.encoder(x)
        features = self.adapter(features)  # 去噪
        return features
```

### 配置

```
adapter_dim: 512
adapter_expansion: 4
adapter_dropout: 0.1
adapter_position: "after_encoder"  # 或 "every_layer"
```

---

## Exp74: Multi-Scale Feature Loss

### 核心想法

不只監督最後一層 encoder output，監督多個中間層的 features。

### 改動

```python
def get_intermediate_features(encoder, audio, layer_indices=[6, 12, 18]):
    """提取指定層的中間特徵"""
    features = []
    x = encoder.preprocess(audio)

    for i, layer in enumerate(encoder.layers):
        x = layer(x)
        if i in layer_indices:
            features.append(x)

    return features

# 訓練時
student_features_list = get_intermediate_features(student_encoder, noisy, [6, 12, 18])
teacher_features_list = get_intermediate_features(teacher_encoder, clean, [6, 12, 18])

# 多尺度 loss
multi_scale_loss = sum(
    MSE(s, t) * weight
    for s, t, weight in zip(student_features_list, teacher_features_list, [0.1, 0.3, 0.6])
)
```

### 配置

```
multi_scale_layers: [6, 12, 18]  # 第 6, 12, 18 層
multi_scale_weights: [0.1, 0.3, 0.6]  # 越深的層權重越大
```

---

## Exp75: Progressive Loss Schedule

### 核心想法

訓練初期用連續空間的 loss（Feature Loss），
後期逐漸加入離散空間的 loss（Token Loss）。

### Schedule

```
Phase 1 (Epoch 1-100):
  - feature_weight: 1.0
  - triplet_weight: 1.0
  - soft_token_weight: 0.0

Phase 2 (Epoch 101-200):
  - feature_weight: 1.0
  - triplet_weight: 1.0
  - soft_token_weight: 0.5  # 開始加入

Phase 3 (Epoch 201-300):
  - feature_weight: 0.5  # 降低
  - triplet_weight: 0.5  # 降低
  - soft_token_weight: 1.0  # 主導
```

### 實作

```python
def get_loss_weights(epoch, total_epochs=300):
    progress = epoch / total_epochs

    if progress < 0.33:  # Phase 1
        return {'feature': 1.0, 'triplet': 1.0, 'soft_token': 0.0}
    elif progress < 0.66:  # Phase 2
        soft_w = (progress - 0.33) / 0.33  # 0 → 1
        return {'feature': 1.0, 'triplet': 1.0, 'soft_token': soft_w}
    else:  # Phase 3
        feature_w = 1.0 - (progress - 0.66) / 0.34 * 0.5  # 1 → 0.5
        return {'feature': feature_w, 'triplet': feature_w, 'soft_token': 1.0}
```

---

## Exp76: Two-Stage Training

### 核心想法

先訓練一個 waveform-level 的去噪網路，
再用去噪後的音訊去匹配 VQ tokens。

### Stage 1: Waveform Denoising

```python
# 目標：Noisy Audio → Clean Audio (waveform level)
# 不經過 VQ，直接監督波形

class WaveformDenoiser(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder  # WavTokenizer encoder
        self.decoder = decoder  # WavTokenizer decoder

    def forward(self, noisy_audio):
        features = self.encoder(noisy_audio)
        clean_audio = self.decoder(features)
        return clean_audio

# Loss: L1 + STFT Loss
loss = L1(denoised_audio, clean_audio) + MultiResolutionSTFTLoss(denoised_audio, clean_audio)
```

### Stage 2: Token Matching

```python
# 輸入：Stage 1 去噪後的音訊
# 目標：匹配 Teacher tokens

denoised_audio = stage1_model(noisy_audio).detach()  # 不更新 Stage 1

# 用去噪後的音訊去提取 features
student_features = student_encoder(denoised_audio)
teacher_features = teacher_encoder(clean_audio)

# Token matching loss
loss = soft_token_loss + feature_loss
```

### 配置

```
# Stage 1
stage1_epochs: 100
stage1_lr: 1e-4
stage1_loss: "l1 + stft"

# Stage 2
stage2_epochs: 200
stage2_lr: 1e-5
stage2_freeze_stage1: True
```

---

## 執行順序建議

| 優先級 | 實驗 | 原因 |
|--------|------|------|
| 1 | Exp71 (Soft Token) | 最簡單，改動最小 |
| 2 | Exp75 (Progressive) | 結合 Exp71，測試 schedule |
| 3 | Exp72 (Contrastive) | 不同的監督方式 |
| 4 | Exp73 (Adapter) | 架構改動 |
| 5 | Exp74 (Multi-Scale) | 需要修改 encoder |
| 6 | Exp76 (Two-Stage) | 最複雜，需要兩階段 |

---

## 快速啟動

```bash
# Exp71: Soft Token Loss
bash exp_1231/run_exp71_soft_token.sh

# Exp72: Contrastive Token Loss
bash exp_1231/run_exp72_contrastive.sh

# Exp73: Denoising Adapter
bash exp_1231/run_exp73_adapter.sh

# Exp74: Multi-Scale Feature Loss
bash exp_1231/run_exp74_multi_scale.sh

# Exp75: Progressive Loss Schedule
bash exp_1231/run_exp75_progressive.sh

# Exp76: Two-Stage Training
bash exp_1231/run_exp76_two_stage.sh
```

---

## 輸出結構

每個實驗完成後會在 `exp_1231/runs/<exp_name>/` 產生以下輸出：

```
exp_1231/runs/exp71_soft_token/
├── config.json           # 實驗配置
├── history.json          # 訓練歷史 (loss, accuracy, lr)
├── training_curves.png   # Loss 和 Accuracy 曲線圖
├── best_model.pt         # 最佳驗證準確率的模型
├── last_model.pt         # 最後一個 epoch 的模型
└── audio_samples/        # 音檔樣本
    └── epoch_X/          # 最佳模型時的 epoch
        ├── train/        # 訓練集樣本
        │   ├── sample_0_noisy.wav
        │   ├── sample_0_clean.wav
        │   ├── sample_0_denoised.wav
        │   ├── sample_1_noisy.wav
        │   ├── sample_1_clean.wav
        │   ├── sample_1_denoised.wav
        │   └── ...
        └── val/          # 驗證集樣本
            ├── sample_0_noisy.wav
            ├── sample_0_clean.wav
            ├── sample_0_denoised.wav
            └── ...
```

### Loss 圖

每個 epoch 都會更新 `training_curves.png`，包含：

1. **Total Loss**: Train/Val 的總 loss 曲線
2. **Token Accuracy**: Train/Val 的 Token 匹配準確率
3. **實驗特定指標**:
   - Exp71: Soft Token Accuracy
   - Exp72: Contrastive Accuracy
   - Exp75: Loss Weights Schedule (phase 變化)
4. **Feature Loss**: Train/Val 的 feature MSE loss
5. **其他 Loss 組成**
6. **Learning Rate**: 學習率變化

### 音檔輸出

音檔會在 **最佳模型更新時** 保存，同時包含 **train** 和 **val** 兩個資料夾：

- `noisy.wav`: 原始帶噪音的輸入
- `clean.wav`: 原始乾淨的目標
- `denoised.wav`: Student 模型去噪後的重建

預設保存 3 個樣本，可聽取比較去噪效果。

---

## 檔案結構

```
exp_1231/
├── README.md                     # 本文件
├── losses.py                     # Loss 函數定義
├── models.py                     # 模型定義 (Adapter)
├── utils.py                      # 共用工具 (plot_metrics, save_audio_samples)
├── train_exp71_soft_token.py     # Exp71 訓練腳本
├── train_exp72_contrastive.py    # Exp72 訓練腳本
├── train_exp73_adapter.py        # Exp73 訓練腳本
├── train_exp75_progressive.py    # Exp75 訓練腳本
├── train_exp76_two_stage.py      # Exp76 訓練腳本
├── run_exp71_soft_token.sh       # Exp71 啟動腳本
├── run_exp72_contrastive.sh      # Exp72 啟動腳本
├── run_exp73_adapter.sh          # Exp73 啟動腳本
├── run_exp75_progressive.sh      # Exp75 啟動腳本
├── run_exp76_two_stage.sh        # Exp76 啟動腳本
└── runs/                         # 實驗輸出目錄
    ├── exp71_soft_token/
    ├── exp72_contrastive/
    └── ...
```
