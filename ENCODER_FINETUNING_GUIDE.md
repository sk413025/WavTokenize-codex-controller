# WavTokenizer Encoder Fine-tuning 指南

## 🎯 核心挑戰

Fine-tuning 預訓練的 VQ-VAE Encoder 需要平衡：
- ✅ **學習新任務**（例如去噪）
- ⚠️ **保留原始特徵**（避免破壞預訓練的表示能力）

---

## 📚 推薦方法（按保守程度排序）

### **方法 1: LoRA (Low-Rank Adaptation)** ⭐ 推薦

#### 📖 文獻支持

- **Whisper + LoRA** (Interspeech 2025): [Mixture of LoRA Experts for Multi-Accent Speech Recognition](https://www.isca-archive.org/interspeech_2025/bagat25_interspeech.pdf)
  - 對 Whisper encoder 的 attention 模組應用 LoRA
  - **關鍵發現**: "accent-related adaptation for the encoder leads to systematic improvement"

- **LoRA 機制分析** (2024): [Mechanistic Interpretability of LoRA-adapted Whisper](https://www.researchgate.net/publication/395402567_Behind_the_Scenes_Mechanistic_Interpretability_of_LoRA-adapted_Whisper_for_Speech_Emotion_Recognition)
  - 首次系統性分析 LoRA 在語音任務中的工作機制
  - **發現**: "LoRA reshapes encoder hierarchies" 但保留原始能力

#### 🔧 實作方式

```python
from peft import LoraConfig, get_peft_model
import sys
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
from decoder.pretrained import WavTokenizer

# 1. 載入預訓練 WavTokenizer
wavtokenizer = WavTokenizer.from_pretrained0802(config, ckpt)

# 2. 配置 LoRA (針對 Encoder 的 Conv 層)
lora_config = LoraConfig(
    r=16,                    # LoRA rank (低秩)
    lora_alpha=32,           # LoRA 縮放係數
    target_modules=[
        # WavTokenizer Encoder (SEANetEncoder) 的卷積層
        "feature_extractor.encodec.encoder.model.0.conv.conv",
        "feature_extractor.encodec.encoder.model.2.conv.conv",
        "feature_extractor.encodec.encoder.model.4.conv.conv",
        "feature_extractor.encodec.encoder.model.6.conv.conv",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION"  # 不是 CAUSAL_LM
)

# 3. 應用 LoRA
model = get_peft_model(wavtokenizer, lora_config)

# 4. 凍結所有非 LoRA 參數（自動完成）
model.print_trainable_parameters()
# 預期輸出: trainable params: ~100K / total: ~50M (< 0.5%)

# 5. 訓練
# 只有 LoRA 參數會更新，原始權重保持不變
```

#### ✅ 優勢

| 特點 | 說明 |
|------|------|
| **極小參數量** | < 1% 的可訓練參數 |
| **不破壞原始權重** | 原始 encoder 權重完全凍結 |
| **易於移除** | 可以隨時移除 LoRA，恢復原始模型 |
| **文獻支持** | Whisper、LLaMA 等大模型驗證有效 |
| **多任務友好** | 可以訓練多個 LoRA 模組，針對不同任務切換 |

#### ⚠️ 注意事項

```python
# LoRA 目標選擇建議
# ✅ 推薦: Encoder 的卷積層
# ❌ 避免: VQ 層（會破壞 codebook 對應關係）
# ⚠️ 謹慎: Backbone（如果想保留生成能力）

# 檢查可用的模組名稱
for name, module in wavtokenizer.named_modules():
    if 'encoder' in name and 'conv' in name:
        print(name, type(module).__name__)
```

---

### **方法 2: Codebook-Only Fine-tuning** 🛡️ 最保守

#### 📖 文獻支持

- **WhisTLE** (2024): [Deeply Supervised Text-Only Domain Adaptation](https://arxiv.org/html/2509.10452.pdf)
  - 使用 frozen encoder + trainable VAE/VQ
  - 證明凍結 encoder 也能有效適應新 domain

- **VQ-VAE Training Strategies**: [GitHub Implementation](https://github.com/AndrewBoessen/VQ-VAE)
  - "Pretraining encoder before training embeddings allows encoder to learn meaningful representations"

#### 🔧 實作方式

```python
# 策略 1: 只微調 Codebook (最保守)
# ════════════════════════════════════════════

# 凍結 Encoder
for param in wavtokenizer.feature_extractor.encodec.encoder.parameters():
    param.requires_grad = False

# 只訓練 VQ codebook (允許 EMA 更新)
for param in wavtokenizer.feature_extractor.encodec.quantizer.parameters():
    param.requires_grad = True  # 如果使用學習式更新

# 訓練時，codebook 會適應新 domain 的特徵分布
# 但 encoder 提取的特徵空間保持不變
```

#### ✅ 優勢

- **最安全**: Encoder 完全不變，不會破壞預訓練特徵
- **適合 Domain Shift**: 當輸入分布改變（例如噪音音頻）
- **快速**: 只更新 codebook（通常用 EMA，甚至不需要梯度）

#### ⚠️ 限制

- **表達能力有限**: 如果 encoder 提取的特徵不適合新任務，codebook 調整無法彌補
- **需要足夠數據**: Codebook 需要看到足夠多新 domain 的樣本

---

### **方法 3: Partial Fine-tuning (Layer-wise)** ⚖️ 平衡

#### 📖 理論基礎

深度網絡的層級特徵：
- **淺層**: 通用低級特徵（邊緣、紋理、頻率成分）
- **深層**: 任務特定高級特徵（語音內容、音色）

**策略**: 凍結淺層，fine-tune 深層

#### 🔧 實作方式

```python
# 策略: 凍結前 N 層，訓練後面的層
# ════════════════════════════════════════════

# WavTokenizer Encoder 結構:
# model.0: Conv1d (第 1 層)
# model.2: Conv1d (第 2 層)
# model.4: Conv1d (第 3 層)
# model.6: Conv1d (第 4 層)

encoder = wavtokenizer.feature_extractor.encodec.encoder

# 凍結前 2 層（保留低級特徵）
for i in [0, 2]:
    for param in encoder.model[i].parameters():
        param.requires_grad = False

# 訓練後 2 層（適應新任務）
for i in [4, 6]:
    for param in encoder.model[i].parameters():
        param.requires_grad = True

# 檢查
trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total = sum(p.numel() for p in encoder.parameters())
print(f"Trainable: {trainable/total*100:.2f}%")
```

#### 📊 選擇凍結層數的指南

| 任務特性 | 建議策略 |
|---------|---------|
| **Domain 接近** (例如乾淨語音 → 乾淨音樂) | 凍結更多層 (前 3 層) |
| **Domain 差異大** (例如乾淨 → 噪音) | 凍結較少 (前 1-2 層) |
| **任務相似** (例如重建 → 去噪) | 凍結深層，訓練淺層 |
| **任務不同** (例如重建 → 分類) | 凍結淺層，訓練深層 |

#### ✅ 優勢

- **可控制**: 精確控制哪些特徵層被調整
- **折衷方案**: 在保留和適應之間平衡

#### ⚠️ 風險

- **仍可能破壞**: 即使只訓練部分層，也可能影響整體特徵
- **需要實驗**: 最佳凍結層數需要嘗試

---

### **方法 4: Adapter Layers** 🔌 插件式

#### 📖 文獻基礎

- **Adapter Tuning** (Google, 2019): 在每層插入小的 bottleneck 模組
- **適用於音頻**: 已在 Wav2Vec 2.0、HuBERT 等模型驗證

#### 🔧 實作方式

```python
import torch.nn as nn

class Adapter(nn.Module):
    """
    Bottleneck adapter module

    Input: (B, C, T)
    Output: (B, C, T) - same shape
    """
    def __init__(self, dim, bottleneck_dim=64):
        super().__init__()
        self.down = nn.Conv1d(dim, bottleneck_dim, 1)
        self.act = nn.GELU()
        self.up = nn.Conv1d(bottleneck_dim, dim, 1)
        self.scale = nn.Parameter(torch.zeros(1))  # 初始為 0，不影響原始輸出

    def forward(self, x):
        # Residual connection
        return x + self.scale * self.up(self.act(self.down(x)))


# 在 Encoder 每層後插入 Adapter
class EncoderWithAdapters(nn.Module):
    def __init__(self, wavtokenizer):
        super().__init__()
        self.encoder = wavtokenizer.feature_extractor.encodec.encoder

        # 凍結原始 encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 在每層後插入 adapter
        self.adapters = nn.ModuleList([
            Adapter(dim=128, bottleneck_dim=32),  # 在 layer 0 後
            Adapter(dim=256, bottleneck_dim=64),  # 在 layer 2 後
            Adapter(dim=512, bottleneck_dim=128), # 在 layer 4 後
            Adapter(dim=1024, bottleneck_dim=256),# 在 layer 6 後
        ])

    def forward(self, x):
        # 手動執行每層 + adapter
        x = self.encoder.model[0](x)
        x = self.adapters[0](x)

        x = self.encoder.model[2](x)
        x = self.adapters[1](x)

        x = self.encoder.model[4](x)
        x = self.adapters[2](x)

        x = self.encoder.model[6](x)
        x = self.adapters[3](x)

        return x
```

#### ✅ 優勢

- **完全凍結原始模型**: Encoder 權重不變
- **小參數量**: Bottleneck 設計限制參數
- **易於移除**: 可以設置 `scale=0` 恢復原始行為

#### ⚠️ 挑戰

- **需要修改前向傳播**: 不如 LoRA 即插即用
- **可能引入瓶頸**: Bottleneck 太小會限制表達能力

---

### **方法 5: Multi-task Learning with Regularization** 🎯 保持能力

#### 📖 核心思想

同時訓練**新任務**和**原始任務**，防止 catastrophic forgetting。

#### 🔧 實作方式

```python
# 損失函數設計
# ════════════════════════════════════════════

def compute_loss(model, noisy_audio, clean_audio):
    # 1. 新任務: 去噪重建
    z_e_noisy = model.encode(noisy_audio)
    z_q_noisy, codes_noisy, vq_loss_noisy = model.vq(z_e_noisy)
    audio_denoised = model.decode(z_q_noisy)

    denoising_loss = F.mse_loss(audio_denoised, clean_audio)

    # 2. 原始任務: 乾淨音頻重建（保持原始能力）
    z_e_clean = model.encode(clean_audio)
    z_q_clean, codes_clean, vq_loss_clean = model.vq(z_e_clean)
    audio_recon = model.decode(z_q_clean)

    reconstruction_loss = F.mse_loss(audio_recon, clean_audio)

    # 3. Codebook Consistency (保持 codebook 不劇變)
    codebook_reg = F.mse_loss(
        model.vq.codebook,
        original_codebook.detach()  # 預訓練的 codebook
    )

    # 總損失
    total_loss = (
        denoising_loss +           # 新任務
        0.5 * reconstruction_loss + # 保持原始能力
        0.1 * codebook_reg +       # Codebook 正則化
        vq_loss_noisy + vq_loss_clean
    )

    return total_loss
```

#### ✅ 優勢

- **明確保護**: 直接在 loss 中保護原始能力
- **靈活控制**: 可調整新舊任務的權重平衡

#### ⚠️ 挑戰

- **需要乾淨數據**: 必須有原始 domain 的數據
- **計算開銷**: 需要雙倍前向傳播

---

## 📊 方法對比總結

| 方法 | 參數量 | 風險 | 實作難度 | 文獻支持 | 推薦度 |
|------|--------|------|----------|----------|--------|
| **LoRA** | < 1% | 低 | 簡單 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Codebook-only** | ~1% | 極低 | 簡單 | ⭐⭐ | ⭐⭐⭐⭐ |
| **Partial Fine-tuning** | 25-75% | 中 | 簡單 | ⭐ | ⭐⭐⭐ |
| **Adapter Layers** | ~5% | 低 | 中等 | ⭐⭐ | ⭐⭐⭐⭐ |
| **Multi-task** | 100% | 低-中 | 複雜 | ⭐⭐ | ⭐⭐⭐ |

---

## 🎯 具體建議

### **針對你的去噪任務**

基於你之前的 Exp5 系列實驗（Transformer 去噪），這裡是最佳方案：

#### **推薦方案 1: Encoder LoRA + Frozen VQ + Frozen Backbone**

```python
# 最佳平衡：保留原始 token 空間，只調整 encoder 特徵提取

from peft import LoraConfig, get_peft_model

# 1. 載入 WavTokenizer
wavtokenizer = WavTokenizer.from_pretrained0802(config, ckpt)

# 2. LoRA 配置（只針對 Encoder）
lora_config = LoraConfig(
    r=16,  # 保守的 rank
    lora_alpha=32,
    target_modules=[
        # 只在 Encoder 的卷積層
        "feature_extractor.encodec.encoder.model.0.conv.conv",
        "feature_extractor.encodec.encoder.model.2.conv.conv",
        "feature_extractor.encodec.encoder.model.4.conv.conv",
        "feature_extractor.encodec.encoder.model.6.conv.conv",
    ],
    lora_dropout=0.1,
)

model = get_peft_model(wavtokenizer, lora_config)

# 3. 確保 VQ 和 Backbone 凍結（PEFT 會自動凍結非 target）
for name, param in model.named_parameters():
    if 'lora' not in name:
        param.requires_grad = False

# 4. 訓練流程
for noisy_audio, clean_audio in dataloader:
    # Noisy → Encoder (LoRA) → VQ (frozen) → codes
    features_noisy, codes_noisy = model.encode(noisy_audio, bandwidth_id=0)

    # Clean → Encoder (LoRA) → VQ (frozen) → codes
    features_clean, codes_clean = model.encode(clean_audio, bandwidth_id=0)

    # Loss: 預測 clean codes (token-level denoising)
    loss = F.cross_entropy(
        logits_from_features(features_noisy),
        codes_clean
    )

    loss.backward()  # 只更新 LoRA 權重
    optimizer.step()
```

**為什麼這樣設計？**
- ✅ **Encoder LoRA**: 學習從噪音音頻提取更好的特徵
- ✅ **Frozen VQ**: 保持 codebook 不變，token 空間一致
- ✅ **Frozen Backbone**: 保留預訓練的 decoder 能力
- ✅ **極小風險**: 原始權重完全不變，隨時可恢復

---

#### **推薦方案 2: Codebook Adaptation (無 Encoder 修改)**

```python
# 最保守方案：完全不動 Encoder

# 1. 凍結整個 Encoder
for param in wavtokenizer.feature_extractor.encodec.encoder.parameters():
    param.requires_grad = False

# 2. 允許 VQ Codebook 適應（使用 EMA 或訓練）
# WavTokenizer 默認使用 EMA 更新 codebook，無需額外設置

# 3. 訓練時，讓 codebook 看到噪音音頻
for noisy_audio in dataloader:
    with torch.no_grad():  # Encoder 凍結
        features, codes, vq_loss = wavtokenizer.feature_extractor(
            noisy_audio, bandwidth_id=0
        )
    # VQ 的 EMA 會自動更新 codebook 以適應新特徵分布

# 4. 驗證：檢查 codebook 是否適應噪音 domain
# 計算量化誤差應該降低
```

**適用場景**：
- 當你擔心任何 encoder 修改
- 當你的去噪任務主要是 domain shift（分布改變）而非特徵改變

---

## ⚠️ 關鍵注意事項

### **1. VQ Codebook 的特殊性**

```python
# ⚠️ 如果你 fine-tune encoder，必須考慮 codebook 對應關係

# 原始情況:
# encoder(clean_audio) → features_clean → VQ → code_123
# code_123 對應的語義: "某個特定的音素/音色"

# Fine-tune 後:
# encoder_finetuned(noisy_audio) → features_noisy → VQ → code_???
# code_??? 可能不再對應原始語義！

# 解決方案:
# 1. Freeze VQ (推薦): 保持 token 語義一致
# 2. Joint fine-tune: Encoder + VQ 一起調整
# 3. 使用 commitment loss: 確保 encoder 輸出接近原始 codebook
```

### **2. 驗證策略**

```python
# 在 fine-tuning 過程中，定期檢查原始能力

def evaluate_original_capability(model, clean_test_data):
    """檢查模型是否還能重建乾淨音頻"""
    with torch.no_grad():
        reconstructed = model(clean_test_data)
        original_loss = F.mse_loss(reconstructed, clean_test_data)

    # 如果 original_loss 增加太多，說明破壞了原始能力
    return original_loss

# 訓練循環中
if epoch % 5 == 0:
    orig_loss = evaluate_original_capability(model, clean_val_data)
    if orig_loss > threshold:
        print("WARNING: Original capability degraded!")
        # 考慮降低學習率或停止訓練
```

### **3. 學習率設置**

```python
# LoRA fine-tuning 建議學習率
optimizer = torch.optim.AdamW([
    {'params': lora_params, 'lr': 5e-5},  # LoRA 層較小學習率
    # 如果有其他可訓練層
    {'params': other_params, 'lr': 1e-5},
])

# 原則: 比從頭訓練小 10-100 倍
# WavTokenizer 原始訓練 lr ~ 1e-4
# Fine-tuning lr ~ 1e-5 to 5e-5
```

---

## 📖 相關文獻與資源

### **Parameter-Efficient Fine-tuning (LoRA)**
- [LoRA for Multi-Accent Speech Recognition](https://www.isca-archive.org/interspeech_2025/bagat25_interspeech.pdf) - Interspeech 2025
- [Mechanistic Interpretability of LoRA-adapted Whisper](https://www.researchgate.net/publication/395402567_Behind_the_Scenes_Mechanistic_Interpretability_of_LoRA-adapted_Whisper_for_Speech_Emotion_Recognition) - 2024

### **VQ-VAE & Neural Audio Codecs**
- [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/pdf/2107.03312) - Google, 2021
- [EnCodec](https://github.com/facebookresearch/encodec) - Meta, 2022
- [Investigating Neural Audio Codecs for Speech Language Models](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/12/SLT2024_CodecInvestigation.pdf) - Microsoft Research, 2024

### **Domain Adaptation with Frozen Encoders**
- [WhisTLE: Text-Only Domain Adaptation](https://arxiv.org/html/2509.10452.pdf) - 2024
- [Learning Source Disentanglement in Neural Audio Codec](https://arxiv.org/html/2409.11228v1) - 2024

### **VQ-VAE Training Best Practices**
- [VQ-VAE with Pretrained Encoder](https://github.com/AndrewBoessen/VQ-VAE) - Implementation Guide
- [Dual Codebook VQ](https://arxiv.org/html/2503.10832v1) - Enhanced Codebook Design

---

## 🎓 實驗建議流程

### **階段 1: Baseline (不修改 Encoder)**

```python
# 1. 使用預訓練 WavTokenizer，完全凍結
# 2. 只訓練你的去噪 Transformer (Exp5-3-1 方式)
# 3. 記錄性能作為 baseline
```

### **階段 2: 保守 Fine-tuning (Codebook Only)**

```python
# 1. 凍結 Encoder，允許 Codebook EMA 更新
# 2. 訓練去噪 Transformer
# 3. 比較與 baseline: 如果提升 < 1%，說明 codebook 調整幫助有限
```

### **階段 3: LoRA Fine-tuning (推薦)**

```python
# 1. Encoder 加 LoRA (rank=8 開始，保守)
# 2. Freeze VQ 和 Backbone
# 3. 訓練
# 4. 如果性能提升明顯且不破壞原始能力 → 成功！
# 5. 如果不夠，嘗試 rank=16 或 rank=32
```

### **階段 4: 更激進方案（如果前面都不夠）**

```python
# 只有在必要時才考慮:
# - Partial fine-tuning (後幾層)
# - Adapter layers
# - Full fine-tuning (最後手段)
```

---

## ✅ 總結與最終建議

### **針對你的去噪任務，推薦順序**：

1. **首選: Encoder LoRA** (rank=16, alpha=32)
   - 文獻支持最強
   - 風險最低
   - 實作簡單

2. **次選: Codebook Adaptation**
   - 極度保守
   - 適合 domain shift 為主的場景

3. **備選: Adapter Layers**
   - 需要更多實作工作
   - 完全可逆

**關鍵原則**：
- ✅ **Always freeze VQ codebook** (保持 token 語義一致)
- ✅ **Start conservative** (小 LoRA rank)
- ✅ **Monitor original capability** (定期測試乾淨音頻重建)
- ✅ **Use multi-task learning** (同時訓練原始任務)

---

## 🔗 Sources

- [Mixture of LoRA Experts for Multi-Accent Speech Recognition](https://www.isca-archive.org/interspeech_2025/bagat25_interspeech.pdf)
- [Mechanistic Interpretability of LoRA-adapted Whisper for Speech Emotion Recognition](https://www.researchgate.net/publication/395402567_Behind_the_Scenes_Mechanistic_Interpretability_of_LoRA-adapted_Whisper_for_Speech_Emotion_Recognition)
- [WhisTLE: Deeply Supervised, Text-Only Domain Adaptation](https://arxiv.org/html/2509.10452.pdf)
- [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/pdf/2107.03312)
- [Microsoft Research: Investigating Neural Audio Codecs](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/12/SLT2024_CodecInvestigation.pdf)
- [Learning Source Disentanglement in Neural Audio Codec](https://arxiv.org/html/2409.11228v1)
