# EXP2: Baseline + Speaker Embedding L2 Loss

## 🎯 實驗目標

探索通過引入 **Speaker Embedding L2 Loss** 作為輔助約束，提升 Token Denoising Transformer 對**未見語者的泛化能力**。

## 💡 核心假設

### 問題診斷
從遮罩實驗（exp/masking）和 zero-shot 實驗（exp/）的結果觀察到：
- ✅ 訓練集語者：降噪效果好
- ❌ 未見語者：泛化性不足

### 假設原因
當前模型可能學到了**特定語者的 token pattern**，導致：
```python
# 可能的過擬合模式
girl1_noisy_tokens → girl1_clean_tokens  # OK
girl2_noisy_tokens → girl2_clean_tokens  # OK
girl9_noisy_tokens → ???                  # 失敗，因為沒見過 girl9 的 pattern
```

### 解決方案
通過 **Speaker Embedding L2 Loss** 作為輔助約束：

```
Loss_total = Loss_CE(pred_tokens, clean_tokens)              ← 主任務：token 翻譯
           + λ * Loss_Speaker(pred_audio, input_audio)      ← 輔助：保持說話人身份
```

**關鍵機制**：
1. **主任務 (CE Loss)**: 學習 noisy → clean 的 token 序列翻譯
2. **輔助約束 (Speaker L2 Loss)**: 強制模型在去噪時保持原始說話人特徵
3. **解耦效果**: 迫使模型分離「噪音去除」和「說話人身份」兩個任務

## 📊 實驗架構

### 模型架構
```
Noisy Tokens (B, T)
    ↓
Frozen Codebook Lookup → (B, T, 512)
    ↓
Positional Encoding
    ↓
Transformer Encoder (4 layers)
    ↓
Output Projection → (B, T, 4096)
    ↓
Predicted Clean Tokens
```

### 損失函數
```python
# 主任務
L_CE = CrossEntropy(pred_logits, target_tokens)

# 輔助約束
pred_audio = WavTokenizer.decode(pred_tokens)
input_audio = WavTokenizer.decode(noisy_tokens)
speaker_emb_pred = ECAPA(pred_audio)
speaker_emb_input = ECAPA(input_audio)
L_Speaker = MSE(speaker_emb_pred, speaker_emb_input)

# 總損失
L_total = L_CE + λ * L_Speaker
```

## 🔬 實驗設置

### 數據集分割
- **訓練集語者**: boy1, boy3-6, boy9-10, girl2-4, girl6-8, girl11
- **驗證集語者 (未見語者)**: girl9, girl10, boy7, boy8

### 超參數
| 參數 | 值 | 說明 |
|------|-----|------|
| `d_model` | 512 | Transformer 維度 |
| `num_layers` | 4 | Transformer 層數 |
| `nhead` | 8 | Attention heads |
| `batch_size` | 8 | Batch size |
| `learning_rate` | 1e-4 | 初始學習率 |
| `num_epochs` | 600 | 訓練輪數 |
| `lambda_speaker` | **[0.1, 0.5, 1.0]** | Speaker loss 權重（對比實驗） |

### 對比實驗組
1. **Baseline** (λ=0): 只有 CE Loss（done/train.py）
2. **Exp2-λ0.1**: CE + 0.1 × Speaker Loss
3. **Exp2-λ0.5**: CE + 0.5 × Speaker Loss
4. **Exp2-λ1.0**: CE + 1.0 × Speaker Loss

## 🚀 使用方法

### 1. 訓練模型

```bash
# Lambda = 0.5 (推薦)
python done/exp2/train_with_speaker.py \
    --input_dirs data/raw/box data/raw/papercup data/raw/plastic \
    --target_dir data/clean/box2 \
    --output_dir ./results/exp2/lambda0.5 \
    --lambda_speaker 0.5 \
    --speaker_model_type ecapa \
    --num_epochs 600 \
    --batch_size 8 \
    --max_sentences_per_speaker 288

# Lambda = 0.1 (輕量約束)
python done/exp2/train_with_speaker.py \
    --input_dirs ... \
    --output_dir ./results/exp2/lambda0.1 \
    --lambda_speaker 0.1 \
    ...

# Lambda = 1.0 (強約束)
python done/exp2/train_with_speaker.py \
    --input_dirs ... \
    --output_dir ./results/exp2/lambda1.0 \
    --lambda_speaker 1.0 \
    ...
```

### 2. 進階選項

```bash
# 從特定 epoch 才開始加入 speaker loss
python done/exp2/train_with_speaker.py \
    --speaker_loss_start_epoch 50 \
    --lambda_speaker 0.5 \
    ...

# 每 N 步才計算一次 speaker loss (加速訓練)
python done/exp2/train_with_speaker.py \
    --compute_speaker_every_n_steps 5 \
    --lambda_speaker 0.5 \
    ...

# 使用 Resemblyzer 而非 ECAPA
python done/exp2/train_with_speaker.py \
    --speaker_model_type resemblyzer \
    --lambda_speaker 0.5 \
    ...
```

## 📈 評估指標

### 1. Token Accuracy
- 訓練集 Token Accuracy
- **驗證集 Token Accuracy (關鍵指標)**

### 2. Speaker Similarity
- 預測音頻 vs 輸入音頻的 speaker embedding 相似度
- 使用 Cosine Similarity 評估

### 3. 主觀聽感
- 檢查 `audio_samples/epoch_XXX/` 中的音頻樣本
- 評估降噪效果和說話人身份保持

### 4. 泛化性對比
```python
# 關鍵問題：對比不同 λ 值
- Baseline (λ=0) 在驗證集的表現
- Exp2-λ0.1 在驗證集的表現
- Exp2-λ0.5 在驗證集的表現
- Exp2-λ1.0 在驗證集的表現

# 期望：λ > 0 的模型在未見語者上表現更好
```

## 📁 文件結構

```
done/exp2/
├── README.md                      # 實驗說明（本文件）
├── loss_with_speaker.py          # CEWithSpeakerLoss 損失函數
├── train_with_speaker.py         # 訓練腳本
├── run_experiments.sh            # 批次實驗腳本（待創建）
└── results/                       # 實驗結果（訓練時自動生成）
    ├── lambda0.1/
    │   ├── config.json
    │   ├── training.log
    │   ├── best_model.pth
    │   ├── loss_curves_epoch_XXX.png
    │   └── audio_samples/
    ├── lambda0.5/
    └── lambda1.0/
```

## 🎓 理論分析

### 為什麼 Speaker Loss 可能提升泛化性？

#### 1. 解耦內容與身份
```python
# 沒有 Speaker Loss：模型可能混淆
model_output = f(noisy_tokens, implicit_speaker_info)

# 有 Speaker Loss：強制解耦
model_output = f(noisy_tokens)  # 只關注噪音去除
subject to: speaker(output) ≈ speaker(input)  # 保持身份
```

#### 2. 正則化效果
- Speaker Loss 作為一種正則化
- 防止模型過度擬合特定語者的 token 分布

#### 3. 預訓練知識遷移
- ECAPA-TDNN 是在大量語者數據上預訓練的
- 其 speaker embedding 空間具有很強的泛化能力
- 通過 L2 Loss，將這種泛化能力傳遞給降噪模型

## ⚠️ 潛在問題與解決方案

### 問題 1: 訓練速度變慢
**原因**: 每個 batch 都需要解碼 tokens → audio

**解決方案**:
```bash
# 每 N 步才計算一次 speaker loss
--compute_speaker_every_n_steps 5
```

### 問題 2: 梯度流問題
**原因**: Token 解碼過程不可微分

**解決方案**:
- 當前實現：使用 hard argmax + stop gradient
- 未來可嘗試：Gumbel-Softmax (soft tokens)

### 問題 3: λ 值難以調整
**原因**: CE Loss 和 Speaker Loss 的尺度不同

**解決方案**:
- 進行網格搜索：λ ∈ {0.1, 0.5, 1.0, 2.0}
- 觀察驗證集表現選擇最佳 λ

## 📊 預期結果

### 成功指標
- ✅ 驗證集 Token Accuracy **提升**
- ✅ 驗證集 Speaker Similarity **維持高值**
- ✅ 主觀聽感：降噪效果好且說話人身份正確

### 失敗指標
- ❌ 驗證集 Token Accuracy **下降** → Speaker loss 干擾了主任務
- ❌ 訓練集 Accuracy 高但驗證集低 → 仍然過擬合
- ❌ Speaker Similarity 下降 → 模型改變了說話人身份

## 🔄 後續實驗方向

### 如果成功
1. 嘗試更多材質（石頭、樹木等）
2. 探索不同的 speaker encoder（WavLM, Wav2Vec2）
3. 研究 λ 的動態調整策略

### 如果失敗
1. 嘗試在 feature space 計算 speaker loss（避免完整解碼）
2. 使用 contrastive learning 而非 L2 loss
3. 探索其他正則化方法（dropout, weight decay）

## 📚 相關文獻

- **Speaker Embedding**: ECAPA-TDNN (Desplanques et al., 2020)
- **Token-based Speech Processing**: Discrete Diffusion (Austin et al., 2021)
- **Disentangled Representation**: β-VAE (Higgins et al., 2017)

## 📝 實驗記錄

| Experiment | λ | Train Acc | Val Acc | Val Speaker Sim | 備註 |
|------------|---|-----------|---------|------------------|------|
| Baseline   | 0 | TBD | TBD | - | 無 speaker loss |
| Exp2-λ0.1  | 0.1 | TBD | TBD | TBD | 輕量約束 |
| Exp2-λ0.5  | 0.5 | TBD | TBD | TBD | 中等約束 |
| Exp2-λ1.0  | 1.0 | TBD | TBD | TBD | 強約束 |

*TBD: 待實驗完成後填寫*
