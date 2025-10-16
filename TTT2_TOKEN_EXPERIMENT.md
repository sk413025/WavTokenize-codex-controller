# TTT2 Token Enhancement 實驗文檔

## 實驗日期
2025-01-15

## 實驗背景

現有的離散模型（`wavtokenizer_transformer_denoising.py`）在句子重建時出現問題，無法正確還原句子。經過分析，發現問題出在 decoder 的 token → feature → audio 重建流程。

## 實驗目的

設計一個新的 Token-based Feature Enhancement 系統，具有以下特性：

1. **凍結的基準模型**: WavTokenizer 的 encoder/decoder 完全凍結，作為穩定的基準
2. **多材質多語者輸入**: Input 為不同材質（噪音）下不同語者的說話音檔
3. **乾淨音檔目標**: Target 為不同語者的乾淨音檔
4. **Token 空間增強**: 在 WavTokenizer 的 token 空間進行特徵增強
5. **通用性**: 模型能夠泛化到不同材質、不同語者

## 架構設計

### 1. 整體流程

```
Noisy Audio (多材質、多語者)
    ↓
WavTokenizer Encoder (凍結)
    ↓
Noisy Tokens [batch_size, seq_len]
    ↓
Token Embedding Layer
    ↓
Noisy Features [batch_size, seq_len, embed_dim]
    ↓
+ Positional Encoding
    ↓
Token Feature Enhancer (Transformer, 可訓練)
    ↓
Enhanced Features [batch_size, seq_len, embed_dim]
    ↓
Feature Projection (投影到 codebook 空間)
    ↓
Enhanced Tokens [batch_size, seq_len]
    ↓
WavTokenizer Decoder (凍結)
    ↓
Enhanced Audio (乾淨、對應語者)
```

### 2. 關鍵模組

#### 2.1 WavTokenizer (凍結)
- **編碼器**: 將音頻轉換為 discrete tokens
  - 輸入: [batch, 1, time] 音頻波形
  - 輸出: [batch, seq_len] discrete token indices (0-4095)
  - 同時輸出 continuous features 用於驗證

- **解碼器**: 將 tokens 重建為音頻
  - 輸入: [n_q, batch, seq_len] discrete codes (n_q=1)
  - 流程: tokens → codes_to_features() → decode() → audio
  - 輸出: [batch, 1, time] 音頻波形

#### 2.2 Token Feature Enhancer (可訓練核心)
```python
TokenFeatureEnhancer(
    embed_dim=512,
    num_layers=4,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1
)
```

**結構**:
- Transformer Encoder (4 層)
  - Multi-Head Self-Attention (8 heads)
  - Feed-Forward Network (2048 dim)
  - Pre-LN (Layer Normalization First)
  - Residual Connections

- Post-Processing Layer
  - Linear + LayerNorm + GELU + Dropout + Linear
  - 進一步精煉增強的特徵

- Residual Weight
  - 可學習的殘差權重，平衡原始與增強特徵

#### 2.3 Token Embedding & Projection
- **Token Embedding**: 將 discrete tokens 投影到 embedding 空間
  - `nn.Embedding(4096, 512)`
  - 建立 token → continuous feature 的映射

- **Positional Encoding**: 添加位置資訊
  - Sinusoidal encoding
  - 支持最長 1000 個 tokens

- **Feature Projection**: 將增強特徵映射回 token 空間
  - Linear(512, 512) → LayerNorm → GELU → Linear(512, 4096)
  - 輸出 logits，通過 argmax 得到 discrete tokens

### 3. 修復的 Decoder 問題

**原問題**: `wavtokenizer_transformer_denoising.py` 的 decoder 無法正確重建句子

**根本原因**:
1. Token 維度處理錯誤: 沒有正確處理 [batch, seq_len] → [n_q, batch, seq_len]
2. `codes_to_features` 輸入格式不符合預期
3. Decoder 期望的 bandwidth_id 設定不當

**修復方案** (在 `TTT2TokenModel.decode_tokens_to_audio()`):
```python
def decode_tokens_to_audio(self, tokens):
    with torch.no_grad():
        # 1. 正確擴展維度: [batch, seq_len] → [1, batch, seq_len]
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        
        # 2. 確保 tokens 在有效範圍 [0, 4095]
        tokens = torch.clamp(tokens, 0, self.codebook_size - 1)
        
        # 3. 正確使用 codes_to_features
        features = self.wavtokenizer.codes_to_features(tokens)
        
        # 4. 使用正確的 bandwidth_id
        bandwidth_id = torch.tensor([0], device=tokens.device)
        audio = self.wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
        
        # 5. 確保音頻維度正確 [batch, 1, time]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        elif audio.dim() == 3 and audio.size(1) != 1:
            audio = audio.mean(dim=1, keepdim=True)
        
        return audio
```

## 損失函數設計

### 多目標損失 (Multi-Objective Loss)

```python
Total Loss = 0.4 * Token_CE + 0.3 * Feature_L2 + 0.2 * Audio_L1 + 0.1 * Token_Smooth
```

#### 1. Token Classification Loss (權重: 0.4)
```python
token_ce = CrossEntropy(token_logits, target_tokens)
```
- **目的**: 預測正確的 target tokens
- **重要性**: 最高，確保 token 序列的準確性

#### 2. Feature L2 Loss (權重: 0.3)
```python
feature_l2 = MSE(enhanced_features, target_features)
```
- **目的**: Enhanced features 接近 target features
- **重要性**: 次高，確保特徵空間的相似性

#### 3. Audio Reconstruction L1 Loss (權重: 0.2)
```python
audio_l1 = L1(enhanced_audio, target_audio_reconstructed)
```
- **目的**: 重建的音頻接近目標音頻
- **重要性**: 中等，直接優化最終輸出質量

#### 4. Token Smoothness Loss (權重: 0.1)
```python
token_smooth = Mean(|enhanced_tokens[t+1] - enhanced_tokens[t]|)
```
- **目的**: 防止 token 序列過於突變
- **重要性**: 最低，輔助正則化

## 訓練配置

### 模型參數
```python
embed_dim = 512          # Token embedding 維度
enhancer_layers = 4      # Transformer 層數
enhancer_heads = 8       # 注意力頭數
enhancer_ff_dim = 2048   # Feed-forward 維度
dropout = 0.1            # Dropout 機率
```

### 訓練參數
```python
batch_size = 8
num_epochs = 100
learning_rate = 1e-4
weight_decay = 1e-5
optimizer = AdamW (beta1=0.9, beta2=0.999)
scheduler = CosineAnnealingLR (T_max=100, eta_min=1e-6)
gradient_clipping = 1.0
```

### 數據集
- **訓練語者** (10人): boy1, boy3, boy4, boy5, boy6, girl2, girl3, girl4, girl6, girl7
- **驗證語者** (2人): girl9, boy7
- **輸入材質**: Box (噪音材質)
- **目標**: Clean (乾淨音檔)
- **句子數**: 每個語者所有可用句子

### 保存策略
- 每 10 個 epoch 保存一次檢查點
- 保存驗證損失最低的最佳模型
- 保存最終模型
- 每 50 個 batch 保存音頻樣本和頻譜圖
- 驗證時保存前 3 個 batch 的樣本

## 實驗執行

### 1. 直接執行
```bash
bash run_ttt2_token.sh
```

### 2. 背景執行
```bash
nohup bash run_ttt2_token.sh > logs/ttt2_token_background_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 3. 指定 GPU
```bash
CUDA_VISIBLE_DEVICES=1 bash run_ttt2_token.sh
```

### 4. 監控訓練
```bash
# 查看即時日誌
tail -f logs/ttt2_token_*.log

# 查看 GPU 使用
watch -n 1 nvidia-smi

# 查看訓練進度
ls -lht results/ttt2_token_enhancement/exp_*/checkpoints/
```

## 預期結果

### 1. 性能指標
- **Token CE Loss**: 應該逐漸下降，最終 < 2.0
- **Feature L2 Loss**: 應該逐漸下降，最終 < 0.5
- **Audio L1 Loss**: 應該逐漸下降，最終 < 0.1
- **Token Smooth Loss**: 應該保持穩定，約 10-50

### 2. 輸出檔案
```
results/ttt2_token_enhancement/exp_YYYYMMDD_HHMMSS/
├── config.json                    # 實驗配置
├── checkpoints/
│   ├── best_model.pth            # 最佳模型
│   ├── checkpoint_epoch_10.pth   # 每 10 epoch 的檢查點
│   ├── checkpoint_epoch_20.pth
│   ├── ...
│   └── final_model.pth           # 最終模型
├── audio_samples/
│   └── epoch_X/
│       ├── batch_Y_noisy.wav     # 輸入噪音
│       ├── batch_Y_target.wav    # 目標乾淨音檔
│       ├── batch_Y_enhanced.wav  # 增強後的音檔
│       ├── batch_Y_noisy_spec.png
│       ├── batch_Y_target_spec.png
│       └── batch_Y_enhanced_spec.png
├── validation_samples/
│   └── epoch_X/
│       └── (類似結構)
└── training_history_epoch_X.png  # 訓練曲線圖
```

### 3. 音頻質量
- **清晰度**: Enhanced audio 應該明顯比 noisy audio 清晰
- **語者一致性**: 保持語者的聲音特徵
- **材質消除**: 成功移除 box 材質的噪音
- **自然度**: 聽起來自然，無明顯失真

## 驗證方法

### 1. 客觀指標 (自動計算)
```python
# Token Accuracy
token_acc = (enhanced_tokens == target_tokens).float().mean()

# Feature Similarity
feature_sim = F.cosine_similarity(enhanced_features, target_features)

# Audio Metrics (需要額外實現)
snr = compute_snr(enhanced_audio, target_audio)
pesq = compute_pesq(enhanced_audio, target_audio)
stoi = compute_stoi(enhanced_audio, target_audio)
```

### 2. 主觀聽測
- 聽比較: noisy → enhanced → target
- 評估: 清晰度、自然度、語者相似度
- 多個語者、多個句子的測試

### 3. 可視化分析
- 頻譜圖比較
- Token 序列比較
- 特徵空間 t-SNE 可視化

## 後續實驗方向

### 1. 模型改進
- [ ] 增加 Enhancer 層數 (4 → 6 or 8)
- [ ] 嘗試不同的 attention 機制 (Sparse, Linear)
- [ ] 添加 Cross-Attention 使用 target 作為 key/value
- [ ] 多尺度特徵融合

### 2. 損失函數優化
- [ ] 動態調整損失權重
- [ ] 添加 Perceptual Loss
- [ ] 添加 Adversarial Loss
- [ ] 引入 Contrastive Learning

### 3. 數據增強
- [ ] 測試其他材質 (glass, paper)
- [ ] 混合多種材質
- [ ] 不同 SNR 的訓練數據
- [ ] 語者增強 (speaker augmentation)

### 4. 通用性測試
- [ ] 跨語者測試 (unseen speakers)
- [ ] 跨材質測試 (unseen materials)
- [ ] 不同長度的音檔
- [ ] 不同情感的語音

## 關鍵差異：vs. 舊模型

| 項目 | 舊模型 (wavtokenizer_transformer_denoising.py) | 新模型 (ttt2_token.py) |
|------|-----------------------------------------------|------------------------|
| **Token 處理** | 直接在 token sequence 上操作 | Token → Embedding → Enhancement → Projection |
| **Decoder 修復** | ❌ Token 維度處理錯誤 | ✅ 正確的維度轉換流程 |
| **特徵空間** | 直接操作 discrete tokens | 在 continuous embedding 空間增強 |
| **架構** | Encoder-Decoder Transformer | Feature Enhancement Transformer |
| **目標** | 序列生成 (teacher forcing) | 特徵增強 (feature refinement) |
| **損失函數** | 單一 CrossEntropy | 多目標損失 (CE + L2 + L1 + Smooth) |
| **通用性** | 針對特定任務 | 設計為通用降噪模型 |

## 實驗記錄規範

根據 `.github/copilot-instructions.md`，每次實驗後需要：

1. **自動命名**: 檔案名包含實驗編號、日期、函式名稱
2. **更新 REPORT.md**: 記錄實驗結果
3. **Git Commit**: 包含以下內容
   - 實驗背景與動機
   - 實驗目的
   - 預期結果
   - 實際執行結果
   - 結果解讀
   - 實驗反思
   - 重現步驟

## 檔案結構

```
c_code/
├── ttt2_token.py              # 新實驗主程式 (本文檔描述)
├── run_ttt2_token.sh          # 執行腳本
├── TTT2_TOKEN_EXPERIMENT.md   # 實驗文檔 (本文件)
├── ttdata.py                  # 數據載入 (共用)
├── config/                    # WavTokenizer 配置
├── results/
│   └── ttt2_token_enhancement/  # 實驗結果目錄
└── logs/                      # 訓練日誌
```

## 參考資料

- WavTokenizer Paper: https://arxiv.org/abs/2308.05734
- Transformer Architecture: "Attention is All You Need"
- 舊模型分析: `wavtokenizer_transformer_denoising.py`
- Token Loss System: `token_loss_system.py`

---

**實驗負責人**: GitHub Copilot  
**創建日期**: 2025-01-15  
**最後更新**: 2025-01-15
