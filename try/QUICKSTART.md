# 快速開始指南

## 🎯 完整訓練流程圖

```
┌─────────────────────────────────────────────────────────────────┐
│                   Frozen Codebook 訓練流程                       │
└─────────────────────────────────────────────────────────────────┘

Step 1: 數據載入
───────────────
    📁 data/raw/box/*.wav (noisy)
    📁 data/clean/box2/*.wav (clean)
         │
         │ AudioDataset
         ▼
    音訊對 (noisy_audio, clean_audio)
         │
         │ [WavTokenizer Encoder] ❄️ FROZEN
         ▼
    Token 對 (noisy_tokens, clean_tokens)
    例: ([2347, 891, ...], [2351, 893, ...])

Step 2: 訓練循環
───────────────
    每個 Batch:
    
    noisy_tokens (B, T)
         │
         │ [Frozen Codebook Lookup] ❄️
         ▼
    embeddings (B, T, 512)
         │
         │ [Transformer Encoder] 🔥 訓練中...
         ▼
    hidden (B, T, 512)
         │
         │ [Output Projection] 🔥 訓練中...
         ▼
    logits (B, T, 4096)
         │
         │ CrossEntropyLoss(logits, clean_tokens)
         ▼
    loss
         │
         │ Backward + Optimizer Step
         ▼
    更新參數 (只有 Transformer + Projection)
    
    ✅ Codebook 保持凍結！

Step 3: 驗證
───────────────
    每個 Epoch 結束:
    
    驗證集 → Token Accuracy
           → Validation Loss
           → 保存最佳模型
           → 繪製訓練曲線

Step 4: 推論
───────────────
    noisy.wav
         │
         │ [Encoder] → noisy_tokens
         │ [Transformer] → clean_tokens
         │ [Decoder] → denoised.wav
         ▼
    denoised.wav
```

---

## 🚀 運行 Frozen Codebook 實驗

### 1. 進入 try 資料夾
```bash
cd /home/sbplab/ruizi/c_code/try
```

### 2. 執行訓練腳本
```bash
bash run_token_denoising_frozen_codebook.sh
```

### 3. 監控訓練進度
```bash
# 在另一個終端
tail -f ../logs/token_denoising_frozen_codebook_*.log
```

---

## 📊 查看結果

### 訓練歷史圖
```bash
# 每 10 epochs 自動更新
ls -lh ../results/token_denoising_frozen_codebook_*/training_history.png
```

### 模型檢查點
```bash
# 最佳模型
ls -lh ../results/token_denoising_frozen_codebook_*/best_model.pt

# 定期保存 (每 100 epochs)
ls -lh ../results/token_denoising_frozen_codebook_*/checkpoint_epoch_*.pt
```

---

## 🔧 自定義配置

### 修改模型大小
編輯 `run_token_denoising_frozen_codebook.sh`:
```bash
--d_model 512              # Transformer 維度 (必須 = 512)
--nhead 8                  # 注意力頭數 (可調整: 4, 8, 16)
--num_layers 6             # Encoder 層數 (可調整: 4, 6, 8, 12)
--dim_feedforward 2048     # 前饋網絡維度 (可調整: 1024, 2048, 4096)
```

### 修改訓練參數
```bash
--batch_size 8             # 批次大小 (依據 GPU 記憶體調整)
--learning_rate 1e-4       # 學習率
--num_epochs 1000          # 訓練輪數
```

### 修改數據分割
```bash
--val_speakers girl9 girl10 boy7 boy8
--train_speakers boy1 boy3 ... (所有非驗證集語者)
```

---

## 📈 預期訓練曲線

### Epoch 1-50: 初始學習
- Token Accuracy: 0% → 10-20%
- Loss: 8.5 → 4-5
- 學習 token 間的基本對應關係

### Epoch 50-200: 快速收斂
- Token Accuracy: 20% → 40-50%
- Loss: 4-5 → 2-3
- 學習時間依賴和上下文信息

### Epoch 200-500: 精細調整
- Token Accuracy: 50% → 60-70%
- Loss: 2-3 → 1-2
- 優化邊界情況和細節

### Epoch 500+: 穩定或過擬合
- 監控 validation loss
- 如果 val loss 上升 → early stopping

---

## 🔍 問題排查

### Q: Token Accuracy 一直是 0%
**可能原因**:
- Learning rate 太小
- Codebook 未正確凍結
- 數據問題

**解決方案**:
```python
# 檢查 codebook 是否凍結
print(model.codebook.requires_grad)  # 應該是 False

# 檢查學習率
print(optimizer.param_groups[0]['lr'])  # 應該是 1e-4

# 檢查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")
```

### Q: Loss 不下降
**可能原因**:
- 數據加載問題
- Token IDs 超出範圍 [0, 4095]
- 模型初始化問題

**解決方案**:
```python
# 檢查 token 範圍
print(f"Token range: [{tokens.min()}, {tokens.max()}]")

# 檢查數據形狀
print(f"Noisy tokens: {noisy_tokens.shape}")
print(f"Clean tokens: {clean_tokens.shape}")
```

### Q: CUDA Out of Memory
**解決方案**:
```bash
# 減少 batch size
--batch_size 4  # 從 8 改為 4

# 或減少模型大小
--num_layers 4  # 從 6 改為 4
--dim_feedforward 1024  # 從 2048 改為 1024
```

---

## 🎯 與現有模型比較

### 同時運行兩個實驗
```bash
# Terminal 1: Frozen Codebook 模型
cd /home/sbplab/ruizi/c_code/try
bash run_token_denoising_frozen_codebook.sh

# Terminal 2: 現有模型 (可訓練 embedding)
cd /home/sbplab/ruizi/c_code
bash run_transformer_large_tokenloss.sh
```

### 比較結果
```bash
# Token Accuracy
grep "Token Accuracy" try/logs/* | tail -20
grep "Token Accuracy" logs/transformer_large_tokenloss* | tail -20

# Loss
grep "Val Loss" try/logs/* | tail -20
grep "Val Loss" logs/transformer_large_tokenloss* | tail -20
```

---

## 📝 實驗記錄

所有實驗會自動記錄到 `../REPORT.md`，包含:
- 實驗編號
- 實驗時間
- 模型配置
- 預期效果
- 實際結果 (待更新)

---

## 🔗 相關文檔

- [`README_FROZEN_CODEBOOK.md`](./README_FROZEN_CODEBOOK.md): 詳細說明
- [`MODEL_COMPARISON_ANALYSIS.md`](./MODEL_COMPARISON_ANALYSIS.md): 模型對比
- [`TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md`](./TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md): 架構解釋
- [`token_denoising_transformer.py`](./token_denoising_transformer.py): 模型代碼
- [`train_token_denoising.py`](./train_token_denoising.py): 訓練代碼

---

**快速測試**:
```bash
# 1 分鐘測試 (只訓練 2 epochs)
python train_token_denoising.py \
    --num_epochs 2 \
    --batch_size 4 \
    --output_dir ../results/test_frozen_codebook
```

**完整訓練**:
```bash
bash run_token_denoising_frozen_codebook.sh
```

🎉 開始探索 Frozen Codebook 降噪！
