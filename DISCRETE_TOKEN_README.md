# 離散 Token 降噪實驗說明

## 概述

這個實驗實現了一個全新的音頻降噪方法：基於離散 Token 的 Transformer 降噪。與傳統的連續特徵方法不同，這個方法完全在離散的 token 空間進行降噪處理。

## 核心理念

### 1. 問題重新定義
- **傳統方法**: 音頻 → 連續特徵 → 處理 → 連續特徵 → 音頻
- **我們的方法**: 音頻 → 離散tokens → 處理 → 離散tokens → 音頻

### 2. 比喻理解
將降噪任務類比為「翻譯問題」：
- **輸入語言**: "帶噪聲學語言" (noisy token sequence)  
- **輸出語言**: "乾淨聲學語言" (clean token sequence)
- **翻譯系統**: Transformer Encoder-Decoder

### 3. 架構組成
```
🧠 Encoder (理解者): 
   - 完整閱讀帶噪 token 序列
   - 通過多層自注意力理解全局模式
   - 識別噪音模式和語音內容

🎯 Decoder (生成者):
   - 逐步生成乾淨的 token 序列  
   - 同時關注自己已生成的內容 (self-attention)
   - 參考 Encoder 的理解結果 (cross-attention)
```

## 文件說明

### 核心文件
- `discrete_token_denoising.py`: 主訓練腳本
- `discrete_inference.py`: 推理腳本  
- `run_discrete_token_experiment.sh`: 自動化實驗腳本

### 訓練腳本功能
1. **數據準備**: 自動從 AudioDataset 提取 token 序列
2. **模型架構**: Transformer Encoder-Decoder
3. **訓練策略**: Teacher forcing + 交叉熵損失
4. **評估指標**: 損失和 token 準確率
5. **可視化**: 自動生成訓練歷史圖

### 推理腳本功能  
1. **單文件處理**: 處理單個音頻文件
2. **批次處理**: 處理整個目錄的音頻
3. **比較功能**: 保存重建的原始音頻用於對比
4. **Token 保存**: 可選保存中間的 token 序列

## 使用方法

### 快速開始
```bash
# 運行完整實驗（訓練+推理）
./run_discrete_token_experiment.sh
```

### 手動訓練
```bash
python discrete_token_denoising.py \
    --config config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    --model_path models/wavtokenizer_large_speech_320_24k.ckpt \
    --output_dir results/my_experiment \
    --num_epochs 100 \
    --batch_size 8
```

### 手動推理
```bash
python discrete_inference.py \
    --model_path results/my_experiment/best_model.pth \
    --input_audio noisy_audio.wav \
    --output_dir denoised_results/
```

## 實驗參數調整

### 模型大小
```bash
# 小模型 (快速測試)
--d_model 256 --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3

# 中等模型 (平衡)  
--d_model 512 --nhead 8 --num_encoder_layers 4 --num_decoder_layers 4

# 大模型 (最佳效果)
--d_model 768 --nhead 12 --num_encoder_layers 6 --num_decoder_layers 6
```

### 訓練策略
```bash
# 快速測試
--max_samples 100 --num_epochs 10 --batch_size 2

# 正常訓練
--max_samples 1000 --num_epochs 50 --batch_size 4  

# 完整訓練
--max_samples 5000 --num_epochs 100 --batch_size 8
```

## 技術優勢

### 1. 簡化處理流程
- 避免複雜的連續特徵工程
- 直接在 WavTokenizer 的 codebook 空間工作
- 利用成熟的 NLP Transformer 架構

### 2. 更好的可解釋性
- Token 序列可以直接檢視和分析
- 容易理解模型的決策過程
- 方便調試和優化

### 3. 端到端訓練
- 單一模型完成整個降噪任務
- 避免多階段訓練的累積誤差
- 更直觀的損失函數 (交叉熵)

### 4. 靈活性
- 支援可變長度序列
- 容易擴展到其他音頻處理任務
- 可以整合其他 NLP 技術 (如 BERT、GPT)

## 預期結果

### 訓練過程
- 損失曲線應該穩定下降
- Token 準確率逐漸提升
- 驗證集表現與訓練集接近

### 推理結果  
- 生成的音頻應該更清晰
- 保留原始語音內容
- 減少背景噪音

## 故障排除

### 常見問題
1. **記憶體不足**: 減少 batch_size 或 max_length
2. **訓練不收斂**: 調整學習率或模型大小
3. **推理效果差**: 增加訓練數據或調整模型架構

### 調試建議
1. 檢查 token 序列的統計分布
2. 監控注意力權重的變化
3. 比較不同 checkpoint 的效果

## 擴展方向

### 短期改進
1. 實現完整的 beam search
2. 添加更多評估指標 (PESQ, STOI 等)
3. 支援多語者和多語言

### 長期研究  
1. 整合預訓練的語言模型
2. 探索更複雜的 token 表示
3. 研究條件生成 (如語者控制)

---

這個實驗代表了音頻降噪領域的一個創新嘗試，將 NLP 的成功經驗應用到音頻處理中。希望這個框架能為後續的研究提供有價值的基礎。
