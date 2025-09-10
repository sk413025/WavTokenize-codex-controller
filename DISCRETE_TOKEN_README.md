# WavTokenizer-Transformer 離散Token降噪系統文檔

## 系統概述

WavTokenizer-Transformer 系統實現了完整的端到端音頻降噪流程，基於預訓練的 WavTokenizer 進行音頻與 Token 的轉換，並使用 Transformer 在離散 Token 空間進行降噪學習。

### 核心架構

```
音頻輸入 → WavTokenizer Encoder (凍結) → 噪聲Tokens → Transformer降噪器 (可訓練) → 乾淨Tokens → WavTokenizer Decoder (凍結) → 音頻輸出
```

## 主要檔案

### 核心系統
- `wavtokenizer_transformer_denoising.py` - 主要訓練腳本
- `wavtokenizer_inference.py` - 推理腳本
- `test_wavtokenizer_transformer.py` - 系統測試腳本
- `token_loss_system.py` - ttt2.py 損失函數移植

### 訓練腳本
- `run_wavtokenizer_crossentropy.sh` - CrossEntropy 損失訓練
- `run_wavtokenizer_tokenloss.sh` - Token Loss 損失訓練

### 支援系統
- `ttdata.py` - 音頻數據集處理
- `ttt2.py` - 原始連續特徵降噪參考系統

## 核心特性

### 1. 端到端音頻處理
- **輸入**: 原始音頻波形
- **輸出**: 降噪後音頻波形
- **內部**: Token 空間處理（對用戶透明）

### 2. 預訓練模型利用
- **WavTokenizer Encoder**: 凍結，將音頻轉換為離散 tokens
- **WavTokenizer Decoder**: 凍結，將 tokens 重建為音頻
- **Transformer**: 可訓練，專注於 token 序列降噪

### 3. 雙重損失函數支持
- **CrossEntropy Loss**: 標準序列建模損失
- **Token Loss**: ttt2.py 高級損失邏輯移植到離散空間

## 系統優勢

### 參數效率
- 總參數: 89.3M
- 凍結參數: 80.6M (WavTokenizer)
- 可訓練參數: 8.7M (Transformer)
- 訓練效率: 只需訓練 10% 的參數

### 音質保證
- 使用經過大規模訓練的 WavTokenizer
- 避免重新訓練編碼解碼器
- 保持高質量音頻重建能力

### 技術創新
- 首次將 ttt2.py 的連續空間損失移植到離散空間
- 建立標準化的音頻 AI 處理范式
- 支持端到端優化和推理

## 使用方法

### 基本訓練

```bash
# CrossEntropy 損失訓練
python wavtokenizer_transformer_denoising.py \
    --config config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    --model_path models/wavtokenizer_large_speech_320_24k.ckpt \
    --output_dir results/crossentropy_training \
    --num_epochs 100 \
    --batch_size 4

# Token Loss 訓練
python wavtokenizer_transformer_denoising.py \
    --use_token_loss \
    --l2_weight 0.3 \
    --consistency_weight 0.4 \
    --manifold_weight 0.1 \
    --output_dir results/tokenloss_training \
    --num_epochs 100
```

### 推理使用

```bash
# 單文件推理
python wavtokenizer_inference.py \
    --checkpoint results/crossentropy_training/best_model.pth \
    --input noisy_audio.wav \
    --output denoised_audio.wav

# 批量推理
python wavtokenizer_inference.py \
    --checkpoint results/tokenloss_training/best_model.pth \
    --input input_dir/ \
    --output output_dir/ \
    --batch
```

### 系統測試

```bash
# 完整系統測試
python test_wavtokenizer_transformer.py
```

## 配置參數

### 模型參數
- `--d_model`: Transformer 模型維度 (默認: 512)
- `--nhead`: 注意力頭數 (默認: 8)
- `--num_encoder_layers`: 編碼器層數 (默認: 6)
- `--num_decoder_layers`: 解碼器層數 (默認: 6)

### Token Loss 權重
- `--l2_weight`: L2 距離損失權重 (默認: 0.3)
- `--consistency_weight`: 內容一致性權重 (默認: 0.4)
- `--manifold_weight`: Manifold 正則化權重 (默認: 0.1)
- `--normalization_weight`: 正規化權重 (默認: 0.1)
- `--coherence_weight`: 連貫性權重 (默認: 0.1)

## 實驗結果

### 系統驗證
- ✅ Token 轉換: 2秒音頻 → 150 tokens
- ✅ 重建質量: MSE 0.568
- ✅ 訓練穩定性: 支持 teacher forcing
- ✅ 推理性能: 端到端音頻處理

### 損失函數比較
- **CrossEntropy**: 標準序列建模，收斂穩定
- **Token Loss**: 語義感知損失，音質提升

## 技術文檔

### 相關文檔
- `MODEL_ARCHITECTURE_EXPLAINED.md` - 詳細架構說明
- `TOKEN_LOSS_EXPERIMENT_RESULTS.md` - 實驗結果分析
- `TOKEN_LOSS_SYSTEM_README.md` - 損失函數系統文檔

### API 參考
- `WavTokenizerTransformerDenoiser` - 主要模型類
- `AudioTokenDataset` - 音頻數據集類
- `train_epoch` - CrossEntropy 訓練函數
- `train_epoch_with_token_loss` - Token Loss 訓練函數

## 故障排除

### 常見問題
1. **CUDA 內存不足**: 減少 `batch_size`
2. **WavTokenizer 載入失敗**: 檢查 config 和 model_path
3. **Token Loss 計算錯誤**: 檢查 embedding_layer 配置

### 調試工具
- 使用 `test_wavtokenizer_transformer.py` 驗證系統
- 檢查生成的音頻文件和比較圖
- 查看訓練日誌中的損失分解信息
