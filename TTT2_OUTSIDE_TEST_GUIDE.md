# TTT2 Outside音檔測試指南

## 📋 概述

本工具用於測試訓練後的TTT2模型在outside音檔上的表現，評估模型的泛化能力。

## 🚀 快速開始

### 1. 環境準備
```bash
conda activate test
```

### 2. 準備outside音檔
```bash
# TTT2測試使用./1n目錄作為outside音檔來源
# 該目錄應該已包含測試音檔

# 檢查1n目錄中的音檔
ls -la ./1n/*.wav

# 如需添加其他音檔，可複製到1n目錄
cp /path/to/your/audio/files/*.wav ./1n/
```

### 3. 運行測試
```bash
# 使用預設參數運行（自動尋找最佳checkpoint）
./run_ttt2_outside_test.sh

# 或手動指定best_model.pth
python test_ttt2_outside.py \
    --checkpoint results/tsne_outputs/b-output4/best_model.pth \
    --outside_dir ./1n \
    --output_dir ttt2_outside_test_results \
    --max_files 10 \
    --audio_length 32000

# 或使用Lightning checkpoint
python test_ttt2_outside.py \
    --checkpoint lightning_logs/version_0/checkpoints/epoch=4-step=20.ckpt \
    --outside_dir ./1n \
    --output_dir ttt2_outside_test_results \
    --max_files 10 \
    --audio_length 32000
```

## 📊 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--checkpoint` | 自動偵測 | TTT2模型checkpoint路徑（優先best_model.pth） |
| `--outside_dir` | `./1n` | outside音檔目錄 |
| `--output_dir` | `ttt2_outside_test_results` | 測試結果輸出目錄 |
| `--max_files` | `10` | 最大測試檔案數量 |
| `--audio_length` | `32000` | 音檔長度(樣本數，對應2秒@16kHz) |

## 🎯 Checkpoint選擇優先級

系統會按以下優先級自動選擇checkpoint：

1. **best_model.pth** (TTT2訓練保存的最佳模型)
   - `results/tsne_outputs/b-output4/best_model.pth`
   - `results/tsne_outputs/output4/best_model.pth`
   - `results/tsne_outputs/output3/best_model.pth`

2. **Lightning checkpoints** (訓練過程檢查點)
   - `lightning_logs/*/checkpoints/*.ckpt`

**推薦使用best_model.pth**，因為它是TTT2訓練過程中保存的最佳性能模型。

## 📁 輸出結構

測試完成後會在輸出目錄中生成：

```
ttt2_outside_test_results/
└── test_YYYYMMDD_HHMMSS/
    ├── TEST_REPORT.md          # 詳細測試報告
    ├── test_report.json        # JSON格式測試數據
    ├── test_001_filename1/     # 第一個測試檔案結果
    │   ├── comparison_original.wav
    │   ├── comparison_enhanced.wav
    │   └── comparison_plot.png
    ├── test_002_filename2/     # 第二個測試檔案結果
    │   └── ...
    └── ...
```

## 📈 評估指標

### SNR (信噪比)
- **計算方式**: `10 * log10(信號功率 / 噪聲功率)`
- **意義**: 衡量增強後音檔相對於原始音檔的信號品質
- **範圍**: 越高越好，負值表示增強效果不佳

### 相關係數
- **計算方式**: 皮爾森相關係數
- **意義**: 原始和增強音檔的線性相關程度
- **範圍**: [-1, 1]，越接近1越好

### RMS差異
- **計算方式**: `sqrt(mean((原始 - 增強)²))`
- **意義**: 原始和增強音檔的均方根差異
- **範圍**: 越小越好

### 頻譜距離
- **計算方式**: 頻域歐幾里得距離
- **意義**: 頻譜特徵的差異程度
- **範圍**: 越小越好

## 🔧 故障排除

### 1. 模型載入失敗
```
❌ 模型載入失敗: xxx
```
**解決方案:**
- 檢查checkpoint路徑是否正確
- 確保config和model檔案存在：
  - `config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml`
  - `models/wavtokenizer_large_speech_320_24k.ckpt`

### 2. 音檔載入失敗
```
載入音檔失敗 xxx: xxx
```
**解決方案:**
- 確保音檔格式支援 (.wav, .mp3, .flac, .m4a)
- 檢查音檔是否損壞
- 確保有足夠的記憶體

### 3. CUDA記憶體不足
```
CUDA out of memory
```
**解決方案:**
- 減少 `--audio_length` 參數
- 減少 `--max_files` 參數
- 使用CPU模式（自動偵測）

### 4. 依賴套件缺失
```
ModuleNotFoundError: No module named 'xxx'
```
**解決方案:**
```bash
pip install librosa soundfile matplotlib
```

## 💡 使用建議

### 1. 測試檔案選擇
- 選擇多樣化的音檔類型（語音、音樂、噪聲等）
- 包含不同長度和品質的音檔
- 考慮包含訓練數據中未見過的條件

### 2. 參數調整
- 對於記憶體有限的系統，減少 `audio_length`
- 對於快速測試，減少 `max_files`
- 對於詳細分析，增加測試檔案數量

### 3. 結果分析
- 關注SNR和相關係數的分佈
- 比較不同類型音檔的表現
- 檢查異常值（極低SNR或相關係數）

## 📚 進階使用

### 1. 批量測試多個checkpoint
```bash
for ckpt in lightning_logs/*/checkpoints/*.ckpt; do
    echo "測試: $ckpt"
    python test_ttt2_outside.py --checkpoint "$ckpt" --output_dir "results_$(basename $ckpt .ckpt)"
done
```

### 2. 自定義評估指標
修改 `compute_audio_metrics()` 函數添加更多指標：
- PESQ (語音品質評估)
- STOI (短時間客觀清晰度)
- 頻譜失真度量

### 3. 可視化比較
生成的 `comparison_plot.png` 包含：
- 時域波形對比
- 頻譜對比
- 差異信號分析

---

## 🆘 支援

如有問題請檢查：
1. 環境配置是否正確
2. 檔案路徑是否存在
3. 依賴套件是否安裝完整

或參考錯誤信息進行故障排除。
