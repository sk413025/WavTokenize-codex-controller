# Token Denoising Transformer 修復總結與啟動指南

**日期**: 2025-10-24  
**實驗編號**: fixed_anti_overfitting

---

## 📋 修復總結

### 問題診斷
前次訓練（epoch 420）發現三大問題：
1. **音頻儲存失敗** - 每 100 epochs 的音頻樣本保存失敗
2. **缺少視覺化** - 沒有 loss 曲線，無法監控訓練
3. **嚴重過擬合** - Train acc 79.55%, Val acc 22.68%, Test acc 0.42%

### 已完成修復

#### 1. 音頻儲存維度修復 ✅
**檔案**: `train_token_denoising_hybrid.py` (行 93-99)
```python
# 檢查並修正維度：codes_to_features 可能返回 4D [1, T, 1, D]
# 需要轉換為 3D [1, T, D] 供 decode 使用
if noisy_features.dim() == 4:
    noisy_features = noisy_features.squeeze(2)
if pred_features.dim() == 4:
    pred_features = pred_features.squeeze(2)
if clean_features.dim() == 4:
    clean_features = clean_features.squeeze(2)
```

#### 2. 新增 Loss 曲線繪製 ✅
**檔案**: `train_token_denoising_hybrid.py` (行 163-229)
- 新增 `plot_loss_curves()` 函數
- 每 50 epochs 自動生成 4 個子圖：
  - Total Loss (Train vs Val)
  - Token Accuracy (Train vs Val)
  - CrossEntropy Loss (Train vs Val)
  - Content & Embedding Losses

#### 3. 防過擬合措施 ✅
**檔案**: `train_token_denoising_hybrid.py`

| 參數 | 舊值 | 新值 | 原因 |
|------|------|------|------|
| `dropout` | 0.1 | **0.2** | 增強正則化 |
| `weight_decay` | 0.01 | **0.05** | 減少過擬合 |
| `num_layers` | 6 | **4** | 降低模型容量 |
| `scheduler` | CosineAnnealing | **ReduceLROnPlateau** | 根據驗證損失動態調整 |

---

## 🚀 快速啟動指南

### 選項 1: 完整訓練（600 epochs）

```bash
cd /home/sbplab/ruizi/c_code/try
bash run_fixed_training.sh
```

**特點**:
- 完整 600 epochs 訓練
- 自動選擇空閒 GPU
- 自動生成實驗報告
- 自動提交 git commit
- 日誌輸出到 `../logs/token_denoising_fixed_*.log`

**監控指令**:
```bash
# 即時查看日誌
tail -f ../logs/token_denoising_fixed_*.log

# 查看訓練進程
ps aux | grep train_token_denoising_hybrid.py
```

---

### 選項 2: 快速測試（2 epochs）

```bash
cd /home/sbplab/ruizi/c_code/try
bash test_fixes.sh
```

**用途**: 驗證所有修復是否正常工作
- 只訓練 2 epochs
- 測試維度處理
- 驗證參數配置
- 檢查 checkpoint 儲存

**成功標準**:
- [ ] 訓練正常完成（無錯誤）
- [ ] checkpoint 已儲存
- [ ] 配置檔案已生成
- [ ] 新參數配置生效（dropout=0.2, num_layers=4）

---

### 選項 3: 背景執行

```bash
cd /home/sbplab/ruizi/c_code/try
chmod +x start_fixed_training_background.sh
bash start_fixed_training_background.sh
```

**特點**:
- 訓練在背景執行
- 不佔用終端
- 可以登出系統繼續訓練

**管理指令**:
```bash
# 查看日誌
tail -f ../logs/token_denoising_fixed_*.log

# 查看進程
ps aux | grep train_token_denoising_hybrid.py

# 停止訓練
kill -SIGTERM <PID>
```

---

## 📊 訓練配置

### 模型架構
- **d_model**: 512
- **num_layers**: 4 (降低)
- **nhead**: 8
- **dim_feedforward**: 2048
- **dropout**: 0.2 (提高)

### 訓練參數
- **batch_size**: 8
- **learning_rate**: 1e-4
- **weight_decay**: 0.05 (提高)
- **num_epochs**: 600
- **scheduler**: ReduceLROnPlateau

### 損失配置
- **CE Loss**: 1.0
- **Content Loss**: 0.5
- **Embed Loss**: 0.3
- **Warmup**: 10 epochs

### 資料配置
- **訓練資料**: 5184 樣本 (288 句 × 18 說話者)
- **驗證集**: girl9, girl10, boy7, boy8
- **Content Batching**: ratio=0.5, min_samples=3

---

## 📁 輸出結構

```
results/token_denoising_fixed_<TIMESTAMP>/
├── config.json                      # 訓練配置
├── best_model.pth                   # 最佳模型
├── checkpoint_epoch_*.pth           # 每 10 epochs 的 checkpoint
├── loss_curves_epoch_*.png          # 每 50 epochs 的損失曲線
└── audio_samples/
    └── epoch_*/                     # 每 100 epochs 的音頻樣本
        ├── sample_0_noisy.wav
        ├── sample_0_predicted.wav
        ├── sample_0_clean.wav
        └── sample_0_spectrogram.png
```

---

## 🎯 預期效果

### Epoch 100
- Val accuracy: > 30%
- Train/Val gap: < 30%
- 音頻樣本正常儲存

### Epoch 200
- Val accuracy: > 50%
- Train/Val gap: < 20%
- Loss 曲線顯示穩定下降

### Epoch 600
- Val accuracy: > 70%
- Train/Val gap: < 15%
- 穩定收斂，無過擬合

---

## 🔍 監控指標

### 每個 Epoch 輸出
- Total Loss (Train / Val)
- CE Loss, Content Loss, Embed Loss
- Token Accuracy (Train / Val)
- Learning Rate

### 每 50 Epochs
- 生成 loss 曲線圖（4 個子圖）

### 每 100 Epochs
- 保存音頻樣本（3 個樣本）
- 保存頻譜圖對比

---

## 🛠 故障排除

### 如果訓練失敗

1. **檢查 GPU 記憶體**
   ```bash
   nvidia-smi
   ```

2. **查看錯誤日誌**
   ```bash
   tail -100 ../logs/token_denoising_fixed_*.log
   ```

3. **減少 batch size**
   修改 `run_fixed_training.sh` 中的 `BATCH_SIZE=8` → `BATCH_SIZE=4`

### 如果仍然過擬合

可以進一步調整參數：
```bash
# 在 run_fixed_training.sh 中修改
DROPOUT=0.3        # 進一步提高
WEIGHT_DECAY=0.1   # 進一步提高
NUM_LAYERS=3       # 進一步降低
```

---

## 📝 相關檔案

### 訓練腳本
- `train_token_denoising_hybrid.py` - 主訓練程式（已修復）
- `run_fixed_training.sh` - 完整訓練腳本
- `test_fixes.sh` - 快速測試腳本
- `start_fixed_training_background.sh` - 背景執行腳本

### 測試腳本
- `simple_test_model.py` - 音頻生成測試
- `eval_model_accuracy.py` - Token 準確率評估

### 文件
- `REPORT.md` - 自動更新的實驗報告
- `../logs/token_denoising_fixed_*.log` - 訓練日誌

---

## ✅ 準備完成檢查清單

- [x] train_token_denoising_hybrid.py 已修復（3 個 bug）
- [x] run_fixed_training.sh 腳本已創建
- [x] test_fixes.sh 測試腳本已創建
- [x] 防過擬合參數已設定
- [x] GPU 自動選擇已配置
- [x] 實驗報告自動更新已配置
- [x] Git commit 自動提交已配置

**現在可以開始訓練了！** 🎉

---

## 🚦 建議執行順序

1. **先測試** (2 分鐘)
   ```bash
   bash test_fixes.sh
   ```

2. **確認測試通過後，執行完整訓練** (約 2-3 天)
   ```bash
   bash run_fixed_training.sh
   ```
   或
   ```bash
   bash start_fixed_training_background.sh
   ```

3. **監控訓練**
   ```bash
   tail -f ../logs/token_denoising_fixed_*.log
   ```

---

**修復完成時間**: 2025-10-24  
**預計訓練時間**: 2-3 天（600 epochs）  
**預期改善**: 消除過擬合，Val accuracy > 70%
