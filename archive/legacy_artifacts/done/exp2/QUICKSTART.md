# EXP2 快速開始指南

## 🚀 開始實驗前的準備

### 1. 確認環境依賴

```bash
# 確認已安裝 speechbrain (用於 ECAPA-TDNN speaker encoder)
pip install speechbrain

# 如果無法安裝 speechbrain，可以使用 resemblyzer
pip install resemblyzer
```

### 2. 快速測試

在開始長時間訓練前，先測試代碼是否正常運行：

```bash
cd /home/sbplab/ruizi/c_code
python done/exp2/test_loss.py
```

**預期輸出**：
```
========================================
測試 CEWithSpeakerLoss 基本功能
========================================
...
✓ 所有測試通過！可以開始正式訓練。
```

如果測試失敗，請檢查錯誤信息並解決依賴問題。

---

## 📊 方案 A: 單個實驗（推薦用於測試）

### 訓練單個模型 (λ=0.5)

```bash
cd /home/sbplab/ruizi/c_code

python done/exp2/train_with_speaker.py \
    --input_dirs data/raw/box data/raw/papercup data/raw/plastic \
    --target_dir data/clean/box2 \
    --output_dir ./results/exp2/lambda0.5 \
    --lambda_speaker 0.5 \
    --num_epochs 600 \
    --batch_size 8 \
    --max_sentences_per_speaker 288
```

**訓練時間估計**: 約 6-12 小時 (取決於 GPU 和數據量)

---

## 🔬 方案 B: 批次實驗（完整對比）

### 一次性運行所有對比實驗

```bash
cd /home/sbplab/ruizi/c_code

# 運行 λ = 0.1, 0.5, 1.0 三個實驗
bash done/exp2/run_experiments.sh
```

**警告**: 這將運行 3 個完整的訓練，總時間約 18-36 小時！

建議在 tmux 或 screen 中運行：
```bash
# 使用 tmux
tmux new -s exp2
bash done/exp2/run_experiments.sh
# Ctrl+B 然後 D 來 detach

# 重新連接
tmux attach -t exp2
```

---

## 📈 訓練過程監控

### 1. 查看實時日誌

```bash
# 查看訓練日誌
tail -f ./results/exp2/lambda0.5/training.log

# 查看最近的損失
grep "Epoch" ./results/exp2/lambda0.5/training.log | tail -20
```

### 2. 檢查損失曲線

訓練過程中會自動生成損失曲線圖：

```bash
# 每 50 epochs 生成一次
ls ./results/exp2/lambda0.5/loss_curves_epoch_*.png

# 查看最新的圖表
eog ./results/exp2/lambda0.5/loss_curves_epoch_600.png
```

### 3. 聆聽音頻樣本

每 100 epochs 會保存音頻樣本：

```bash
# 查看保存的樣本
ls ./results/exp2/lambda0.5/audio_samples/

# 聆聽 epoch 600 的樣本
cd ./results/exp2/lambda0.5/audio_samples/epoch_600/
mpv sample_0_predicted.wav  # 預測的音頻
mpv sample_0_noisy.wav      # 輸入的噪音音頻
mpv sample_0_clean.wav      # 目標的乾淨音頻
```

---

## 🎯 重要參數說明

### λ (lambda_speaker) 的選擇

| λ 值 | 含義 | 適用場景 |
|------|------|----------|
| 0.0  | 無 Speaker Loss (baseline) | 對比實驗 |
| 0.1  | 輕量約束 | 不確定 speaker loss 是否有幫助時 |
| **0.5**  | **中等約束（推薦）** | **大部分情況的起點** |
| 1.0  | 強約束 | 如果 0.5 效果好，可以嘗試更強約束 |

### 其他關鍵參數

```bash
# 從第 50 epoch 才開始加入 speaker loss
--speaker_loss_start_epoch 50

# 每 5 步才計算一次 speaker loss（加速訓練 5 倍）
--compute_speaker_every_n_steps 5

# 使用 resemblyzer 而非 ECAPA
--speaker_model_type resemblyzer
```

---

## 📊 評估實驗結果

### 1. 對比驗證集表現

```bash
# 查看各實驗的最佳驗證損失
grep "保存最佳模型" ./results/exp2/lambda0.1/training.log | tail -1
grep "保存最佳模型" ./results/exp2/lambda0.5/training.log | tail -1
grep "保存最佳模型" ./results/exp2/lambda1.0/training.log | tail -1
```

### 2. 檢查泛化性

關鍵指標：**驗證集 Token Accuracy**

```bash
# 查看最後幾個 epoch 的驗證集表現
grep "Val   -" ./results/exp2/lambda0.5/training.log | tail -10
```

**期望結果**：
- ✅ 驗證集 Accuracy 比 baseline (λ=0) 更高
- ✅ 驗證集 Speaker Loss 較低（說話人身份保持良好）
- ✅ 訓練集和驗證集的 gap 較小（泛化性好）

### 3. 主觀評估

聆聽驗證集語者（girl9, girl10, boy7, boy8）的音頻樣本：

```bash
# 播放驗證集樣本（如果訓練腳本有保存）
cd ./results/exp2/lambda0.5/audio_samples/epoch_600/
mpv sample_*_predicted.wav
```

評估標準：
- 噪音是否有效去除？
- 說話人身份是否保持？
- 是否有明顯的失真？

---

## 🐛 常見問題

### Q1: ECAPA-TDNN 下載失敗

**問題**: `speechbrain` 無法下載預訓練模型

**解決方案**:
```bash
# 方案 1: 手動下載模型
mkdir -p pretrained_models
cd pretrained_models
wget https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/...

# 方案 2: 改用 resemblyzer
--speaker_model_type resemblyzer
```

### Q2: CUDA Out of Memory

**問題**: GPU 記憶體不足

**解決方案**:
```bash
# 減少 batch size
--batch_size 4  # 原本是 8

# 或每 N 步才計算 speaker loss
--compute_speaker_every_n_steps 5
```

### Q3: 訓練速度太慢

**問題**: 加入 speaker loss 後訓練變慢

**解決方案**:
```bash
# 每 5 步計算一次 speaker loss（加速 5 倍）
--compute_speaker_every_n_steps 5

# 或從後期才開始加入 speaker loss
--speaker_loss_start_epoch 100
```

### Q4: 驗證集表現沒有提升

**可能原因**:
1. λ 值太大，干擾了主任務 → 嘗試更小的 λ (0.1)
2. λ 值太小，約束不足 → 嘗試更大的 λ (1.0)
3. Speaker Loss 計算有問題 → 檢查 speaker embedding 是否正常

---

## 📝 實驗後續

### 將結果填入實驗記錄

完成實驗後，將結果填入 [`README.md`](README.md) 的實驗記錄表格：

```markdown
| Experiment | λ | Train Acc | Val Acc | Val Speaker Sim | 備註 |
|------------|---|-----------|---------|------------------|------|
| Baseline   | 0 | 95.2% | 82.1% | - | 無 speaker loss |
| Exp2-λ0.1  | 0.1 | 94.8% | 83.5% | 0.92 | ✓ 輕微提升 |
| Exp2-λ0.5  | 0.5 | 94.5% | **85.2%** | 0.94 | ✓✓ 明顯提升 |
| Exp2-λ1.0  | 1.0 | 93.1% | 84.0% | 0.96 | ✗ 過度約束 |
```

### 下一步方向

如果實驗成功（驗證集表現提升）：
1. 嘗試更多噪音材質
2. 探索動態調整 λ 的策略
3. 嘗試其他 speaker encoder（WavLM, Wav2Vec2）

如果實驗失敗（驗證集表現未提升）：
1. 在 feature space 計算 speaker loss（避免完整解碼）
2. 使用 contrastive learning 而非 L2 loss
3. 探索其他正則化方法

---

## 📞 需要幫助？

如有問題，請檢查：
1. [`README.md`](README.md) - 詳細的實驗說明
2. [`loss_with_speaker.py`](loss_with_speaker.py) - 損失函數實現
3. [`train_with_speaker.py`](train_with_speaker.py) - 訓練腳本
4. [`test_loss.py`](test_loss.py) - 測試腳本

祝實驗順利！🚀
