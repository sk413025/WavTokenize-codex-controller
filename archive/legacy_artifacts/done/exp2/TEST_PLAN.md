# EXP2 測試計劃

## 🎯 測試目標

快速驗證 **Baseline + Speaker Embedding L2 Loss** 框架是否正常工作。

## 📋 測試步驟

### 第一步：最小化測試（必做）

使用極少數據快速測試所有組件：

```bash
cd /home/sbplab/ruizi/c_code
bash done/exp2/test_minimal.sh
```

**配置**：
- 語者數：14 位（全部）
- 每位語者句子數：**1 句**
- 訓練 epochs：**100**
- Batch size：**14** （一個 batch = 全部語者）
- Lambda：0.5
- Transformer layers：2（加速訓練）

**預期時間**：約 10-20 分鐘

**預期結果**：
- ✅ 訓練正常啟動
- ✅ Speaker Loss 正常計算（不為 0）
- ✅ Loss 逐漸下降
- ✅ 能正常保存 checkpoint 和音頻樣本

---

### 第二步：檢查測試結果

#### 1. 查看訓練日誌

```bash
# 查看完整日誌
cat ./results/exp2/test_minimal/training.log

# 查看最後 20 行
tail -20 ./results/exp2/test_minimal/training.log

# 檢查 loss 趨勢
grep "Epoch" ./results/exp2/test_minimal/training.log
```

**關鍵檢查點**：
- [ ] CE Loss 是否正常？（應該從 ~8-10 開始下降）
- [ ] Speaker Loss 是否正常？（應該不為 0，且有數值）
- [ ] Token Accuracy 是否提升？（從 0% 開始上升）

#### 2. 查看配置

```bash
cat ./results/exp2/test_minimal/config.json
```

確認所有參數都正確。

#### 3. 聆聽音頻樣本

```bash
# 查看生成的樣本
ls ./results/exp2/test_minimal/audio_samples/

# 播放最後一個 epoch 的樣本
cd ./results/exp2/test_minimal/audio_samples/epoch_100/
mpv sample_0_predicted.wav
mpv sample_0_noisy.wav
mpv sample_0_clean.wav
```

**評估標準**：
- 噪音是否有去除？
- 聲音是否清晰？
- 說話人身份是否正確？

---

### 第三步：如果測試成功

如果最小化測試通過，可以進行完整實驗：

#### 選項 A：單個完整實驗

```bash
cd /home/sbplab/ruizi/c_code

python done/exp2/train_with_speaker.py \
    --input_dirs data/raw/box data/raw/papercup data/raw/plastic \
    --target_dir data/clean/box2 \
    --output_dir ./results/exp2/lambda0.5_full \
    --lambda_speaker 0.5 \
    --num_epochs 600 \
    --batch_size 8 \
    --max_sentences_per_speaker 288
```

#### 選項 B：批次對比實驗

```bash
# 運行所有 λ 值的對比實驗
bash done/exp2/run_experiments.sh
```

---

## 🐛 常見問題排查

### 問題 1: ECAPA-TDNN 下載失敗

**症狀**：
```
❌ 無法載入 ECAPA
```

**解決方案**：
```bash
# 檢查 speechbrain 是否安裝
python -c "import speechbrain; print('OK')"

# 如果未安裝
pip install speechbrain

# 或改用 resemblyzer
--speaker_model_type resemblyzer
```

### 問題 2: WavTokenizer 路徑錯誤

**症狀**：
```
✗ WavTokenizer 載入失敗
```

**解決方案**：
```bash
# 查找實際路徑
find /home/sbplab/ruizi -name "wavtokenizer*.yaml" -type f 2>/dev/null

# 使用正確路徑運行
python done/exp2/train_with_speaker.py \
    --wavtokenizer_config /path/to/correct/config.yaml \
    --wavtokenizer_checkpoint /path/to/correct/checkpoint.pth \
    ...
```

### 問題 3: CUDA Out of Memory

**症狀**：
```
RuntimeError: CUDA out of memory
```

**解決方案**：
```bash
# 減少 batch size
--batch_size 4  # 從 14 改為 4

# 或減少 transformer layers
--num_layers 2  # 從 4 改為 2

# 或降低計算 speaker loss 的頻率
--compute_speaker_every_n_steps 5
```

### 問題 4: Speaker Loss 一直為 0

**可能原因**：
1. Speaker encoder 未正確載入
2. 音頻解碼失敗
3. 設置了 `speaker_loss_start_epoch` 太大

**解決方案**：
```bash
# 檢查日誌中的 speaker encoder 初始化
grep "ECAPA" ./results/exp2/test_minimal/training.log

# 確保從 epoch 0 開始
--speaker_loss_start_epoch 0
```

---

## 📊 成功標準

### 最小化測試

- [x] 訓練能正常啟動和完成
- [x] Speaker Loss > 0 且正常計算
- [x] CE Loss 從 ~8-10 下降到 <5
- [x] Token Accuracy 從 0% 上升到 >20%
- [x] 能保存 checkpoint 和音頻樣本

### 完整實驗

- [x] 訓練集 Token Accuracy > 85%
- [x] 驗證集 Token Accuracy > 70%
- [x] 驗證集表現 **優於** baseline (λ=0)
- [x] Speaker Similarity 維持 >0.85
- [x] 主觀聽感：降噪效果好且說話人身份正確

---

## 📝 記錄實驗結果

完成測試後，將結果填入 [README.md](README.md) 的實驗記錄表格：

| Experiment | λ | Train Acc | Val Acc | Val Speaker Sim | 備註 |
|------------|---|-----------|---------|------------------|------|
| Test Minimal | 0.5 | ??% | ??% | ?? | 1句/人, 100 epochs |
| Full Exp | 0.5 | ??% | ??% | ?? | 288句/人, 600 epochs |

---

## 🚀 開始測試

準備好後，執行：

```bash
cd /home/sbplab/ruizi/c_code
bash done/exp2/test_minimal.sh
```

預期 10-20 分鐘完成。訓練過程中可以實時查看：

```bash
# 另開一個終端
tail -f ./results/exp2/test_minimal/training.log
```

祝測試順利！🎉
