# 音頻重建失敗診斷總結報告

**實驗編號**: EXP20251021_01  
**診斷日期**: 2025-10-21  
**生成函式**: diagnose_voice_reconstruction, analyze_loss_components, investigate_training_failure

---

## 📋 執行摘要

### 問題現象
訓練 452+ epochs 後，模型**完全無法重建人聲內容**：
- ❌ Token 準確率 = 0.00%
- ❌ 頻譜相關性 = 0.517 (偏低)
- ❌ CE Loss = 8.59 (接近隨機猜測的 log(4096)=8.32)

### 根本原因
**Learning Rate 過小！**
- 當前 LR: **2e-6** (0.000002)
- 正常範圍: 1e-4 到 1e-3
- **比正常值小 50-500 倍！**

### 原因分析
使用了 **OneCycleLR scheduler**，在長訓練中會把 LR 降到極低：
- Epoch 100: LR = 2.0e-6
- Epoch 200: LR = 2.0e-6  
- Epoch 300: LR = 2.0e-6
- Epoch 400: LR = 2.0e-6

OneCycleLR 不適合 1000 epochs 的長訓練，在 40% 進度時已將 LR 降到初始值的 1/25。

---

## 🔍 詳細診斷結果

### 1. 音頻統計
```
Input (Noisy):
  RMS: 0.106437
  Max Amplitude: 0.990
  Energy: 1018.68

Enhanced (Model Output):
  RMS: 0.122170  ✅ 振幅正常
  Max Amplitude: 0.990
  Energy: 1342.11

Target (Clean):
  RMS: 0.106626
  Max Amplitude: 0.990
  Energy: 1022.32
```

**結論**: ✅ 音頻振幅正常，不是音量問題

### 2. 頻譜分析
```
頻譜 Pearson 相關係數: 0.5174
頻譜 MSE: 415.83 dB²

頻帶能量分佈差異:
  低頻 (0-25%):   -6.78 dB
  中低 (25-50%):  -7.41 dB  
  中高 (50-75%):  -9.80 dB  ⚠️ 差異最大
  高頻 (75-100%): -2.98 dB
```

**結論**: ⚠️ 頻譜相關性偏低，中高頻能量不足

### 3. Token 準確率
```
Token 統計:
  Input tokens:  281 個
  Target tokens: 281 個
  Predicted:     400 個 (含 padding)

Token 準確率: 0.00% (0/281)  ❌❌❌

Token 多樣性:
  Input:     207/4096 (5.05%)
  Target:    213/4096 (5.20%)
  Predicted: 129/4096 (3.15%)  ⚠️ 偏低

Mode Collapse 檢查:
  Top 1 token: 7.47%
  Top 5 tokens: 25.62%
```

**結論**: ❌ 完全沒有預測對任何 token，但無嚴重 mode collapse

### 4. 損失組成分析
```
各損失組件（Epoch 400）:
  CE Loss:     85.86  (95.1%)  ✅ 占比正常
  L2 Loss:      0.46  ( 0.5%)
  Coherence:    2.43  ( 2.7%)
  Manifold:     1.54  ( 1.7%)
  -------------------------
  Total:       90.29  (100%)

Raw CE Loss: 8.5856  ❌ 接近隨機 (log(4096)=8.32)
```

**結論**: ✅ CE Loss 權重配置正確 (95.1%)，但 ❌ Raw CE Loss 幾乎沒下降

### 5. Learning Rate 歷史
```
Epoch 100: LR = 2.000e-6
Epoch 200: LR = 2.000e-6
Epoch 300: LR = 2.001e-6
Epoch 400: LR = 2.002e-6

初始設定: 5e-5 (0.00005)
實際使用: 2e-6 (0.000002)  ❌ 降低了 25 倍！
```

**結論**: ❌❌❌ **這是根本問題！LR 太小導致模型無法學習**

---

## 💡 問題原因總結

### 主要問題
**OneCycleLR scheduler 不適合長訓練**

1. **OneCycleLR 行為**:
   - 0-10%: LR 從 0 升到 max_lr (5e-5)
   - 10-100%: LR 從 max_lr 降到接近 0
   - Epoch 400/1000 = 40%: 已進入快速下降階段

2. **實際影響**:
   - LR 從 5e-5 降到 2e-6 (降低 25 倍)
   - 梯度更新極小，參數幾乎不變
   - CE Loss 無法下降 (保持在 8.59)
   - Token 準確率始終為 0%

3. **為什麼之前沒發現**:
   - Checkpoint 中的 loss 一直是 1000000.0 (validation loss 計算錯誤)
   - Log 中沒有記錄 CE Loss 的具體數值
   - 只看到 Total Loss 在下降 (實際主要是 auxiliary losses 在降)

### 次要問題
1. **CE Loss weight = 10.0 可能仍不足**
   - 雖然占比 95.1%，但 raw CE loss 沒下降
   - 建議測試 15.0-20.0

2. **Model 容量可能不足**
   - d_model=256 對 4096 vocab 可能偏小
   - 建議測試 d_model=512

---

## 🛠️ 修復方案

### 方案 A：修復 Learning Rate (建議)

1. **停止當前訓練**
   ```bash
   kill 3620908
   ```

2. **修改訓練腳本**:
   - 移除 OneCycleLR
   - 使用固定 LR 或溫和的 StepLR
   - 增加初始 LR 到 1e-4 或 2e-4

3. **修改配置**:
   ```bash
   # 在 run_transformer_large_tokenloss.sh 中
   --learning_rate 1e-4 \           # 增加 2 倍
   --use_scheduler False \          # 禁用 scheduler
   --ce_weight 15.0 \               # 可選：進一步增加 CE weight
   ```

4. **從頭開始訓練**:
   - 不要從 Epoch 400 繼續 (LR 已經太小)
   - 重新初始化 optimizer
   - 預期 50-100 epochs 就能看到明顯改善

### 方案 B：保守修復

如果想保留已訓練的模型：

1. **修改 scheduler**:
   ```python
   # 改用 ConstantLR 或 StepLR
   scheduler = optim.lr_scheduler.ConstantLR(
       optimizer, 
       factor=1.0,  # 不改變 LR
       total_iters=args.num_epochs
   )
   ```

2. **重置 optimizer 但保留模型參數**:
   ```python
   # 載入 checkpoint，但不載入 optimizer_state_dict
   # 使用新的 LR = 1e-4 初始化 optimizer
   ```

3. **從 Epoch 400 繼續訓練**:
   - 使用固定 LR = 1e-4
   - 訓練 200-400 epochs
   - 監控 CE Loss 是否開始下降

---

## 📊 預期結果

### 修復後應該看到：

**前 50 epochs**:
- CE Loss: 8.59 → 5.0-6.0
- Token Accuracy: 0% → 10-20%
- 頻譜相關性: 0.52 → 0.65-0.75

**100 epochs 後**:
- CE Loss: < 4.0
- Token Accuracy: 30-50%
- 頻譜相關性: > 0.80
- 音頻應該開始有可辨識的人聲

**200-300 epochs**:
- CE Loss: < 2.0
- Token Accuracy: > 60%
- 頻譜相關性: > 0.90
- 音頻品質接近 target

---

## 📁 生成檔案

### 頻譜圖
- `results/diagnosis_EXP20251021_01_20251021_005311/spectrogram_comparison_EXP20251021_01.png`
- 顯示 Enhanced vs Target 的頻譜差異

### 診斷報告
- `results/diagnosis_EXP20251021_01_20251021_005311/diagnosis_report_EXP20251021_01.txt`
- 包含詳細統計數據

---

## ✅ 下一步行動

### 立即執行 (優先級 1)

1. **停止當前訓練**
   ```bash
   ps aux | grep wavtokenizer_transformer_denoising.py
   kill <PID>
   ```

2. **修改訓練腳本**
   - 設定 LR = 1e-4 (或 2e-4)
   - 禁用 OneCycleLR scheduler
   - 可選：增加 CE weight 到 15.0

3. **重新開始訓練**
   - 從頭訓練，不使用 Epoch 400 checkpoint
   - 預期 50-100 epochs 內看到明顯改善

### 監控指標 (優先級 2)

訓練過程中每 10 epochs 檢查：
- [ ] CE Loss 是否持續下降
- [ ] Token Accuracy 是否 > 0%
- [ ] Learning Rate 保持在 1e-4 附近
- [ ] Audio samples 是否開始有人聲

### 長期優化 (優先級 3)

訓練收斂後：
- [ ] 測試更大的 model (d_model=512)
- [ ] 測試更高的 CE weight (20.0-30.0)
- [ ] 測試不同的 optimizer (AdamW, Lion)
- [ ] 加入 label smoothing

---

## 📝 總結

**根本問題**: OneCycleLR scheduler 把 learning rate 降到太小 (2e-6)，導致模型無法學習

**證據**:
1. ✅ CE Loss weight 配置正確 (95.1%)
2. ❌ Raw CE Loss = 8.59 (幾乎沒下降)
3. ❌ Token 準確率 = 0%
4. ❌ LR = 2e-6 (比正常值小 50-500 倍)

**解決方案**: 
- 修改 LR 到 1e-4
- 禁用 OneCycleLR
- 重新開始訓練

**預期時間**: 50-100 epochs 內看到明顯改善

---

**報告生成時間**: 2025-10-21 00:53  
**實驗編號**: EXP20251021_01  
**診斷函式**: diagnose_voice_reconstruction, analyze_loss_components, investigate_training_failure
