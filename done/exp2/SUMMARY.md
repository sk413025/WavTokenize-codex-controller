# EXP2: Baseline + Speaker Loss 實驗框架總結

## ✅ 已完成的工作

### 📁 核心文件

| 文件 | 功能 | 狀態 |
|------|------|------|
| `loss_with_speaker.py` | CE + Speaker L2 Loss 損失函數 | ✅ 完成 |
| `train_with_speaker.py` | 完整訓練腳本 | ✅ 完成 |
| `run_experiments.sh` | 批次實驗腳本 (λ=0.1, 0.5, 1.0) | ✅ 完成 |
| `test_minimal.sh` | 最小化測試腳本 | ✅ 完成 |
| `verify_data_paths.sh` | 數據路徑驗證腳本 | ✅ 完成 |
| `README.md` | 詳細實驗說明 | ✅ 完成 |
| `QUICKSTART.md` | 快速開始指南 | ✅ 完成 |
| `TEST_PLAN.md` | 測試計劃 | ✅ 完成 |
| `test_loss.py` | 損失函數測試腳本 | ✅ 完成 |

### 🎯 實驗設計

**核心假設**：引入 Speaker Embedding L2 Loss 作為輔助約束，強制模型在去噪時保持說話人身份，從而提升對未見語者的泛化能力。

**損失函數**：
```python
L_total = L_CE(pred_tokens, clean_tokens)              # 主任務
        + λ * L_Speaker(pred_audio, input_audio)      # 輔助約束
```

**對比實驗**：
- Baseline (λ=0)：只有 CE Loss
- Exp2-λ0.1：輕量約束
- Exp2-λ0.5：中等約束（推薦）
- Exp2-λ1.0：強約束

### 📊 數據配置

**已驗證的路徑**：
- ✅ 輸入資料夾：
  - `data/raw/box` (5184 個音檔)
  - `data/raw/papercup` (5184 個音檔)
  - `data/raw/plastic` (5184 個音檔)
- ✅ 目標資料夾：
  - `data/clean/box2` (5184 個音檔)
- ✅ 語者分布：18 位語者 (boy1-10, girl2-4, girl6-11)
  - 訓練集：14 位
  - 驗證集：4 位 (girl9, girl10, boy7, boy8) - **未見語者**

---

## 🚀 快速開始

### 步驟 1: 最小化測試（推薦先做）

```bash
cd /home/sbplab/ruizi/c_code
bash done/exp2/test_minimal.sh
```

**配置**：
- 14 位語者，每位 1 句話
- 100 epochs，batch size 14
- 約 10-20 分鐘完成

### 步驟 2: 檢查測試結果

```bash
# 查看訓練日誌
tail -20 ./results/exp2/test_minimal/training.log

# 聆聽音頻樣本
ls ./results/exp2/test_minimal/audio_samples/epoch_100/
```

**成功標準**：
- ✅ Speaker Loss > 0
- ✅ CE Loss 下降
- ✅ Token Accuracy 上升
- ✅ 音頻樣本有降噪效果

### 步驟 3: 完整實驗

如果測試成功，運行完整實驗：

```bash
# 選項 A: 單個實驗 (λ=0.5)
python done/exp2/train_with_speaker.py \
    --input_dirs data/raw/box data/raw/papercup data/raw/plastic \
    --target_dir data/clean/box2 \
    --output_dir ./results/exp2/lambda0.5_full \
    --lambda_speaker 0.5 \
    --num_epochs 600 \
    --batch_size 8 \
    --max_sentences_per_speaker 288

# 選項 B: 批次對比實驗（所有 λ 值）
bash done/exp2/run_experiments.sh
```

---

## 📖 文檔導航

| 文檔 | 用途 |
|------|------|
| [README.md](README.md) | 詳細實驗設計、理論分析、評估方法 |
| [QUICKSTART.md](QUICKSTART.md) | 一步步操作指南、常見問題 |
| [TEST_PLAN.md](TEST_PLAN.md) | 測試步驟、成功標準、問題排查 |

---

## 🔍 關鍵檢查點

### 訓練前

- [ ] 數據路徑正確（運行 `verify_data_paths.sh`）
- [ ] speechbrain 已安裝（用於 ECAPA-TDNN）
- [ ] WavTokenizer 可載入（檢查配置路徑）

### 訓練中

- [ ] Speaker Loss 正常計算（不為 0）
- [ ] CE Loss 逐漸下降
- [ ] Token Accuracy 逐漸上升
- [ ] GPU 記憶體使用正常

### 訓練後

- [ ] 驗證集 Token Accuracy > baseline
- [ ] Speaker Similarity 維持高值 (>0.85)
- [ ] 音頻樣本降噪效果好
- [ ] 說話人身份保持正確

---

## 💡 核心創新點

1. **解耦內容與身份**
   - 主任務（CE Loss）：學習噪音 → 乾淨的映射
   - 輔助約束（Speaker Loss）：保持說話人身份

2. **預訓練知識遷移**
   - 使用凍結的 ECAPA-TDNN speaker encoder
   - 將其在大量語者上學到的泛化能力傳遞給降噪模型

3. **正則化效果**
   - Speaker Loss 作為約束，防止過度擬合特定語者

---

## 📊 預期實驗結果

### 成功指標

| 指標 | Baseline (λ=0) | Exp2 (λ=0.5) | 期望 |
|------|----------------|--------------|------|
| 訓練集 Accuracy | ~95% | ~94% | 略降（正常） |
| 驗證集 Accuracy | ~82% | **>85%** | **提升** ✅ |
| Speaker Similarity | - | >0.90 | 高值 ✅ |
| 泛化 Gap | 13% | **<10%** | **縮小** ✅ |

### 失敗指標

- ❌ 驗證集 Accuracy 下降 → λ 太大
- ❌ Speaker Similarity 低 → 模型改變了說話人
- ❌ Gap 仍然很大 → 仍過擬合

---

## 🔄 後續方向

### 如果成功
1. 測試更多材質組合
2. 探索動態 λ 調整策略
3. 嘗試其他 speaker encoder（WavLM, Wav2Vec2）

### 如果失敗
1. 在 feature space 計算 speaker loss（避免解碼）
2. 使用 contrastive learning 替代 L2 loss
3. 探索其他正則化方法

---

## 🎓 理論基礎

**為什麼 Speaker Loss 能提升泛化性？**

1. **強制解耦**：模型不能依賴語者特定的 token pattern
2. **預訓練遷移**：ECAPA 的泛化能力通過 L2 約束傳遞
3. **正則化**：防止模型記住訓練集語者的噪音模式

**類比**：
```
沒有 Speaker Loss:
  model(girl1_noisy) → girl1_clean  (記住 girl1 的模式)
  model(girl9_noisy) → ❌ 失敗       (沒見過 girl9)

有 Speaker Loss:
  model(any_speaker_noisy) → clean  (只學噪音去除)
  約束: speaker(output) = speaker(input)  (保持身份)
  model(girl9_noisy) → ✅ 成功       (泛化到 girl9)
```

---

## 📞 問題排查

遇到問題？按順序檢查：

1. **數據問題**：運行 `bash done/exp2/verify_data_paths.sh`
2. **依賴問題**：確認 speechbrain 安裝
3. **路徑問題**：檢查 WavTokenizer 配置路徑
4. **記憶體問題**：減少 batch_size 或 num_layers
5. **其他問題**：查看 [QUICKSTART.md](QUICKSTART.md) 常見問題章節

---

## ✨ 實驗框架特色

- ✅ **完整性**：從數據驗證到訓練到評估的完整流程
- ✅ **靈活性**：支持多種配置和超參數調整
- ✅ **可擴展性**：易於添加新的損失項或約束
- ✅ **穩定性**：完善的錯誤處理和日誌記錄
- ✅ **文檔齊全**：詳細的說明文檔和使用指南

---

## 🎯 立即開始

```bash
cd /home/sbplab/ruizi/c_code
bash done/exp2/test_minimal.sh
```

**預期時間**：10-20 分鐘
**預期結果**：驗證所有組件正常工作

實驗順利！🚀
