# Token Masking 實驗 - 快速開始指南

## ✨ 已完成的工作

我已經為你的 Token Denoising Transformer 設計並實現了完整的 Token Masking 實驗套件！

### 📦 新增文件

```
done/
├── mask.py                     # ✅ Token Masker 核心實現
├── train_mask.py               # ✅ 帶遮罩的訓練腳本
├── run_mask_dynamic.sh         # ✅ 方案 A 執行腳本
├── run_mask_weighted.sh        # ✅ 方案 B 執行腳本
├── run_mask_progressive.sh     # ✅ 方案 C 執行腳本
├── MASKING_EXPERIMENTS.md      # ✅ 完整實驗指南
└── README_MASKING.md           # ✅ 本文件
```

### 🎯 三種遮罩策略

#### 方案 A: Dynamic Masking（建議首選）
- **遮罩方式**: 固定 20% 隨機遮罩
- **適合場景**: 首次嘗試 masking，驗證有效性
- **執行命令**: `bash done/run_mask_dynamic.sh`

#### 方案 B: Weighted Loss Masking
- **遮罩方式**: 10% 遮罩 + 遮罩位置損失權重 2x
- **適合場景**: 強化模型對遮罩位置的學習
- **執行命令**: `bash done/run_mask_weighted.sh`

#### 方案 C: Progressive Masking
- **遮罩方式**: 5% → 30% 漸進式增長（前 100 epochs）
- **適合場景**: 課程學習，長時間訓練（200+ epochs）
- **執行命令**: `bash done/run_mask_progressive.sh`

---

## 🚀 立即開始

### 步驟 1: 驗證環境

```bash
# 確認在 test 環境
conda activate test

# 測試 mask.py 是否正常
cd done
python mask.py
```

**預期輸出**: 應該看到三種策略的測試結果，遮罩比例正常

### 步驟 2: 運行第一個實驗（Dynamic Masking）

```bash
cd done
bash run_mask_dynamic.sh
```

**預期行為**:
- 自動檢測可用 GPU
- 開始訓練，顯示進度條
- 每 50 epochs 保存 loss 曲線
- 每 100 epochs 保存音頻樣本

### 步驟 3: 監控訓練

```bash
# 實時查看日誌
tail -f done/logs/mask_dynamic_*.log

# 過濾關鍵信息
tail -f done/logs/mask_dynamic_*.log | grep -E "(Epoch|Loss|Acc|Mask)"
```

### 步驟 4: 查看結果

```bash
# 訓練完成後，查看結果目錄
ls done/results/mask_dynamic_*/

# 查看 loss 曲線
ls done/results/mask_dynamic_*/loss_curves_*.png

# 聽音頻樣本
ls done/results/mask_dynamic_*/audio_samples/epoch_*/
```

---

## 📊 實驗建議順序

### 第一階段：驗證 Masking 有效性（1-2 天）

1. **運行 Dynamic Masking** (10%, 200 epochs)
   ```bash
   bash done/run_mask_dynamic.sh
   ```

2. **對比無遮罩版本**
   - 檢查 loss 是否正常下降
   - 對比驗證集 accuracy
   - 聽音頻樣本品質

3. **判斷標準**:
   - ✅ Loss 穩定下降 → Masking 有效
   - ❌ Loss 不穩定 → 降低遮罩比例到 5%

### 第二階段：對比三種策略（3-5 天）

1. **同時運行三種策略**（使用不同 GPU）
   ```bash
   # Terminal 1
   bash done/run_mask_dynamic.sh

   # Terminal 2
   bash done/run_mask_weighted.sh

   # Terminal 3
   bash done/run_mask_progressive.sh
   ```

2. **對比指標**:
   - 訓練穩定性（loss 曲線平滑度）
   - 收斂速度（達到最佳 val loss 的 epoch 數）
   - 最終驗證 loss
   - 音頻主觀品質

### 第三階段：超參數調優（根據需要）

找出最佳策略後，調整超參數：

#### Dynamic / Weighted
```bash
# 修改 run_mask_dynamic.sh
MASK_RATIO=0.15  # 嘗試 0.05, 0.10, 0.15, 0.20
```

#### Progressive
```bash
# 修改 run_mask_progressive.sh
PROGRESSIVE_END_RATIO=0.40  # 嘗試 0.20, 0.30, 0.40
PROGRESSIVE_EPOCHS=150      # 嘗試 50, 100, 150
```

---

## 🔍 預期實驗結果

### Dynamic Masking vs 無遮罩

| 指標 | 無遮罩 (train.py) | Dynamic (預期) | 改善 |
|------|-------------------|----------------|------|
| Train Acc | ~70% | ~68% | -2% (正常) |
| Val Acc | ~30% | ~35% | +5% ✅ |
| Val Loss | Baseline | -5~10% | 改善 ✅ |
| 音頻品質 | 基準 | 相近或更好 | ✅ |

**結論**: 略微犧牲訓練 accuracy，換取更好的泛化能力

### 三種策略對比

| 指標 | Dynamic | Weighted | Progressive |
|------|---------|----------|-------------|
| 訓練穩定性 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 收斂速度 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 泛化能力 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 實現複雜度 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

**推薦**:
- **首選**: Dynamic Masking（簡單有效）
- **進階**: Progressive Masking（泛化能力最強）

---

## 💡 常見問題速查

### Q: 遮罩會讓訓練變慢嗎？
**A**: 會略微增加 5-10% 計算開銷，但泛化提升值得這個代價。

### Q: 10% 遮罩比例是否合適？
**A**:
- ✅ 訓練穩定 → 可增加到 15-20%
- ❌ 訓練不穩定 → 降低到 5%

### Q: 驗證時要不要遮罩？
**A**: **要**。當前實現在驗證時也使用遮罩，保持訓練/驗證一致性。

### Q: 如何調整遮罩比例？
**A**: 編輯對應的 `.sh` 文件：
```bash
MASK_RATIO=0.15  # 改為 15%
```

### Q: Progressive 的最終遮罩率設多高？
**A**:
- 保守: 20% (小數據集)
- 標準: 30% (推薦)
- 激進: 40% (大數據集 + 充足訓練時間)

---

## 📝 Git 分支管理

### 當前狀態
```bash
# 查看當前分支
git branch
# * feature/token-masking

# 查看改動
git status
# On branch feature/token-masking
# ...已提交 mask.py, train_mask.py, run_mask_*.sh
```

### 合併到主分支（實驗完成後）

```bash
# 確認實驗成功後，合併到主分支
git checkout c_code
git merge feature/token-masking

# 或保留分支繼續實驗
git checkout c_code
git merge --no-ff feature/token-masking  # 保留完整歷史
```

---

## 📚 詳細文檔

- **完整實驗指南**: [done/MASKING_EXPERIMENTS.md](./MASKING_EXPERIMENTS.md)
  - 理論背景
  - 實現細節
  - 評估指標
  - 超參數調優建議

- **實驗記錄**: [REPORT.md](../REPORT.md)
  - 記錄所有實驗結果
  - 對比分析

---

## ✅ 快速檢查清單

實驗前確認：
- [ ] 已激活 `test` conda 環境
- [ ] 已測試 `mask.py`（應該正常運行）
- [ ] WavTokenizer 路徑正確
- [ ] GPU 記憶體充足（至少 8GB）
- [ ] 數據集路徑正確（`../data/raw/box`, `../data/clean/box2`）

實驗中記錄：
- [ ] 訓練 loss 曲線趨勢
- [ ] 驗證 loss 是否改善
- [ ] Token accuracy 變化
- [ ] 音頻樣本品質（主觀評測）

實驗後分析：
- [ ] 對比無遮罩版本
- [ ] 對比三種策略
- [ ] 更新 REPORT.md

---

## 🎉 預祝實驗順利！

如有任何問題：
1. 查看詳細指南：`done/MASKING_EXPERIMENTS.md`
2. 檢查訓練日誌：`done/logs/mask_*.log`
3. 查看配置文件：`done/results/*/config.json`

**建議**: 先用 **Dynamic Masking** 跑一輪（約 1-2 小時），確認訓練正常後再啟動其他策略！
