# 執行清單 Checklist

**實驗**: Token Denoising with Frozen Codebook  
**日期**: 2025-10-22

---

## ✅ 前置檢查

- [ ] 已閱讀 `QUICKSTART.md`
- [ ] 已理解 `SUMMARY.md` 中的核心差異
- [ ] 確認 GPU 可用 (`nvidia-smi`)
- [ ] 確認數據路徑存在:
  ```bash
  ls -lh ../data/raw/box/
  ls -lh ../data/clean/box2/
  ```

---

## 🚀 執行步驟

### Step 1: 進入資料夾
```bash
cd /home/sbplab/ruizi/c_code/try
```

### Step 2: 檢查腳本權限
```bash
ls -lh run_token_denoising_frozen_codebook.sh
# 應該看到 -rwxr-xr-x (有 x 執行權限)
```

### Step 3: 執行訓練
```bash
bash run_token_denoising_frozen_codebook.sh
```

### Step 4: 監控訓練 (另一個終端)
```bash
tail -f ../logs/token_denoising_frozen_codebook_*.log
```

---

## 📊 關鍵指標監控

### Epoch 1-50
- [ ] Token Accuracy > 0% (確認模型在學習)
- [ ] Loss 下降趨勢明確
- [ ] 無 CUDA OOM 錯誤

### Epoch 50-100
- [ ] Token Accuracy > 10%
- [ ] Loss < 5.0
- [ ] Validation Loss 不上升 (無過擬合)

### Epoch 100-200
- [ ] Token Accuracy > 30%
- [ ] Loss < 3.0
- [ ] 音訊樣本可辨識

### Epoch 200+
- [ ] Token Accuracy > 50%
- [ ] Loss < 2.0
- [ ] 與現有模型對比

---

## 🔍 檢查點

### 每 10 Epochs
- [ ] 查看 `training_history.png` 曲線
- [ ] 檢查 Loss 是否平滑下降
- [ ] 檢查 Accuracy 是否穩定上升

### 每 50 Epochs
- [ ] 聽音訊樣本 (在 `../results/.../audio_samples/`)
- [ ] 檢查頻譜圖 (是否連續)
- [ ] 記錄關鍵指標

### 每 100 Epochs
- [ ] 保存檢查點
- [ ] 更新 REPORT.md
- [ ] 與現有模型對比

---

## 🐛 問題排查

### 問題 1: Token Accuracy 一直是 0%
**檢查**:
```python
# 在訓練腳本中添加 debug
print(f"Codebook requires_grad: {model.codebook.requires_grad}")  # 應該是 False
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")  # 應該是 1e-4
```

### 問題 2: CUDA Out of Memory
**解決**:
```bash
# 修改 run_token_denoising_frozen_codebook.sh
export TTT_BATCH_SIZE=4  # 從 8 改為 4
```

### 問題 3: Loss 不下降
**檢查**:
```bash
# 檢查數據
ls -lh ../data/raw/box/*.wav | head -5
ls -lh ../data/clean/box2/*.wav | head -5

# 確認語者分割正確
grep "訓練集大小" ../logs/*.log
```

---

## 📝 實驗記錄

### 開始時間
- [ ] 記錄: `________________`

### GPU 使用
- [ ] GPU ID: `____`
- [ ] 記憶體: `____ GB`

### 關鍵時間點
- [ ] Epoch 50 時間: `________`
- [ ] Epoch 100 時間: `________`
- [ ] Epoch 200 時間: `________`

### 最佳結果
- [ ] Best Token Accuracy: `_____%`
- [ ] Best Loss: `_______`
- [ ] Epoch: `______`

---

## 🎯 對比實驗

### 與現有模型比較

| 指標 | 現有模型 | Frozen Codebook | 差異 |
|------|----------|-----------------|------|
| Token Accuracy | ____% | ____% | ____% |
| Loss (Epoch 200) | ____ | ____ | ____ |
| 訓練時間/epoch | ____s | ____s | ____s |
| 音訊質量 (主觀) | ____/10 | ____/10 | ____ |

---

## ✅ 實驗完成

- [ ] 訓練至少 200 epochs
- [ ] 保存最佳模型
- [ ] 聽音訊樣本
- [ ] 更新 REPORT.md
- [ ] 提交 Git Commit
- [ ] 撰寫結論

---

## 📧 結果報告模板

```markdown
## Frozen Codebook 實驗結果

**實驗 ID**: frozen_codebook_YYYYMMDD_HHMM  
**訓練時間**: XX 小時  
**最佳 Epoch**: XXX

### 關鍵指標
- Token Accuracy: XX.XX%
- Validation Loss: X.XXX
- 訓練時間/epoch: XX 秒

### 與現有模型對比
- Token Accuracy 差異: ±XX%
- 音訊質量: [相當/更好/較差]
- 訓練效率: [更快/相當/較慢]

### 結論
[Frozen Codebook 是否有效? 為什麼?]

### 建議
[後續實驗方向]
```

---

## 🔗 快速連結

- [QUICKSTART.md](./QUICKSTART.md) - 快速開始指南
- [SUMMARY.md](./SUMMARY.md) - 完整總結
- [MODEL_COMPARISON_ANALYSIS.md](./MODEL_COMPARISON_ANALYSIS.md) - 詳細對比
- [README_FROZEN_CODEBOOK.md](./README_FROZEN_CODEBOOK.md) - 完整說明

---

**最後更新**: 2025-10-22  
**用途**: 實驗執行前的完整檢查清單  
**建議**: 印出來或在旁邊打開，逐項檢查
