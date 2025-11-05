# Git Commit 策略規劃

## Commit 範圍

### 要包含的檔案

**1. 實驗報告（主要成果）**
```
PLATEAU_MECHANISM_ANALYSIS.md          # 詳細機轉分析，含 ASCII 圖示
PLATEAU_DIAGNOSIS_SUMMARY.md           # 簡潔摘要
TRAINING_PLATEAU_DIAGNOSIS_20251105.md # 完整診斷報告
EXPERIMENT_REPRODUCTION_GUIDE.md       # 實驗重現指南
```

**2. 分析工具（可重現性）**
```
done/exp/analyze_token_distribution.py        # Token 分布分析工具
done/exp/analyze_token_accuracy_inference.py  # Accuracy 反推工具
```

**3. Commit Message**
```
COMMIT_MESSAGE_DRAFT.md  # 完整 commit message
```

**4. 總報告更新（可選）**
```
REPORT.md  # 更新章節：訓練平台期診斷 (2025-11-05)
```

### 不包含的檔案

**訓練腳本（已在之前 commit）**
```
done/exp/train_zeroshot_full_cached_analysis.py  # 已在 fa1b686
done/exp/data_zeroshot.py
done/exp/model_zeroshot.py
```

**數據檔案（太大）**
```
done/exp/data/train_cache.pt  # 1.1GB
done/exp/data/val_cache.pt    # 400MB
results/zeroshot_100epochs_20251105_002300/  # 訓練中
```

## Commit 順序與分類

### 選項 A: 單一大 Commit（推薦）

**優點**: 
- 所有分析成果完整記錄
- Commit message 涵蓋所有細節
- 易於追蹤完整實驗流程

**Commit Title**:
```
診斷訓練平台期：Token Distribution Mismatch 導致 Val Acc 僅 37%
```

**檔案清單**:
```
PLATEAU_MECHANISM_ANALYSIS.md
PLATEAU_DIAGNOSIS_SUMMARY.md
TRAINING_PLATEAU_DIAGNOSIS_20251105.md
EXPERIMENT_REPRODUCTION_GUIDE.md
COMMIT_MESSAGE_DRAFT.md
done/exp/analyze_token_distribution.py
done/exp/analyze_token_accuracy_inference.py
REPORT.md (updated)
```

### 選項 B: 多個小 Commit

**Commit 1: 分析工具**
```
Title: 新增 Token 分布分析與準確率反推工具
Files:
  done/exp/analyze_token_distribution.py
  done/exp/analyze_token_accuracy_inference.py
```

**Commit 2: 實驗報告**
```
Title: 訓練平台期診斷報告：Distribution Mismatch 分析
Files:
  TRAINING_PLATEAU_DIAGNOSIS_20251105.md
  EXPERIMENT_REPRODUCTION_GUIDE.md
```

**Commit 3: 機轉分析**
```
Title: Token Distribution Mismatch 機轉模型與改進方向
Files:
  PLATEAU_MECHANISM_ANALYSIS.md
  PLATEAU_DIAGNOSIS_SUMMARY.md
  COMMIT_MESSAGE_DRAFT.md
```

**不推薦原因**: 實驗是一個完整單元，拆分會失去連貫性

## 詳細 Commit Message

見 `COMMIT_MESSAGE_DRAFT.md`，結構如下：

1. **標題** (50 字以內)
2. **實驗背景** - 訓練平台期現象
3. **實驗動機** - 需要診斷根本原因
4. **實驗目的** - 找出問題、建立機轉、提出方向
5. **實驗方法** - 4 種分析方法
6. **實驗結果** - 4 項核心發現 + 數據
7. **實驗解讀** - 機轉模型、因果鏈、數據支持
8. **改進方向** - 3 項具體方案
9. **重現指南** - 環境需求、步驟、檔案
10. **關鍵洞察** - 本質困難
11. **下一步** - 後續實驗計劃

## 執行步驟

### Step 1: 更新 REPORT.md

在 `REPORT.md` 新增章節：

```markdown
## 實驗 7: 訓練平台期診斷 (2025-11-05)

### 問題
100-epoch 訓練中發現嚴重平台期：Train Acc 54%, Val Acc 37%

### 診斷結果
見詳細報告：
- `PLATEAU_MECHANISM_ANALYSIS.md` - 機轉分析
- `PLATEAU_DIAGNOSIS_SUMMARY.md` - 簡潔摘要
- `TRAINING_PLATEAU_DIAGNOSIS_20251105.md` - 完整報告

### 核心發現
1. Token 453 在 Val 佔 18.65%，Train 僅 13.57%
2. 15 個 tokens 有 distribution mismatch，累計 10.94%
3. Mismatch 只能解釋 30% 的 gap，70% 來自泛化不足
4. Padding 不是問題（Val padding <0.3%）

### 下一步
實作 Speaker-Adaptive Token Distribution Modeling
```

### Step 2: Stage 所有檔案

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised

git add PLATEAU_MECHANISM_ANALYSIS.md
git add PLATEAU_DIAGNOSIS_SUMMARY.md
git add TRAINING_PLATEAU_DIAGNOSIS_20251105.md
git add EXPERIMENT_REPRODUCTION_GUIDE.md
git add COMMIT_MESSAGE_DRAFT.md
git add done/exp/analyze_token_distribution.py
git add done/exp/analyze_token_accuracy_inference.py
git add REPORT.md
```

### Step 3: 複製 Commit Message

```bash
# 方法 1: 直接使用檔案內容
git commit -F COMMIT_MESSAGE_DRAFT.md

# 方法 2: 手動編輯
git commit
# 然後在編輯器中貼上 COMMIT_MESSAGE_DRAFT.md 的內容
```

### Step 4: 驗證 Commit

```bash
# 檢查 commit 內容
git show HEAD

# 檢查 commit 的檔案清單
git show --name-only HEAD

# 確認 commit message 完整性
git log -1 --pretty=format:"%B" HEAD
```

### Step 5: Push (可選)

```bash
git push origin main
```

## Commit Message 最佳實踐

### 標題格式
```
診斷訓練平台期：Token Distribution Mismatch 導致 Val Acc 僅 37%
```

### Body 格式（從 COMMIT_MESSAGE_DRAFT.md）

**必須包含**:
- ✅ 實驗背景（Training plateau at epoch 38）
- ✅ 實驗動機（需要診斷根本原因）
- ✅ 實驗方法（4 種分析）
- ✅ 實驗結果（4 項核心發現 + 數據）
- ✅ 實驗解讀（機轉模型、因果鏈）
- ✅ 重現步驟（完整指令）
- ✅ 改進方向（3 項方案）

**可選包含**:
- 待驗證假設
- 關鍵洞察
- 下一步行動

## 時間戳記錄

- **實驗開始**: 2025-11-05 00:23:00 (啟動 100-epoch 訓練)
- **診斷開始**: 2025-11-05 01:30:00 (Epoch 38 觀察到平台期)
- **分析完成**: 2025-11-05 02:30:00 (完成所有分析與報告)
- **Commit 時間**: 將為執行 `git commit` 的時間

## 檔案大小檢查

```bash
# 確認要 commit 的檔案大小
du -h PLATEAU_MECHANISM_ANALYSIS.md           # ~20KB
du -h PLATEAU_DIAGNOSIS_SUMMARY.md            # ~8KB
du -h TRAINING_PLATEAU_DIAGNOSIS_20251105.md  # ~15KB
du -h EXPERIMENT_REPRODUCTION_GUIDE.md        # ~10KB
du -h done/exp/analyze_token_distribution.py  # ~10KB
du -h done/exp/analyze_token_accuracy_inference.py  # ~6KB
du -h COMMIT_MESSAGE_DRAFT.md                 # ~12KB

# 總計約 80KB，符合 commit 大小要求
```

## 相關 Commits 參考

**Previous Commit** (fa1b686):
```
完成 3-epoch 測試訓練：Zero-Shot Speaker Denoising with Cached Data

- 實驗背景：驗證 GPU 3 cached data 訓練流程
- 方法：3-epoch 快速測試，使用 14/4 speaker split
- 結果：Train Acc 39.6%, Val Acc 29.3%
- 重現：見 EXPERIMENT_LOG_20251104.md
```

**This Commit** (待建立):
```
診斷訓練平台期：Token Distribution Mismatch 導致 Val Acc 僅 37%

[從 COMMIT_MESSAGE_DRAFT.md 複製完整內容]
```

**Next Commit** (規劃中):
```
實作 Speaker-Adaptive Token Distribution Modeling

- 背景：診斷發現 distribution mismatch 問題
- 方法：新增 speaker → token distribution prior mapping
- 預期：Val Acc 提升至 45-50%
```

## 檢查清單

執行 commit 前確認：

- [ ] 所有分析報告檔案已建立
- [ ] 分析工具腳本可執行
- [ ] Commit message 包含完整實驗記錄
- [ ] REPORT.md 已更新
- [ ] 檔案大小合理（總計 <100KB）
- [ ] 重現步驟已驗證
- [ ] 無敏感資料或大檔案
- [ ] Commit message 符合中文 Google docstring 風格

## 建議的完整 Commit Message

見 `COMMIT_MESSAGE_DRAFT.md` 完整內容（約 350 行）

## 參考文檔

- `.github/copilot-instructions.md` - Commit message 規範
- `EXPERIMENT_REPRODUCTION_GUIDE.md` - 重現步驟
- `PLATEAU_MECHANISM_ANALYSIS.md` - 詳細分析
- `PLATEAU_DIAGNOSIS_SUMMARY.md` - 簡潔摘要

