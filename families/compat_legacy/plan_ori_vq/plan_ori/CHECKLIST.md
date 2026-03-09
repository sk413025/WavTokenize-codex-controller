# Plan Original - 執行清單

**快速參考**: 按順序勾選以下項目

---

## 📖 第一步：閱讀理解 (30 分鐘)

- [ ] 閱讀 `README.md` - 理解方案背景
- [ ] 閱讀 `PLAN.md` - 理解實驗計劃
- [ ] 閱讀 `SPEC.md` - 理解技術規格
- [ ] 閱讀 `AGENT_TASK.md` - 理解任務細節

---

## 💻 第二步：實現代碼 (4-6 小時)

### 2.1 模型實現

- [ ] 創建 `models_single_vq_ema.py`
- [ ] 實現 `SingleVQWithEMA` class
  - [ ] `__init__` 方法
  - [ ] `_ema_update` 方法
  - [ ] `forward` 方法
  - [ ] `get_codebook_usage` 方法
- [ ] 實現 `TeacherStudentSingleVQ` class
  - [ ] 繼承 `TeacherStudentIntermediate`
  - [ ] 載入預訓練 codebook
  - [ ] 替換 quantizer
- [ ] Code review 自檢

### 2.2 訓練腳本

- [ ] 創建 `train_single_vq_ema.py`
- [ ] 基於 `train_long_v2.py` 修改
- [ ] 修改模型初始化
- [ ] 確認 loss 計算正確
- [ ] 確認 metrics logging

### 2.3 執行腳本

- [ ] 創建 `run_exp_ori_short.sh`
- [ ] 設定超參數
- [ ] 測試腳本可執行

---

## 🧪 第三步：測試驗證 (2-3 小時)

### 3.1 單元測試

- [ ] 創建 `test_single_vq_ema.py`
- [ ] 測試初始化
- [ ] 測試 forward pass
- [ ] 測試 EMA update
- [ ] 測試 dead-code reset
- [ ] **執行**: `python families/compat_legacy/plan_ori_vq/plan_ori/test_single_vq_ema.py`
- [ ] 所有測試通過 ✅

### 3.2 Smoke Test

- [ ] **執行 10 steps**:
  ```bash
  python families/compat_legacy/plan_ori_vq/plan_ori/train_single_vq_ema.py \
    --output_dir test_smoke \
    --steps 10 \
    --batch_size 2
  ```
- [ ] 無錯誤完成
- [ ] 檢查輸出文件生成
- [ ] 檢查 loss 值合理

---

## 🚀 第四步：Short-run 實驗 (8-10 小時)

### 4.1 啟動訓練

- [ ] 確認 GPU 可用: `nvidia-smi`
- [ ] 確認資料路徑存在
- [ ] **執行**: `bash families/compat_legacy/plan_ori_vq/plan_ori/run_exp_ori_short.sh 0`
- [ ] 記錄開始時間

### 4.2 監控進度

每 200 steps 檢查:
- [ ] Step 200: 記錄 metrics
  - [ ] entropy: _____
  - [ ] top10: _____
  - [ ] used: _____
  - [ ] P1 gate: ✅/❌
- [ ] Step 400: 記錄 metrics
- [ ] Step 600: 記錄 metrics
- [ ] Step 800: 記錄 metrics
- [ ] Step 1000: 記錄 final metrics
  - [ ] entropy ≥5.0: ✅/❌
  - [ ] top10 ≤0.5: ✅/❌
  - [ ] used ≥410: ✅/❌
  - [ ] **P2 gate**: ✅/❌

### 4.3 問題處理

如遇問題:
- [ ] 記錄錯誤訊息
- [ ] 檢查 log 文件
- [ ] 檢查 GPU memory
- [ ] 查看參考資源

---

## 📊 第五步：分析結果 (2-3 小時)

### 5.1 運行分析腳本

- [ ] 創建 `analyze_results.py`
- [ ] **執行**: `python families/compat_legacy/plan_ori_vq/plan_ori/analyze_results.py <output_dir>`
- [ ] 查看 metrics curves
- [ ] 查看 P2/P3 判定

### 5.2 對比分析

填寫對比表格:

| Method | Entropy | Top-10 | Used | Status |
|--------|---------|--------|------|--------|
| Baseline | 6.07 | 19.7% | 740 | ❌ |
| RVQ | 9.03 | 15.8% | 1089/layer | ✅ |
| Plan Ori | _____ | _____ | _____ | _____ |

---

## 📝 第六步：撰寫文檔 (2-3 小時)

### 6.1 填寫 RESULTS.md

- [ ] 創建 `RESULTS.md`
- [ ] 填寫 final metrics
- [ ] 填寫對比分析
- [ ] 回答科學問題:
  - [ ] Q1: 預訓練 + EMA 能否避免 collapse？
  - [ ] Q2: Warm start vs Cold start？
  - [ ] Q3: 單層 vs 多層必要性？
- [ ] 填寫決策與後續步驟

### 6.2 更新其他文檔

- [ ] 更新 `README.md` 狀態
- [ ] 更新 `families/compat_legacy/plan_ori_vq/README.md`
- [ ] 提交 git commit

---

## 🎯 第七步：決策點

### 如果 P2 PASSED ✅

- [ ] 決定是否進行 long-run (300 epochs)
- [ ] 評估 GPU 資源
- [ ] 評估時間預算
- [ ] 與 RVQ 進行詳細對比
- [ ] 考慮作為主要方案

### 如果 P2 FAILED ❌

- [ ] 深入分析失敗原因
- [ ] 寫入 ablation study
- [ ] 終止方案 A
- [ ] 返回 RVQ 方案
- [ ] 總結經驗教訓

---

## ✅ 最終驗收

### 代碼品質

- [ ] 所有單元測試通過
- [ ] Smoke test 成功
- [ ] Code 有註解
- [ ] 符合 SPEC.md

### 實驗完整性

- [ ] Short-run 1000 steps 完成
- [ ] Metrics 完整記錄
- [ ] 視覺化圖表生成
- [ ] P2 gate 判定明確

### 文檔完整性

- [ ] RESULTS.md 填寫完整
- [ ] 分析結論清晰
- [ ] 後續決策明確
- [ ] 所有文檔已更新

---

## 📌 快速命令參考

```bash
# 1. 單元測試
python families/compat_legacy/plan_ori_vq/plan_ori/test_single_vq_ema.py

# 2. Smoke test
python families/compat_legacy/plan_ori_vq/plan_ori/train_single_vq_ema.py \
  --output_dir test_smoke --steps 10 --batch_size 2

# 3. Short-run
bash families/compat_legacy/plan_ori_vq/plan_ori/run_exp_ori_short.sh 0

# 4. 監控進度
tail -f <output_dir>/train.log

# 5. 分析結果
python families/compat_legacy/plan_ori_vq/plan_ori/analyze_results.py <output_dir>

# 6. 檢查 GPU
nvidia-smi
```

---

## 📞 需要幫助？

遇到問題時查看:
1. `AGENT_TASK.md` - 詳細指引
2. `SPEC.md` - 技術規格
3. `exp_0128/phase3/residual_vq/models_rvq.py` - RVQ 參考實現
4. `exp_0128/phase3-2/SUMMARY.md` - Phase 3-2 經驗

---

**提示**: 按順序勾選，完成一項再進行下一項！

**預計總時間**: 2-3 天 (實現 6h + 測試 3h + 實驗 10h + 分析 3h + 文檔 3h)

**最後更新**: 2026-02-11
