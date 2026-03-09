# M2 Definition (2026-02-23)

## 1) 名詞定義（避免混淆）

本專案目前同時存在兩個「M2」語意：

1. **里程碑 M2（分析流程）**
   - 定義：現況盤點（training progress / artifacts / missing list）
   - 來源：`spec_coverage_status_20260219.json` 的 `M2_current_inventory`
2. **模組 M2（SPEC 分析模組）**
   - 定義：感知音質評估（PESQ/STOI）
   - 來源：`families/deps/t453_weighted_baseline/COMMIT_5e859b0_VAL_AUDIO_ANALYSIS_SPEC.md`

本文件將兩者都明確化，以供後續驗收使用。

---

## 2) 里程碑 M2（現況盤點）定義與狀態

### 定義
- 盤點項目：run 完整性、輸出可用性、缺失清單、可追溯證據。

### 現況
- `M2_current_inventory = 100`
- 來源：`families/deps/t453_weighted_baseline/analysis_commit_5e859b0/spec_coverage_status_20260219.json`

### 盤點結論
- 已有可稽核的 run / summary / history / log 對齊資料。
- M2（里程碑語意）判定：**已完成**。

---

## 3) SPEC 模組 M2（PESQ/STOI）定義與狀態

### 規格定義（原文對齊）
- Epoch: `050, 100, 150, 200, 220, 222(best), 250, 300(final)`
- Split: `train`, `val`
- 每個 split/epoch 主要評估樣本數：`N>=200`，資源不足可 `N>=100`
- 指標：`PESQ_noisy`, `PESQ_recon`, `ΔPESQ`, `STOI_noisy`, `STOI_recon`, `ΔSTOI`，含 mean/std/95% CI
- 規格來源：`families/deps/t453_weighted_baseline/COMMIT_5e859b0_VAL_AUDIO_ANALYSIS_SPEC.md`

### 現況
- 已產出：`audio_quality_by_epoch.json/.md`
- 主要分析採用：`N=100`（可追溯）
- `epoch222` 原生 checkpoint 缺失，已以 replay 補件並明確標註替代證據。
- 證據：
  - `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/audio_quality_by_epoch.md`
  - `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/audio_quality_epoch222_replay.md`

### 判定
- M2（SPEC 模組語意）：**已完成（含缺口註記與補件）**。

---

## 4) 跨實驗可比較性（驗收補充）

### 定義
- 需確認比較集合包含：`0206_plan_ori`, `0206_v2`, `0216`, `0217`
- 且各 run 皆有 300 epoch 佐證（summary/history/log）

### 現況
- `comparison_set` 已包含上述四組
- `all_epoch300 = true`
- 證據：`families/deps/t453_weighted_baseline/runs/t453_weighted_epoch_20260217_104843/cross_experiment_300epoch_check_20260219.json`

---

## 5) 與目前決策的關係

- M1 最小改動（`t453_min_weight 0.2 -> 0.3`）驗收結果為 **No-Go**。
- 此 No-Go 不影響「里程碑 M2」與「SPEC 模組 M2」已完成的事實；
  影響的是後續是否進入下一輪設定改動。
- 證據：`families/deps/t453_weighted_baseline/analysis_commit_5e859b0/M1_ACCEPTANCE_EVALUATION_20260222.md`
