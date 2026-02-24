# Commit 5e859b0 最終決策報告（執行＋管控）

日期：2026-02-19  
範圍：`exp_0216/runs/augmented_long_20260216`（不改設定先評估）

## 1) 根因分析（證據鏈）

### Primary：H2 目標函數與感知音質不一致（Strong Support）
- 事實：在 `N=100`、`epoch={050,100,150,200,220,250,300}` 下，train/val 的 `ΔPESQ`、`ΔSTOI` 全部為負。
- 事實：`feature_mse` 改善，但感知指標未轉正。
- 證據：
  - `exp_0217/analysis_commit_5e859b0/audio_quality_by_epoch.md`
  - `exp_0217/analysis_commit_5e859b0/statistical_tests_summary.md`

### Secondary：H1 資料分布差異（Partial Support）
- 事實：T453 分桶在 val@epoch300 出現顯著差異（FDR 後仍顯著的比較存在）。
- 證據：
  - `exp_0217/analysis_commit_5e859b0/stratified_quality_report.md`
  - `exp_0217/analysis_commit_5e859b0/statistical_tests_summary.md`

### Secondary：H3 容量/量化瓶頸（Partial Support）
- 事實：`with_vq` 相對 `no_vq` 有負向差距，但不足以單獨定主因。
- 證據：
  - `exp_0216/PESQ_STOI_novq_comparison.json`
  - `THESIS_ARCHITECTURE.md`

### H4 增強副作用（Insufficient Evidence）
- 事實：無直接對照 ablation 證據可歸因。
- 證據：
  - `exp_0217/analysis_commit_5e859b0/hypothesis_scoring.json`

---

## 2) 是否修改設定（Go/No-Go）

- 決策：**Go（進入最小改動設計階段；尚未執行改訓練）**
- 原因：
  1. SPEC 必要輸出已達 `10/10`。
  2. Trigger A 已達成（val `ΔPESQ/ΔSTOI` 在 N>=100 下持續非正）。
  3. Trigger B 已達成（統計檢定顯示 `p < 0.05`）。

證據：
- `exp_0217/analysis_commit_5e859b0/spec_coverage_status_20260219.json`
- `exp_0217/analysis_commit_5e859b0/next_experiment_recommendation.md`
- `exp_0217/analysis_commit_5e859b0/statistical_tests_summary.json`

---

## 3) 最小改動方案與驗收標準（若執行）

### 最小改動（單因子）
- 僅調整：`t453_min_weight: 0.2 -> 0.3`
- 其他條件固定：seed、資料、模型、batch、lr、增強與訓練流程。

### 驗收標準
1. val mean `ΔPESQ` 相對基線提升 `>= +0.03`  
2. val mean `ΔSTOI` 相對基線提升 `>= +0.01`  
3. `best_val_mse` 退化 `<= 1%`  
4. `P2` 持續通過；`P3` 維持監控，不作硬 gate

基線來源：
- `exp_0217/analysis_commit_5e859b0/audio_quality_by_epoch.md`

---

## 4) 驗收進度判定

### 已滿足
- 文件對齊（M1）
- 現況盤點（M2）
- 根因分析（M3）
- 決策輸出（M4）
- 最小改動規格（M5）

### 唯一剩餘缺口（可追溯）
- `epoch_222` 無 checkpoint（`checkpoint_epoch222.pt` 不存在），該節點音質數值不可得。
- 本報告未以替代推論填補，已明確標記 `missing_checkpoint`。

證據：
- `exp_0217/analysis_commit_5e859b0/audio_quality_by_epoch.md`
- `exp_0217/analysis_commit_5e859b0/hypothesis_scoring.json`

### 補件進展（本輪新增）
- 已完成 `epoch220 -> epoch222` 同設定 replay，產生補件 checkpoint 與音質評估：
  - `exp_0217/analysis_commit_5e859b0/replay_from220_to222/checkpoints/checkpoint_epoch222_replay.pt`
  - `exp_0217/analysis_commit_5e859b0/audio_quality_epoch222_replay.md`
- 補件結果（N=100）：
  - train: `ΔPESQ=-0.4894`, `ΔSTOI=-0.0122`
  - val: `ΔPESQ=-0.3004`, `ΔSTOI=-0.0591`
- 說明：此為補件替代證據，可用於驗收連續性；但不等同原始 run 的原生 epoch222 artifact。

---

## 5) M1 執行驗收結果（2026-02-22 更新）

M1 執行實驗：
- Run: `exp_0217/runs/t453_m1_minw03_epoch100_debug`
- 設定：`t453_min_weight: 0.2 -> 0.3`（其餘固定）
- 完成度：`100 epochs`，checkpoint `010~100` 全部存在

驗收結果（對照第 3 節門檻）：
1. Val `ΔPESQ` 提升 `>= +0.03`
   - 結果：**Fail**（`epoch050/060/100` 相對 baseline 的增益均為負值）
2. Val `ΔSTOI` 提升 `>= +0.01`
   - 結果：**Fail**（`epoch050/060/100` 相對 baseline 的增益均為負值）
3. `best_val_mse` 退化 `<= 1%`
   - Baseline: `0.038064`（exp_0216）
   - M1 best: `0.038994`
   - 退化：`+2.44%`
   - 結果：**Fail**
4. `P2` 持續通過
   - 結果：**Pass**
5. `P3` 監控（非硬 gate）
   - 結果：`fail`（監控紀錄）

結論：
- M1 最小改動方案之完整驗收判定：**No-Go（未達門檻）**。

證據：
- `exp_0217/analysis_commit_5e859b0/M1_ACCEPTANCE_EVALUATION_20260222.md`
- `exp_0217/analysis_commit_5e859b0/audio_quality_m1_epoch050.json`
- `exp_0217/analysis_commit_5e859b0/audio_quality_m1_epoch060.json`
- `exp_0217/analysis_commit_5e859b0/audio_quality_m1_epoch100.json`
- `exp_0217/runs/t453_m1_minw03_epoch100_debug/summary.json`

---

## 6) M2 定義補件（2026-02-23 更新）

### M2 的兩個語意（明確區分）
1. 里程碑 M2：現況盤點（current inventory）
2. SPEC 模組 M2：PESQ/STOI 感知音質評估

### 現況判定
- 里程碑 M2：**已完成**（`M2_current_inventory=100`）
- SPEC 模組 M2：**已完成**（含 `epoch222` 缺口註記與 replay 補件）

### 補充驗收證據
- 跨實驗比較集合已涵蓋：`0206_plan_ori / 0206_v2 / 0216 / 0217`
- 且皆有 `300 epoch` 證據：`all_epoch300=true`

### 文件入口
- `exp_0217/analysis_commit_5e859b0/M2_DEFINITION_20260223.md`
- `exp_0217/analysis_commit_5e859b0/M2_EXECUTION_GATE_20260223.md`
- `exp_0217/runs/t453_weighted_epoch_20260217_104843/cross_experiment_300epoch_check_20260219.json`

---

## 7) M2 執行驗收結果（2026-02-23 更新）

M2 執行實驗：
- Run: `exp_0217/runs/t453_m2_interw002_epoch100_20260223`
- 設定：`intermediate_weight: 0.03 -> 0.02`（單因子；其餘固定）
- 完成度：`100 epochs`，checkpoint `010~100` 已生成

驗收結果（沿用第 3 節門檻）：
1. Val `ΔPESQ` 提升 `>= +0.03`
   - `epoch050` 增益：`+0.004286`
   - `epoch100` 增益：`-0.006192`
   - 結果：**Fail**
2. Val `ΔSTOI` 提升 `>= +0.01`
   - `epoch050` 增益：`-0.006757`
   - `epoch100` 增益：`-0.005400`
   - 結果：**Fail**
3. `best_val_mse` 退化 `<= 1%`
   - Baseline: `0.038064`（exp_0216）
   - M2 best: `0.039052`
   - 退化：`+2.59%`
   - 結果：**Fail**
4. `P2` 持續通過
   - 結果：**Pass**
5. `P3` 監控（非硬 gate）
   - 結果：`fail`（監控紀錄）

結論：
- M2 最小改動方案之完整驗收判定：**No-Go（未達門檻）**。

證據：
- `exp_0217/analysis_commit_5e859b0/M2_ACCEPTANCE_EVALUATION_20260223.md`
- `exp_0217/analysis_commit_5e859b0/audio_quality_m2_epoch050.json`
- `exp_0217/analysis_commit_5e859b0/audio_quality_m2_epoch100.json`
- `exp_0217/runs/t453_m2_interw002_epoch100_20260223/summary.json`
