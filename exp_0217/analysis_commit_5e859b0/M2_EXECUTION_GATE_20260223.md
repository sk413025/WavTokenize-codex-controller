# M2 Execution Gate (2026-02-23)

## 1) 目的

在 M1（`t453_min_weight 0.2 -> 0.3`）驗收 No-Go 後，定義是否可進入下一輪最小改動（以下稱執行 M2）之 gate，避免跳步改設定。

---

## 2) 進入執行 M2 的前置條件（已核對）

1. 不改設定先評估已完成。
   - 證據：`exp_0217/analysis_commit_5e859b0/FINAL_DECISION_REPORT_20260219.md`
2. 根因分析有主因且可追溯。
   - 主因：`H2_objective_mismatch`（strong support）
   - 證據：`exp_0217/analysis_commit_5e859b0/hypothesis_scoring.json`
3. M1 驗收已完成且為 No-Go。
   - 證據：`exp_0217/analysis_commit_5e859b0/M1_ACCEPTANCE_EVALUATION_20260222.md`
4. 比較基線可追溯且跨實驗完整。
   - `0206_plan_ori / 0206_v2 / 0216 / 0217`，且 `all_epoch300=true`
   - 證據：`exp_0217/runs/t453_weighted_epoch_20260217_104843/cross_experiment_300epoch_check_20260219.json`

---

## 3) 執行 M2 的硬限制（不可違反）

1. 單因子改動（only one parameter changes），且是**相對基線 `exp_0216`**。
2. 其他設定固定（seed/data/model/batch/lr/augmentation/training loop）。
3. 驗收門檻沿用 M1 標準，不可降標：
   - val mean `ΔPESQ` gain `>= +0.03`
   - val mean `ΔSTOI` gain `>= +0.01`
   - `best_val_mse` degradation `<= 1%`
   - `P2` pass；`P3` 監控（非硬 gate）
4. 結論必須附來源檔案與數值。

門檻來源：`exp_0217/analysis_commit_5e859b0/next_experiment_recommendation.md`

---

## 4) M2 候選軸（依證據排序）

### 優先軸（建議）
- **Loss 權重軸**（對應 H2 objective mismatch）
- 理由：
  1. H2 為 primary root cause（strong support）。
  2. M1 已測 sampling 軸且未達門檻（No-Go）。
  3. 訓練程式中 loss 權重為顯式可控參數，且直接參與總損失：
     - `L_total = lambda_quant*L_quant + intermediate_weight*L_inter + beta_commit*L_commit`
  4. 證據：
     - `exp_0217/analysis_commit_5e859b0/hypothesis_scoring.json`
     - `exp_0217/train_t453_weighted.py`
     - `exp_0217/runs/t453_m1_minw03_epoch100_debug/config.json`

### 暫不優先軸
- 再次調整 `t453_min_weight`（同 sampling 軸）
- 理由：M1 已在此軸執行且驗收未過。

---

## 5) 可執行的 M2 最小改動模板（尚未執行）

> 下面是「可執行模板」，不是已完成結果。

- 建議候選：`intermediate_weight: 0.03 -> 0.02`（單因子）
- 其餘參數固定與 `exp_0216` 基線相同（含 `t453_min_weight=0.2` 不改）。
- 執行長度：先 `100 epochs`（與 M1 同尺度，便於可比），後再視結果決定是否擴到 `300 epochs`。
- 驗收：沿用第 3 節門檻，並以固定 `N=100` 索引做音質比較。

---

## 6) 目前 Gate 判定

- **Gate 結論：Go（可進入執行 M2 規劃/啟動）**
- 但 M2 仍屬「待執行」狀態，尚未有新訓練結果。
