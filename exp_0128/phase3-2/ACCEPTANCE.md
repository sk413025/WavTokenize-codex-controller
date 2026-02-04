# Phase 3-2 驗收：RVQ Fix 是否值得繼續（Pass/Fail Gate）

> 本文件用來「定義成功長什麼樣、什麼情況要停損/pivot」。
> 目標是把 Phase 3-2 變成可執行、可量化、可決策的流程。

## A. 產出物驗收（必須）

每次實驗 run 結束後，必須在對應 `run_exp6*_TIMESTAMP/` 內看到：
- `config.json`（包含所有 loss 權重、EMA/threshold 等）
- `summary.json`（final_metrics + success 判定）
- `metrics_history.json`
- `loss_history.json`
- `training_curves.png`

若缺任一項，視為 **FAIL（流程不合格）**，需先修好 logging/輸出再談模型好壞。

## B. 可靠性驗收（P0：先能跑完）

### P0 必要條件（任何 Phase 3-2 實驗都要滿足）
- 不 crash（包含 eval）
- 無 NaN/Inf（loss / audio / features）
- 1000 steps 可完整跑完（或符合早停規則並留下 artifacts）

## C. Collapse 緩解驗收（P1：最小可用）

> 這是 Phase 3-2 的第一個決策門檻：**只要 P1 過不了，就不值得做長跑或更複雜的加法**。

### P1 通過條件（任一設定在 step 200 達成即可）
以 layer0 為主（因為 layer0 與 baseline 單層 VQ 最可比）：

- `layer0_top10_mass <= 0.95`（不再 100% 壟斷）
- `layer0_used_codes >= max(0.02 * K, 20)`，其中 `K=rvq_codebook_size`
  - 1024：至少 20 codes（或更高）
  - 512：至少 20 codes（≈3.9%）
- `feature_mse <= 0.1`（仍維持基本對齊能力）

若 step 200 同時滿足：
- `layer0_top10_mass > 0.95` 且 `layer0_used_codes < 0.01 * K`
則標記 **collapse_flag=true** 並可早停（此 run 視為 FAIL）。

## D. 目標達成驗收（P2：值得繼續 RVQ 路線）

### P2 通過條件（在 step 1000 以 final_metrics 判定）

- `layer0_entropy >= 5.0`
- `layer0_top10_mass <= 0.5`
- `layer0_used_codes >= 0.10 * K`
- `joint_diversity >= 0.30`（多層 RVQ 應該帶來組合多樣性；5c baseline 已達 ~0.33，但 layer0 仍崩）
- `feature_mse <= 0.1`

> P2 的意義：不僅「不 collapse」，而且有足夠 code usage 讓 tokens 具備表徵力。

## E. 伸展目標（P3：對齊原 Phase 3 的嚴格成功判準）

若 P2 達成，再以原 Phase 3 目標作為 stretch：
- `layer0_entropy > 6.5`
- `layer0_top10_mass < 0.15`
- `joint_diversity > 0.7`
- `feature_mse < 0.1`

P3 未達成不等於失敗；P2 達成即代表 RVQ 路線值得進入更長訓練與更完整音質評估。

## F. Pivot（放棄 RVQ 路線）條件

若完成以下最小探索仍無法通過 P1，建議 **停止 RVQ**：

1) Exp 6a（quantized alignment + encoder commitment）  
2) Exp 6b（β sweep 至少 4 組）  
3) Exp 6c（EMA + dead-code reset 至少 2 組 threshold）

**Pivot 觸發**：
- 上述組合全部 FAIL（P1 無一通過）

Pivot 後建議方向（擇一）：
- 多頭獨立單層 VQ（避免 RVQ residual dynamics）
- Gumbel-Softmax / soft quantization
- 直接 distill teacher codes（以 token supervision 為主）
- 重新設計 encoder（提升可量化性），再回來做 VQ

---

## G. 與實驗目標的關聯（為什麼用這些門檻）

Phase 1/2 已證實：collapse 不是資料分布或簡單正則能解；Phase 3 嘗試架構解法但失敗。  
因此 Phase 3-2 的驗收重點必須同時檢查：

- **多樣性（防 collapse）**：`top10_mass/used_codes/entropy/joint_diversity`
- **可用性（不只是亂用 codes）**：`feature_mse` 不可失控

只要能先達到 P1/P2，就表示「quantizer 被拉回訓練主路徑」，RVQ 才符合我們要的：**穩定、可離散化、可用於下游的 audio token 表徵**。

