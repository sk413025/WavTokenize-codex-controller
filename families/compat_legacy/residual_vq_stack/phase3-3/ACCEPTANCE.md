# Phase 3-3 驗收：Hot-Code Branching 是否有效（與是否值得開 Phase 3-3）

> 本文件定義 Phase 3-3 的 pass/fail 與停損規則，避免做出「分裂了但沒有改善/不可歸因」的實驗。

## A. 產出物（必須）

每次 run 結束後，必須包含：
- `config.json`
- `summary.json`
- `metrics_history.json`
- `loss_history.json`
- `training_curves.png`
- `split_history.json`（Phase 3-3 新增）

缺任一項視為 FAIL（流程問題，需先修復）。

## B. 前置條件（建議 Gate）

若 Phase 3-2 尚未通過 P1（layer0 不再 top10=100%、used_codes 不再個位數），則：
- Phase 3-3 的任何結果都可能不可解釋（split 但主目標不吃 quantized）。
- 建議先完成 Phase 3-2，再啟動 Phase 3-3。

## C. P0：可執行性

- 不 crash、無 NaN/Inf
- split event 發生後訓練可繼續（不會因為 embedding 重置造成 shape/optimizer 錯誤）

## D. P1：Split “真的被用到”

對每個 split group（parent + children）在 split 後的第一個 eval step（例如 step+200）：

- `child_active_count >= 2`  
  - 定義：在 children 中，至少有 2 個 child 的 usage > 0（避免只有單一 child 接手原本的 collapse）

推薦更強版本（若 token 數夠）：
- group 內 usage entropy `H(p_children)` >= `0.7 * log(k)`

P1 未通過表示：split 沒有創造有效分流（子分支內立即 collapse 或根本不使用）。

## E. P2：對 collapse 指標有實質改善

在 final（step 1000）判定：

- `layer0_top10_mass` 相對 Phase 3（5x）明顯下降  
  - 5x 典型值：≈ 1.0
  - Phase 3-3 目標：<= 0.95（最低門檻）
- `layer0_used_codes` 相對 5x 明顯上升  
  - 5x 典型值：~7~10
  - Phase 3-3 目標：>= 20（最低門檻）
- `feature_mse <= 0.1`（仍維持可用性）

若 P1 通過但 P2 不通過：表示 split 有造成分流，但尚未解決整體 collapse（可能需要更大 k 或配合 commitment/EMA）。

## F. P3：回答核心假設（speech/noise 是否分開）

> P3 是用來回答你提出的假設，不一定要當作“成功”門檻，但應該要能提供可重現證據。

選一個 noise proxy（SNR 或 per-frame 差異）後，對 split group 的 children：

- `proxy_separation` 明顯 > 0  
  - 例：children 的 proxy 均值差異 >= 0.5 * pooled std（效果量）  
  - 或：KS test p-value 很小（僅作參考；要注意樣本數）
  - 或：mutual information > 門檻（例如 > 0.01 bits）

若 P2 達成但 P3 不達成：代表 split 幫助多樣性，但未必真的把 speech/noise 分開。

## G. 停損 / Pivot

若做完以下最小組合仍無法通過 P2，建議停止 Phase 3-3：

1) One-shot split（k=3）  
2) Periodic split（interval=200）  
3) Split + group balance（僅一種正則）  

Pivot 方向（擇一）：
- 直接處理 collapse 的根因（Phase 3-2 的 loss/commitment/EMA）
- 改用多頭單層 VQ / Gumbel-softmax / token supervision distillation

---

## H. 為什麼這與實驗目標相關

Phase 3 的失敗顯示：單看 feature_mse 並不能保證 token 表徵有用；token 可能只剩幾個，無法承載語義差異。  
Phase 3-3 的驗收同時要求：
- token 使用多樣性（避免 collapse）
- split group 內確實產生分流（否則只是形式上增加 code）
- 若可能，對 noise proxy 有分離跡象（直接驗證 “speech/noise 混在同一 code” 假設）

這些都直接對齊我們的終極目標：得到可離散化、可分辨、且 decoder 可用的音訊 token。

