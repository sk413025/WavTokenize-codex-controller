# 提示詞（#2）：演算法實驗工程師 Agent — Exp0124「短跑 invariance 原型」與決策

你是「演算法實驗工程師 Agent」。任務：根據 Exp0124 的文件與目前結論，實作並跑完 **Primary Proposed Fix 的短跑驗證**（teacher‑anchored noise‑invariant training），用最小成本判斷是否值得投入長訓練或需改走 disentanglement/factorization。

---

## 0) 必讀文件（開工前先讀完）

1. 規劃：`exp_0124/27e564a_valid_token_collapse_planning.md`
2. 規格：`exp_0124/27e564a_valid_token_collapse_spec.md`
3. 驗收：`exp_0124/27e564a_valid_token_collapse_acceptance.md`
4. 現況結論（含 Primary Proposed Fix）：`exp_0124/token_collapse_27e564a/CONCLUSION.md`
5. 現況進度：`exp_0124/token_collapse_27e564a/PROGRESS.md`

參考（上一輪 gap/anti-collapse 的既有工具與結果）：
- `epx_0123/train_valid_gap_58a9b71/CONCLUSION.md`
- `epx_0123/train_valid_gap_58a9b71/ablation_anticollapse_summary*.md`
- `epx_0123/train_valid_gap_58a9b71/ablation_anticollapse*/lambda_*/metrics.json`

---

## 1) 你的目標（你要交付的決策）

在短跑（2k samples、800–1000 optimizer steps）的成本內，產出「可否證」結論：
1) `L_invar` 是否真的能降低 noise‑sensitivity（降低 controlled pairs 的 `token_change_rate`）？  
2) `L_invar` 是否能改善 collapse 指標（entropy↑、top‑k mass↓、KL↓）？  
3) strict acc 是否至少不惡化，且有機會在長跑改善？  

最後你必須在 `exp_0124/token_collapse_27e564a/CONCLUSION.md` 更新一段 **Decision**：
- **Go**：值得投入長訓練驗證 Primary Fix  
- **No‑Go / Pivot**：不值得，改做 probe 或 disentanglement/factorization（並說明原因）

---

## 2) 產出位置（全部統一放這裡）

建立並只使用：
- `exp_0124/token_collapse_27e564a/invariance_short_run/`

並在該目錄內新增：
- `PROGRESS.md`（此短跑專用進度）
- `runs/`（每個 λ 一個子目錄）
- `summary.json`、`summary.md`（總表）

同時你需要回填總進度到：
- `exp_0124/token_collapse_27e564a/PROGRESS.md`

---

## 3) 實作規範（避免「不小心做出常數 token」）

你要實作的目標函數是「**teacher anchor + invariance**」，不是純 invariance：

- `L_anchor`（必做）：student(x_noisy) 對齊 teacher(clean) 的 **token-level / per-frame** 對齊  
  - 可用 teacher codes 作 target 的 CE/KL（建議用 soft logits/soft assignment 版本更穩）
- `L_invar`（要測的變因）：同一 clean 的兩個 noisy views（x1/x2）之 student 表徵一致性  
  - 建議在 **soft assignment / logits / continuous features** 上做（不要直接 hard codes 相等）
  - 必要時做 sequence-level global shift 對齊後再算一致性（避免被對齊誤差污染）
- （可選）`L_div`：輕量 diversity regularizer（只用來防 trivial collapse；不要壓過 anchor）

你必須做 **ablation**：只改 `λ_invar`，其它都固定，否則無法判斷成效是否來自 invariance。

---

## 4) 你要跑的實驗矩陣（最小但可判定）

固定設定（除非遇到 OOM）：
- data：train subset 2k（固定 seed 抽樣），val subset 500（固定 seed 抽樣）
- steps：800–1000 optimizer steps（非 micro steps）
- batch_size / grad_accum：依 GPU 調整，但要在 config 記錄「有效 batch」

ablation（至少 3 點）：
- `λ_invar = 0.0`（anchor only baseline）
- `λ_invar = small`（例如 0.05）
- `λ_invar = mid`（例如 0.1）

（可選第四點）：
- `λ_invar = high`（例如 0.2；僅用來確認是否會推向 collapse）

每個 setting 都要跑完同一套 eval（見下節）。

---

## 5) 必跑 eval 指標（每個 setting 都要有）

在同一個 checkpoint（短跑結束）上評估並輸出到該 setting 的 `metrics.json`：

### (A) strict token acc（主指標之一）
- `val_strict_acc_frame_weighted`
- `train_strict_acc_frame_weighted`（只需同 subset）

### (B) collapse 指標（一定要）
- val：entropy、top‑k mass、unique、KL(student||teacher)
- train：同上（對照）

### (C) noise sensitivity（對應你們的 Top-1：H‑SUP/joint encoding）
- controlled pairs：`token_change_rate`（同 clean，不同 noise views）
- 至少 N=20–50 pairs（固定 seed）

### (D) VQ margin（對應 Top-2：H‑VQ）
- train/val margin 的 mean/p50/p90（或至少 mean/p50）
- 若 margin 沒改善，代表 invariance 可能沒碰到 root cause 或需要更強 anchor/量化穩定性策略

---

## 6) 成功/失敗判準（短跑用，避免只看 strict acc）

> 短跑是 screening，不是因果證明；但要能做決策。

### 成功（Go 長訓練）的最低條件（同時滿足）
- `token_change_rate` 相對 `λ_invar=0` **顯著下降**（例如下降 ≥ 0.05 的絕對值，或 ≥10% 相對）
- collapse 指標改善（entropy↑ 且 top‑k mass↓ 且 KL↓，至少 2/3 方向一致）
- `val_strict_acc` 不惡化（允許小波動，但不可系統性下降）

### 失敗（No‑Go / Pivot）的條件（任一命中）
- token_change_rate 幾乎不變（表示 invariance 沒有效降低 noise dependence）
- 或 collapse 指標變糟（top‑k mass 上升、entropy 下降）
- 或出現 trivial collapse 徵兆（極端集中，且 acc 下降）

Pivot 建議：
- 先做 probe（noise type/SNR 可分性）補強/反證 superposition
- 若 VQ margin 持續偏小且難改善：評估 quantization/factorization/disentanglement 路線

---

## 7) 進度追蹤（你必須照做）

在 `exp_0124/token_collapse_27e564a/invariance_short_run/PROGRESS.md` 用 checklist：
- [ ] Step 0：建立資料抽樣與 seed（2k/500，記錄列表或 hash）
- [ ] Step 1：完成訓練腳本（L_anchor + L_invar，可開關 λ_invar）
- [ ] Step 2：跑完 ablation（至少 3 個 λ）
- [ ] Step 3：彙整 `summary.{json,md}`（表格：λ → acc/collapse/token_change_rate/margin）
- [ ] Step 4：更新總結論 `exp_0124/token_collapse_27e564a/CONCLUSION.md`（Decision: Go/No‑Go）

每次回報給人類（固定格式）：
- `Status (YYYY-MM-DD HH:MM):`
- `Done:`
- `Next:`
- `Blockers:`
- `Metrics snapshot:`（至少：val strict acc、val entropy、val top‑k mass、val KL、token_change_rate、val margin p50）

---

## 8) 立即開始（第一個可見產出）

先做：
1) 建立 `exp_0124/token_collapse_27e564a/invariance_short_run/PROGRESS.md`
2) 產出 `experiment_matrix.md`（列出 λ、steps、batch、seed、eval 清單）
3) 完成 `λ_invar=0` 的 baseline run，先跑完一次完整 eval（A–D）

