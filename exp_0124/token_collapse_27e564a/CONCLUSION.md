# CONCLUSION: Valid token collapse (commit 27e564a)

## 核心回答（Q1–Q4）

### Q1) valid token collapse Top‑3 root causes（含排序）

**#1 H‑SUP / joint encoding（superposition）**
- **證據**：Controlled pairs 顯示「同 clean、不同 noise」時 student token 變動極大：`token_change_rate` mean ≈ **0.897**（`superposition_pair_tests.json`）。
- **解讀**：token 序列對 noise 高敏感，與「token(signal, noise_train) 聯合編碼」一致；這會在 valid noise 分佈改變時觸發 collapse。

**#2 H‑VQ（量化不穩定 / margin 變小）**
- **證據**：VQ margin（d2‑d1）在 val 明顯更小：mean **0.0138** vs train **0.0197**；p50 **0.0101** vs **0.0131**（`vq_margin_stats_train_val.json`）。
- **解讀**：margin 變小代表量化不穩定、碼選擇更容易漂移，會放大 strict acc 下降與 collapse。

**#3 H‑LOSS（現有正則僅改善分佈，未解決 token‑level 對齊）**
- **證據**：Exp0123 的 KL/entropy 正則改善 KL 與 top‑k mass，但 strict acc 提升有限（`ablation_anticollapse_summary*_*.md`）。
- **解讀**：batch‑level distribution regularizer 不等價於 per‑frame / per‑sample 對齊，導致 collapse 只是「症狀改善」，無法真正解 strict acc。

**非主因（目前證據）**
- **H‑SPK**：speaker 條件化分佈較平（`collapse_by_speaker.json`），不支持單一 speaker shift 為主因。
- **H‑SIL/H‑ENERGY**：energy bins 有差異但非單一區間壟斷（`collapse_by_energy.json`）。
- **H‑NOISE（僅SNR）**：SNR bins 有差異，但 collapse 不集中於單一 SNR 區間（`collapse_by_snr.json`）。

---

### Q2) Student 是否學到 `token(signal, noise_train)`（聯合編碼）？
**結論：** **支持（但尚需更強證據）**。
- **支持點**：Controlled pairs 中「同 clean 不同 noise」token change rate ≈ **0.897**，表徵高度 noise‑dependent（`superposition_pair_tests.json`）。
- **保留點**：alignment_drop 均值接近 0（baseline 已低），需用 **probe（noise type/SNR 可分性）** 或「同內容多噪聲的對齊穩定性」補強。
- **缺口**：尚未完成 probe；若 probe 顯示 noise 可被線性解碼且內容對齊下降，則可更強支持 superposition。

---

### Q3) Noise‑invariant training 與 disentanglement / factorization 是否值得做？
**結論：值得做，但需以證據導向的最小原型驗證。**
- **Noise‑invariant training**：高 token_change_rate 顯示強 noise 敏感，值得加 **teacher anchor + invariance**。
- **Disentanglement / factorization**：若 invariance 無法改善 strict acc，且 VQ margin 問題持續，才值得投入（成本高）。
- **風險**：純 invariance 可能導致 trivial 常數 token，因此必須保留 teacher anchor（per‑frame KL/CE）。

---

### Q4) Proposed Fix（Primary）+ 下一步驗證（1–3 天可啟動）

**Primary Proposed Fix（可落地）**
> **Noise‑invariant training + teacher anchor（token‑level KL/CE）**
- **做法**：對同一 clean 生成兩個 noisy views (x1/x2)，同時最小化：
  - `L_anchor`: student(x1/x2) 對齊 teacher(clean) 的 **per‑frame KL/CE**
  - `L_invar`: student(x1) vs student(x2) 在 soft assignment 上的一致性（可先做 global‑shift alignment）
  - （可選）`L_div`: 輕量 entropy/unique regularizer 防止常數 token
- **對應 root causes**：
  - H‑SUP（減少 noise‑dependent token）
  - H‑LOSS（用 per‑frame anchor 修正對齊問題）
  - H‑VQ（提升穩定性，間接改善 margin）

**下一步驗證實驗（至少 2 個）**
1) **短跑 invariance 原型（1–2 天）**
   - 設定：N=2k samples，steps=800–1000
   - 成功門檻：val strict acc ↑（至少 +5–10% 相對提升）且 collapse 指標改善（entropy↑、top‑k mass↓、KL↓）
   - 失敗門檻：strict acc 無改善或 collapse 指標惡化

2) **Probe（噪音可分性）**
   - 用 encoder features / token hist 線性回歸噪音型別或 SNR
   - 成功門檻：noise 可被準確解碼且與內容對齊下降呈負相關 → 支持 superposition

> **短跑不是因果證明**：短跑若只改善分佈、不改善 strict acc，須進一步拉長步數或改 loss 設計。

---

## 附：主要證據檔案
- Step A: `metrics_collapse_overview.json`
- Step B: `token_entropy_vs_acc_val.json`, `token_entropy_vs_acc_val.png`, `case_studies.md`
- Step C: `collapse_by_speaker.json`, `collapse_by_snr.json`, `collapse_by_energy.json`
- Step D: `vq_margin_stats_train_val.json`, `vq_margin_hist_train_val.png`
- Step E: `superposition_pair_tests.json`, `superposition_pair_plots.png`

---

## Acceptance self-check（對照 27e564a_valid_token_collapse_acceptance.md）
- M1: ✅（Step A 完成）
- M2: ✅（Step B 完成）
- M3: ✅（Step E controlled pairs 完成）
- M4: ✅（Top‑3 root causes 已排序並附證據）
- M5: ✅（下一步驗證實驗 ≥2）
- M6: ✅（Primary Proposed Fix 已提出）
- S1: ✅（VQ margin 完成）
- S2: ✅（speaker/SNR/energy 完成）
- C2: ✅（短跑限制已說明）

---

## Decision (Exp0124-2 short run: noise‑invariant training)

**結論：No‑Go（暫不投入長訓練，需調整或轉向）**

**依據（λ_invar ∈ {0.0, 0.05, 0.10}，N=2k/500，steps=800）：**
- **token_change_rate**：0.9366 → 0.9262 → 0.9218（絕對下降 < 0.05，未達成功門檻）。
- **collapse 指標**：entropy ↑、top‑k mass ↓ 在 0.05/0.10 有改善；KL 在 0.10 下降、0.05 上升（改善不穩定）。
- **val strict acc**：0.00605 → 0.00677 → 0.00631（未惡化，僅小幅提升）。

**判斷：** 目前 L_invar 對「noise‑sensitivity」的降低不足，未達 Go 門檻；雖有分佈改善，但不足以證明可投入長訓練。

**下一步（Pivot / 調整方向）：**
1) **強化 invariance 設計**：改用 sequence‑level global shift 對齊後再算一致性，或在 feature 層做一致性；同時提高 λ 或拉長 steps 重新確認 token_change_rate 是否顯著下降。  
2) **Probe / disentanglement**：先做 noise‑type/SNR probe，若 noise 可線性解碼且內容對齊變差，轉向 factorization/disentanglement 原型。
