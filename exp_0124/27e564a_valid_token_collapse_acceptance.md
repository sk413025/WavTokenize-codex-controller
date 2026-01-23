# Exp 0124 — 驗收：Commit `27e564a` valid token collapse 成因分析與解法

## 1. 驗收目的

確認我們對「valid token collapse」的解釋不是猜測，而是：
- 有可重現的量化證據
- 能把 superposition（與其他假設）逐條支持/反證
- 產出可行的解法路線與下一步驗證實驗（能在 1–3 天內啟動）

---

## 2. 必須達成（MUST）

### M1：Collapse 被一致量化且可重現
- `exp_0124/token_collapse_27e564a/metrics_collapse_overview.json` 存在
- 內含：student/teacher 的 `entropy`、`top_k_mass`、`unique`、`KL(student||teacher)`（train/val 都要）
- 並清楚記錄：checkpoint/run、抽樣 seed（或全量）、版本/commit

### M2：有 per-sample 層級證據（不是只看平均）
- `token_entropy_vs_acc_val.(png|json)` 存在
- 能回答：collapse 是「少數極端樣本」還是「整體退化」？與 strict acc 的關聯強度如何？

### M3：Superposition（H-SUP）有明確判定（支持/不支持/證據不足）
至少完成其一：
- controlled pairs（同 clean、不同 noise）的量化結果（`superposition_pair_tests.json`）
- 或 probe 指標（noise/content 可分性）結果

且在 `exp_0124/token_collapse_27e564a/CONCLUSION.md` 中寫出：
1) 你採用的 superposition 操作性定義  
2) 證據或反證  
3) 下一步若要更強證據還缺什麼

### M4：至少 3 個「非 superposition」候選原因被檢查並排序
必須在 `CONCLUSION.md` 中給出 Top-3 root causes（可含 superposition），並且每一項都有：
- 關鍵證據（數字/圖檔名）
- 為何足以/不足以解釋 collapse

### M5：下一步驗證實驗是「可執行」而不是口號
在 `CONCLUSION.md` 中列出下一步（至少 2 個）：
- 每個實驗需包含：目的、方法、預期能否證的結果、需要的資源（steps/epochs/資料）
- 並與某個 root cause 明確對應（例如針對 H-CAP 做 factorized token 原型）

### M6：提出「可落地的解法建議」（至少 1 個 primary）
在 `CONCLUSION.md` 中必須有一段 **Proposed Fix**（或同等標題），至少包含：
- primary 解法（擇一或組合）：`noise-invariant training + teacher anchor`、`token-level 對齊（per-frame）`、`disentanglement/factorization`
- 為什麼它對應到你的 Top-3 root causes（逐點對應）
- 最小驗證方式與成功門檻（至少包含 collapse 指標與 strict acc 的方向性要求）

---

## 3. 應該達成（SHOULD）

### S1：VQ margin（H-VQ）分析完成
- `vq_margin_stats_train_val.json` 與 `vq_margin_hist_train_val.png` 產出
- 能回答：valid 的量化不穩定（margin 小）是否是 collapse/acc 的重要因素？

### S2：Speaker / Noise / Silence 的條件化分析完成
至少完成其中兩項：
- `collapse_by_speaker.json`
- `collapse_by_snr.json`
- `collapse_by_energy.json`

並在 `CONCLUSION.md` 寫出「collapse 是否集中於某些條件」。

---

## 4. 可選達成（COULD）

### C1：提出並跑一個最小原型（解法的因果方向）
例如（擇一即可）：
- token-level CE/KL 對齊（per-frame）的小規模訓練/微調
- factorized token（content/noise）的小原型
- noise-invariant consistency 約束（同 clean 不同 noise）
- disentanglement loss（顯式分離 signal/noise representation，例如兩路表示或 adversarial）

要求：
- 至少能展示 collapse 指標朝正向變化（entropy↑、top‑k mass↓、KL↓）
- 並且 strict acc 不惡化（或有小幅提升）

### C2：把「短跑不等於因果」的限制寫清楚
若只做 200–400 steps 的短跑，必須在結論中明確聲明其限制，並提出需要更長訓練才能驗證的項目。

---

## 5. Reviewer Checklist（快速檢查）

- [ ] `exp_0124/token_collapse_27e564a/metrics_collapse_overview.json`
- [ ] `exp_0124/token_collapse_27e564a/token_entropy_vs_acc_val.png`
- [ ] `exp_0124/token_collapse_27e564a/superposition_pair_tests.json`（或等價 probe 證據）
- [ ] `exp_0124/token_collapse_27e564a/CONCLUSION.md`（逐條判定 + Top-3 root causes + 下一步）
- [ ] （should）`vq_margin_stats_train_val.json`
- [ ] （should）至少兩個 collapse_by_* 條件化結果
