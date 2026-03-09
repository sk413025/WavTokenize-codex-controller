# Exp 0125 — 驗收：Commit `589e6d` valid token collapse（TracIn 診斷 LoRA 影響）

## 1. 驗收目的

確認我們對 valid token collapse 的解釋不是猜測，而是：
- 有可重現的量化證據（collapse/acc/margin）
- 用 TracIn 把 val 失敗樣本連回 train samples 的影響力
- 能提出至少 1 個可落地解法（Proposed Fix）與下一步驗證實驗

---

## 2. 必須達成（MUST）

### M1：Collapse 與 failure set 定義清楚且可重現
必須存在：
- `exp_0125/tracin_token_collapse_589e6d/metrics_overview.json`
- `exp_0125/tracin_token_collapse_589e6d/failure_set.json`

且包含：
- collapse 指標（entropy/top‑k mass/unique/KL）與 strict acc（train/val）
- failure/success selection 參數（N、threshold、seed）

### M2：TracIn 計算完成（至少 TracIn-CP 近似）
必須存在：
- `exp_0125/tracin_token_collapse_589e6d/tracin_scores.csv`

且至少覆蓋：
- val failure set 的 ≥ 50 個樣本
- train candidate set 的 ≥ 2000 個樣本（或清楚說明為何只能更少）

### M3：確認「只算 LoRA」是否成立（參數集合驗證）
必須在 `metrics_overview.json` 或獨立 `param_scope.json` 記錄：
- optimizer 實際更新的參數集合摘要（name pattern / count）
- 若有非 LoRA trainable params，必須說明是否已納入 TracIn

### M4：Influence → 噪音材質 的證據鏈
必須存在：
- `exp_0125/tracin_token_collapse_589e6d/proponents_profile.json`
- `exp_0125/tracin_token_collapse_589e6d/opponents_profile.json`

且 `CONCLUSION.md` 必須明確回答：
- 高 influence train samples 是否顯著偏向「雜訊材質強」？（用 SNR/energy/noise type 或 proxy）
- 這個結論是否能排除「只是內容相似」等混淆因子？（至少提出一個控制比較）

### M5：Top-3 root causes + Proposed Fix（可落地）
`exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md` 必須包含：
- Top-3 root causes（每項附關鍵證據檔名與數字）
- Proposed Fix（至少 1 個 primary）與其對應 root cause 的理由
- 下一步驗證實驗（至少 2 個），包含目的/方法/成功判準/資源需求

---

## 3. 應該達成（SHOULD）

### S1：至少兩種 loss 版本的 TracIn 對照
例如：
- `L_train`（訓練目標）
- `L_anchor`（token-level 對齊）

並在結論中說明：
- 哪一個更能解釋 collapse？
- 兩者是否一致指向同一批 train samples？

### S2：至少一個最小反事實驗證（downweight/filter）
若影響力分析強烈指向某類 train 子集（噪音材質強），應做：
- downweight/filter 該子集的短跑微調
- 比較 collapse 指標與 strict acc 是否改善（方向性即可）

### S3：音質觀點的交叉檢查（對齊研究目標）
若已有保存/可重建音檔，應至少做其一：
- 以 PESQ/STOI/SI‑SDR 挑選「音質最差」的 val failure set，對其做 TracIn opponents/proponents 分析；或
- 在反事實短跑後回報至少一個音質指標不惡化（避免只改善 token 指標但聽感變差）

---

## 4. 可選達成（COULD）

### C1：更貼近學術定義的 TracIn（多 checkpoint）
若可取得多 checkpoint（或可重新訓練保存），則：
- 用多 checkpoint 加權求和（TracIn-CP）
- 檢查結果在不同 checkpoint 子集下是否穩健

### C2：結合 a901ca0 的 probe 結果做交叉驗證
- influence 高的 train samples 是否也具有更高的 noise probe 可解碼性？
- 若不一致，代表 collapse 可能不是單純 noise-dependent，而是 VQ/對齊/其他因素

---

## 5. Reviewer Checklist（快速檢查）

- [ ] `exp_0125/tracin_token_collapse_589e6d/metrics_overview.json`
- [ ] `exp_0125/tracin_token_collapse_589e6d/failure_set.json`
- [ ] `exp_0125/tracin_token_collapse_589e6d/tracin_scores.csv`
- [ ] `exp_0125/tracin_token_collapse_589e6d/proponents_profile.json`
- [ ] `exp_0125/tracin_token_collapse_589e6d/opponents_profile.json`
- [ ] `exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md`
- [ ] （should）至少 2 種 loss 的 TracIn 對照
- [ ] （should）至少 1 個 downweight/filter 的短跑反事實驗證
