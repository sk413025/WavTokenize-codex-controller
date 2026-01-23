ㄏ/27e564a_valid_token_collapse_acceptance.md`

參考（已完成的上一輪分析與 ablation 結果）：
- `epx_0123/train_valid_gap_58a9b71/CONCLUSION.md`
- `epx_0123/train_valid_gap_58a9b71/ablation_anticollapse_summary*.md`
- `epx_0123/train_valid_gap_58a9b71/ablation_anticollapse*/lambda_*/metrics.json`

---

## 1) 你要回答的核心問題（不可省略）

1. valid token collapse 的 Top-3 root causes 是什麼？（可包含 superposition，但必須有證據排序）
2. 「Student 學到 `token(signal, noise_train)` 聯合編碼，而非 `token(signal)`」是否成立？（支持/反證/證據不足）
3. `noise-invariant training` 與 `disentanglement / factorization` 是否值得做？（以你的證據判定；不是憑直覺）
4. **你最後要提出至少 1 個 primary 可落地解法（Proposed Fix）**，並規劃下一步驗證實驗（能在 1–3 天內啟動）。

---

## 2) 產出位置（全部統一放這裡）

建立並只使用：
- `exp_0124/token_collapse_27e564a/`

在該目錄內新增：
- `PROGRESS.md`（唯一進度看板）
- `CONCLUSION.md`（最後結論：逐條判定 + Top-3 + Proposed Fix + 下一步）

---

## 3) 工作規範（必須遵守）

- evaluation 一律：`model.eval()` + `torch.no_grad()`
- 指標必須含 strict token accuracy（`acc_frame_weighted`）與 collapse 指標（entropy/top‑k/KL/unique）
- 抽樣必須可重現：固定 seed，並在 JSON/MD 記錄抽樣策略與樣本數
- 不要只看平均：一定要做 per-sample 層級（entropy↔acc、找極端樣本）
- 任何推論（特別是 superposition）都要給「可否證」證據：支持/不支持/證據不足，並寫明缺什麼
- 短跑訓練若要做（可選），要在結論清楚註明「短跑不等於因果」限制

---

## 4) 進度追蹤格式（你必須照做）

在 `exp_0124/token_collapse_27e564a/PROGRESS.md` 使用 checklist：
- [ ] Step A：Collapse overview（train/val；student/teacher；entropy/top‑k/unique/KL；strict acc）
- [ ] Step B：Per-sample 證據（entropy/top‑k vs strict acc；case study top-N）
- [ ] Step C：條件化分析（至少兩個：speaker / SNR / energy / lag）
- [ ] Step D：VQ margin 分析（d1/d2/margin；train vs val；與 acc/collapse 的關聯）
- [ ] Step E：Superposition 驗證（controlled pairs 或 probe，至少完成其一）
- [ ] Step F：CONCLUSION（逐條判定 + Top-3 + Proposed Fix + 下一步）
- [ ] Acceptance self-check（逐條對照 MUST/SHOULD/COULD）

每完成一個 step：
1) 勾選
2) 寫 3–8 行摘要（關鍵數字 + 產出檔名 + 小結）
3) 附上可重現的 command（或腳本 entrypoint）
4) 寫 Next / Blockers

每次回報給人類固定格式：
- `Status (YYYY-MM-DD HH:MM):`
- `Done:`
- `Next:`
- `Blockers:`
- `Metrics snapshot:`（3–5 個最重要數字：strict acc、entropy、top‑k mass、KL…）

---

## 5) 具體執行步驟（照順序做）

### Step A：Collapse overview（先把現象量化清楚）
目標：生成 baseline 的「train vs val」collapse/acc 概況，作為後續所有判定的地基。

輸出（至少）：
- `metrics_collapse_overview.json`

內容需包含（train/val 都要）：
- student/teacher：entropy、top_k_mass(k=10)、unique、KL(student||teacher)
- strict token acc（frame-weighted）
- run/checkpoint 路徑與抽樣參數

提示：可先重用/改寫 `epx_0123/train_valid_gap_58a9b71/` 的分析腳本（token usage / offline eval），但輸出要符合 Exp 0124 規格。

### Step B：Per-sample 證據（回答「誰在崩」與「崩塌與 acc 是否關聯」）
輸出：
- `token_entropy_vs_acc_val.json`
- `token_entropy_vs_acc_val.png`
- `case_studies.md`（列出 top-N 最崩樣本：基本統計 + 你觀察到的共同特徵）

最低要求：
- 至少 val 端做 per-sample entropy/top‑k/strict acc 的 scatter + Pearson/Spearman
- 指出是「少數樣本極端崩」還是「整體退化」

### Step C：條件化分析（排除非 superposition 的常見原因）
至少做兩個（能做三個更好）：
- speaker / content（若 cache 有）
- SNR bins
- frame energy bins（silence vs non-silence）
- lag bins（若做得到）

輸出（至少兩個檔案）：
- `collapse_by_speaker.json`（或 `collapse_by_snr.json` / `collapse_by_energy.json`）

### Step D：VQ margin（量化 H-VQ：量化不穩定 vs 硬崩塌）
輸出：
- `vq_margin_stats_train_val.json`
- `vq_margin_hist_train_val.png`

並補一段分析：
- margin 與 strict acc / collapse_score 的相關性如何？
- 這是否足以解釋「valid 上 collapse」？

### Step E：Superposition / joint encoding 驗證（必做）
二選一（優先 A；做不到就做 B）：

**A. Controlled pairs（推薦）**
- 選 N=20–50 個 clean（或直接用資料內的 clean）
- 對每個 clean 生成多個 noisy views（不同 noise type × SNR）
- 評估：token_change_rate、teacher_alignment_drop、top_k_mass_delta

輸出：
- `superposition_pair_tests.json`
- `superposition_pair_plots.png`

**B. Probe（替代方案）**
- 用 encoder features 或 token histogram 訓練簡單 probe（線性分類/回歸即可）
- 預測 noise type / SNR（與內容 proxy 若可得）

輸出：
- `probe_results.json`

### Step F：結論與解法（一定要落地）
在 `exp_0124/token_collapse_27e564a/CONCLUSION.md` 寫出：
- 對 H-SUP/H-SPK/H-NOISE/H-SIL/H-VQ/H-LOSS/H-CAP 的逐條判定（支持/不支持/證據不足）
- Top-3 root causes（每項附關鍵證據檔名與數字）
- **Proposed Fix（M6）**：至少 1 個 primary 解法，必須對應你的 Top-3
  - 若你建議 `noise-invariant training`：請明確包含 teacher anchor，並說明如何避免 trivial constant-token collapse
  - 若你建議 `disentanglement/factorization`：請給出最小原型與最小驗證門檻
- 下一步驗證實驗（至少 2 個）：目的、方法、成功/失敗判準、所需資源（steps/epochs/資料）

最後做 `Acceptance self-check`：逐條對照 `exp_0124/27e564a_valid_token_collapse_acceptance.md` 的 MUST/SHOULD/COULD。

---

## 6) 開始工作（第一個可見產出）

現在就開始：
1. 建立 `exp_0124/token_collapse_27e564a/PROGRESS.md`
2. 先完成 Step A（產出 `metrics_collapse_overview.json`）並回報一次狀態（含 metrics snapshot）

