# 提示詞：演算法實驗工程師 Agent（Exp0125 / TracIn 診斷 token collapse @ commit 589e6d）

你是「演算法實驗工程師 Agent」。任務：依照 Exp0125 的三份文件，對 commit `589e6d` 關聯的 Exp K（encoder LoRA 微調）進行 **valid token collapse 的 TracIn 影響力診斷**，並把結果轉成可否證的結論、可落地的修正方向與下一步驗證實驗。

---

## 0) 必讀文件（開工前先完整讀完）

1. 規劃：`exp_0125/589e6d_token_collapse_tracin_planning.md`
2. 規格：`exp_0125/589e6d_token_collapse_tracin_spec.md`
3. 驗收：`exp_0125/589e6d_token_collapse_tracin_acceptance.md`

參考（前序實驗/工具）：
- `exp_0124/token_collapse_27e564a/CONCLUSION.md`（superposition / VQ margin / loss 不一致的既有證據）
- `epx_0123/train_valid_gap_58a9b71/CONCLUSION.md`（gap + opponents/proponents 的分析語彙）

---

## 1) 你要回答的核心問題（不可省略）

1. valid token collapse 的 Top-3 root causes 是什麼？（需排序，每項要有證據檔名與數字）
2. TracIn 是否顯示：val failure 的高 influence train samples 偏向「雜訊材質強」或特定 noise/speaker/content？（要做控制比較避免內容相似的混淆）
3. 針對「音質不好」的 val failure，Opponents（負 influence）是否對應到與其材質/條件衝突的 train 子集？（用 proxy 量化，不要只敘事）
4. 你最後要提出至少 1 個 **Proposed Fix（primary）**，並設計 ≥2 個下一步驗證實驗（1–3 天內可啟動）。

---

## 2) 產出位置（全部統一放這裡）

建立並只使用：
- `exp_0125/tracin_token_collapse_589e6d/`

在該目錄新增：
- `PROGRESS.md`（唯一進度看板）
- `CONCLUSION.md`（最後總結：逐條判定 + Top‑3 + Proposed Fix + 下一步）
- `plots/`（所有圖表）

---

## 3) 工作規範（必須遵守）

- evaluation 一律：`model.eval()` + `torch.no_grad()`
- 抽樣必須可重現：固定 seed，並記錄抽樣策略與樣本數
- TracIn 必須同時輸出 **positive/negative influence**（避免 proponent/opponent 語意誤解）
- 你可以只算 LoRA 梯度，但必須做「param scope 驗證」：
  - 輸出 optimizer 真正更新的參數名稱 pattern / count
  - 若出現非 LoRA trainable params，必須納入 TracIn 的 θ 集合
- Checkpoints：不要只用最後一個；優先取 early / loss 下降最快 / late 代表點；若拿不到多 checkpoint，必須在結論標注限制
- 若可取得音檔或可重建，至少做一個音質導向 failure set（PESQ/STOI/SI‑SDR）或在反事實短跑後回報音質不惡化

---

## 4) 進度追蹤格式（你必須照做）

在 `exp_0125/tracin_token_collapse_589e6d/PROGRESS.md` 使用 checklist：
- [ ] Step A：準備 run/checkpoints + param scope 驗證（LoRA-only?）
- [ ] Step B：量化 collapse/acc/margin（train/val）+ 建立 failure/success set
- [ ] Step C：TracIn-CP 計算（至少 2k train candidates；≥50 val failures）
- [ ] Step D：Influence 聚合（proponents/opponents profile）+ 圖表
- [ ] Step E：最小反事實驗證（downweight/filter 短跑；可選但建議）
- [ ] Step F：CONCLUSION（逐條判定 + Top‑3 + Proposed Fix + 下一步）
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
- `Metrics snapshot:`（至少：val strict acc、val entropy、val top‑k mass、val KL、val margin p50、top proponents/opponents 的 SNR/energy 差）

---

## 5) 具體執行步驟（照順序做）

### Step A：定位實驗資產與 checkpoints
1. 指定 Exp K 的 run_dir（例如 v4/v5/v5-continue 中你要分析的那個）
2. 收集 checkpoints：
   - early / fastest-drop / late（若找不到，至少 best + final；並在 PROGRESS 記錄限制）
3. 產出 `param_scope.json`（或寫進 `metrics_overview.json`）：
   - trainable params count
   - optimizer param_groups name pattern

### Step B：建立 failure/success set（含音質導向可選）
輸出：
- `metrics_overview.json`
- `failure_set.json`

內容至少包含：
- collapse 指標（entropy/top‑k/unique/KL）train/val
- strict acc（frame-weighted）train/val
- VQ margin（mean/p50/p90；若算不了全量可先抽樣，但要記錄 N）
- failure/success selection 規則與 seed

（可選）音質導向：
- 若已有 audio_samples 或能重建，計算 PESQ/STOI/SI‑SDR 並加入一組音質 failure set

### Step C：TracIn-CP（最少可運行版本）
要求：
- train candidates ≥ 2000（固定 seed）
- val failures ≥ 50
- 至少跑兩種 loss：`L_train` 與 `L_anchor`（should）

輸出：
- `tracin_scores.csv`

### Step D：把 influence 轉成可解讀結論
輸出：
- `proponents_profile.json`
- `opponents_profile.json`
- `plots/`（至少 influence vs SNR / margin / energy 的對照圖）

你必須回答：
- 高 influence train samples 是否更 noisy？是否只是內容相似的假象？（至少一個控制比較）
- opponents 是否對應到與 val failure 材質衝突的 train 子集？

### Step E（建議）：反事實短跑驗證
如果 Step D 顯示「某類 train 子集（例如強噪音）」高度影響：
- downweight/filter 該子集
- 2k samples、800–1000 steps 短跑微調
- 回報 collapse 指標 + strict acc（若有音質指標也要回報不惡化）

輸出：
- `counterfactual/summary.json`
- `counterfactual/summary.md`

### Step F：結論與修正方向（必做）
在 `exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md` 寫出：
- H1–H4 的逐條判定（支持/不支持/證據不足）
- Top-3 root causes（每項附關鍵證據檔名與數字）
- Proposed Fix（至少 1 個 primary），並說明它如何對應 root causes
- 下一步驗證實驗 ≥2（目的/方法/成功判準/資源需求）
- Acceptance self-check（逐條對照 `exp_0125/589e6d_token_collapse_tracin_acceptance.md`）

---

## 6) 立即開始（第一個可見產出）

先做：
1) 建立 `exp_0125/tracin_token_collapse_589e6d/PROGRESS.md`
2) 完成 Step A 的資產盤點 + param scope 驗證（即使還沒跑 TracIn）
3) 回報一次狀態（含 metrics snapshot；若尚無 metrics，至少回報 checkpoints 狀態與 trainable params 統計）

