# Exp 0125 — 規格：Commit `589e6d` valid token collapse（TracIn 診斷 LoRA 影響）

## 1. 範圍（Scope）

本規格定義：
- 如何選定 valid token collapse 的 failure/success sets
- 如何在 **LoRA 微調**設定下執行 TracIn（TracIn-CP）影響力分析
- 如何把 influence 與「噪音材質/內容」特徵連結，形成可否證結論
- 如何做最小反事實驗證（downweight/filter）

不在本規格內：
- 完整 TracIn 論文級別多 checkpoint / 全量每步追蹤（允許近似，但要標注限制）

---

## 2. 主要輸入（Inputs）

### 2.1 版本
- 目標 commit：`589e6d286cce5bb42a6f174b15eabc824c994a84`
- 參考 commit：`a901ca0b9cd46009239de55a10eb43654dac1af1`（invariance/probe 的負結果與工具可重用）

### 2.2 實驗資產
至少需要：
- Exp K 的 run directory（含 checkpoint / config / history）
- Train/Val caches（或 dataset loader 能還原 train/val split）

---

## 3. 指標定義（Metrics）

### 3.1 Collapse 指標（val 為主，train 對照）
對 student/teacher 的 token histogram（masked frames）計算：
- `entropy`
- `top_k_mass(k=10)`
- `unique`
- `KL(student || teacher)`

每個樣本（per-sample）也要計算：
- `entropy_per_sample`
- `top_k_mass_per_sample`
- `collapse_score_per_sample`（建議：`z(top_k_mass) - z(entropy)`）

### 3.2 Accuracy 指標
- strict masked token accuracy：`acc_frame_weighted`

（可選）：
- global-shift tolerant（sequence-level single offset）

### 3.3 VQ margin（量化穩定性）
對每個 frame 計算：
- `d1`、`d2`、`margin=d2-d1`

輸出：
- train/val 的 mean、p10/p50/p90
- margin 與 strict acc / collapse_score 的相關性

---

## 4. Failure/Success Set 定義（Selection）

### 4.1 Val failure set（至少一種，建議兩種交集）
可選策略：
1. `collapse_score_per_sample` top-N（例如 N=100）
2. `strict_acc_per_sample` bottom-N（例如 N=100）
3. （可選，但推薦對齊研究目標）**音質導向**：PESQ / STOI / SI‑SDR bottom-N  
   - 需明確指定重建音檔的產生方式（teacher/noisy/student）與 sample rate（PESQ 通常需 16k resample）

建議最終使用：
- `failure_set = union(top collapse, bottom acc)`
- 並保留 `success_set = top acc 且非 collapse` 作對照（N=100）

### 4.2 Train candidate set（TracIn 的對象）
依資源分三層：
- Tier 1（快）：train 抽樣 2k（固定 seed）
- Tier 2（中）：train 全量（若可負擔）
- Tier 3（精）：只算「疑似噪音強」或「與 failure 同 content/speaker」的子集做對照

---

## 5. TracIn 設計（TracIn-CP：checkpoint approximation）

### 5.1 參數子空間：只算 LoRA 可以嗎？
**原則**：TracIn 計算應針對「訓練中被 optimizer 更新的參數」。  
在 Exp K（encoder LoRA 微調）中，若確實只有 LoRA 參數（以及可能的少量 bias/LayerNorm）`requires_grad=True`，則：
- **只計算 LoRA 梯度是正確且等價的**（因為其他參數梯度為 0，不影響 dot product）
- 同時顯著降低記憶體與計算成本

**必做驗證**（寫入輸出 JSON）：
- 列出 optimizer param_groups 的 parameter name pattern（應只包含 `lora_*` 相關）
- 統計 trainable params 數量；若出現非 LoRA trainable，需納入 TracIn 參數集合

### 5.2 TracIn score 定義（需明確定義符號與解讀）

採用 TracIn-CP 近似（多 checkpoint 時更接近學術定義）：

對於 train example `t` 與 val example `v`，在 checkpoint `c`：
- `g_t^c = ∇_θ L(t; θ_c)`（對 trainable θ）
- `g_v^c = ∇_θ L(v; θ_c)`
- `score_c(t,v) = η_c * <g_t^c, g_v^c>`

總分：
- `TracIn(t,v) = Σ_c score_c(t,v)`

輸出時同時提供：
- top-K **positive**（梯度方向一致）
- top-K **negative**（梯度方向相反）

> 注意：不同論文/實作對「proponent/opponent」用語與符號可能不同；本規格要求 **同時報正負**，並在報告中用「positive/negative influence」避免語意誤解。

**η_c（步長）選擇建議**：
- 若能從訓練紀錄取得該 checkpoint 的 learning rate（scheduler），可用 `η_c = lr_c` 作近似
- 使用 AdamW 時這仍是近似（有效步長受動量/二階估計影響）；需在報告中標注「η_c 近似方式」

### 5.3 使用的 loss（L）選擇
至少提供兩個版本（避免只用一個 loss 得出偏結論）：

**L_train（訓練目標）**
- 與 Exp K 訓練一致（final loss + intermediate loss），用於回答「哪些 train samples 驅動 LoRA 更新方向」

**L_anchor（token-level anchor）**
- 用 teacher codes 做 target 的 token-level CE/KL（或 soft logits matching）
- 用於回答「哪些 train samples 驅動 ‘對齊 teacher’ 或 ‘偏離 teacher’」

可選（若要直接對 collapse）：
- **L_collapse**：以 soft assignment 的負 entropy / top-k mass proxy 作為 differentiable collapse objective

---

## 6. 計算流程（Protocol）

### Step A：準備資料與標註特徵
對 train/val 每個樣本準備：
- 基本 id：speaker/content/sentence/filename/path（若 cache 可得）
- noise proxy：SNR、energy/silence ratio、noise type（若可得）
- alignment proxy：lag（可抽樣）
- VQ margin（可抽樣或全量）

### Step B：建立 failure/success set
輸出：`failure_set.json`
- 列出 failure/success id
- 記錄 selection 參數（N、threshold、seed）

### Step C：TracIn-CP 計算
checkpoint 選擇優先序：
1. 多 checkpoint（例如每 N epochs 保存，或 best+final+collapse-start）
2. 建議至少包含三個代表點（若可取得）：
   - 收斂前期（early）
   - Loss 下降最快時期（可用 `history.json` 的斜率最大區段近似）
   - 訓練末期（late）
2. 若只有 best/final，仍可先做「單 checkpoint influence」作近似（需在結論標注限制）

輸出：`tracin_scores.csv`
欄位至少含：
- val_id、train_id、checkpoint_id、score、sign
- train 的 noise proxy（SNR/noise type/energy）、val 的同欄位（方便聚合）

### Step D：聚合與統計（把 influence 轉成可解讀結論）
對每個 val failure：
- top-K positive / negative train examples 的分佈（SNR/noise type/energy/margin）

對全體 failure set 聚合：
- 高 influence train 子集的 noise proxy 分佈 vs random baseline
- effect size（例如 mean SNR 差、KS test、或簡單 bootstrap CI）

輸出：
- `proponents_profile.json`
- `opponents_profile.json`
- `plots/influence_vs_snr.png`、`plots/influence_vs_margin.png` 等

---

## 7. 因果方向的最小驗證（Counterfactual）

若 Phase D 顯示「高 influence train samples 明顯噪音更強」，至少做一個最小驗證：
- **Downweight / filter**：把 top-M 高 influence（依 failure set 聚合）train samples 降權或移除
- 用短跑（例如 2k samples、800–1000 steps）微調重跑

成功判準（短跑版）：
- val collapse 指標改善（entropy↑、top‑k mass↓、KL↓）
- strict acc 不惡化（或小幅提升）

補充（若 failure set 以音質導向為主）：
- 至少回報一個音質指標（STOI/PESQ/SI‑SDR）不惡化，避免只改善 token 指標但聽感退化

---

## 8. 輸出目錄（Outputs）

固定輸出到：
- `exp_0125/tracin_token_collapse_589e6d/`

最少包含：
- `failure_set.json`
- `metrics_overview.json`
- `tracin_scores.csv`
- `proponents_profile.json`、`opponents_profile.json`
- `CONCLUSION.md`
