# Exp 0124 — 規劃：Commit `27e564a` valid token collapse 成因分析與解法

## 0. 背景與觀測現象（從 commit `27e564a` 出發）

Commit `27e564a29c52a11cca5fb1290f694c6808a4007e`（exp0123 Step I anti-collapse ablation）已做的事與關鍵觀察：
- 針對 Exp K v5（`58a9b71`）的 train/val gap，做了 **anti-collapse 正則**短跑（Entropy regularizer / KL-to-teacher）。
- **分佈對齊（KL）與 entropy/top‑k mass**在部分設定有改善，但 **val strict token accuracy 提升非常有限**（例如 baseline ~0.869% → best ~0.902%）。
- 結論暗示：token collapse 可能是 symptom（表徵退化後的結果），或者需要更長訓練 / 更結構性的改動才會反映到 strict acc。

本 Exp 0124 的目標：把「valid 上 token collapse」從現象層，往下拆成 **可驗證的 root causes**，並提出能對應 root cause 的解法與驗證路線。

---

## 1. 分析目標（要回答的問題）

1. **valid token collapse 的根因是什麼？**（可能多因一果，需排序）
2. Student 是否學到的是 `token(signal, noise_train)` 的**聯合編碼**（joint encoding），而非 `token(signal)` 的「純淨編碼」？若是，證據與反證是什麼？
2. 你提出的「superposition：signal/noise 同時被壓在同一 token」是否成立？若成立，證據是什麼？若不成立，反證是什麼？
3. 除了 superposition，還有哪些更可能的原因（資料分佈 / speaker shift / VQ 量化行為 / loss 設計 / capacity）？
4. 哪一類解法最可能有效：**loss/regularization**、**資料/評估流程**、還是 **架構/量化策略**？

---

## 2. Token collapse 的「可操作定義」

以 valid split 為主，定義 collapse 需同時滿足「相對 teacher」的退化：
- `entropy(student_codes)` 明顯下降
- `top_k_mass(student_codes)` 明顯上升（少數 token 佔比過大）
- `KL(student || teacher)` 上升（分佈偏離 teacher）
- `unique_codes(student)` 明顯少於 teacher 或少於 train

補充（避免只看全局平均）：同時計算 **per-sample collapse**（例如 per-sample entropy/top‑k mass）用來定位「是少數樣本極端崩塌」或「整體崩塌」。

---

## 3. 核心假設（Hypotheses）— 包含 superposition 與其他可能性

### H-SUP：Superposition / Entanglement（你提出的假設）⭐
直覺：noisy = signal + noise，模型把兩者資訊混疊在同一組離散 token 內；遇到 OOD noise/speaker 時，token 退化成「預設碼」導致 collapse。

等價描述（便於對齊研究語言）：
- Student 可能學到的是 `token(signal, noise_train)` 的聯合編碼；當 valid 的 noise/speaker 分佈改變時，聯合編碼失配，導致 tokens 退化成少數「安全碼」（collapse）。

可驗證預測：
- 固定同一段 clean content，換不同 noise type / SNR 時，**student token 序列大幅變動**，且變動主要被少數 token 吸收（top‑k mass 上升）。
- token（或 encoder features）同時可線性解碼出「內容」與「噪音」特徵，但兩者互相干擾：提升噪音可辨識度時，內容對齊（teacher codes）變差。

### H-SPK：Speaker / Domain shift（train/val speaker disjoint）
若 valid speaker 全為 unseen，student 可能在 unseen speaker 上退化成少數 token（collapse）而非精確對齊 teacher。

可驗證預測：collapse severity 與 speaker_id 強相關；或只要改成 speaker-matched 的 split，collapse 明顯緩解。

### H-NOISE：Noise type / spectral shift（SNR 不夠代表難度）
即使 mean SNR 差異不大，不同 noise 的頻譜特性可能造成 encoder feature 偏移，導致 VQ mapping 崩壞與 collapse。

可驗證預測：collapse 集中在特定 noise cluster（可用 noise spectrum / modulation 特徵聚類近似）。

### H-SIL：Silence / low-energy frame 主導
valid 的靜音/低能量比例更高時，student 可能把大量 frame 映射到「silence tokens」，造成 top‑k mass 上升，看起來像 collapse。

可驗證預測：collapse tokens 的出現與 frame energy 強負相關；排除 silence frames 後 collapse 顯著降低。

### H-VQ：Quantizer assignment margin / codebook geometry 問題
feature 到 codebook 的「最近/次近」距離差（margin）若在 valid 更差，token equality 會更不穩定；或 feature drift 導致集中落到少數 code。

可驗證預測：valid 的 (d2-d1) margin 分佈顯著變小（不穩定）或顯著變大但集中在少數 code（硬崩塌）。

### H-LOSS：目前 loss 只在「全局分佈」層面抑制 collapse，不足以提升 strict acc
例如 entropy regularizer 會提高多樣性，但不保證靠近 teacher codes；KL-to-teacher 若是 batch-level distribution，也不等於 per-frame / per-sample 的對齊。

可驗證預測：分佈指標改善（entropy、KL）但 strict acc 無改善（commit `27e564a` 已觀察到），且錯誤呈現「內容 token 對不回來」而非「只是不夠多樣」。

### H-CAP：容量不足（codebook size/token rate/單路 token 無法同時承載 signal+noise）
這是 superposition 的「結構性版本」：不是訓練問題，而是表示瓶頸；需要 factorization（雙路 token / residual / product quantization）。

可驗證預測：不論怎麼正則化，valid strict acc 上限被卡住；但採用 factorized 表示後（即使簡單），collapse 顯著下降且 strict acc 提升。

---

## 4. 分析路線圖（由便宜到昂貴，先找能否證的證據）

### Phase A：重現 collapse + 定位「誰在崩」
產出：
- valid 全量（或固定 seed 抽樣）的 token entropy/top‑k/unique/KL
- per-sample collapse vs strict acc 的散點圖與相關性
- 找出 top-N 最崩的樣本（case study：波形、SNR、lag、speaker）

### Phase B：條件化分析（驗證 H-SPK/H-NOISE/H-SIL/H-VQ）
把 collapse/acc 分別按以下因子分桶：
- speaker_id、content_id（若 cache 有）
- SNR bins、lag bins
- frame energy bins（silence vs non-silence）
- quantizer margin bins（d2-d1）

目標：回答「collapse 是否主要集中在某些條件」。

### Phase C：Superposition 專屬驗證（H-SUP/H-CAP）
用「同 clean、不同 noise」的 controlled pairs 做三件事：
1. **Noise sensitivity of tokens**：同 clean content，換 noise type/SNR，student token 的變動率（Hamming / Jaccard / top‑k mass 變化）。
2. **Content consistency**：同 clean content，student 是否仍能對齊 teacher clean codes（strict/tolerant/global-shift）。
3. **可分性檢驗**：用簡單 probe 從 encoder feature（或 token histogram）預測 noise type 與內容 proxy（如 content_id）；檢查「同一表徵是否同時承載兩者且互相干擾」。

### Phase D：提出解法並做最小驗證（以因果證據為目標）
依 root cause 類型分流：
- 若 H-LOSS 為主：改成 **per-frame** 的 teacher 對齊（例如 token-level CE/KL、或 soft-logits matching），而非只管 batch-level 分佈。
- 若 H-SPK/H-NOISE 為主：做 speaker/noise 覆蓋或 domain adaptation（資料補齊、分桶 reweight、或 conditioning）。
- 若 H-SUP（joint encoding）為主：優先試 **Noise-invariant training（同一 clean 配不同 noise 輸出一致）+ Teacher anchor（維持對齊 clean teacher）**，避免只做 invariance 反而鼓勵「輸出常數 token」的崩塌。
- 若 H-CAP/H-SUP 為主且 invariance 無法改善：做 factorization / disentanglement（content token + noise token；或 residual/product quantization；或顯式分離 content/noise representation），用最小原型驗證是否能解 collapse。

---

## 5. 預期產出物（Artifacts）

建議集中在：
- `exp_0124/token_collapse_27e564a/`

包含（至少）：
- `metrics_collapse_overview.json`（train/val + baseline/ablation 對照）
- `token_entropy_vs_acc_val.(png|json)`（全量或大樣本）
- `collapse_by_speaker.json`、`collapse_by_snr.json`、`collapse_by_energy.json`
- `vq_margin_stats_train_val.json`、`vq_margin_hist.png`
- `superposition_pair_tests.json`、`superposition_pair_plots.png`
- `CONCLUSION.md`：H-SUP/H-SPK/H-NOISE/H-SIL/H-VQ/H-LOSS/H-CAP 的逐條判定 + Top-3 root causes + 下一步解法建議

---

## 6. 為什麼這跟研究目標有關（Research Goal Link）

我們的研究目標是：在 noisy 輸入下，Student 仍能產生「接近 clean Teacher」的高品質離散 token（供下游 audio LM/codec/去噪任務使用）。  
valid token collapse 代表：
- token 序列資訊量下降（entropy 低、top‑k mass 高），**下游模型看到的是「退化表徵」**；
- train 上看似學到，但 valid 上崩塌，意味著目前方法缺乏泛化，研究結論不可交付；
- 若 collapse 的根因是 superposition/capacity，則需要結構性設計（factorization）而不是只調 loss。

因此，Exp 0124 的分析重點是把 collapse 拆成可驗證、可否證的因素，才能對準研究問題做真正有效的改進。
