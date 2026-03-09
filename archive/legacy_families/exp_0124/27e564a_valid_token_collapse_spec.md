# Exp 0124 — 規格：Commit `27e564a` valid token collapse 成因分析與解法驗證

## 1. 範圍（Scope）

本規格定義：
- 如何量化與定位 **valid token collapse**
- 如何驗證「superposition（signal/noise 同 token）」與其他假設
- 如何對候選解法做最小可行的驗證（以因果證據為目標）

不在本規格內：
- 大規模架構重寫的完整工程落地（但會定義最小原型驗證）

---

## 2. 主要輸入（Inputs）

### 2.1 版本
- 目標 commit：`27e564a29c52a11cca5fb1290f694c6808a4007e`
- 參考資料夾：`epx_0123/train_valid_gap_58a9b71/`（含 Step F/G/H/I 產出）

### 2.2 基準 checkpoint / run
至少包含：
- baseline：Exp K v5 best checkpoint（例如 `exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848/best_model.pt`）
- ablation：Step I 產出的 best setting（例如 KL-to-teacher λ=0.1 的 `metrics.json` 與其對應 checkpoint/weight state；若只有短跑且未存 checkpoint，則至少對照其輸出的 collapse 指標）

---

## 3. 指標定義（Metrics）

### 3.1 Token collapse 指標（split: val 為主，train 作對照）
對 student 與 teacher 分別計算：
- `entropy`：token histogram 的 Shannon entropy
- `top_k_mass(k=10)`：top-10 token 佔比
- `unique`：出現過的 code 數
- `KL(student || teacher)`：以 counts/histogram 計算

加強（用於定位崩塌樣本）：
- `entropy_per_sample`、`top_k_mass_per_sample`
- `collapse_score_per_sample`（建議：`z(top_k_mass) - z(entropy)` 或其他標準化組合）

### 3.2 Accuracy 指標（對照 collapse 的實際傷害）
至少回報：
- strict masked token accuracy（`acc_frame_weighted`）

可選（用於評估對齊敏感度，但不要混作主結論）：
- global-shift tolerant（sequence-level single offset）
- per-frame tolerant 上界（僅作上界參考）

### 3.3 Superposition/Entanglement 指標（H-SUP/H-CAP）

#### (A) Pair invariance（同 clean、不同 noise）
對每個 clean sample 生成多個 noisy 版本（noise type×SNR）後計算：
- `token_change_rate`：noisy1 vs noisy2 的 token 差異比例（mask 後）
- `teacher_alignment_drop`：不同 noise 下對 teacher 的 strict acc 下降量
- `top_k_mass_delta`：不同 noise 下 top-k mass 的變化（是否被少數 token 吸收）

#### (B) Predictability（probe）
以 encoder feature（或 token histogram）做簡單 probe：
- `noise_type_accuracy` / `SNR_regression_R2`
- `content_proxy_accuracy`（如 content_id 或 sentence_id；視資料是否有 metadata）

判讀：
- 若能同時很好預測 noise 與內容，且 noise 變動會造成內容對齊顯著變差，支持 entanglement/superposition。

### 3.4 VQ assignment / margin 指標（H-VQ）
對每個 frame 計算：
- `d1`：到最近 codebook entry 的距離
- `d2`：到次近 entry 的距離
- `margin = d2 - d1`

輸出 train vs val 的：
- `margin_hist`
- `mean/percentiles`（p10/p50/p90）
- 以及 margin 與 strict acc / collapse_score 的相關性

### 3.5 Silence/energy 指標（H-SIL）
以 frame energy 或 RMS 作 proxy：
- `silence_ratio`：低於閾值的 frame 比例（閾值需固定並記錄）
- `collapse_on_non_silence`：排除 silence frames 後的 collapse/acc

---

## 4. 實驗設計（Protocol）

### 4.1 控制條件（Controls）
- evaluation 一律 `model.eval()` + `torch.no_grad()`
- 固定 `encoder_stride=320`（與既有分析一致）
- 抽樣必須可重現：固定 seed、記錄抽樣策略
- 所有結果輸出到：`exp_0124/token_collapse_27e564a/`

### 4.2 必要比較（Comparisons）
1. baseline vs（若可得）best anti-collapse setting：collapse 指標與 strict acc 的差異
2. train vs val：collapse 指標差異是否主要來自 val
3. per-condition：speaker / SNR / lag / energy / margin 的分桶結果
4. pair tests：同 clean、不同 noise 的 invariance 指標

---

## 5. 產出物格式（Outputs）

建議固定結構：
- `exp_0124/token_collapse_27e564a/`
  - `metrics_collapse_overview.json`
  - `token_entropy_vs_acc_val.json`
  - `token_entropy_vs_acc_val.png`
  - `collapse_by_speaker.json`
  - `collapse_by_snr.json`
  - `collapse_by_energy.json`
  - `vq_margin_stats_train_val.json`
  - `vq_margin_hist_train_val.png`
  - `superposition_pair_tests.json`
  - `superposition_pair_plots.png`
  - `CONCLUSION.md`（逐條判定 + Top-3 root causes + 解法與下一步）

JSON 欄位需包含：
- timestamp、commit id、run/checkpoint path、抽樣參數（seed、N）、重要超參數（k、threshold）

---

## 6. 解法（Mitigation）候選與最小驗證規格

> 目的：不是列清單，而是每個解法都要能「對應某個 root cause」，並有最小驗證方式。

### 6.1 Loss/Regularization 路線（對應 H-LOSS/H5）
- token-level 對齊：以 teacher codes 為 target，做 per-frame 的 soft logits matching / CE / KL（比 batch-level distribution 更直接）
- conditional KL：按 SNR/noise type 分桶做 KL-to-teacher（避免全局平均掩蓋局部崩塌）

最小驗證：
- 短跑可做「collapse 指標是否改善」；但 strict acc 的因果驗證需 `max_steps>=1000` 或更接近 full training

### 6.2 Data/Domain 路線（對應 H-SPK/H-NOISE/H-SIL）
- speaker/noise 覆蓋補齊、reweight、或加入 conditioning（speaker embedding / noise embedding）
- silence-aware：把 silence frames 從主要 token acc 指標中分離，或在 loss 中降低 silence 對 collapse 指標的支配

最小驗證：
- 做 split 重切（speaker-matched / noise-matched）或分桶 eval，觀察 collapse 是否顯著緩解

### 6.3 Architecture/Quantization 路線（對應 H-SUP/H-CAP/H-VQ）
- factorized tokens：content token + noise token（雙 codebook 或 residual VQ）
- product quantization / residual quantization：提高容量、降低 entanglement
- noise-invariant objective：同 clean 不同 noise 的一致性約束（**必須**保留對 teacher 的對齊，否則容易被「輸出常數 token」的 trivial 解吸走）

最小驗證：
- 在小 subset 上先證明：factorization 能降低 collapse_score 且 strict acc 提升（即使幅度不大，也要方向一致且可重現）

### 6.4 Noise-invariant training / Disentanglement（對應 H-SUP/H-CAP/H-LOSS）

#### (A) Noise-invariant training（推薦先做）
定義（避免 trivial collapse）：
- 對同一段 clean，生成兩個 noisy views：`x1=clean+noise_a`、`x2=clean+noise_b`
- 同時最小化：
  - `L_anchor`：student(x1/x2) 對齊 teacher(clean)（token-level CE/KL 或 soft-logits matching）
  - `L_invar`：student(x1) 與 student(x2) 的一致性（建議在 **soft assignment / logits / feature** 上做；必要時先做 sequence-level global shift 對齊再算）
  - （可選）`L_div`：輕量 diversity regularizer（避免 invariance 把模型推向常數輸出）

最小驗證（建議 2-stage）：
1. Offline（無訓練）：先量化「同 clean 不同 noise」下的 `token_change_rate` 與 `teacher_alignment_drop`，作為 baseline。
2. Short run（訓練）：固定小資料、固定 steps，比較 `metrics_collapse_overview.json` 中的 entropy/top‑k/KL 與 strict acc 方向是否一致改善。

#### (B) Disentanglement（若 invariance 仍無法改善或顯示容量瓶頸再做）
最小原型（兩路）：
- `z_content`：用來對齊 teacher（輸出 tokens）
- `z_noise`：用來預測 noise type / SNR（aux head）或用 adversarial 讓 `z_content` 不可辨識 noise
- independence 約束：`z_content ⟂ z_noise`（例如 orthogonality / gradient reversal / HSIC 等，選最簡單可跑者）

最小驗證：
- 先看 collapse 指標是否改善（entropy↑、top‑k mass↓、KL↓），再看 strict acc 是否不惡化/提升
