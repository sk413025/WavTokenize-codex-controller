# Exp 0125 — 規劃：Commit `589e6d` valid token collapse（TracIn 診斷 LoRA 影響）

## 0. 背景與動機

**目標 commit**：`589e6d286cce5bb42a6f174b15eabc824c994a84`（Exp K v4/v5/v5-continue 完整訓練）  
**參考 commit**：`a901ca0b9cd46009239de55a10eb43654dac1af1`（exp0124-4：invariance + noise probe；維持 No-Go）

目前問題：在 Exp K（encoder LoRA 微調）實驗中，**valid 出現 token collapse**（token 分佈高度集中、entropy 下降、top‑k mass 上升，並伴隨 strict token accuracy 低/不穩定）。  
既有經驗（a901ca0）顯示：
- 各種 invariance 設計未能顯著降低 token_change_rate（仍 >0.92）
- noise type 可被線性 probe 解碼（superposition 部分支持），但與 strict acc 的相關性很弱

因此：需要一個能把「失敗樣本」連回「訓練資料與更新方向」的診斷工具，定位 **到底是哪些 train examples 在 LoRA 微調時把模型推向 noise-dependent / collapse 的解**。  
本規劃採用 **TracIn**（Tracing Gradient Descent）做 influence diagnosis。

---

## 1. 分析目標（要回答的問題）

1. valid token collapse 的 Top-3 root causes 是什麼？（需排序、可否證）
2. token collapse 是否源自 **noise-dependent / joint encoding**（把 noise 當 token 的一部分）？
3. 哪些 train samples 對 val 失敗樣本最有影響力？這些 train samples 的共同特徵是什麼（噪音材質強、特定 noise type、特定 speaker/content、對齊偏移、VQ margin 小…）？
4. **可落地解法**是什麼（至少 1 個 primary），以及下一步驗證實驗如何設計（1–3 天可啟動）？

---

## 2. 核心假設（Hypotheses）

### H1：Noise-dependent（joint encoding / superposition）
Student 學到的是 `token(signal, noise_train)` 的聯合編碼；valid noise/speaker 分佈改變時，模型退化到少數「安全碼」→ collapse。

### H2：VQ / margin 問題放大 collapse
valid 的 quantizer assignment margin（d2-d1）更小 → code 選擇不穩定；或 margin 結構性偏移 → 集中落到少數 codes。

### H3：資料驅動（某些「雜訊特徵強烈」train examples 主導 LoRA 更新）
LoRA 微調的梯度方向被特定 train 子集主導；若這些樣本具有強噪音材質，會把 noise 納入 token 形成 noise-dependent 失敗。

### H4：Loss 與目標不一致（batch-level 分佈正則不足以保證 token-level 對齊）
即使改善分佈指標（KL/top‑k mass），strict acc 仍不提升（類似 Exp0123 anti-collapse ablation 的觀察）。

---

## 3. 方法概述：用 TracIn 追溯「誰把模型推向失敗」

### 3.1 我們要診斷的「失敗樣本」
先定義 valid failure set（例如 top‑N collapse_score / bottom‑N strict acc），並保留對照的 valid success set（top‑N strict acc + no collapse）。

若研究目標更貼近「主觀/客觀音質」，可加入 **音質導向 failure set**（用來對齊「還原乾淨聲音」的目標）：
- 以 STOI / PESQ / SI‑SDR 等指標挑選 val 中最差的 top‑N（並保留最佳 top‑N 作對照）
- 再對這些「音質差」樣本做 TracIn proponents/opponents，能更直接回答「為什麼還原會失敗」

### 3.2 TracIn 的核心輸出
對每個 val failure example `v`，找出 train examples `{t}` 中：
- **High proponents**：與 `v` 的梯度方向高度一致（或在你定義的 objective 上推動同方向）
- **High opponents**：與 `v` 的梯度方向高度相反（在 loss-based 版本下常被視為“有害”或衝突樣本）

並把這些 top-K train samples 做「材質/噪音」分析：SNR、noise type、能量分佈、lag、VQ margin、speaker/content 等。

> 關鍵：TracIn 的結論應該被用來提出 **可反駁的可執行修正**（例如移除/降權某些子集後，collapse 是否下降），而不是只停在敘事。

---

## 4. 分析路線圖（由便宜到昂貴）

### Phase A：重現 collapse + 建立 failure/success set
- 產出：collapse 指標（entropy/top‑k/KL/unique）與 strict acc（train/val）
- 選出：val failure set、val success set

### Phase B：TracIn-CP（checkpoint-level TracIn）影響力分析
- 只在 **trainable 參數子空間**計算梯度（Exp K 若僅 LoRA，可只算 LoRA）
- checkpoint 盡量不要只拿最後一個：優先取「收斂前期 / loss 下降最快 / 訓練末期」等代表點；若缺少多 checkpoint，先用 1–3 checkpoints 做 TracIn-CP 近似（並在結論中標注限制）

### Phase C：把 influence 與「噪音材質」連起來
檢查 high proponents/opponents 的 train 樣本是否：
- 系統性偏向低 SNR / 特定 noise type / 特定 speaker/content
- 具有更小 VQ margin / 更高對齊偏移 / 更高 silence ratio

### Phase D：最小反事實驗證（因果方向）
至少做一個快速驗證：
- downweight / filter high-influence train subset，再短跑微調（或只做 offline reweight proxy）看 val collapse 是否下降
- 或對 high-influence subset 做 targeted augmentation / invariance / anchor 強化

---

## 5. 產出物（Artifacts）

建議集中於：
- `exp_0125/tracin_token_collapse_589e6d/`

至少包含：
- `failure_set.json`（val failure/success 定義、樣本 id 列表）
- `metrics_overview.json`（collapse+acc+margin 的 train/val 概覽）
- `tracin_scores.csv`（每個 val failure 的 top-K proponents/opponents）
- `proponents_profile.json`、`opponents_profile.json`（SNR/noise/speaker/content/lag/margin 統計）
- `plots/`（influence 分佈、SNR/margin 分佈對照、case study）
- `CONCLUSION.md`（逐條判定 + Top-3 root causes + Proposed Fix + 下一步）

---

## 6. 為什麼這跟研究目標有關（Research Goal Link）

研究目標是「**還原出該語者的乾淨聲音**」。要做到這件事，模型必須學會：
- **材質（Noise / channel / environment）**與
- **內容（Semantic / speaker identity / phonetic content）**

的解耦（disentanglement）。

valid token collapse 代表：模型在 unseen/shifted 條件下，輸出的離散表徵退化成少數 token，通常意味著：
- 表徵把 noise 當成 token 的一部分（noise-dependent），或
- 量化/表徵穩定性不足（margin 小），導致內容對齊失敗。

TracIn 讓我們能把「val 的失敗輸出」追溯到「哪些 train examples 的梯度更新在 LoRA 微調中最有影響力」。  
若 high-influence train samples 具有強噪音材質，且移除/降權後 collapse 可被緩解，這就提供了**可操作的證據**：目前訓練流程未達到材質/內容解耦，應轉向更合適的目標函數（teacher anchor + invariance）、資料策略、或 factorization/disentanglement。
