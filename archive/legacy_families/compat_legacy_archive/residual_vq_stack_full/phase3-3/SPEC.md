# Phase 3-3 規格：Hot-Code Branching / Codebook Split（k-way 分流）

> 目標：提供一個可控、可 ablation 的「hot code 分裂」機制，用來測試
> “speech/noise 是否被壓縮在同一個 code” 的假設，並觀察是否能減緩 collapse。

## 1. 適用位置（先做最小改動）

**預設只對 layer0 做 split**（因為 layer0 最接近 baseline 單層 VQ，且 Phase 3 的 collapse 主要體現在 layer0）。

若後續有效，再擴展到多層（layer0..layerN）。

## 2. Split 機制定義

### 2.1 名詞
- `K`：layer0 codebook size
- `c*`：hot code index（使用率最高）
- `children(c*)`：分裂出來的子 codes indices（k-1 個）
- `k`：分流數（預設 3）

### 2.2 Hot code 選取

每次 split event 時，使用最近一個 window 的 usage 統計（或 eval 統計）選取 hot codes：
- `hot_k`：選 top-`hot_k` 個（預設 1）

選取依據（擇一）：
- usage count 最大
- 或 top-10 mass 對應的 codes

### 2.3 子 code 來源（建議：重用 unused）

因為 collapse 通常造成大量 unused codes，優先重用它們：
- 在 layer0 找 `usage_count == 0` 的 indices
- 選出 `k-1` 個作為 children

若 unused 不足，才允許擴增 codebook（較大改動，先不要）。

### 2.4 初始化方式（兩階段，先簡後繁）

#### Init-A（最簡：噪音擾動）
令 parent embedding 為 `e_parent`：
- `e_child_j = e_parent + Normal(0, split_init_std)`
- `split_init_std` 建議從 `1e-3 ~ 1e-2` 掃描

#### Init-B（較好：local kmeans / 1D split）
收集被指派到 `c*` 的 residual vectors（或 encoder outputs）作為樣本 `X`，做 `k`-means 得到 `k` 個中心：
- 用這些中心替換 `e_parent` 與 children embeddings

> Init-B 比較能「真的分開混合模式」，但實作較複雜，先做 A 再做 B。

## 3. 指派規則（不改 decoder；盡量不改現有 quantize）

### 3.1 方案 A（推薦）：仍用最近鄰
Split 後直接讓最近鄰在 `{parent ∪ children}` 的局部區域自然分裂 Voronoi cell：
- 不需要顯式 gate
- decoder 仍只看 quantized vectors

### 3.2 方案 B（可選）：顯式分支（Hierarchical）
若方案 A 會再次 collapse 到單一 child，可做：
- 若 argmin 選到 parent 或其 children，就在這個 group 裡做第二次選擇（例如 softmax gating）

## 4. 正則（只針對 group，避免全域 entropy 的 Phase2 問題）

### 4.1 子分支 balance loss（可選）
對每個 split group 的 children usage（在 window 內）計算分佈 `p`：
- `L_group_entropy = -H(p)`（最小化負 entropy = 最大化 entropy）
- 或 `L_group_kl = KL(p || Uniform)`

僅對「被分配到該 group 的 tokens」統計，不做全域 entropy。

### 4.2 子 embedding repulsion（可選）
鼓勵 children embeddings 彼此分開：
- `L_repulse = Σ_{i<j} exp(-||e_i - e_j||^2 / σ^2)`

## 5. 需要新增的紀錄（artifacts）

每個 run 必須輸出：
- `split_history.json`：記錄每次 split event：
  - step、被 split 的 parent codes、children indices、init 方法、init_std/kmeans_iters
- `usage_history.json`（可選）：每個 eval step 的 top-N codes usage

## 6. CLI / Config（建議新增參數）

建議在 training script 增加：
- `--enable_hot_split`（bool）
- `--split_layer`（default 0）
- `--split_k`（default 3）
- `--split_hot_k`（default 1）
- `--split_interval`（default 0=one-shot；或 200）
- `--split_init_std`（default 1e-3）
- `--split_init_method`（`noise` / `kmeans`）
- `--split_balance_weight`（default 0）
- `--split_repulse_weight`（default 0）

## 7. 評估：如何回答“speech/noise 是否分開”

建議使用 noise proxy（至少一個）：

1) sample-level SNR（若 dataloader 可提供，或開 `compute_snr=True`）  
2) per-frame proxy：以 encoder stride 對齊後，計算 `||noisy-clean||` 或 `||student- teacher||` 的 frame-wise 值

對 split group 的不同 children：
- 比較 proxy 的均值/分佈差異（例如 effect size / KS test / mutual information）

> 不要求完美分類，只要子 codes 對 proxy 產生穩定可重現的分離，即支持 H1。

