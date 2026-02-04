# Phase 3-3 規劃：Hot-Code Branching（常見碼分流 / Codebook Split）

> 本階段是針對你的假設：「噪音與語音可能被壓縮在同一個常見 code」而設計的實驗路線。
> 核心想法是：**找出最常出現的 code（hot code），將其“分裂/分流”成多個子 code**，讓 encoder 有足夠容量把原本混在一起的模式（speech/noise）拆開。

## 0. 與 Phase 3 / Phase 3-2 的關係（前置條件）

### Phase 3（RVQ 5a/5b/5c）觀察
在 `55f7255` 的結果中，layer0 幾乎完全 collapse（`top10_mass≈1.0`），而 `feature_mse` 仍可很低，表示：模型可能能在 **極少 codes** 下達成對齊。

### Phase 3-2（先修主路徑）
Phase 3-2 的重點是把 training objective「拉到 quantized features」上，並補正 commitment/EMA，避免 quantizer 被繞過。

**Phase 3-3 的前置條件（建議）**：
- Phase 3-2 至少通過 P1（不再 top10=100%、used_codes 不再個位數），否則 hot-code split 會變成「分裂了也用不到」。

## 1. 問題定義（本階段要驗證什麼）

> 如果某個 code 被過度使用，它可能承載了多種語義（speech/noise/silence），導致 token 表徵不可分辨。  
> Phase 3-3 以「hot code 分裂」去驗證：**把該 code 的 Voronoi cell 拆細，能否提升可分性與多樣性**。

### 主要假設（H）

- **H1（容量不足/混合模式）**：hot code 對應的 feature cluster 實際上是多模態（例如 speech vs noise），但被迫落在同一個 code。
  - 預期：split 後子 codes 會呈現不同的 noise proxy 分佈（例如低 SNR vs 高 SNR）。
- **H2（訓練動力學/繞過 quantizer）**：hot code 只是 collapse 的症狀，split 只會產生多個“看起來不同但實際仍不使用”的 codes。
  - 預期：split 後仍是單一子 code 壟斷，或很快又回到 top10≈1.0。

## 2. 機制概念（Hot-Code Branching 是什麼）

你提出的「把常見 code 分成三種分流」可以有兩種實作型態：

### 方案 A：Codebook Split（最簡單、最像先例）
- 找到 hot code `c*`
- 從未使用/低使用的 codes 中挑 `k-1` 個，將其 embedding 初始化在 `c*` 附近（加小擾動或用 local kmeans）
- 仍然用「最近鄰」指派（argmin distance），但因為附近多了幾個候選，`c*` 的區域被切成更細的子區域

### 方案 B：Hierarchical Branch（顯式分支）
- 第一步選 parent code（粗分類）
- 若 parent 是 hot code，啟用該 parent 專屬的小型 sub-codebook 做第二次量化（k-way）
- quantized 向量可用 `parent_vec + child_vec`（類 residual），decoder 一樣只看向量

建議先做 **方案 A**（改動小、最容易做 ablation 與歸因），若有效再升級到 B。

## 3. 實驗矩陣（Phase 3-3）

> 建議固定在「Phase 3-2 最佳設定」的訓練骨架上，只替換/新增 split 機制。

### Exp 7a：One-shot Split（k=3）
**目的**：驗證 split 本身是否能讓 hot code 不再壟斷。  
**做法**：
- 在 step 0 或 step 50 時，找 layer0 的 top-1 hot code
- split 成 3 個近鄰 codes（重用 unused codes）
- 不加額外正則

### Exp 7b：Periodic Split（每 N steps）
**目的**：如果 collapse 會“追上”新 codes，定期 split 觀察是否能維持多樣性。  
**做法**：
- 每 `split_interval`（例如 200 steps）找 top-1 / top-3 hot codes
- 對每個 hot code split 成 3 子 codes（若有足夠 unused codes）

### Exp 7c：Split + Intra-group Balance（只針對子分支）
**目的**：避免 split 後仍只有單一子 code 被選中（子分支內再次 collapse）。  
**做法**（擇一，避免混太多）：
- 對子分支加 entropy / balance penalty（只在 “屬於該 group 的 token” 上計算）
- 或對子 embedding 加 repulsion（鼓勵子 embedding 分開）

### Exp 7d：Split + Noise proxy 分離驗證
**目的**：直接回應你的假設「speech/noise 混在同一 code」。  
**做法**：
- 在 eval 時對每個 token 計算 noise proxy（例如 per-frame `||noisy-clean||`、或使用 `compute_snr=True` 的 sample-level SNR）
- 檢查 split 後不同子 codes 是否對應到不同的 proxy 分佈（見 Acceptance）

## 4. 指標（除了 collapse，也要回答“分開了嗎”）

### 4.1 Collapse / 多樣性（沿用 Phase 3-2）
- `layer0_top10_mass`
- `layer0_used_codes`
- `layer0_entropy`
- `joint_diversity`（若仍是 RVQ）
- `feature_mse`

### 4.2 Branching 的有效性（新）

對每個 split group（parent + children）：
- `group_entropy`：子 codes 的使用分佈 entropy（避免子分支內 collapse）
- `child_min_usage`：最少被用到的 child 是否 > 門檻
- `proxy_separation`：子 codes 的 noise proxy 均值是否顯著分離（或 mutual information）

## 5. 風險與停損

- 若 Phase 3-2 尚未讓 quantized features 進入主訓練目標，split 很可能無效（因為 quantizer 可被繞過）。
- 若 split 後仍 `top10_mass≈1.0` 且 `used_codes` 沒改善，代表這不是“混合模式”問題，而是更根本的 collapse 動力學 → 應 pivot。

## 6. 為什麼這與我們實驗目標相關

我們的總目標是得到「可離散化、可分辨、且 decoder 可用」的 token 表徵。  
Phase 3 顯示：模型可以靠少數 token 就把 feature_mse 做好，導致 tokens 不具可解釋/可用性。  
Phase 3-3 的 hot-code split 直接針對你提出的核心風險：**不同語義（speech/noise）被壓縮到同一 token**。  
如果 split 後能把一個 hot code 拆成對應不同 noise proxy 的子 codes，代表 tokens 的“可分性”真的提升，這對後續任何下游（去噪/壓縮/辨識）都是關鍵。

