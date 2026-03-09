# Phase 3-2 規劃：RVQ Collapse 修復與後續實驗路線

> 目標：在 `55f725556f4074a545f558edcc9b800bd8205ec`（Phase 3 RVQ 5a/5b/5c）已確認「嚴重 collapse」之後，
> 用最小必要的架構/訓練目標修正，驗證 RVQ 是否有機會成為可用解法；若仍無法改善，建立明確 pivot 條件改走其他方法。

## 0. 背景（Phase 3 既有結果摘要）

Commit `55f7255` 的 short-run（1000 steps）結果：

- Exp 5a（2 layers, 2048/layer）：`layer0_used_codes=8/2048 (0.39%)`、`layer0_top10_mass≈1.0`
- Exp 5b（4 layers, 1024/layer）：`layer0_used_codes=7/1024 (0.68%)`、`layer0_top10_mass=1.0`
- Exp 5c（8 layers, 512/layer）：`layer0_used_codes=10/512 (1.95%)`、`layer0_top10_mass≈1.0`、`joint_diversity=0.331`（相對最佳）

結論：**layer0 近乎完全 collapse**（top-10 質量 100%）且使用碼數遠低於 baseline 單層 VQ（約 18% 使用率）。

## 1. Phase 3-2 問題定義

我們的實驗目標是「避免 token collapse，同時維持 teacher-student 的特徵對齊能力」：

- **多樣性目標**：提升 code usage、entropy，降低 top-k mass（避免少數 token 壟斷）。
- **可用性目標**：quantized features 仍能支撐 decoder 產生可用音訊（或至少保持 teacher feature alignment）。

Phase 3（5a/5b/5c）顯示：可以把 `feature_mse` 壓低，但卻以「極端 collapse」達成，這違背了我們要建立「穩定、可離散化、可泛化」的 token 表徵目標。

## 2. 核心假設（為什麼會 collapse）

> Phase 3-2 主要在驗證：**collapse 是否源於訓練目標/梯度路徑使 quantizer 可被繞過**。

### H1：主訓練目標沒有直接約束 quantized features

目前的主 loss 對齊的是 `student_encoder_out`（pre-quant）而不是 `student_quantized`（post-quant）。
結果是模型可以在不「付出代價」的情況下讓 codes collapse，只要 encoder out 能對齊 teacher，就會被主 loss 獎勵。

### H2：目前名為 `commitment_loss` 的項其實只更新 codebook，未形成 encoder commitment

目前 RVQ 內部的 loss 形態接近 `mse(residual.detach(), q)`，主要在讓 codebook 去追 residual（codebook loss），
但缺少標準 VQ-VAE 的 encoder commitment（`mse(residual, q.detach())`）與/或 EMA codebook update + dead-code reset。

### H3：中間層監督可能過強，導致模型傾向「複製 teacher」而非學會穩定可量化表徵

Phase 3 log 顯示 `inter_loss` 量級遠大於 main loss，可能在早期就把 encoder 推向一個「容易被少量碼表示」的區域。

## 3. Phase 3-2 路線總覽（先修正必要的，再談加法）

優先順序：

1. **讓 quantized features 進入主訓練目標**（不再讓 quantizer 被繞過）
2. **補上正確的 encoder commitment**（必要時再調權重/排程）
3. **再加入 EMA / dead-code reset（穩定性與可恢復性）**
4. 若仍不足，再加「多樣性正則」做 ablation（避免回到 Phase 2 的錯誤正則化）

## 4. 實驗矩陣（Phase 3-2）

> 建議先固定在 4-layer/1024（類似 Exp 5b）做快速迭代，成功後再擴到 8-layer/512。

### Exp 6a：Quantized Alignment（最小必要修復）

**目的**：驗證「把 loss 打到 quantized features」是否能阻止 layer0 collapse。  
**改動**：
- 主對齊 loss：改用 `student_quantized` vs `teacher_encoder_out`
- RVQ loss：加入真正的 encoder commitment（見 SPEC）
- 先不做 EMA

**預期**：
- step 200 開始 `layer0_top10_mass` 明顯 < 1.0
- `layer0_used_codes` 明顯 > 1%（至少回到十幾到數十個 codes）

### Exp 6b：Commitment β sweep（找出有效區間）

**目的**：在 Exp 6a 基礎上掃描 encoder commitment 權重 β，觀察多樣性與 feature_mse 的 trade-off。  
**建議掃描**：β ∈ {0.25, 0.5, 1.0, 2.0}  
**停損**：任一設定若 step 200 仍 `top10≈1.0` 且 `used_codes < 1%`，直接標記無效。

### Exp 6c：EMA + dead-code reset（提升穩定性與可恢復性）

**目的**：若 Exp 6a/6b 仍會 collapse 或難以維持多樣性，引入標準 VQ 的 EMA codebook 更新與 dead-code 置換。  
**改動**：
- 每層 codebook 改為 EMA 更新（參考 repo 內既有實作，見 SPEC）
- 設定 dead-code threshold（例如 2 或 5）並在訓練中置換

**預期**：
- used codes 不會快速掉到個位數
- joint_diversity 在多層設定會顯著提升

### Exp 6d（可選）：Diversity Regularization ablation（只在仍崩潰時做）

只在 6a~6c 仍無法達到「基本不 collapse」時才做，避免重演 Phase 2 的錯誤：
- **per-layer usage entropy**（以 layer0 為主）或
- **temporal anti-collapse**（連續幀重複 token 懲罰）

### Exp 6e：Pivot Gate（決策點）

若完成 Exp 6c（含 EMA + 正確 loss）仍無法達到 **驗收 P1**（見 ACCEPTANCE），則判定：

- RVQ 在目前 teacher-student + intermediate supervision 設定下不值得再投入
- 轉向替代方案（例：多頭獨立單層 VQ、Gumbel-Softmax、直接使用 teacher codes 做 distillation、或改 encoder 以提升可量化性）

## 5. 里程碑與資源

- 每個 Exp 建議先跑 short-run（400~1000 steps）判定是否值得繼續
- GPU：優先用 2080 Ti（8 layers 設定較吃顯存）
- 產出物：每個 run 需有 `summary.json / metrics_history.json / loss_history.json / training_curves.png`

---

## 6. 與實驗目標的關聯（為什麼 Phase 3-2 必要）

Phase 1/2 的結論已指向：**collapse 是訓練動力學與 encoder 表徵的問題，單靠 sampling、entropy reg、refresh 都無法治本**。  
Phase 3 引入 RVQ 的目的，是用「架構約束」逼迫多樣性；但 5a/5b/5c 失敗顯示目前實作/訓練目標仍可繞過 quantizer。  
因此 Phase 3-2 的核心就是：**把 quantizer 拉回訓練主路徑**，讓「不用多碼」不再是最容易的解，這才與我們的最終目標一致。

