# Phase 3-2 規格：RVQ Fix（Loss / Commitment / EMA）實作與實驗配置

> 本文件定義 Phase 3-2 的「需要改什麼、怎麼跑、記錄什麼」。
> 以能快速迭代、可回溯、可驗收為優先。

## 1. 範圍與非目標

### 範圍（本階段要解）
- 修正 RVQ 訓練目標，使 quantizer 不再被繞過（避免 layer0 collapse）。
- 提供最小必要的 commitment 機制（encoder commitment / codebook update），並可做參數掃描。
- 視需要導入 EMA codebook 更新 + dead-code reset（提升穩定性與可恢復性）。

### 非目標（暫不做）
- 不追求「最終音質最佳化」：Phase 3-2 先以 collapse mitigation + 可訓練穩定性為主。
- 不一次引入多個新正則（避免不可歸因）；所有新項目都要可 ablation。

## 2. 參考基線（Phase 3 現況）

主要程式：
- `exp_0128/phase3/residual_vq/models_rvq.py`
- `exp_0128/phase3/residual_vq/train_rvq_short_run.py`

Phase 3 collapse 特徵：
- `layer0_top10_mass ≈ 1.0`
- `layer0_used_codes` 掉到個位數
- `feature_mse` 仍然很低（代表主目標可在 collapse 下達成）

## 3. Loss 定義（Phase 3-2 核心）

令：
- `z_e = student_encoder_out`（pre-quant）
- `z_q = student_quantized`（post-quant，RVQ 輸出）
- `t_e = teacher_encoder_out`
- `S = student_intermediates`, `T = teacher_intermediates`

### 3.1 主要對齊（Distortion / Alignment）

**規格**：新增（或替換）一個「對齊 quantized features」的主 loss：

- `L_quant = masked_mse(z_q, t_e)`（可選擇加 cosine）

可選 ablation：
- `L_pre = masked_mse(z_e, t_e)`（保留但權重預設 0 或很小）

> 理由：若主目標只對齊 pre-quant，quantizer 可以被繞過，collapse 不會被懲罰。

### 3.2 Intermediate supervision

沿用現有 intermediate loss（例如 L4/L8 的 MSE）：
- `L_inter = IntermediateSupervisionLossV6(S, T)`

> 但需要可調權重（因為它在 Phase 3 量級顯著較大，可能淹沒其他目標）。

### 3.3 正確的 VQ/RVQ commitment（必要修正）

對每一層 RVQ（layer i），令：
- `r_i`：該層輸入 residual（含 gradient）
- `q_i`：對應 codebook embedding（lookup 結果）

**需要同時支援兩種更新模式：**

#### 模式 A：梯度式 codebook update + encoder commitment（先做）

- **Encoder commitment**（讓 encoder 去貼 codebook）：  
  `L_commit_i = mse(r_i, q_i.detach())`
- **Codebook loss**（讓 codebook 去追 encoder）：  
  `L_codebook_i = mse(r_i.detach(), q_i)`

總和：
- `L_commit = Σ_i L_commit_i`
- `L_codebook = Σ_i L_codebook_i`

#### 模式 B：EMA codebook update + encoder commitment（穩定版）

- 保留 `L_commit = Σ_i mse(r_i, q_i.detach())`
- 移除 `L_codebook`，改由 EMA 更新 codebook（見 4.2）

> 注意：Phase 3 現況的 `commitment_loss` 接近 `mse(r.detach(), q)`，只更新 codebook，缺少 encoder commitment。

### 3.4 總 loss（可控權重）

必須在 `train_rvq_short_run.py` 支援下列權重（CLI flags）：

- `λ_quant`：`L_quant` 權重（預設 1.0）
- `λ_pre`：`L_pre` 權重（預設 0.0）
- `λ_inter`：`L_inter` 權重（建議先從 0.25~1.0 掃描）
- `β_commit`：`L_commit` 權重（β sweep 主要參數）
- `λ_codebook`：`L_codebook` 權重（僅模式 A；模式 B = 0）

總 loss：

```
L_total =
  λ_quant   * L_quant
+ λ_pre     * L_pre
+ λ_inter   * L_inter
+ β_commit  * L_commit
+ λ_codebook* L_codebook
```

## 4. RVQ Quantizer 實作規格

### 4.1 現有計算的保留要求

- 距離計算需維持 memory-efficient（避免 `torch.cdist` OOM）。
- 量化需維持 straight-through estimator。
- 需要回傳：
  - `quantized`（z_q）
  - `all_layer_codes`（[n_layers, B, T]）
  - `losses`：`L_commit`/`L_codebook` 分項（供 log）

### 4.2 EMA codebook update（模式 B）

**實作建議**：直接參考 repo 內既有 EMA codebook 實作：
- `/home/sbplab/ruizi/WavTokenizer-main/encoder/quantization/core_vq.py`（`EuclideanCodebook`）

每層需維護 buffer：
- `cluster_size[K]`
- `embed_avg[K, D]`
- `embed[K, D]`（或對應 embedding weight）
- `decay`、`eps`、`threshold_ema_dead_code`

更新流程（training 時）：
1) 取得 indices（assignment）
2) one-hot 計數更新 `cluster_size`
3) 用 EMA 更新 `embed_avg`
4) 以平滑後的 cluster size 正規化得到新 `embed`
5) 若 `cluster_size < threshold`，從目前 batch residual 取樣替換 dead codes（dead-code reset）

**規格要求**：
- EMA 更新必須在 `torch.no_grad()` 中進行
- dead-code reset 需可開關並可調 threshold

## 5. 指標與記錄（不改口徑，增加必要資訊）

沿用 `evaluate_collapse_metrics()` 的 RVQ-specific 指標：
- `layer0_entropy`
- `layer0_top10_mass`
- `layer0_used_codes`
- `joint_diversity`
- `feature_mse`

新增（Phase 3-2）建議 log：
- `loss_quant`（L_quant）
- `loss_pre`（L_pre）
- `loss_inter`（L_inter）
- `loss_commit`（L_commit）
- `loss_codebook`（L_codebook，若啟用）
- 每層 `layer_i_used / layer_i_entropy`（已有）
- 早停判斷欄位：`collapse_flag`（見 ACCEPTANCE）

## 6. 實驗配置（Phase 3-2 IDs）

> 每個實驗都要輸出 timestamped run dir，並保存：`config.json / summary.json / metrics_history.json / loss_history.json / training_curves.png`

### Exp 6a：Quantized Alignment

固定：
- layers=4, codebook=1024（對齊 Exp 5b 方便比較）
- steps=1000, batch=8, grad_accum=2, lr=1e-4

變動（預設）：
- `λ_quant=1.0`
- `λ_pre=0.0`
- `λ_inter=0.5`
- `β_commit=0.25`
- `λ_codebook=1.0`（模式 A）

### Exp 6b：β sweep

在 Exp 6a 基礎上，掃描：
- `β_commit ∈ {0.25, 0.5, 1.0, 2.0}`

### Exp 6c：EMA + dead-code reset

在 6b 最佳 β 設定上：
- 啟用 EMA（模式 B）
- 設定 `decay=0.99`（可掃描 0.95~0.999）
- 設定 `threshold_ema_dead_code ∈ {2, 5, 10}`

### Exp 6d（可選）：Diversity regularization ablation

僅在 6c 仍無法達到 P1 時啟動，且一次只加一個項目：
- `λ_entropy_layer0 * H(p(layer0_codes))` 或
- temporal penalty（連續幀 token 重複懲罰）

## 7. 早停與 Debug 規格

為節省算力，允許早停：

- 若 step 200 時同時滿足：
  - `layer0_top10_mass > 0.95`
  - `layer0_used_codes < 0.01 * codebook_size`
則標記 `collapse_flag=true`，可停止該 run（並保留 artifacts）。

## 8. 風險與對策

- 風險：intermediate supervision 仍壓過 quantized alignment  
  對策：必須可調 `λ_inter`，必要時做 warmup（先 train quantized alignment 200 steps 再加 intermediate）。

- 風險：EMA dead-code reset 造成訓練震盪  
  對策：提高 decay、降低 threshold，或只在固定 interval 做 reset。

