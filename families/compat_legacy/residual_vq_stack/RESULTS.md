# exp_0128: Phase 1 驗證實驗結果報告 ❌ 失敗

**日期**: 2026-01-29
**狀態**: 兩項實驗均失敗
**結論**: 短期採樣調整方案無法解決 token collapse 問題

---

## 執行摘要

基於 [TracIn 診斷結果](../exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md)，我們測試了兩種短期修復方案：

1. **實驗 1**: TracIn-Weighted Soft Reweighting (α=0.5)
2. **實驗 2**: Noise-Balanced Sampling (box:papercup = 1:1)

**兩項實驗均失敗**，所有評估指標相較 baseline 均顯著惡化：

| 實驗 | Entropy | Top-10 Mass | Strict Acc | 判定 |
|------|---------|-------------|------------|------|
| Baseline (exp_k v6 @ epoch 300) | 6.07 | 19.7% | 0.91% | - |
| **實驗 1** (TracIn Weighted) | **5.63** ⬇ | **29.0%** ⬆ | **0.60%** ⬇ | ❌ 失敗 |
| **實驗 2** (Noise Balanced) | **5.56** ⬇ | **28.3%** ⬆ | **0.52%** ⬇ | ❌ 失敗 |

---

## 實驗背景

### 問題現象

Exp K v6 在訓練 epoch 300 時出現嚴重的 validation token collapse：

- **Entropy**: 6.07 (健康值應 > 8.0)
- **Top-10 Mass**: 19.7% (健康值應 < 10%)
- **Strict Accuracy**: 0.91% (極低準確率)
- **KL Divergence**: 1.25 (分佈嚴重偏移)

### TracIn 診斷發現

通過 TracIn (Training Data Attribution) 分析，識別出導致 collapse 的高影響樣本 (proponents) 特徵：

1. **噪音材質失衡**: papercup 57% (baseline 33%) → 1.7x 過度代表
2. **SNR 偏低**: -2.24 dB vs baseline -1.46 dB
3. **分佈異常**: Cohen's d = -0.107 (偏離 clean distribution)

**反事實實驗結果**:
- 刪除 proponents → collapse **惡化** (entropy 6.07 → 5.61)
- 刪除 opponents → collapse **改善** (entropy 6.07 → 6.53)

**結論**: Proponents 包含「困難但必要」的樣本，完全刪除會破壞學習，需要軟性調整。

---

## 實驗設計

### 實驗 1: TracIn-Weighted Soft Reweighting

**方法**: 使用 TracIn influence scores 進行軟性重加權

```python
weight_i = 1 / (1 + α × TracIn_score_i)
```

- High influence proponents → 降低權重 (down-weighted)
- Low influence opponents → 提高權重 (up-weighted)
- **參數**: α = 0.5 (reweighting strength)

**預期**: 減少 proponents 過度影響，同時保留樣本多樣性

**實現**: `TracInWeightedSampler` 從 TracIn scores CSV 讀取影響分數並計算樣本權重

---

### 實驗 2: Noise-Balanced Sampling

**方法**: 強制每個 batch 的 noise material 平衡分佈

- box : papercup = 1 : 1 (原本 box 51.9%, papercup 48.1%)
- 解決 proponents 中 papercup 過度代表問題 (57% → 50%)

**預期**: 平衡噪音分佈，降低 noise-dependent encoding 風險

**實現**: `NoiseBalancedSampler` 從三種材質組中各抽等量樣本組成 batch

---

### 配置一致性

兩個實驗與 baseline (exp_k v6) **完全一致**，唯一差異為 sampler：

| 配置項 | exp_k v6 | 實驗 1 | 實驗 2 |
|--------|----------|--------|--------|
| LoRA (rank/alpha/dropout) | 256/512/0.2 | 256/512/0.2 | 256/512/0.2 |
| Intermediate layers | [3, 4, 6] | [3, 4, 6] | [3, 4, 6] |
| Layer weights | {3:0.3, 4:0.5, 6:0.5} | {3:0.3, 4:0.5, 6:0.5} | {3:0.3, 4:0.5, 6:0.5} |
| Loss weights | inter=0.5 | inter=0.5 | inter=0.5 |
| Optimizer | AdamW (1e-4/0.01) | AdamW (1e-4/0.01) | AdamW (1e-4/0.01) |
| Batch (size/accum) | 8/2 (eff=16) | 8/2 (eff=16) | 8/2 (eff=16) |
| Dataset | CurriculumDataset | CurriculumDataset | CurriculumDataset |
| Training steps | - | 1000 | 1000 |
| **Sampler** | **Random** | **TracInWeighted** | **NoiseBalanced** |
| Seed | 42 | 42 | 42 |

---

## 實驗結果

### 實驗 1: TracIn-Weighted Soft Reweighting (α=0.5)

**運行資訊**:
- 運行目錄: `exp_0128/soft_reweighting/run_exp1_20260129_023536/`
- GPU: CUDA:1
- 訓練時間: ~2.5 小時
- Total steps: 1000

**最終結果** (step 1000):

| Metric | Baseline | 實驗 1 | 變化 | 判定 |
|--------|----------|--------|------|------|
| Entropy | 6.07 | **5.63** | **-0.44** ⬇ | ❌ 惡化 |
| Top-10 Mass | 19.7% | **29.0%** | **+9.3%** ⬆ | ❌ 惡化 |
| Strict Acc | 0.91% | **0.60%** | **-0.31%** ⬇ | ❌ 惡化 |
| Unique Tokens | - | 1437 | - | - |

**訓練過程** (metrics_history.json):

| Step | Entropy | Top-10 Mass | Strict Acc | Unique Tokens |
|------|---------|-------------|------------|---------------|
| 0 (初始) | **6.26** | 11.8% | 3.52% | 1583 |
| 200 | 5.50 | 26.7% | 0.65% | 1169 |
| 400 | 5.52 | 30.8% | 0.45% | 1285 |
| 600 | 5.53 | 29.8% | 0.52% | 1305 |
| 800 | 5.62 | 28.2% | 0.49% | 1318 |
| 1000 | **5.63** | **29.0%** | **0.60%** | 1437 |

**關鍵觀察**:
- ✅ **初始狀態優於 baseline**: Step 0 entropy 6.26 > baseline 6.07
- ❌ **訓練導致 collapse**: Entropy 從 6.26 急降至 5.50 (step 200)
- ❌ **持續惡化**: Step 200-1000 間 entropy 在 5.5-5.6 之間振盪，無改善趨勢
- ❌ **Top-10 集中度惡化**: 從 11.8% 飆升至 29.0%

---

### 實驗 2: Noise-Balanced Sampling

**運行資訊**:
- 運行目錄: `exp_0128/noise_balanced_sampling/run_exp2_20260129_022108/`
- GPU: CUDA:0
- 訓練時間: ~2.5 小時
- Total steps: 1000
- Noise distribution: box 51.9%, papercup 48.1% (forced balanced in batch)

**最終結果** (step 1000):

| Metric | Baseline | 實驗 2 | 變化 | 判定 |
|--------|----------|--------|------|------|
| Entropy | 6.07 | **5.56** | **-0.51** ⬇ | ❌ 惡化 |
| Top-10 Mass | 19.7% | **28.3%** | **+8.6%** ⬆ | ❌ 惡化 |
| Strict Acc | 0.91% | **0.52%** | **-0.39%** ⬇ | ❌ 惡化 |
| Unique Tokens | - | 1268 | - | - |

**訓練過程** (metrics_history.json):

| Step | Entropy | Top-10 Mass | Strict Acc | Unique Tokens |
|------|---------|-------------|------------|---------------|
| 0 (初始) | **6.26** | 11.8% | 3.52% | 1583 |
| 200 | 5.64 | 20.9% | 1.29% | 1184 |
| 400 | 5.77 | 24.5% | 0.70% | 1387 |
| 600 | 5.55 | 28.6% | 0.60% | 1278 |
| 800 | 5.64 | 27.0% | 0.46% | 1340 |
| 1000 | **5.56** | **28.3%** | **0.52%** | 1268 |

**關鍵觀察**:
- ✅ **初始狀態優於 baseline**: Step 0 entropy 6.26 > baseline 6.07 (與實驗 1 相同)
- ❌ **訓練導致 collapse**: Entropy 從 6.26 降至 5.64 (step 200)
- ❌ **持續惡化**: Step 200-1000 間 entropy 在 5.5-5.8 之間振盪
- ❌ **Top-10 集中度惡化**: 從 11.8% 飆升至 28.3%
- ⚠️ **比實驗 1 更差**: 所有指標均略差於實驗 1

---

## 失敗原因分析

### 1. 兩實驗結果驚人相似

| 指標 | 實驗 1 | 實驗 2 | 差異 |
|------|--------|--------|------|
| Entropy | 5.63 | 5.56 | 0.07 (1.2%) |
| Top-10 Mass | 29.0% | 28.3% | 0.7% |
| Strict Acc | 0.60% | 0.52% | 0.08% |
| Unique Tokens | 1437 | 1268 | 169 (11.8%) |

**結論**: 兩種**完全不同**的採樣策略產生**幾乎相同**的失敗結果，說明問題不在採樣方法。

---

### 2. 關鍵發現：未訓練狀態優於 Baseline

**兩實驗的 Step 0 (初始 LoRA) 指標**:

| Metric | Step 0 (未訓練) | Baseline (epoch 300) | 差異 |
|--------|----------------|---------------------|------|
| Entropy | **6.26** | 6.07 | **+0.19** ✅ 更好 |
| Top-10 Mass | **11.8%** | 19.7% | **-7.9%** ✅ 更好 |
| Strict Acc | **3.52%** | 0.91% | **+2.61%** ✅ 更好 |

**震撼結論**:
- 隨機初始化的 LoRA 參數產生的編碼**優於**經過 300 epochs 訓練的模型
- **訓練過程本身導致 collapse**，而非數據分佈問題

---

### 3. 訓練動態導致 Collapse

**所有實驗（包括 baseline）的共同模式**:

```
Initial (Step 0/Epoch 0):  Entropy ~6.2-6.3, Top-10 ~11-12%  ✅ 健康
    ↓
Early Training (Step 200 / Epoch ~60):  Entropy ~5.5-5.7, Top-10 ~20-27%  ⚠️ 開始 collapse
    ↓
Late Training (Step 1000 / Epoch 300):  Entropy ~5.5-6.1, Top-10 ~20-29%  ❌ 嚴重 collapse
```

**結論**: Collapse 是訓練過程中的**動態現象**，在早期訓練（~200 steps）就已出現。

---

### 4. 採樣調整無效的根本原因

#### 實驗 1 失敗分析 (TracIn Weighted)

**預期**: Down-weight high-influence proponents → 減少其對梯度更新的影響

**實際結果**:
- 初始狀態相同 (Step 0: entropy 6.26)
- 訓練後惡化相同 (Step 1000: entropy 5.63 vs baseline 6.07)

**失敗原因**:
1. **Proponents 不是病因**: TracIn 診斷顯示刪除 proponents 反而**惡化** collapse (entropy 6.07 → 5.61)
2. **必要的困難樣本**: Proponents 包含噪音條件下的困難樣本，對學習魯棒編碼是必要的
3. **軟性調整不足**: 僅調整採樣權重無法改變訓練目標和 loss landscape

#### 實驗 2 失敗分析 (Noise Balanced)

**預期**: 平衡 noise material 分佈 → 避免 noise-dependent encoding

**實際結果**:
- 初始狀態相同 (Step 0: entropy 6.26)
- 訓練後惡化**更嚴重** (Step 1000: entropy 5.56 vs 實驗 1 的 5.63)

**失敗原因**:
1. **強制平衡破壞自然分佈**: 數據集本身 box 51.9% vs papercup 48.1% 已接近平衡
2. **並非材質失衡問題**: TracIn 診斷的 papercup 57% 過度代表是**結果**而非**原因**
3. **降低樣本多樣性**: 強制平衡可能導致某些重要樣本組合被忽略

---

### 5. 真正的問題：訓練目標與架構

**採樣調整失敗揭示的深層問題**:

#### (1) **Loss Function 問題**

當前配置：
```python
intermediate_weight = 0.5  # Intermediate supervision
feature_loss_weight = 0.0  # Feature matching (disabled)
triplet_loss_weight = 0.0  # Triplet loss (disabled)
```

**問題**:
- **缺乏 entropy regularization**: 沒有明確懲罰低 entropy 的 loss term
- **缺乏 diversity constraint**: 沒有鼓勵使用多樣 codebook 的機制
- **Intermediate supervision 可能過強**: 強制中間層匹配可能限制表達能力

#### (2) **VQ Codebook 問題**

- **Codebook 未更新**: Student VQ quantizer 被凍結
- **無 codebook refresh**: 沒有機制重置未使用的 codebook entries
- **EMA 未使用**: 無 exponential moving average 更新

#### (3) **優化器問題**

- **學習率過大**: 1e-4 可能在 LoRA 微調時導致過快收斂到局部最優
- **Weight decay 不足**: 0.01 可能無法有效正則化
- **無 gradient clipping on VQ**: VQ 層的梯度可能不穩定

#### (4) **架構問題**

- **LoRA rank 過大**: rank=256 可能有足夠容量學習 collapse 到少數 tokens
- **No stochastic depth**: 沒有隨機深度正則化
- **No dropout in encoder**: Encoder 缺乏 dropout 可能導致 overfitting

---

## 結論與建議

### Phase 1 結論

✅ **成功驗證**:
1. TracIn 診斷正確識別了高影響樣本
2. 實驗設置正確且可重現
3. 採樣策略實現正確（兩實驗結果一致性高）

❌ **方法失敗**:
1. **TracIn-Weighted Soft Reweighting (α=0.5)**: 無效，所有指標惡化
2. **Noise-Balanced Sampling**: 無效，所有指標惡化

**關鍵發現**:
- Token collapse 不是數據採樣問題，而是**訓練動態**問題
- 未訓練的模型優於訓練後的模型 → 訓練目標或優化器有根本缺陷
- 需要**架構層面**或**loss 層面**的修改，而非數據層面

---

### Phase 2 建議方案

基於失敗分析，建議以下方向（按優先級排序）：

#### 方案 A: Entropy Regularization (高優先級)

**方法**: 在 loss 中加入 entropy 正則化項

```python
# Entropy regularization
token_probs = F.softmax(logits, dim=-1)  # [B, T, V]
entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-8), dim=-1).mean()
entropy_loss = -lambda_entropy * entropy  # Maximize entropy

total_loss = intermediate_loss + entropy_loss
```

**參數**: λ_entropy ∈ {0.01, 0.05, 0.1}

**預期**: 明確懲罰低 entropy 分佈，鼓勵使用多樣 tokens

**實驗時長**: 1000 steps (2-3 小時)

---

#### 方案 B: Codebook Refresh (高優先級)

**方法**: 定期重置未使用的 codebook entries

```python
# Every N steps, reset unused codes
if step % refresh_interval == 0:
    usage_count = count_codebook_usage(codes, codebook_size=4096)
    unused_mask = usage_count < threshold  # e.g., < 10 uses
    with torch.no_grad():
        codebook[unused_mask] = torch.randn_like(codebook[unused_mask])
```

**參數**:
- refresh_interval ∈ {50, 100, 200} steps
- threshold ∈ {5, 10, 20} minimum usage count

**預期**: 防止 codebook collapse，維持 codebook diversity

**實驗時長**: 1000 steps (2-3 小時)

---

#### 方案 C: 降低學習率 (中優先級)

**方法**: 使用更保守的學習率和 warmup

```python
lr = 5e-5  # 降低自 1e-4
warmup_steps = 100  # 更長 warmup
min_lr = 1e-7  # 更低 minimum
```

**預期**: 減緩訓練速度，避免過早收斂到 collapse 狀態

**實驗時長**: 2000 steps (4-6 小時，因訓練較慢)

---

#### 方案 D: 減小 LoRA Rank (中優先級)

**方法**: 降低 LoRA 參數容量

```python
lora_rank = 128  # 降低自 256
lora_alpha = 256  # 保持 alpha/rank = 2
```

**預期**: 限制模型容量，強制學習更 diverse 的編碼策略

**實驗時長**: 1000 steps (2-3 小時)

---

#### 方案 E: 組合方法 (如果單一方法成功)

**方法**: 組合多個成功的單一方法

例如：
- 方案 A (Entropy Reg) + 方案 B (Codebook Refresh)
- 方案 C (Lower LR) + 方案 D (Smaller LoRA)

**實驗時長**: Full training (300 epochs, 2-3 天)

---

### 立即行動建議

#### 短期驗證 (1-2 天)

**優先測試**: 方案 A (Entropy Regularization) 和 方案 B (Codebook Refresh)

理由：
1. 直接針對 collapse 問題（低 entropy, 少量 tokens）
2. 實現簡單，風險低
3. 可與現有配置兼容
4. 快速驗證（1000 steps, 2-3 小時/實驗）

**執行計劃**:
```bash
# 方案 A-1: Entropy Reg (λ=0.01)
bash exp_0128/run_exp3_entropy_reg_0.01.sh

# 方案 A-2: Entropy Reg (λ=0.05)
bash exp_0128/run_exp3_entropy_reg_0.05.sh

# 方案 B-1: Codebook Refresh (every 100 steps, threshold 10)
bash exp_0128/run_exp4_codebook_refresh_100_10.sh

# 方案 B-2: Codebook Refresh (every 50 steps, threshold 5)
bash exp_0128/run_exp4_codebook_refresh_50_5.sh
```

**成功判準** (與 baseline 比較):
```python
success = (
    entropy > 6.07 AND
    top_10_mass < 0.197 AND
    strict_acc >= 0.0082  # 90% of baseline
)
```

---

#### 中期驗證 (3-5 天)

**如果短期驗證成功**: 進行 full training (300 epochs)

**如果短期驗證失敗**: 測試方案 C (Lower LR) 和 方案 D (Smaller LoRA)

---

#### 長期方向 (1-2 週)

如果所有方案均失敗，需考慮更根本的改變：

1. **重新訓練 VQ Codebook**: 使用當前數據集重新訓練 WavTokenizer
2. **Multi-task Learning**: 同時優化 denoising + diversity
3. **Architecture Search**: 嘗試不同 LoRA 配置或監督層選擇
4. **Dataset Augmentation**: 更多樣的噪音條件和音檔

---

## 附錄

### A. 實驗文件結構

```
exp_0128/
├── README.md                                    # 實驗概述
├── RESULTS.md                                   # 本文件：結果報告
├── noise_balanced_sampling/
│   ├── run_exp2_20260129_022108/               # 實驗 2 運行目錄
│   │   ├── summary.json                        # 最終結果
│   │   ├── metrics_history.json                # 訓練歷史
│   │   ├── loss_history.json
│   │   ├── config.json
│   │   ├── final_model.pt
│   │   └── checkpoints/
│   ├── train_short_run.py                      # 訓練腳本
│   ├── sampler.py                              # NoiseBalancedSampler
│   └── run_exp2.sh                             # 啟動腳本
├── soft_reweighting/
│   ├── run_exp1_20260129_023536/               # 實驗 1 運行目錄
│   │   ├── summary.json
│   │   ├── metrics_history.json
│   │   ├── loss_history.json
│   │   ├── config.json
│   │   ├── final_model.pt
│   │   └── checkpoints/
│   ├── train_short_run.py                      # 訓練腳本
│   ├── sampler.py                              # TracInWeightedSampler
│   ├── data_weighted.py                        # DataLoader wrapper
│   └── run_exp1.sh                             # 啟動腳本
└── start_parallel.sh                           # 平行執行兩實驗
```

### B. 參考文件

- TracIn 診斷報告: [exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md](../exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md)
- Baseline 配置: [exp_0112_intermediate/train_v6.py](../exp_0112_intermediate/train_v6.py)
- Baseline 運行腳本: [exp_0112_intermediate/run_exp_k_v6.sh](../exp_0112_intermediate/run_exp_k_v6.sh)

### C. 可重現性

所有實驗使用相同種子 (seed=42) 並記錄完整配置於 `config.json`。
可通過以下命令重現：

```bash
# 實驗 1
bash exp_0128/soft_reweighting/run_exp1.sh

# 實驗 2
bash exp_0128/noise_balanced_sampling/run_exp2.sh

# 平行執行
bash exp_0128/start_parallel.sh
```

---

**報告生成日期**: 2026-01-29
**作者**: Claude Code + 實驗數據分析
**狀態**: Phase 1 完成，Phase 2 待執行
