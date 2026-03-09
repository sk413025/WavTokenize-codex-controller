# exp_0128 Phase 2: 架構層面修復方案

**日期**: 2026-01-29
**狀態**: 準備執行
**基於**: [Phase 1 失敗分析](../RESULTS.md)

---

## 背景

Phase 1 的兩項採樣調整實驗（TracIn-Weighted 和 Noise-Balanced）均失敗，揭示了關鍵發現：

**核心洞察**: Token collapse 不是數據採樣問題，而是**訓練動態**問題
- 未訓練的模型（Step 0）優於訓練後的模型（Epoch 300）
- Entropy: 6.26 vs 6.07 (+0.19) ✅
- Top-10 Mass: 11.8% vs 19.7% (-7.9%) ✅
- **結論**: 訓練過程本身導致 collapse

因此 Phase 2 轉向**架構層面**和** loss 層面**的修復方案。

---

## Phase 2 實驗設計

### 高優先級實驗（本階段）

#### 實驗 3: Entropy Regularization

**方法**: 在 loss 中明確加入 entropy 正則化項，懲罰低 entropy 的 token 分佈

```python
# Compute token entropy
token_probs = softmax(token_counts)
entropy = -sum(token_probs * log(token_probs))

# Entropy regularization loss (negative to maximize entropy)
entropy_loss = -lambda_entropy * entropy

# Total loss
total_loss = intermediate_loss + main_loss + entropy_loss
```

**參數掃描**:
- **實驗 3a**: λ = 0.01 (保守，微調)
- **實驗 3b**: λ = 0.05 (中等強度)
- **實驗 3c**: λ = 0.1 (激進，強制 diversity)

**預期**:
- 明確鼓勵模型使用多樣化的 tokens
- 直接提升 entropy，降低 top-k mass
- 可能略微犧牲 strict accuracy（trade-off）

**執行**:
```bash
# 實驗 3a (λ=0.01)
bash exp_0128/phase2/entropy_regularization/run_exp3a_lambda_0.01.sh

# 實驗 3b (λ=0.05)
bash exp_0128/phase2/entropy_regularization/run_exp3b_lambda_0.05.sh

# 實驗 3c (λ=0.1)
bash exp_0128/phase2/entropy_regularization/run_exp3c_lambda_0.1.sh
```

---

#### 實驗 4: Codebook Refresh

**方法**: 定期檢查 codebook 使用情況，重置未使用的 codebook entries

```python
# Every refresh_interval steps:
if step % refresh_interval == 0:
    # Find unused codes (usage < threshold)
    unused_codes = find_codes_with_usage_less_than(threshold)

    # Reinitialize with random vectors
    codebook[unused_codes] = random_init(std=codebook.std())

    # Reset usage count
    usage_count[unused_codes] = 0
```

**參數掃描**:
- **實驗 4a**: interval=100 steps, threshold=10 (保守，少量重置)
- **實驗 4b**: interval=50 steps, threshold=5 (激進，頻繁重置)

**預期**:
- 防止 codebook collapse（部分 codes 從未使用）
- 維持 codebook diversity
- 強制模型探索更多 code space

**執行**:
```bash
# 實驗 4a (interval=100, threshold=10)
bash exp_0128/phase2/codebook_refresh/run_exp4a_interval_100_thresh_10.sh

# 實驗 4b (interval=50, threshold=5)
bash exp_0128/phase2/codebook_refresh/run_exp4b_interval_50_thresh_5.sh
```

---

## 實驗配置

### 一致性保證

所有 Phase 2 實驗與 Phase 1 和 baseline (exp_k v6) **完全一致**：

| 配置項 | 值 | 說明 |
|--------|-----|------|
| **Model** | TeacherStudentIntermediate | 同 exp_k v6 |
| **LoRA** | rank=256, alpha=512, dropout=0.2 | 同 exp_k v6 |
| **Intermediate Layers** | [3, 4, 6] | 同 exp_k v6 |
| **Layer Weights** | {3: 0.3, 4: 0.5, 6: 0.5} | 同 exp_k v6 |
| **Loss Weights** | intermediate=0.5, main=0.5 | 同 exp_k v6 |
| **Optimizer** | AdamW (lr=1e-4, wd=0.01) | 同 exp_k v6 |
| **Batch Size** | 2 (effective=16 with grad_accum=2) | 同 Phase 1 |
| **Training Steps** | 1000 | 同 Phase 1 |
| **Sampler** | Random (標準 DataLoader) | 同 baseline |
| **Seed** | 42 | 同所有實驗 |

### 唯一差異

- **實驗 3**: 加入 `entropy_loss` term (λ ∈ {0.01, 0.05, 0.1})
- **實驗 4**: 加入 codebook refresh 機制 (interval ∈ {50, 100}, threshold ∈ {5, 10})

---

## 成功判準

與 baseline (exp_k v6 @ epoch 300) 比較：

| Metric | Baseline | 成功條件 | 說明 |
|--------|----------|---------|------|
| **Entropy** | 6.07 | **> 6.07** | 必須提升 |
| **Top-10 Mass** | 19.7% | **< 19.7%** | 必須降低 |
| **Strict Accuracy** | 0.91% | **≥ 0.82%** | 允許下降 10% |

**組合判定**:
```python
success = (
    entropy > 6.07 AND
    top_10_mass < 0.197 AND
    strict_acc >= 0.0082  # 90% of baseline
)
```

---

## 預期時間表

### 短期驗證（1-2 天）

**並行執行策略**:
- GPU 0: 實驗 3a (λ=0.01) + 實驗 3c (λ=0.1) + 實驗 4a (interval=100)
- GPU 1: 實驗 3b (λ=0.05) + 實驗 4b (interval=50)

**每個實驗**:
- 1000 steps × ~9 sec/step ≈ 2.5 小時
- 總計：2-3 小時（並行執行）

**里程碑**:
- **Day 1 上午**: 啟動所有 5 個實驗
- **Day 1 下午**: 收集結果，分析成功/失敗
- **Day 2**: 如果成功，準備 full training；如果失敗，測試 Phase 2 中優先級方案（Lower LR, Smaller LoRA）

---

## 文件結構

```
exp_0128/phase2/
├── README.md                                    # 本文件
├── entropy_regularization/
│   ├── train_entropy_reg.py                    # Entropy regularization 訓練腳本
│   ├── run_exp3a_lambda_0.01.sh                # 實驗 3a (λ=0.01)
│   ├── run_exp3b_lambda_0.05.sh                # 實驗 3b (λ=0.05)
│   ├── run_exp3c_lambda_0.1.sh                 # 實驗 3c (λ=0.1)
│   ├── exp3a_lambda_0.01/                      # 實驗 3a 輸出 (運行後生成)
│   ├── exp3b_lambda_0.05/                      # 實驗 3b 輸出
│   └── exp3c_lambda_0.1/                       # 實驗 3c 輸出
├── codebook_refresh/
│   ├── train_codebook_refresh.py               # Codebook refresh 訓練腳本
│   ├── run_exp4a_interval_100_thresh_10.sh     # 實驗 4a (interval=100, threshold=10)
│   ├── run_exp4b_interval_50_thresh_5.sh       # 實驗 4b (interval=50, threshold=5)
│   ├── exp4a_interval_100_thresh_10/           # 實驗 4a 輸出 (運行後生成)
│   └── exp4b_interval_50_thresh_5/             # 實驗 4b 輸出
└── RESULTS.md                                   # Phase 2 結果報告 (執行後創建)
```

---

## 技術實現細節

### Entropy Regularization 實現

**核心函數** (in `train_entropy_reg.py`):
```python
def compute_token_entropy_loss(student_codes, codebook_size=2048):
    """
    計算 student codes 的 entropy

    Returns:
        entropy_loss: -H(p) (negative to maximize entropy)
        entropy_value: H(p) 的實際值（用於 logging）
    """
    # Flatten all tokens
    tokens_flat = student_codes.flatten()

    # Count token frequencies
    token_counts = torch.bincount(tokens_flat, minlength=codebook_size).float()

    # Compute probability distribution
    token_probs = token_counts / token_counts.sum()

    # Remove zero probabilities
    token_probs = token_probs[token_probs > 0]

    # Compute entropy: H(p) = -sum(p * log(p))
    entropy = -(token_probs * torch.log(token_probs + 1e-8)).sum()

    # Return negative entropy as loss
    return -entropy, entropy.item()
```

**訓練循環修改**:
```python
# Standard losses
loss_main = ...
loss_inter = ...

# NEW: Entropy regularization
loss_entropy, batch_entropy = compute_token_entropy_loss(student_codes)

# Total loss
total_loss = (
    0.5 * loss_inter +
    0.5 * loss_main +
    lambda_entropy * loss_entropy
)
```

---

### Codebook Refresh 實現

**核心類** (in `train_codebook_refresh.py`):
```python
class CodebookUsageTracker:
    def __init__(self, codebook_size=2048):
        self.usage_count = torch.zeros(codebook_size, dtype=torch.long)
        self.refresh_history = []

    def update(self, codes):
        """更新 codebook 使用統計"""
        counts = torch.bincount(codes.flatten(), minlength=self.codebook_size)
        self.usage_count += counts

    def refresh_codebook(self, model, threshold, step):
        """重置未使用的 codebook entries"""
        # Find unused codes
        unused_codes = (self.usage_count < threshold).nonzero()

        # Get codebook from model
        vq_layer = model.student.feature_extractor.encodec.quantizer.vq.layers[0]
        codebook = vq_layer._codebook.embed  # [2048, dim]

        # Reinitialize with random vectors
        std = codebook.std().item()
        codebook[unused_codes] = torch.randn_like(codebook[unused_codes]) * std

        # Reset usage count
        self.usage_count[unused_codes] = 0

        # Log refresh event
        self.refresh_history.append({
            'step': step,
            'num_refreshed': len(unused_codes),
            'unused_codes': unused_codes.tolist(),
        })
```

**訓練循環修改**:
```python
# Track codebook usage
codebook_tracker.update(student_codes)

# Periodic refresh
if step % refresh_interval == 0:
    num_refreshed = codebook_tracker.refresh_codebook(
        model, usage_threshold, step
    )
    print(f"Refreshed {num_refreshed} codes")
```

---

## 監控指標

每個實驗會記錄：

### 1. Loss Components
- `total_loss`: 總 loss
- `loss_main`: Main loss (feature + triplet)
- `loss_inter`: Intermediate supervision loss
- `loss_entropy`: Entropy regularization loss（僅實驗 3）

### 2. Collapse Metrics (每 200 steps)
- `entropy`: Token 分佈 entropy
- `top_10_mass`: 前 10 個 tokens 的機率質量
- `strict_acc`: Student-Teacher token 嚴格匹配率
- `unique_tokens`: 使用的唯一 token 數量

### 3. Codebook Statistics（僅實驗 4）
- `num_used`: 被使用的 codebook entries 數量
- `num_unused`: 未使用的 codebook entries 數量
- `num_refreshed`: 每次 refresh 重置的 entries 數量
- `total_refreshes`: 總 refresh 次數

### 4. Visualizations
- Training loss curves
- Entropy progression (batch + validation)
- Top-10 mass progression
- Codebook refresh events（僅實驗 4）

---

## 下一步

### 如果 Phase 2 成功（任一實驗成功）

1. **Full Training**: 將成功的方法訓練 300 epochs
2. **組合測試**: 測試 Entropy Reg + Codebook Refresh 組合
3. **超參數優化**: 精調成功方法的參數

### 如果 Phase 2 失敗（所有實驗失敗）

測試 Phase 2 中優先級方案：

**方案 C: 降低學習率**
```bash
# Lower LR: 1e-4 → 5e-5
python train.py --lr 5e-5 --warmup_steps 100 --steps 2000
```

**方案 D: 減小 LoRA Rank**
```bash
# Smaller LoRA: rank 256 → 128
python train.py --lora_rank 128 --lora_alpha 256 --steps 1000
```

### 如果所有方案失敗

需考慮更根本的改變（Phase 3）：
- 重新訓練 VQ Codebook（使用當前數據集）
- Multi-task Learning（denoising + diversity）
- Architecture Search（不同 LoRA 配置或監督層）
- Dataset Augmentation（更多樣噪音條件）

---

## 參考

- [Phase 1 結果報告](../RESULTS.md)
- [TracIn 診斷分析](../../exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md)
- [Baseline 配置 (exp_k v6)](../../exp_0112_intermediate/train_v6.py)

---

**創建日期**: 2026-01-29
**最後更新**: 2026-01-29
**狀態**: 準備執行 ✅
