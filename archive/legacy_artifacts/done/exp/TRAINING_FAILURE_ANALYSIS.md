# 訓練正確率低的深度機制分析

## 📊 觀察到的現象（Commit 159908c 實驗結果）

### 訓練停滯表現

**Epoch 34 的最終結果**（3 個實驗幾乎相同）：
| 實驗 | Initial Acc | Final Acc | 增長 | Train Acc | Val Acc |
|------|-------------|-----------|------|-----------|---------|
| Baseline | 17.48% | 17.75% | +0.27% | 8-11% | 17-18% |
| Exp1 (α=0.5) | 17.48% | 17.78% | +0.30% | 8-11% | 17-18% |
| Exp2 (α=0.7) | 17.48% | 17.75% | +0.27% | 8-11% | 17-18% |

### 異常現象

1. **極低的訓練 accuracy (8-11%)**
   - 遠低於驗證 accuracy (17-18%)
   - 違反常識（訓練集通常更容易）

2. **模型 Collapse**
   - 初期：84% 預測集中在 Token 453
   - 中期：30-20% 預測仍集中在 Token 453
   - 後期：100% 預測集中在 Token 244

3. **三個實驗無差異**
   - Baseline (CE), Soft Target (α=0.5), Soft Target (α=0.7) 表現幾乎相同
   - 表明 soft target 沒有起作用

4. **梯度流動正常**
   - PDB 調試確認：52/52 參數有梯度
   - 排除梯度消失問題

---

## 🔍 關鍵數據發現

### Token 分布極度不平衡（實際訓練數據）

```
Token    0:  46.45%  ← 壓倒性的 majority class
Token  453:   7.00%  ← 第二高頻
Token  244:   0.24%  ← 極稀有（但模型卻 collapse 到這裡！）
Token 1145:   0.21%
... (其他 1830 個 tokens)

使用的 tokens: 1834 / 4096 (44.8%)
未使用的 tokens: 2262 / 4096 (55.2%)  ← 超過一半的 codebook 從未出現！
```

### 模型配置

```python
模型: ZeroShotDenoisingTransformer
  - Parameters: 14.8M
  - d_model: 512
  - nhead: 8
  - num_layers: 4
  - dim_feedforward: 2048
  - dropout: 0.1
  - Vocab size: 4096
  - Frozen codebook: (4096, 512)

訓練配置:
  - Batch size: 28
  - Learning rate: 1e-4
  - Optimizer: AdamW
  - Warm-up epochs: 50 → 5 (修改後)
  - Total epochs: 200
```

---

## 💡 可能的機制假設

### 假設 1: 極端 Class Imbalance 主導一切 ⭐⭐⭐⭐⭐

**機制**：
```
Token 0 佔 46.45% + Token 453 佔 7.00% = 53.45% 的數據
→ 模型發現：總是預測這兩個 token 可以快速降低 loss
→ Output layer 的 weight 向 Token 0, 453 偏移
→ 其他 2260+ tokens 的 weight 訓練不足
→ 即使後來加入 class weights，output layer 已經"固化"
```

**證據**：
1. ✅ Token 0 (46.45%) 從未被 down-weight，但也造成問題
2. ✅ 模型從 Token 453 collapse 轉移到 Token 244
3. ✅ Training Acc (8-11%) < Val Acc (17-18%) ← 異常！
   - 可能原因：Training set 的 token 分布更不平衡
   - Val set 恰好 Token 0/453 佔比較少

**為什麼 class weights 不夠**：
```python
# 當前策略
weights = torch.ones(4096)
weights[453] = 0.01  # 只降低 Token 453

# 問題
Token 0 (46.45%) 未被降權  ← 最大問題！
Token 244 (0.24%) 仍然是 1.0 權重，但數據太少無法學習
```

**改進方向**：
```python
# 應該使用 inverse frequency weighting
freq = torch.tensor([token 出現頻率 for each token])
weights = 1.0 / (freq + epsilon)
weights = weights / weights.sum() * len(weights)  # 歸一化

# 或者
weights = (1.0 / freq) ** 0.5  # 平方根緩和極端值
```

---

### 假設 2: Soft Target 的 Temperature 設置不當 ⭐⭐⭐⭐

**機制**：
```python
# 當前設置
temperature = 2.0
soft_targets = F.softmax(distances / temperature, dim=-1)

# 問題：distances 的 scale 未知
如果 distances 已經很小（例如 [-10, -5, -2]）：
  distances / 2.0 = [-5, -2.5, -1]  ← 仍然很尖銳！
  softmax 後仍然接近 one-hot

如果 distances 很大（例如 [-1000, -500, -200]）：
  distances / 2.0 = [-500, -250, -100]  ← 過於平滑！
  softmax 後幾乎 uniform
```

**應該做的**：
```python
# 1. 分析 distances 的實際 scale
distances_std = target_distances.std()
distances_range = target_distances.max() - target_distances.min()

# 2. Adaptive temperature
temperature = distances_std * some_factor

# 3. 或者先 normalize distances
distances_normalized = (distances - distances.mean(dim=-1, keepdim=True)) / distances.std(dim=-1, keepdim=True)
soft_targets = F.softmax(distances_normalized / temperature, dim=-1)
```

**證據**：
1. ✅ 三個實驗（α=0.5, 0.7, baseline）無差異 ← soft target 可能無效
2. ⚠️ 缺乏 distances 分布的統計數據

---

### 假設 3: Codebook 的 55% 從未使用 → Representation Collapse ⭐⭐⭐⭐

**機制**：
```
訓練數據只使用 1834 / 4096 tokens (44.8%)
→ Output projection 的 2262 個 token heads 從未被訓練
→ 這些 heads 的 weight 保持隨機初始化
→ 模型無法學習這些 tokens 的表示
→ 即使遇到這些 tokens，模型也只會預測訓練過的 1834 個
→ 實際有效的類別數：1834，而非 4096
```

**為什麼會這樣**：
```
數據集: EARS (single speaker or limited speakers)
→ 音素/發音變化有限
→ WavTokenizer 的 4096 個 tokens 設計用於多樣化的語音
→ 單一說話者只使用 codebook 的子集
```

**改進方向**：
1. **使用多說話者數據集** - 增加 token 覆蓋率
2. **Codebook pruning** - 只保留常用的 2000 個 tokens
3. **Hierarchical codebook** - 粗粒度 → 細粒度分層預測

**證據**：
1. ✅ 只有 44.8% 的 tokens 被使用
2. ✅ 模型 collapse 到稀有 token (Token 244, 0.24%) ← 異常！
   - 正常情況：collapse 到高頻 token
   - 現象：collapse 到低頻 token
   - 可能原因：Token 244 的 output head "恰好"產生高 logit

---

### 假設 4: 訓練/驗證分布不一致 ⭐⭐⭐

**機制**：
```
Train Acc (8-11%) < Val Acc (17-18%)  ← 反常！

可能原因：
1. Training set 的 Token 0/453 佔比更高（更不平衡）
2. Validation set 的 Token 分布更均勻
3. 或者 training 時使用的 noisy tokens 更難預測
```

**驗證方法**：
```python
# 分別統計 train/val 的 token 分布
train_token_dist = analyze_token_distribution(train_dataset)
val_token_dist = analyze_token_distribution(val_dataset)

# 計算 KL divergence
kl_div = compute_kl(train_token_dist, val_token_dist)
if kl_div > threshold:
    print("Train/Val 分布不一致！")
```

**改進方向**：
- Stratified split：確保 train/val 的 token 分布一致
- 或者：oversampling 稀有 tokens

---

### 假設 5: Warm-up 策略失效 ⭐⭐⭐

**機制**：
```python
# Warm-up 設計（原始）
if epoch < warmup_epochs:  # warmup_epochs = 50
    alpha = 0.0  # 只用 CE loss
else:
    alpha = target_alpha  # 開始用 soft target

# 問題
Epoch 1-50: 只用 CE → 模型學會預測 Token 0/453
Epoch 51+: 切換到 soft target → 但 output layer 已經"固化"在 Token 0/453
→ Soft target 無法改變已經 collapse 的模型
```

**改進後（修改為 warmup_epochs=5）**：
```
理論上更好：更早引入 soft target
但實際：仍然 collapse
→ 表明問題不在 warm-up，而在更根本的地方
```

---

### 假設 6: Entropy Regularization 太弱 ⭐⭐

**機制**：
```python
entropy_weight = 0.01
entropy_reg = -entropy_weight * entropy

# 問題：entropy_weight 可能太小
loss = α * soft_loss + (1-α) * hard_loss + entropy_reg
     ≈ 5.0              ≈ 5.0              ≈ -0.01 * 8.29
     ≈ 5.0              ≈ 5.0              ≈ -0.08

# Entropy regularization 的貢獻只有 0.08，相比 loss ~10 太小了！
```

**應該做的**：
```python
# 動態調整 entropy weight
if current_entropy < threshold:  # 預測過於集中
    entropy_weight = 0.1  # 增加 10 倍
else:
    entropy_weight = 0.01
```

---

### 假設 7: 模型容量不足以區分 4096 類 ⭐⭐

**機制**：
```
模型參數: 14.8M
任務: 4096-class classification

對比：
  - ResNet-50 (25M params) for ImageNet (1000-class)
  - ViT-B (86M params) for ImageNet (1000-class)
  - GPT-2 Small (117M params) for vocab ~50k

我們的模型: 14.8M params for 4096-class
→ 每個 class 平均只有 3600 個參數！
→ 可能不足以學習 4096 個精細的類別邊界
```

**改進方向**：
- 增加模型容量：num_layers=6, d_model=768 → ~35M params
- 或者：Hierarchical prediction (先預測 coarse category，再 fine-grained)

---

### 假設 8: Frozen Codebook 導致表示學習受限 ⭐⭐

**機制**：
```python
self.register_buffer('codebook', codebook)  # Frozen!

# Token embedding = codebook[token_id]
token_emb = self.codebook[noisy_token_ids]  # (B, T, 512)

# 問題
Codebook 是為 VQ-VAE 訓練的，不一定適合 denoising 任務
→ Token embeddings 可能無法有效表示 noisy → clean 的關係
→ Transformer 只能在固定的 embedding space 中學習
```

**改進方向**：
```python
# 方案 1: 可學習的 token embedding
self.token_embedding = nn.Embedding(4096, 512)
# 用 codebook 初始化，但允許微調
self.token_embedding.weight.data.copy_(codebook)

# 方案 2: Learned projection
token_emb_frozen = self.codebook[token_ids]
token_emb = self.token_proj(token_emb_frozen)  # 可學習的投影
```

---

## 🎯 最可能的根本原因（排序）

### 1. 極端 Class Imbalance + 不完整的 Class Weighting ⭐⭐⭐⭐⭐

**綜合證據最強**：
- Token 0 (46.45%) 未被處理
- Token 453 (7.00%) 被 down-weight 後，模型轉向 Token 244
- Train Acc < Val Acc（訓練集分布更不平衡）

**立即可驗證**：
```python
# 計算 inverse frequency weights
weights = compute_inverse_frequency_weights(token_distribution)

# 重新訓練
model_new = train_with_new_weights(weights)
```

---

### 2. Soft Target 的 Temperature/Scale 設置不當 ⭐⭐⭐⭐

**為什麼三個實驗無差異**：
- 如果 soft target 本質上是 one-hot（temperature 太低）
- 或者 uniform（temperature 太高）
- 那麼 α=0.5, 0.7 都沒意義

**立即可驗證**：
```python
# 分析 soft targets 的熵
soft_targets = F.softmax(distances / temperature, dim=-1)
entropy = -(soft_targets * torch.log(soft_targets + 1e-10)).sum(dim=-1).mean()
print(f"Soft target entropy: {entropy:.4f}")
print(f"Max entropy (uniform): {np.log(4096):.4f}")  # 8.29

# 如果 entropy < 1.0：太尖銳（接近 one-hot）
# 如果 entropy > 7.0：太平滑（接近 uniform）
```

---

### 3. Codebook 覆蓋率不足 (55% 未使用) ⭐⭐⭐⭐

**為什麼模型 collapse**：
- 2262 個 tokens 從未出現 → output heads 未訓練
- 模型實際只需區分 1834 個 classes
- 但 loss function 仍然計算 4096-way softmax
- 導致 gradients 分散到未使用的 heads

**立即可驗證**：
```python
# 只訓練出現過的 tokens
used_tokens = get_used_tokens(dataset)  # 1834 個
mask = create_token_mask(used_tokens)   # (4096,) boolean

# 在 loss 中 mask 掉未使用的 tokens
logits_masked = logits.clone()
logits_masked[:, :, ~mask] = -1e10  # 未使用的設為極小值
loss = F.cross_entropy(logits_masked, targets)
```

---

## 🔬 建議的實驗驗證順序

### 實驗 1: 修復 Class Imbalance（最優先）⏰ 1-2 天

```python
# 實施完整的 inverse frequency weighting
freq = compute_token_frequency(dataset)
weights = (1.0 / (freq + 1e-6)) ** 0.5  # 平方根緩和極端值
weights = weights / weights.mean()       # 歸一化

# 訓練
loss_fn = SoftTargetLoss(
    temperature=2.0,
    alpha=0.5,
    class_weights=weights,  # ← 全局 weights
    entropy_weight=0.01
)
```

**預期**：
- Train Acc 提升到 15-20%
- Val Acc 提升到 25-30%
- Token 預測更均勻

---

### 實驗 2: 分析並修復 Soft Target ⏰ 半天

```python
# 步驟 1: 分析 distances 分布
distances_stats = analyze_distances(dataset)
print(f"Mean: {distances_stats['mean']:.4f}")
print(f"Std: {distances_stats['std']:.4f}")
print(f"Range: [{distances_stats['min']:.4f}, {distances_stats['max']:.4f}]")

# 步驟 2: 測試不同 temperature
for temp in [0.5, 1.0, 2.0, 5.0, 10.0]:
    soft_targets = F.softmax(distances / temp, dim=-1)
    entropy = compute_entropy(soft_targets)
    print(f"Temp={temp}: Entropy={entropy:.4f}")

# 步驟 3: 選擇最佳 temperature
best_temp = select_best_temperature(entropy_values)
```

**預期**：
- 找到合適的 temperature（可能是 5.0-10.0，而非 2.0）
- Soft target 的 entropy 在 3.0-5.0 之間（既不太尖也不太平）

---

### 實驗 3: 限制到常用 Tokens ⏰ 半天

```python
# 只保留出現次數 > threshold 的 tokens
threshold = 1000  # 出現至少 1000 次
used_tokens = [t for t, count in token_counts.items() if count > threshold]
print(f"Used tokens: {len(used_tokens)} / 4096")

# 創建 token mapping
token_map = {old_id: new_id for new_id, old_id in enumerate(used_tokens)}

# 訓練只輸出 used_tokens
model = ZeroShotDenoisingTransformer(..., vocab_size=len(used_tokens))
```

**預期**：
- 模型更容易訓練（fewer classes）
- Accuracy 提升到 30-40%

---

### 實驗 4: 增加 Entropy Weight ⏰ 1 天

```python
# 動態 entropy weight
def get_entropy_weight(current_entropy, target_entropy=5.0):
    if current_entropy < target_entropy:
        return 0.1  # 預測太集中，強烈鼓勵多樣化
    else:
        return 0.01
```

---

## 📝 總結

### 核心問題

**Token 0 (46.45%) + Token 453 (7.00%) = 53.45% 的數據極度不平衡**

這是所有問題的根源：
1. 模型優化 shortcut：總是預測 Token 0/453
2. Output layer 向這兩個 tokens 偏移
3. 即使後來加入 class weights (只降 Token 453)，Token 0 仍未被處理
4. 模型從 Token 453 collapse 轉移到其他 token (Token 244)

### 最可能的機制

```
極端不平衡的數據
    ↓
模型發現預測 Token 0/453 可快速降低 loss
    ↓
Output layer 向 Token 0/453 "固化"
    ↓
加入 class weight (只降 Token 453)
    ↓
模型轉向預測其他高 logit 的 token (Token 244)
    ↓
但 Token 244 只有 0.24% 數據 → 無法學好
    ↓
訓練停滯
```

### 推薦的解決方案

**短期（立即可做）**：
1. ✅ 實施完整的 inverse frequency weighting（包含 Token 0）
2. ✅ 分析 distances 分布，調整 temperature
3. ✅ 增加 entropy weight 到 0.1

**中期（1-2 週）**：
1. 增加數據多樣性（多說話者）
2. 限制到常用的 ~2000 tokens
3. 增加模型容量（num_layers=6, d_model=768）

**長期（1-2 月）**：
1. Hierarchical token prediction
2. 可學習的 token embedding（微調 codebook）
3. 改進評估指標（Top-5 Acc, Perceptual metrics）
