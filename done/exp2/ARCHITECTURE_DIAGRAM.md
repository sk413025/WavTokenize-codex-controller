# 梯度流與凍結參數架構圖

## 🎯 關鍵問題

**Q: 使用 soft probabilities 計算 features 會動到凍結的 codebook 嗎？**
**A: 不會！Codebook 仍然凍結，只是「讀取」方式不同。**

---

## 📐 完整架構圖

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FORWARD PASS (帶梯度)                                │
└─────────────────────────────────────────────────────────────────────────────┘

輸入: Noisy Tokens (B, T)
  │
  ├──────────────────────────────────────────────────────────────────┐
  │                                                                    │
  │  [主任務: Token 去噪]                      [輔助任務: Speaker 保持]
  │                                                                    │
  ▼                                                                    ▼
┌──────────────────────┐                                    ┌──────────────────┐
│  Frozen Codebook     │                                    │  Frozen Codebook │
│  Lookup (不可微)     │                                    │  (凍結，只讀)    │
│  (4096, 512)         │                                    │  (4096, 512)     │
│  ⛔ No Gradient      │                                    │  ⛔ No Gradient  │
└──────────────────────┘                                    └──────────────────┘
  │                                                                    ▲
  │ embeddings (B, T, 512)                                            │ 只讀取
  │ 🔒 Frozen                                                         │ 不更新
  ▼                                                                    │
┌──────────────────────┐                                              │
│  Positional Encoding │                                              │
│  ✅ Trainable        │                                              │
└──────────────────────┘                                              │
  │                                                                    │
  ▼                                                                    │
┌──────────────────────┐                                              │
│  Transformer Encoder │                                              │
│  (4 layers)          │                                              │
│  ✅ Trainable        │                                              │
└──────────────────────┘                                              │
  │                                                                    │
  │ hidden (B, T, 512)                                                │
  │ ✅ Has Gradient                                                   │
  ▼                                                                    │
┌──────────────────────┐                                              │
│  Output Projection   │                                              │
│  Linear(512 → 4096)  │                                              │
│  ✅ Trainable        │                                              │
└──────────────────────┘                                              │
  │                                                                    │
  │ pred_logits (B, T, 4096)                                         │
  │ ✅ Has Gradient                                                   │
  ├────────────────────────┬─────────────────────────────────────────┘
  │                        │
  │                        │
  ▼                        ▼
┌──────────────────────┐  ┌──────────────────────────────────────────┐
│  CrossEntropy Loss   │  │  Softmax (可微！)                        │
│                      │  │  pred_probs = softmax(pred_logits)       │
│  Target: clean_tokens│  │  (B, T, 4096)                            │
│                      │  │  ✅ Has Gradient                         │
└──────────────────────┘  └──────────────────────────────────────────┘
  │                        │
  │ L_CE                   │ pred_probs (B, T, 4096)
  │                        │ ✅ Has Gradient
  │                        ▼
  │                      ┌──────────────────────────────────────────┐
  │                      │  Soft Lookup (可微的矩陣乘法！)          │
  │                      │                                            │
  │                      │  soft_features = pred_probs @ codebook    │
  │                      │                                            │
  │                      │  (B,T,4096) @ (4096,512) = (B,T,512)     │
  │                      │  ✅ Has Gradient                         │
  │                      │                                            │
  │                      │  ⚠️  Codebook 只是「被讀取」            │
  │                      │       不會被更新！仍然凍結！             │
  │                      └──────────────────────────────────────────┘
  │                        │
  │                        │ soft_features (B, 512, T)
  │                        │ ✅ Has Gradient
  │                        ▼
  │                      ┌──────────────────────────────────────────┐
  │                      │  WavTokenizer Decoder                     │
  │                      │  (凍結，但保持梯度流)                    │
  │                      │                                            │
  │                      │  pred_audio = decoder(soft_features)      │
  │                      │  (B, audio_len)                           │
  │                      │  ✅ Has Gradient (通過 decoder)          │
  │                      │  🔒 Decoder weights frozen               │
  │                      └──────────────────────────────────────────┘
  │                        │
  │                        │ pred_audio (B, audio_len)
  │                        │ ✅ Has Gradient
  │                        ▼
  │                      ┌──────────────────────────────────────────┐
  │                      │  Speaker Encoder (ECAPA-TDNN)            │
  │                      │  (凍結，但保持梯度流)                    │
  │                      │                                            │
  │                      │  pred_emb = speaker_encoder(pred_audio)   │
  │                      │  (B, 256)                                 │
  │                      │  ✅ Has Gradient (通過 encoder)          │
  │                      │  🔒 Encoder weights frozen               │
  │                      └──────────────────────────────────────────┘
  │                        │
  │                        │ pred_emb (B, 256)
  │                        │ ✅ Has Gradient
  │                        ▼
  │                      ┌──────────────────────────────────────────┐
  │                      │  L2 Loss with Target                      │
  │                      │                                            │
  │                      │  target_emb = speaker_encoder(input_audio)│
  │                      │  (frozen, no grad)                        │
  │                      │                                            │
  │                      │  L_speaker = MSE(pred_emb, target_emb)    │
  │                      └──────────────────────────────────────────┘
  │                        │
  │                        │ L_speaker
  ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Total Loss = L_CE + λ × L_speaker                              │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
BACKWARD PASS (梯度回傳)

┌─────────────────────────────────────────────────────────────────┐
│  梯度回傳路徑:                                                   │
│                                                                   │
│  L_total → L_CE → pred_logits → Transformer → 參數更新 ✅       │
│          ↓                                                       │
│          L_speaker → pred_emb → pred_audio → soft_features      │
│                    → pred_probs → pred_logits → Transformer     │
│                    → 參數更新 ✅                                │
│                                                                   │
│  ⚠️  注意: Codebook, WavTokenizer, Speaker Encoder 都不更新！  │
│      它們只是「凍結的函數」，用來傳遞梯度                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 關鍵細節解析

### 1. Codebook 的作用

```python
# 傳統方式 (Hard Lookup)
features = codebook[discrete_tokens]  # 索引操作，不可微
# Codebook: 只被「索引」，不參與梯度計算

# 我們的方式 (Soft Lookup)
soft_features = pred_probs @ codebook  # 矩陣乘法，可微
# Codebook: 被「讀取」為矩陣，參與前向計算，但不更新
```

**數學上的等價性**：
```
Hard: features[t] = codebook[argmax(probs[t])]
Soft: features[t] = Σ probs[t,i] × codebook[i]
                    i=0 to 4095

Hard: 選擇概率最大的那一個 codebook entry
Soft: 對所有 codebook entries 做加權平均（權重 = 概率）
```

### 2. 為什麼 Codebook 不會被更新？

```python
# 在 model 初始化時
self.register_buffer('codebook', codebook)  # 註冊為 buffer，不是 parameter
# Buffer 意味著：
# 1. 會被保存到 checkpoint
# 2. 會跟隨 model.to(device)
# 3. 但不會出現在 model.parameters() 中
# 4. 優化器不會更新它！

# 驗證
for name, param in model.named_parameters():
    print(name, param.requires_grad)
# 輸出中不會有 'codebook'！
```

### 3. Soft Lookup 的梯度流

```
Forward:
  pred_probs: (B, T, 4096) - 每個 token position 的概率分布
  codebook: (4096, 512) - 凍結的查找表
  soft_features: (B, T, 512) - 加權平均的結果

Backward:
  ∂L/∂soft_features → ∂L/∂pred_probs (通過矩陣乘法的梯度)

  ⚠️ 關鍵：
  ∂(pred_probs @ codebook)/∂pred_probs = codebook^T

  所以梯度可以回傳到 pred_probs！
  但 codebook 本身不需要梯度（凍結）
```

---

## 🧮 數學推導

### Soft Lookup 的梯度

設：
- `z = pred_probs @ codebook`  (soft features)
- `pred_probs`: (B, T, 4096), 每個元素 p_ij
- `codebook`: (4096, 512), 每個元素 c_kl
- `z`: (B, T, 512), 每個元素 z_il

則：
```
z_il = Σ p_ik × c_kl
       k

∂z_il/∂p_ij = c_jl  (如果 k=j)
            = 0     (否則)
```

**結論**：
- 梯度 `∂L/∂pred_probs` 可以通過 codebook 計算
- 但 codebook 本身不需要梯度！
- 梯度流：`L → z → pred_probs → pred_logits → Transformer`

---

## 🔬 WavTokenizer Decoder 的作用

### 凍結但保持梯度流

```python
# WavTokenizer decoder 是凍結的
for param in wavtokenizer.parameters():
    param.requires_grad = False  # 凍結權重

# 但 decoder 仍然是一個可微分的函數！
# 輸入有梯度 → 輸出也有梯度

# 類比：
# f(x) = 2x + 3
# 其中 2 和 3 是「凍結的常數」
# 但 df/dx = 2 仍然存在！
# x 的梯度可以通過 f 回傳
```

**數學表示**：
```
pred_audio = Decoder(soft_features; θ_decoder)

其中 θ_decoder 是凍結的參數

∂pred_audio/∂soft_features 仍然存在！
（通過 decoder 的前向傳播計算）

但 ∂L/∂θ_decoder 不需要計算（凍結）
```

---

## 📊 對比：三種方法

### 方法 1: Hard Tokens (原始)

```
pred_tokens = argmax(pred_logits)  ← 不可微！
features = codebook[pred_tokens]    ← 不可微！
✗ 梯度被切斷
```

### 方法 2: Gumbel-Softmax

```
pred_tokens_soft = gumbel_softmax(pred_logits, τ)  ← 可微
features = pred_tokens_soft @ codebook              ← 可微
✓ 梯度流暢
✗ 需要調整溫度參數 τ
✗ 訓練和測試不一致（需要退火）
```

### 方法 3: 直接 Soft Lookup (我們的方法)

```
pred_probs = softmax(pred_logits)   ← 可微
soft_features = pred_probs @ codebook  ← 可微
✓ 梯度流暢
✓ 不需要額外超參數
✓ 實現簡單
✓ Codebook 仍然凍結
```

---

## 🎯 回答你的問題

### Q1: 會動到凍結的 codebook 嗎？

**A: 不會！**

- Codebook 註冊為 `buffer`，不是 `parameter`
- 優化器不會更新它
- 只是在前向計算時被「讀取」為矩陣
- 類似於查表，只是查表方式變成「軟查表」

### Q2: Decoder 會無法辨識嗎？

**A: 不會！**

原因：
1. **Soft features 接近 Hard features**：
   - 當模型訓練良好時，`pred_probs` 會集中在某一個 token 上
   - 此時 `soft_features ≈ codebook[argmax(pred_probs)]`
   - 和 hard lookup 幾乎一樣！

2. **Decoder 本身是連續函數**：
   - Decoder 輸入是 continuous features，不是 discrete tokens
   - Soft features 仍然在 decoder 訓練時見過的 feature space 中
   - Decoder 可以正常處理

3. **數學上的平滑性**：
   ```
   Hard: 選一個 entry
   Soft: 多個 entries 的加權平均

   當概率集中時，Soft → Hard
   ```

### Q3: 這會影響推理嗎？

**A: 推理時仍然用 Hard Tokens！**

```python
# 訓練時
pred_probs = softmax(pred_logits)
soft_features = pred_probs @ codebook  # Soft lookup
pred_audio = decoder(soft_features)
# 用於計算 speaker loss

# 推理時
pred_tokens = argmax(pred_logits)  # Hard tokens
features = codebook[pred_tokens]    # Hard lookup
pred_audio = decoder(features)
# 和原來一模一樣！
```

**推理不受影響！**

---

## 💡 直覺理解

### 類比：投票系統

**Hard Tokens (傳統)**：
```
投票結果：[0.6, 0.3, 0.1, 0.0, ...]
決策：選票數最多的候選人 → 候選人 0
輸出：候選人 0 的政策
```

**Soft Tokens (我們的方法)**：
```
投票結果：[0.6, 0.3, 0.1, 0.0, ...]
決策：按得票率加權所有候選人的政策
輸出：0.6 × 政策_0 + 0.3 × 政策_1 + 0.1 × 政策_2 + ...
```

當得票集中時（訓練良好）：
```
投票結果：[0.95, 0.03, 0.02, 0.0, ...]
Hard: 政策_0
Soft: 0.95 × 政策_0 + 0.03 × 政策_1 + 0.02 × 政策_2
    ≈ 政策_0 (幾乎相同！)
```

---

## 🔧 實現驗證

可以用以下代碼驗證 codebook 確實沒被更新：

```python
# 記錄初始 codebook
codebook_before = model.codebook.clone()

# 訓練 100 steps
for _ in range(100):
    loss.backward()
    optimizer.step()

# 檢查 codebook
codebook_after = model.codebook

assert torch.equal(codebook_before, codebook_after), "Codebook 被更新了！"
print("✅ Codebook 確實沒有被更新")
```

---

## 📚 總結

| 特性 | 狀態 | 說明 |
|------|------|------|
| Codebook 更新 | ❌ | 註冊為 buffer，不會被優化器更新 |
| 梯度流 | ✅ | 通過矩陣乘法，梯度完整回傳到 pred_logits |
| Decoder 兼容性 | ✅ | Soft features 仍在合理範圍內 |
| 推理一致性 | ✅ | 推理時仍使用 hard tokens |
| 額外開銷 | 小 | 只是一個矩陣乘法 |

**結論：完全可行，不會有任何問題！** ✅
