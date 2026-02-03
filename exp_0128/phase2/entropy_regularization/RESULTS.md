# Exp 3: Entropy Regularization Results

## 實驗目的
通過 Entropy Regularization 明確懲罰低 entropy 分佈，鼓勵使用多樣化的 codebook tokens。

## 方法
```python
total_loss = intermediate_loss + lambda_entropy * entropy_loss
entropy_loss = -entropy  # Maximize entropy (負值表示最小化負 entropy)
```

## Baseline (exp_k v6 @ epoch 300)
- Entropy: 6.07
- Top-10 Mass: 19.7%
- Strict Accuracy: 0.91%

## 成功判準
- Val entropy > 6.07
- Val top-10 mass < 19.7%
- Val strict acc >= 0.82% (90% of baseline)

---

## 實驗結果

### ❌ Exp 3a: λ=0.01
**狀態**: 未完成（數據不完整）
- 初始評估: Entropy 6.31, Top-10 10.3%, Strict Acc 1.81%
- 訓練步數: 部分完成

### ❌ Exp 3b: λ=0.05
**狀態**: 完成但失敗
- 初始評估: Entropy 6.26, Top-10 11.8%, Strict Acc 3.52%
- 最終評估: Entropy 5.60, Top-10 32.8%, Strict Acc 0.63%
- 變化: Entropy ↓0.47, Top-10 Mass ↑13.1%, Strict Acc ↓0.28%
- **結論**: Collapse 惡化

### ❌ Exp 3c: λ=0.10
**狀態**: 完成但失敗
- 初始評估: Entropy 6.26, Top-10 11.8%, Strict Acc 3.52%
- 最終評估: Entropy 5.57, Top-10 30.7%, Strict Acc 0.56%
- 變化: Entropy ↓0.50, Top-10 Mass ↑11.0%, Strict Acc ↓0.35%
- **結論**: λ 過大，Collapse 惡化更嚴重

---

## 關鍵發現

### 1. 初始模型優於訓練後模型 ✓
所有實驗的 Step 0（LoRA initialized）指標都優於 baseline：
- Entropy: 6.26 vs 6.07 (baseline)
- Top-10 Mass: 11.8% vs 19.7% (baseline)
- Strict Acc: 3.5% vs 0.91% (baseline)

這證實了 Phase 1 的假設：**問題出在訓練過程中發生的 collapse**。

### 2. Entropy Regularization 失敗的原因 ❌

#### 問題 1: 正則化目標錯誤
```python
# 當前實作（錯誤）
entropy_loss = -entropy  # 最大化 token distribution entropy
```

但實際上：
- Token distribution entropy 在訓練中不斷增加（batch 內 token 分佈變均勻）
- 卻無法阻止 **sequential collapse**（連續幀使用相同 token）

#### 問題 2: 忽略了時序結構
Entropy regularization 只考慮 batch 內的 token 統計，忽略了：
- **時序上的 collapse**：連續多幀使用相同 token
- **空間上的 collapse**：同一音頻內所有時間步使用少數 token

#### 問題 3: 訓練不穩定
較大的 λ (0.05, 0.1) 導致：
- 訓練過程中 entropy 與 reconstruction loss 衝突
- 最終兩者都變差（entropy 下降，accuracy 也下降）

---

## 結論

❌ **Entropy Regularization 方法失敗**

原因：
1. **正則化目標不匹配**：Token distribution entropy ≠ Anti-collapse
2. **忽略時序結構**：無法阻止連續幀使用相同 token
3. **訓練不穩定**：Entropy loss 與 reconstruction loss 衝突

建議：
- 放棄 global entropy regularization
- 轉向 **Codebook Refresh** (Exp 4)：直接重置未使用的 codebook entries
- 或考慮 **temporal diversity regularization**：懲罰連續幀使用相同 token

---

## 實驗資訊

- 訓練步數: 1000 steps
- Batch size: 8 (grad_accum=2, effective=16)
- Learning rate: 1e-4
- LoRA rank: 256, alpha: 512
- Evaluation interval: 200 steps
- 完成時間: 2026-02-02
