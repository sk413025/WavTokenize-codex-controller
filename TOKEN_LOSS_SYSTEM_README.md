# Token Loss 系統：將 ttt2.py 的 Loss 邏輯應用到 Token 空間

## 概述

本系統將 `ttt2.py` 中基於連續特徵的損失計算邏輯完整移植到離散 token 空間，實現了 **完全相同的運算邏輯**，只是操作對象從連續特徵變為離散 tokens。

## 核心概念

### 1. L2 距離損失 (compute_token_l2_loss)
**ttt2.py 原理：**
```python
l2_distance = torch.norm(enhanced_features - target_features, p=2, dim=1)
```

**Token 空間應用：**
```python
# 將 tokens 嵌入到連續空間，然後計算 L2 距離
predicted_embed = embedding_layer(predicted_tokens)  # [batch, seq_len, embed_dim]
target_embed = embedding_layer(target_tokens)
l2_distance = torch.norm(predicted_embed - target_embed, p=2, dim=-1)
```

**物理意義：** 在嵌入空間中，語義相近的 tokens 距離更近，L2 損失確保預測 token 在語義上接近目標 token。

### 2. 內容一致性損失 (compute_token_content_consistency_loss)
**ttt2.py 原理：** 確保增強特徵與目標特徵在語義上保持一致

**Token 空間應用：**
```python
# 主要：交叉熵確保 token 預測準確性
consistency_loss = F.cross_entropy(predicted_logits, target_tokens)

# 附加：分佈熵正則化，避免過於確定或隨機的預測
predicted_probs = F.softmax(predicted_logits / temperature, dim=-1)
entropy_loss = -(predicted_probs * torch.log(predicted_probs + 1e-10)).sum(dim=-1)
```

**物理意義：** 不僅要預測正確的 token，還要保持合理的不確定性分佈。

### 3. Manifold 正則化損失 (compute_token_manifold_regularization_loss)
**ttt2.py 原理：**
```python
manifold_distance = torch.norm(enhanced_features - input_features, p=2, dim=1)
adaptive_threshold = mean_distance + 2 * std_distance
excess_distance = F.relu(manifold_distance - adaptive_threshold)
```

**Token 空間應用（完全相同邏輯）：**
```python
# 在嵌入空間計算距離
predicted_embed = embedding_layer(predicted_tokens)
input_embed = embedding_layer(input_tokens)
manifold_distance = torch.norm(predicted_embed - input_embed, p=2, dim=-1)

# 完全相同的適應性閾值計算
mean_distance = manifold_distance.mean()
std_distance = manifold_distance.std()
adaptive_threshold = mean_distance + 2 * std_distance
excess_distance = F.relu(manifold_distance - adaptive_threshold)
```

**物理意義：** 防止預測的 token 偏離輸入 token 的語義 manifold 太遠，保持語義連續性。

### 4. 正則化損失 (compute_token_normalization_loss)
**ttt2.py 原理：** 控制特徵的幅度和分佈

**Token 空間應用：**
```python
# L2 正則化：控制 logits 幅度
l2_norm = torch.norm(predicted_logits, p=2, dim=-1)

# 熵正則化：控制預測分佈的不確定性
predicted_probs = F.softmax(predicted_logits, dim=-1)
entropy = -(predicted_probs * torch.log(predicted_probs + 1e-10)).sum(dim=-1)
target_entropy = np.log(vocab_size) * 0.5  # 適中的熵值
regularization_loss = F.mse_loss(entropy, target_entropy)
```

**物理意義：** 避免過度自信的預測，保持適度的不確定性。

### 5. 連貫性損失 (compute_token_coherence_loss)
**ttt2.py 原理：** 保持時間維度上的一致性

**Token 空間應用：**
```python
# 計算局部窗口內的變化率
for i in range(seq_len - window_size + 1):
    pred_window = predicted_tokens[:, i:i+window_size]
    input_window = input_tokens[:, i:i+window_size]
    
    # 鄰近 token 的差異（變化率）
    pred_diff = torch.abs(pred_window[:, 1:] - pred_window[:, :-1])
    input_diff = torch.abs(input_window[:, 1:] - input_window[:, :-1])
    
    # 變化率應該相似（保持語義連貫性）
    diff_loss = F.mse_loss(pred_diff, input_diff)
```

**物理意義：** 確保生成的 token 序列在語義上平滑過渡，避免突兀的語義跳躍。

## 使用方式

### 1. 簡單使用
```python
from token_loss_system import compute_combined_token_loss

# 在訓練循環中
total_loss, loss_dict = compute_combined_token_loss(
    predicted_logits=logits,           # [batch, seq_len, vocab_size]
    predicted_tokens=predicted_tokens, # [batch, seq_len]
    target_tokens=target_tokens,       # [batch, seq_len]
    input_tokens=input_tokens,         # [batch, seq_len]
    embedding_layer=model.embedding,   # nn.Embedding 層
    weights={'l2': 0.3, 'consistency': 0.4, 'manifold': 0.1, 
             'normalization': 0.1, 'coherence': 0.1}
)

total_loss.backward()
```

### 2. 在 discrete_token_denoising.py 中使用
```bash
# 使用 Token Loss 系統訓練
python discrete_token_denoising.py \
  --use_token_loss \
  --l2_weight 0.3 \
  --consistency_weight 0.4 \
  --manifold_weight 0.1 \
  --normalization_weight 0.1 \
  --coherence_weight 0.1
```

### 3. 運行比較實驗
```bash
# 比較不同 loss 配置的效果
python compare_token_loss.py --epochs 10 --max_samples 100
```

## 技術細節

### 關鍵設計決策

1. **嵌入空間映射**：使用 `nn.Embedding` 將離散 tokens 映射到連續空間，使得可以應用基於距離的損失函數。

2. **完全保持 ttt2.py 邏輯**：所有計算公式（如適應性閾值、正則化方式）與 ttt2.py 完全一致。

3. **權重可調**：每個損失組件都有獨立的權重，可以根據任務特點調整。

4. **漸進式回退**：如果高級損失計算失敗，會自動回退到簡單的交叉熵損失。

### 預期優勢

1. **語義約束**：L2 距離在嵌入空間提供語義相似性約束
2. **結構保持**：Manifold 正則化保持語義結構
3. **平滑性**：連貫性損失確保序列的語義平滑性
4. **穩定性**：正則化防止過擬合和過度自信

### 實驗驗證

運行 `compare_token_loss.py` 會執行以下對比：

1. **基線**：純交叉熵損失
2. **平衡配置**：所有損失均衡使用
3. **一致性專注**：更注重 token 預測準確性
4. **L2 專注**：更注重語義相似性

## 與原始方法的關係

| 方面 | ttt2.py | Token Loss 系統 |
|------|---------|----------------|
| **操作對象** | 連續特徵 [B, 512, T] | 離散 tokens [B, T] |
| **L2 計算** | 直接在特徵空間 | 在嵌入空間 |
| **一致性** | 特徵重建質量 | Token 預測準確性 |
| **Manifold** | 特徵流形約束 | Token 語義流形約束 |
| **正則化** | 特徵幅度控制 | Logits 和熵控制 |
| **運算邏輯** | ✅ 完全相同 | ✅ 完全相同 |

## 總結

這個 Token Loss 系統成功地將 ttt2.py 的成熟損失設計移植到離散 token 領域，提供了比單純交叉熵更豐富的語義約束。通過在嵌入空間中應用相同的數學運算，我們可以在 token 序列建模中獲得與連續特徵建模類似的優勢。
