# 驗證流程優化記錄 - 2025/10/21

## 📋 改進概述

本次改進優化了 `WavTokenizerTransformerDenoiser` 的驗證流程，使代碼更清晰、更易維護。

## 🎯 核心改進

### 1. **引入 `return_logits` 參數**

在 `forward()` 方法中添加 `return_logits` 參數，允許在 `eval` 模式下也能返回 logits 用於損失計算。

#### 修改前的問題：
```python
# 舊方法：需要在驗證時切換模式
def validate_epoch(...):
    model.eval()
    with torch.no_grad():
        # 需要臨時切換回 train 模式才能獲取 logits？
        # 或者需要特殊的條件判斷？
        output = model(noisy_audio, clean_audio)
```

#### 修改後的優雅方案：
```python
# 新方法：通過參數控制返回格式
def validate_epoch(...):
    model.eval()  # 只需要在開頭設置一次
    with torch.no_grad():
        # 直接使用 return_logits=True，即使在 eval 模式也返回 logits
        output = model(noisy_audio, clean_audio, return_logits=True)
        logits = output['logits']
        target_tokens = output['target_tokens']
        # ... 計算損失 ...
```

### 2. **forward() 方法的智能條件判斷**

```python
def forward(self, noisy_audio, clean_audio=None, return_logits=False):
    """完整的前向傳播：Audio → Tokens → Transformer → Tokens → Audio
    
    Args:
        noisy_audio: 輸入的噪聲音頻 [B, 1, T]
        clean_audio: 乾淨音頻（訓練/驗證時提供）[B, 1, T]
        return_logits: 強制返回 logits 格式（用於驗證），即使在 eval 模式也返回 logits
                     - True: 返回 {'logits', 'target_tokens', 'noisy_tokens', 'clean_tokens'}
                     - False (推理): 返回 {'denoised_audio', 'denoised_tokens', 'noisy_tokens'}
    
    Returns:
        dict: 根據模式返回不同的內容
            訓練/驗證模式 (training=True 或 return_logits=True):
                - logits: [B, L, 4096] Transformer 輸出的 token 機率分佈
                - target_tokens: [B, L] 目標 token 序列
                - noisy_tokens: [B, L-1] 噪聲 token 序列（已調整長度）
                - clean_tokens: [B, L_original] 原始乾淨 token 序列
            推理模式 (training=False 且 return_logits=False):
                - denoised_audio: [B, 1, T] 降噪後的音頻
                - denoised_tokens: [B, L] 預測的 token 序列
                - noisy_tokens: [B, L] 原始噪聲 token 序列
    """
    
    noisy_tokens = self.encode_audio_to_tokens(noisy_audio)
    
    # 關鍵條件：訓練模式 OR 需要返回 logits（用於驗證）
    # 這樣在 model.eval() + return_logits=True 時也能計算 loss
    if (self.training or return_logits) and clean_audio is not None:
        # 訓練/驗證邏輯：返回 logits 和 tokens
        ...
        return {
            'logits': logits,
            'target_tokens': target_tokens,
            'noisy_tokens': noisy_tokens_adjusted,
            'clean_tokens': clean_tokens
        }
    else:
        # 推理邏輯：返回 audio 和 tokens
        ...
        return {
            'denoised_audio': denoised_audio,
            'denoised_tokens': denoised_tokens,
            'noisy_tokens': noisy_tokens
        }
```

### 3. **validate_epoch() 的清晰實現**

```python
def validate_epoch(model, dataloader, criterion, device):
    """驗證一個 epoch
    
    優化設計：
        - 使用 model.eval() 設置評估模式（開頭設置一次即可）
        - 通過 return_logits=True 強制返回 logits，即使在 eval 模式
        - 避免了在驗證時切換回 train 模式的混亂
        - 保持梯度關閉 (torch.no_grad())，節省內存
    """
    model.eval()  # 只需要在開頭設置一次
    
    with torch.no_grad():  # 禁用梯度計算
        for batch in dataloader:
            noisy_audio, clean_audio = batch[0], batch[1]
            
            # 關鍵：使用 return_logits=True
            output = model(noisy_audio, clean_audio, return_logits=True)
            
            logits = output['logits']
            target_tokens = output['target_tokens']
            
            # 計算損失...
            loss = criterion(logits, target_tokens)
```

## ✅ 優勢總結

### 1. **代碼清晰度 ⬆️**
- 不需要在驗證時切換 `model.train()` / `model.eval()`
- 通過明確的參數 `return_logits` 控制行為
- 邏輯分離清晰：訓練/驗證 vs 推理

### 2. **維護性 ⬆️**
- 單一職責：`forward()` 根據參數決定返回內容
- 易於理解：一看就知道 `return_logits=True` 是用於驗證
- 減少條件判斷的複雜度

### 3. **安全性 ⬆️**
- 驗證時保持 `model.eval()` 狀態，確保 Dropout/BatchNorm 行為正確
- 使用 `torch.no_grad()` 確保不計算梯度
- 避免意外的模式切換導致的 bug

### 4. **靈活性 ⬆️**
- 可以在推理時也返回 logits（設置 `return_logits=True`）
- 支持不同的使用場景：
  - 訓練：`model.train()` + `forward(noisy, clean)`
  - 驗證：`model.eval()` + `forward(noisy, clean, return_logits=True)`
  - 推理：`model.eval()` + `forward(noisy)`

## 🔍 使用示例

### 訓練時
```python
model.train()
optimizer.zero_grad()

output = model(noisy_audio, clean_audio)  # return_logits=False (default)
# 因為 model.training=True，所以會返回 logits

loss = criterion(output['logits'], output['target_tokens'])
loss.backward()
optimizer.step()
```

### 驗證時
```python
model.eval()
with torch.no_grad():
    output = model(noisy_audio, clean_audio, return_logits=True)
    # 即使 model.training=False，因為 return_logits=True，也會返回 logits
    
    loss = criterion(output['logits'], output['target_tokens'])
    # 計算驗證損失...
```

### 推理時
```python
model.eval()
with torch.no_grad():
    output = model(noisy_audio)  # 不提供 clean_audio
    # 或者
    output = model(noisy_audio, return_logits=False)
    
    denoised_audio = output['denoised_audio']
    # 直接獲得降噪後的音頻
```

## 📊 改進對比表

| 特性 | 改進前 | 改進後 |
|------|--------|--------|
| **驗證時的模式切換** | 可能需要臨時切換模式 | 保持 `eval()` 模式，通過參數控制 |
| **代碼可讀性** | 需要理解複雜的條件判斷 | 一目了然：`return_logits=True` |
| **維護難度** | 較高，邏輯分散 | 較低，邏輯集中在參數 |
| **出錯風險** | 模式切換可能忘記恢復 | 無模式切換風險 |
| **靈活性** | 有限 | 高，支持多種使用場景 |

## 🎯 結論

這次改進通過引入 `return_logits` 參數，使得驗證流程更加優雅和安全。主要優勢在於：

1. ✅ **語義清晰**：參數名稱直接表達意圖
2. ✅ **邏輯分離**：訓練/驗證/推理的邏輯清晰分離
3. ✅ **安全可靠**：避免模式切換帶來的風險
4. ✅ **易於維護**：代碼結構清晰，易於理解和修改

這是一個優秀的軟體工程實踐案例！🎉

---

**修改日期**: 2025/10/21  
**實驗ID**: large_tokenloss_FIXED_LR_202510210121  
**當前訓練狀態**: Epoch 1, Token Accuracy 8-15% ✅ (LR 修復已生效)
