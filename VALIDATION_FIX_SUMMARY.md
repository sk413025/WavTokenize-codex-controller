# 驗證函數問題診斷報告

## 🔍 問題根因分析

### 核心問題
**驗證損失恆為 1000000 的原因**：模型在驗證模式下返回的輸出字典格式與訓練模式不同。

### 詳細分析

#### 1. 模型輸出格式差異

**訓練模式** (`self.training = True`):
```python
return {
    'logits': logits,              # [B, L, vocab_size]
    'target_tokens': target_tokens,
    'noisy_tokens': noisy_tokens,
    'clean_tokens': clean_tokens
}
```

**推理模式** (`self.training = False`):
```python
return {
    'denoised_audio': denoised_audio,  # 沒有 'logits'！
    'denoised_tokens': denoised_tokens,
    'noisy_tokens': noisy_tokens
}
```

#### 2. 驗證函數的錯誤調用

```python
def validate_epoch(model, dataloader, criterion, device):
    model.eval()  # ❌ 設置為評估模式，self.training = False
    
    with torch.no_grad():
        output = model(noisy_audio, clean_audio)
        
        # ❌ 嘗試訪問 'logits'，但推理模式沒有這個鍵
        logits = output['logits']  # KeyError: 'logits'
```

#### 3. 錯誤處理邏輯

```python
except Exception as e:
    logging.error(f"驗證批次 {batch_count} 出錯，跳過: {e}")
    # 所有批次都出錯，最終 valid_batches = 0
    
# 最後返回
if valid_batches > 0:
    avg_loss = total_loss / valid_batches
else:
    avg_loss = 1e6  # 返回 1000000
```

---

## 💡 解決方案

### 方案 1：在驗證時強制使用訓練模式的輸出（推薦）

修改模型的 `forward()` 方法，添加一個參數控制輸出格式：

```python
def forward(self, noisy_audio, clean_audio=None, return_logits=False):
    """
    Args:
        return_logits: 即使在評估模式也返回 logits（用於驗證）
    """
    noisy_tokens = self.encode_audio_to_tokens(noisy_audio)
    
    # 修改條件：訓練模式 OR 需要返回 logits
    if (self.training or return_logits) and clean_audio is not None:
        # 返回包含 logits 的字典
        ...
        return {
            'logits': logits,
            'target_tokens': target_tokens,
            'noisy_tokens': noisy_tokens_adjusted,
            'clean_tokens': clean_tokens
        }
    else:
        # 推理模式
        ...
        return {
            'denoised_audio': denoised_audio,
            'denoised_tokens': denoised_tokens,
            'noisy_tokens': noisy_tokens
        }
```

然後修改驗證函數：

```python
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    
    with torch.no_grad():
        # 添加 return_logits=True 參數
        output = model(noisy_audio, clean_audio, return_logits=True)
        
        logits = output['logits']  # ✅ 現在可以正常訪問
        target_tokens = output['target_tokens']
```

### 方案 2：修改驗證函數邏輯（更簡單，但不優雅）

直接在驗證時臨時切換回訓練模式：

```python
def validate_epoch(model, dataloader, criterion, device):
    model.eval()  # 關閉 dropout 等
    
    with torch.no_grad():
        # 臨時設置 training=True 以獲取正確的輸出格式
        model.training = True
        output = model(noisy_audio, clean_audio)
        model.training = False
        
        logits = output['logits']
        target_tokens = output['target_tokens']
```

⚠️ **注意**：這個方案會影響 BatchNorm、Dropout 的行為

### 方案 3：完全重寫驗證邏輯（最複雜）

不使用 `model.eval()`，而是手動計算驗證損失：

```python
def validate_epoch(model, dataloader, criterion, device):
    # 保持訓練模式，但關閉梯度
    was_training = model.training
    model.train()  # 保持訓練模式
    
    with torch.no_grad():
        for batch in dataloader:
            output = model(noisy_audio, clean_audio)
            logits = output['logits']
            target_tokens = output['target_tokens']
            # 計算損失...
    
    # 恢復原始狀態
    if not was_training:
        model.eval()
```

---

## 🎯 推薦實施步驟

### 步驟 1：修改模型 forward 方法

```python
# 在 wavtokenizer_transformer_denoising.py 的模型類中
def forward(self, noisy_audio, clean_audio=None, return_logits=False):
    """
    Args:
        noisy_audio: 輸入的噪聲音頻
        clean_audio: 乾淨音頻（訓練時提供）
        return_logits: 強制返回 logits 格式（用於驗證），即使在 eval 模式
    """
    noisy_tokens = self.encode_audio_to_tokens(noisy_audio)
    
    # 修改條件判斷
    if (self.training or return_logits) and clean_audio is not None:
        # ... 訓練模式的邏輯（返回 logits）
        return {
            'logits': logits,
            'target_tokens': target_tokens,
            'noisy_tokens': noisy_tokens_adjusted,
            'clean_tokens': clean_tokens
        }
    else:
        # ... 推理模式的邏輯
        return {
            'denoised_audio': denoised_audio,
            'denoised_tokens': denoised_tokens,
            'noisy_tokens': noisy_tokens
        }
```

### 步驟 2：修改驗證函數調用

```python
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    # ... 其他代碼 ...
    
    with torch.no_grad():
        # 添加 return_logits=True
        output = model(noisy_audio, clean_audio, return_logits=True)
        
        logits = output['logits']
        target_tokens = output['target_tokens']
        # ... 其他代碼 ...
```

### 步驟 3：測試驗證函數

運行一個小的驗證測試：

```python
# 測試驗證是否正常工作
python -c "
import torch
from wavtokenizer_transformer_denoising import *

# 加載模型
model = torch.load('results/transformer_large_tokenloss_large_tokenloss_202510190523/best_model.pth')

# 測試驗證函數
# ...
"
```

---

## 📋 實施檢查清單

- [ ] 修改 `forward()` 方法，添加 `return_logits` 參數
- [ ] 修改驗證函數，傳入 `return_logits=True`
- [ ] 測試驗證函數是否正常返回損失
- [ ] 重新啟動訓練，觀察驗證損失是否正常
- [ ] 確認驗證損失不再是 1000000
- [ ] 檢查 Best Model 是否根據驗證損失正確保存

---

## 🚀 修復後的預期效果

修復後，驗證日誌應該顯示：

```
2025-10-20 XX:XX:XX,XXX - INFO - Epoch 301/1000
2025-10-20 XX:XX:XX,XXX - INFO - Train Loss: 3.6XXX, Val Loss: 3.8XXX, Val Acc: 0.25XX
                                                        ^^^^^^^^ 不再是 1000000！
```

預期驗證損失範圍：3.5 - 4.5（應該略高於訓練損失）

---

**報告生成時間**: 2025-10-20  
**問題嚴重性**: 🔴 高（影響模型選擇和過擬合判斷）  
**修復難度**: 🟢 低（只需修改 1-2 個函數）  
**預計修復時間**: 15-30 分鐘
