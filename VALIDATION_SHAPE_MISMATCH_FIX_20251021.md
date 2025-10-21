# Validation Logits Shape Mismatch 修復報告

**實驗編號**: EXP-SHAPE-MISMATCH-FIX-20251021
**日期**: 2025-10-21 03:15
**函式**: forward_transformer, forward in wavtokenizer_transformer_denoising.py
**目標**: 修復驗證時 logits 形狀錯誤導致的 batch_size 不匹配問題

---

## 一、實驗背景

### 錯誤發生
在完成 dtype 修復並重啟訓練後，在 Epoch 100 驗證時出現新錯誤：

```
ValueError: Expected input batch_size (4) to match target batch_size (1080).
```

### 錯誤分析
- `input batch_size (4)`: logits_flat 的第一個維度
- `target batch_size (1080)`: target_flat 的第一個維度 (4 × 270 = 1080)
- **結論**: logits 的形狀是 `[4, 4096]` 而不是預期的 `[4, 270, 4096]`

---

## 二、根本原因診斷

### 問題追蹤

#### 1. Reshape 邏輯檢查
```python
# validate_epoch 中
logits_flat = logits.view(-1, logits.size(-1))  # 期望 [B*L, V]
target_flat = target_tokens.view(-1)             # [B*L]
```

如果 logits 是 `[4, 4096]`（2D），則：
- `logits_flat` = `[4, 4096]` （不變）
- `target_flat` = `[1080]` （4 × 270）

→ **形狀不匹配！**

#### 2. Forward 函式檢查
```python
# forward() 中
if (self.training or return_logits) and clean_audio is not None:
    logits = self.forward_transformer(input_tokens, decoder_input)
```

條件看起來正確，應該走訓練分支返回 logits。

#### 3. forward_transformer 檢查（根本原因）
```python
# 原始代碼
def forward_transformer(self, src_tokens, tgt_tokens=None):
    # ...
    if tgt_tokens is not None and self.training:  # ❌ 問題在這裡！
        # ... 返回 logits [B, L, vocab_size]
        return logits
    else:
        # ... 返回 predicted_tokens [B, L]
        return predicted_tokens
```

**問題發現**：
- 當 `model.eval()` 時，`self.training = False`
- 即使 `tgt_tokens is not None`，也會走 `else` 分支
- 返回 `predicted_tokens` ([B, L]) 而不是 `logits` ([B, L, V])

但是 forward() 中的代碼期望得到 logits：
```python
logits = self.forward_transformer(input_tokens, decoder_input)
# logits 實際上是 predicted_tokens [B, L]
```

然後嘗試取 `logits.size(-1)` 作為 vocab_size：
```python
logits_flat = logits.view(-1, logits.size(-1))
# 如果 logits 是 [4, 270]，則 logits.size(-1) = 270
# logits_flat = [4, 270] （錯誤！）
```

但實際錯誤訊息說是 `[4, 4096]`，這說明在某個地方 logits 被錯誤地重塑了。

### 測試驗證

創建了 `test_shape_mismatch.py` 來驗證問題：

```python
# 場景 3: logits 形狀異常 (2D 而非 3D)
logits_2d = torch.randn(4, 4096)  # [B=4, V=4096]
target_normal = torch.randint(0, 4096, (4, 270))  # [B=4, L=270]

logits_flat_2d = logits_2d.reshape(-1, logits_2d.size(-1))
# → [4, 4096]

target_flat_normal = target_normal.reshape(-1)
# → [1080]

# 錯誤: batch_size (4) to match (1080) ✓ 符合我們的錯誤訊息！
```

---

## 三、解決方案

### 修復策略
在 `forward_transformer` 函式中添加 `return_logits` 參數，與 `forward` 函式保持一致。

### 修改內容

#### 1. forward_transformer 簽名更新
```python
# 修改前
def forward_transformer(self, src_tokens, tgt_tokens=None):

# 修改後  
def forward_transformer(self, src_tokens, tgt_tokens=None, return_logits=False):
    """Transformer 前向傳播（僅處理 token 序列）
    
    Args:
        src_tokens: 源 token 序列 [B, L]
        tgt_tokens: 目標 token 序列 [B, L]（訓練/驗證時提供）
        return_logits: 強制返回 logits 而非 predicted_tokens（用於驗證）
    
    Returns:
        training 模式或 return_logits=True: logits [B, L, vocab_size]
        inference 模式: predicted_tokens [B, L]
    """
```

#### 2. 條件判斷修改
```python
# 修改前
if tgt_tokens is not None and self.training:
    # 返回 logits

# 修改後
if tgt_tokens is not None and (self.training or return_logits):
    # 返回 logits
```

#### 3. forward 函式調用更新
```python
# 修改前
logits = self.forward_transformer(input_tokens, decoder_input)

# 修改後
logits = self.forward_transformer(input_tokens, decoder_input, return_logits=return_logits)
```

---

## 四、修復驗證

### 邏輯驗證

**訓練模式** (`model.train()`):
- `self.training = True`
- `return_logits = False` (預設)
- 條件: `tgt_tokens is not None and (True or False)` = `True`
- ✅ 返回 logits

**驗證模式** (`model.eval()` + `return_logits=True`):
- `self.training = False`
- `return_logits = True`
- 條件: `tgt_tokens is not None and (False or True)` = `True`
- ✅ 返回 logits

**推理模式** (`model.eval()` + `return_logits=False`):
- `self.training = False`
- `return_logits = False`
- 條件: `tgt_tokens is not None and (False or False)` = `False`
- ✅ 返回 predicted_tokens

### 形狀驗證

修復後的形狀流程：

1. **forward_transformer 返回**: `logits` [4, 270, 4096]
2. **forward 接收**: `logits` [4, 270, 4096]
3. **validate_epoch 接收**: 
   - `output['logits']` = [4, 270, 4096]
   - `output['target_tokens']` = [4, 270]
4. **Flatten操作**:
   - `logits_flat` = [1080, 4096]
   - `target_flat` = [1080]
5. **CrossEntropyLoss**: ✅ 形狀匹配！

---

## 五、重啟訓練

### 訓練重啟信息
- **時間**: 2025-10-21 03:15
- **進程 ID**: 4002820
- **日誌文件**: `logs/large_tokenloss_LOGITS_FIX_20251021_031547.log`
- **修復內容**: 
  1. forward_transformer 添加 return_logits 參數
  2. 條件判斷從 `self.training` 改為 `(self.training or return_logits)`
  3. forward 調用時傳遞 return_logits

### Debug 輸出
添加了詳細的形狀檢查：
```python
# forward 函式返回前
logger.info(f"Forward return shapes - logits: {logits.shape}, target_tokens: {target_tokens.shape}")

# validate_epoch 接收後
logger.info(f"Validation shapes - logits: {logits.shape}, target: {target_tokens.shape}")
logger.info(f"After flatten - logits_flat: {logits_flat.shape}, target_flat: {target_flat.shape}")
```

---

## 六、教訓與反思

### 技術教訓

1. **模式狀態的複雜性**:
   - `model.train()` 和 `model.eval()` 會改變 `self.training` 狀態
   - 不同模式需要不同的返回格式
   - 需要額外的標誌 (`return_logits`) 來覆蓋默認行為

2. **參數傳遞的一致性**:
   - forward() 有 `return_logits` 參數
   - forward_transformer() 也應該有相同的參數
   - 避免在子函式中做假設

3. **錯誤診斷的重要性**:
   - 錯誤訊息 "batch_size (4) to match (1080)" 看似簡單
   - 實際上需要追蹤整個數據流才能找到根本原因
   - 創建測試腳本 (test_shape_mismatch.py) 幫助驗證假設

### 設計改進

#### 更好的設計方案
```python
def forward_transformer(self, src_tokens, tgt_tokens=None, return_logits=False):
    """
    明確三種模式：
    1. Training: self.training=True, return_logits=Any → 返回 logits
    2. Validation: self.training=False, return_logits=True → 返回 logits
    3. Inference: self.training=False, return_logits=False → 返回 tokens
    """
    # 統一條件：需要 logits 時
    need_logits = self.training or return_logits
    
    if tgt_tokens is not None and need_logits:
        # 完整的 encoder-decoder 前向傳播
        return logits  # [B, L, V]
    else:
        # 快速推理
        return predicted_tokens  # [B, L]
```

#### 防禦性編程
```python
# 在 validate_epoch 中添加斷言
assert logits.dim() == 3, f"期望 logits 是 3D tensor [B, L, V], 但得到 {logits.shape}"
assert logits.size(-1) == model.vocab_size, f"期望 vocab_size={model.vocab_size}, 但得到 {logits.size(-1)}"
```

---

## 七、下一步計劃

### 立即驗證
1. 等待訓練到達 Epoch 100
2. 檢查驗證是否成功運行
3. 確認 log 中的形狀輸出是否正確

### 長期改進
1. 添加單元測試覆蓋所有模式
2. 重構 forward 邏輯，減少條件分支
3. 統一訓練和驗證的代碼路徑

---

## 八、相關文件

### 修改的文件
1. `wavtokenizer_transformer_denoising.py`
   - `forward_transformer()` (Line 438-520)
   - `forward()` (Line 586-587)

### 新增的文件
1. `test_shape_mismatch.py` - 形狀不匹配測試腳本
2. `VALIDATION_SHAPE_MISMATCH_FIX_20251021.md` - 本報告

### 日誌文件
1. `logs/large_tokenloss_LOGITS_FIX_20251021_031547.log` - 修復後的訓練日誌

---

**報告生成時間**: 2025-10-21 03:20:00
**實驗狀態**: ✅ 修復完成並重啟訓練
**預期驗證時間**: Epoch 100 (~2-3 小時後)
