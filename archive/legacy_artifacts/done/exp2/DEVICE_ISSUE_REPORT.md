# EXP2 Device Issue 完整報告

## 當前狀態

訓練框架已經基本完成，但有一個**嚴重的device mismatch問題**阻止Speaker Loss計算。

## 核心問題

WavTokenizer的 `decode()` 方法返回的audio tensor **始終在CPU上**，即使：
- 輸入features已經在GPU上
- bandwidth_id在GPU上
- 調用了 `.to(self.device)`

錯誤訊息：
```
RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
```

## 已嘗試的解決方案（均失敗）

### 嘗試 1: 在decode後加 `.to(self.device)`
```python
audio = self.wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
return audio.to(self.device)  # ❌ 沒有效果
```

### 嘗試 2: 在speaker encoder調用前加 `.to(self.device)`
```python
pred_audio = pred_audio.to(self.device)
noisy_audio = noisy_audio.to(self.device)
pred_emb = self.speaker_encoder(pred_audio)  # ❌ 仍然失敗
```

### 嘗試 3: 使用 `.contiguous()`
```python
return audio.to(self.device).contiguous()  # ❌ 沒有效果
```

##根本原因分析

可能的原因：
1. **WavTokenizer內部強制CPU**：WavTokenizer的decode方法內部可能有 `.cpu()` 調用
2. **SpeechBrain的特殊處理**：Speaker Encoder (ECAPA-TDNN) 可能需要特定格式的input
3. **torch.no_grad()的副作用**：在no_grad context中，`.to()` 可能不會真正移動tensor

## 潛在解決方案

###方案A：使用cuda()而不是to(device)
```python
if torch.cuda.is_available():
    pred_audio = pred_audio.cuda()
    noisy_audio = noisy_audio.cuda()
```

### 方案B：在no_grad外面調用decode
移除 `_decode_tokens_to_audio` 中的 `with torch.no_grad()` wrapper

### 方案C：創建新的GPU tensor
```python
pred_audio_gpu = torch.empty_like(pred_audio, device='cuda')
pred_audio_gpu.copy_(pred_audio)
```

### 方案D：檢查WavTokenizer源碼
找到為什麼decode總是返回CPU tensor

## 測試狀態

- ✅ 訓練循環正常運行
- ✅ CE Loss正常計算和下降
- ✅ Token Accuracy正常提升
- ❌ **Speaker Loss始終為0.0000（因為計算失敗）**
- ❌ 無法驗證梯度流是否正確

## 下一步建議

1. **最優先**：嘗試方案B - 移除no_grad wrapper看是否解決
2. 如果B失敗，嘗試方案A - 使用`.cuda()`
3. 如果還失敗，需要debug WavTokenizer源碼
4. 最後手段：直接使用CPU版本的speaker encoder（會很慢）

##文件位置

- 損失函數：`done/exp2/loss_with_speaker.py`
- 問題方法：
  - `_decode_soft_features_to_audio()` (line 70-89)
  - `_decode_tokens_to_audio()` (line 91-137)
  - `_compute_speaker_loss_from_features()` (line 139-180)

## 訓練日誌

```
Epoch 1/10
  Train - Loss: 8.0513, CE: 8.0513, Speaker: 0.0000, Acc: 9.35%
Epoch 2/10
  Train - Loss: 7.1406, CE: 7.1406, Speaker: 0.0000, Acc: 20.87%
...
```

注意：Speaker Loss = 0.0000 表示計算失敗，不是真的為0。
