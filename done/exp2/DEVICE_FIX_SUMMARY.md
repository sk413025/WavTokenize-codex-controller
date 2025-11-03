# Device (CPU/GPU) 錯誤修正

## 問題

訓練時 Speaker Loss 一直是 0.0000，檢查後發現錯誤：
```
⚠️  Speaker loss computation failed: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
```

## 根本原因

WavTokenizer的 `decode()` 方法返回的音頻tensor可能在CPU上，而Speaker Encoder (ECAPA-TDNN) 的權重在GPU上，導致device mismatch。

## 已實施的修正

### 1. [loss_with_speaker.py:89](done/exp2/loss_with_speaker.py#L89)
在 `_decode_soft_features_to_audio()` 的return語句加上 `.to(self.device)`

### 2. [loss_with_speaker.py:126](done/exp2/loss_with_speaker.py#L126)
在 `_decode_tokens_to_audio()` 中，decode前先確保features在GPU上：
```python
features = features.to(self.device)
```

### 3. [loss_with_speaker.py:134](done/exp2/loss_with_speaker.py#L134)
在 `_decode_tokens_to_audio()` 的return語句加上 `.to(self.device)`

## 期待結果

修正後，Speaker Loss 應該要 > 0（預期範圍：0.01 - 0.10），而不是一直停留在 0.0000。

## 測試方法

```bash
bash done/exp2/run_test_quick.sh
# 檢查log中的 Speaker Loss 值
tail -f results/exp2/test_minimal/training.log | grep "Speaker"
```

## 當前狀態

修正已完成，等待測試驗證。
