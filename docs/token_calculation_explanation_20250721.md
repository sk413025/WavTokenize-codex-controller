# WavTokenizer 離散編碼與 Token 計算問題解釋

## 2025-07-21

## 問題總結

在 `verify_wavtokenizer.py` 腳本中，我們發現 token 計算使用了不正確的維度，導致每秒 token 數被報告為非常低（0.25-0.38 tokens/s）。正確實作後，每秒 token 數應該約為 75 tokens/s，符合 WavTokenizer 模型的設計規格。

## 離散編碼結構解釋

WavTokenizer 模型在編碼音頻時，會生成離散編碼（discrete code），其形狀通常為：
```
[batch_size, num_codebooks, seq_length]
```

其中：
- `batch_size`：批次大小，通常為 1（處理單個音頻時）
- `num_codebooks`：編碼本數量，在我們的模型中通常是 8 或 16
- `seq_length`：序列長度，即時間維度上的 token 數量

## 問題原因

在原始的 `verify_wavtokenizer.py` 中，token 數量計算使用了以下方式：
```python
tokens_per_second = discrete_code.shape[1] / audio_length_seconds
```

這裡使用了 `shape[1]`，它實際上是編碼本的數量（num_codebooks），而非時間維度上的 token 數量。編碼本數量通常是固定的（8 或 16），這就是為什麼計算結果非常低的原因。

## 正確的計算方式

正確的 token 計算應該使用序列長度，即最後一維：
```python
tokens = discrete_code.shape[-1]  # 使用最後一維（seq_length）
tokens_per_second = tokens / audio_length_seconds
```

使用 `shape[-1]` 可以安全地訪問最後一維，不管維度的總數是幾個。

## 驗證結果

使用修正後的計算方法，我們得到了預期的結果：每秒約 75 個 token，這符合 WavTokenizer 的設計規格。

## 最佳實踐

為了避免類似的混淆，建議：

1. **明確記錄張量的維度含義**：在代碼註釋中明確記錄每個維度的含義。
2. **使用語義明確的變量名**：如 `num_tokens = discrete_code.shape[-1]` 而不是直接使用形狀索引。
3. **使用打印調試**：在不確定形狀結構時，先打印出整個形狀和每個維度的值。

## 範例代碼

```python
# 正確的 token 計算方式
discrete_code_shape = discrete_code.shape
print(f"離散編碼形狀: {discrete_code_shape}")  # 例如: [1, 8, 1500]

batch_size = discrete_code_shape[0]
num_codebooks = discrete_code_shape[1]
seq_length = discrete_code_shape[2]  # 或 discrete_code_shape[-1]

print(f"批次大小: {batch_size}")
print(f"編碼本數量: {num_codebooks}")
print(f"序列長度 (token 數): {seq_length}")

audio_length_seconds = wav.shape[1] / 24000  # 假設採樣率為 24kHz
tokens_per_second = seq_length / audio_length_seconds

print(f"音頻長度: {audio_length_seconds:.2f} 秒")
print(f"總 token 數: {seq_length}")
print(f"每秒 token 數: {tokens_per_second:.2f}")
```

## 維度解釋圖

```
離散編碼張量 (discrete_code):
[batch_size, num_codebooks, seq_length]
    ^            ^             ^
    |            |             |
    |            |             +-- 時間維度上的 token 數量 (discrete_code.shape[-1])
    |            |
    |            +-- 編碼本數量 (discrete_code.shape[1])
    |
    +-- 批次大小 (discrete_code.shape[0])
```

透過這次錯誤修正，我們更清楚地理解了 WavTokenizer 模型中離散編碼的結構，確保了正確的 token 計算方式。
