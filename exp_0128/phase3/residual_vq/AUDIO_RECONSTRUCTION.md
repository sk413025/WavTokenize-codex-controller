# RVQ 音頻重建方案

## 問題

RVQ 使用多層 codebook，輸出多層 codes，而原始 WavTokenizer decoder 只能處理單層 codes。

## 解決方案：使用 Quantized Vectors

### 核心思路

**不使用 codes，而是直接使用 quantized vectors**

```python
# RVQ 輸出
z_q = q0 + q1 + q2 + q3  # [batch, 512, time]

# Teacher decoder 接受 quantized vectors
audio = teacher.decoder(z_q)  # [batch, 1, audio_length]
```

### 為什麼這樣可行？

#### 原始 WavTokenizer 流程

```python
# Encoding
audio → encoder → z → quantizer → codes
                              ↓
                        quantized (z_q)
                              ↓
                         decoder → audio

# Decoder 實際輸入
# 不是 codes，而是 quantized vectors!
decoder(quantized_vectors) → audio
```

**關鍵發現**: Decoder 不直接使用 codes，而是使用 quantized vectors！

#### RVQ 流程

```python
# Student (RVQ)
audio → encoder → z → RVQ → multi_codes
                         ↓
                    quantized (z_q = q0+q1+q2+q3)
                         ↓
                    teacher.decoder → audio ✅

# Teacher decoder 只看 quantized vectors 的形狀
# z_q: [batch, 512, time] ← 符合 decoder 期望
```

## 實作細節

### 1. 在 `models_rvq.py` 中添加 decode 方法

```python
class TeacherStudentRVQ(TeacherStudentIntermediate):
    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        使用 teacher decoder 重建音頻

        Args:
            quantized: [batch, dim, time] - RVQ 量化後的向量

        Returns:
            audio: [batch, 1, audio_length] - 重建的音頻
        """
        with torch.no_grad():
            audio = self.teacher.feature_extractor.encodec.decoder(quantized)
        return audio
```

### 2. 在 `train_rvq_short_run.py` 中保存音頻

```python
def save_audio_samples(model, data_loader, device, output_dir, step, ...):
    # Forward pass
    outputs = model(clean_audio, noisy_audio)

    # Get RVQ quantized vectors
    student_quantized = outputs['student_quantized']  # [batch, 512, time]

    # Reconstruct using teacher decoder
    reconstructed = model.decode(student_quantized)  # [batch, 1, audio_length]

    # Save audio files
    torchaudio.save('reconstructed.wav', reconstructed[0].cpu(), 16000)
```

## 輸出結果

訓練時會在以下位置保存音頻樣本：

```
run_exp5b_TIMESTAMP/
└── audio_samples/
    ├── step_000000/
    │   ├── val_sample0_clean.wav        # 乾淨音頻（目標）
    │   ├── val_sample0_noisy.wav        # 噪音音頻（輸入）
    │   ├── val_sample0_reconstructed.wav # RVQ 重建（輸出）
    │   ├── val_sample1_clean.wav
    │   ├── val_sample1_noisy.wav
    │   └── val_sample1_reconstructed.wav
    ├── step_000200/
    │   └── ...
    ├── step_000400/
    │   └── ...
    └── ...
```

## 評估音頻質量

### 比較三種音頻

1. **Clean** (clean.wav): 原始乾淨音頻
2. **Noisy** (noisy.wav): 加入噪音的音頻（模型輸入）
3. **Reconstructed** (reconstructed.wav): RVQ 重建的音頻（模型輸出）

### 預期結果

如果 RVQ 訓練成功：
- **Reconstructed 應該接近 Clean**
- 但 codebook usage 更多樣化（解決 collapse）

### 音頻質量指標

可以使用以下指標評估：
```python
# SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
si_snr = compute_si_snr(clean, reconstructed)

# PESQ (Perceptual Evaluation of Speech Quality)
pesq_score = compute_pesq(clean, reconstructed)

# STOI (Short-Time Objective Intelligibility)
stoi_score = compute_stoi(clean, reconstructed)
```

## 與原始方法的對比

### 原始 WavTokenizer

```
Encoding:
  audio → encoder → quantizer → codes → (lookup) → quantized

Decoding:
  codes → (lookup) → quantized → decoder → audio
```

### RVQ 方法

```
Encoding:
  audio → encoder → RVQ → multi_codes → (sum) → quantized

Decoding:
  quantized → decoder → audio  ← 直接用 quantized，跳過 codes
```

**關鍵差異**: 我們直接使用 quantized vectors，不需要經過 codes lookup！

## 技術細節

### Quantized Vectors 的格式

```python
# 形狀
quantized.shape = [batch, 512, time_frames]

# 例如
batch = 8
dim = 512
time_frames = 75  # 對應 1.5 秒 @ 16kHz

# Decoder 輸入
quantized: [8, 512, 75]

# Decoder 輸出
audio: [8, 1, 24000]  # 1.5 秒 @ 16kHz
```

### RVQ 如何生成 Quantized Vectors

```python
# RVQ forward
z_q = 0
residual = z

for layer in range(n_layers):
    # 找到最近的 code
    indices = argmin(distance(residual, codebook[layer]))

    # 獲取對應的向量
    q = codebook[layer][indices]

    # 累積
    z_q += q  # 關鍵：疊加所有層

    # 更新殘差
    residual = residual - q

# 最終 quantized vectors
# z_q = q0 + q1 + q2 + q3
# 包含所有層的資訊！
```

## 常見問題

### Q1: 為什麼不用 codes 重建？

**A**: 因為 decoder 實際上不需要 codes！

原始流程中，codes 只是用來 lookup codebook 獲取 quantized vectors。
既然 RVQ 已經給我們 quantized vectors (q0+q1+q2+q3)，就可以直接用。

### Q2: 這樣會損失資訊嗎？

**A**: 不會！quantized vectors 包含**所有層的資訊**。

```
單層 VQ: codes → quantized (只有一層資訊)
RVQ: multi_codes → quantized (所有層資訊的總和)
```

### Q3: 音頻質量會下降嗎？

**A**: 理論上會更好！

- RVQ 使用多層逼近，更精確
- Quantized vectors 更接近原始 encoder output
- 表達能力更強（1024^4 vs 4096）

### Q4: 可以用 student decoder 嗎？

**A**: 可以，但通常用 teacher decoder。

```python
# 方案 A: Teacher decoder (推薦)
audio = model.teacher.feature_extractor.encodec.decoder(quantized)

# 方案 B: Student decoder (如果 student 有 decoder)
audio = model.student.feature_extractor.encodec.decoder(quantized)
```

通常用 teacher decoder 因為：
1. Teacher 是預訓練好的，decoder 質量有保證
2. Student 主要訓練 encoder，decoder 可能不穩定

## 實驗建議

### 檢查音頻質量

1. **初始階段** (Step 0-200):
   - 聽 reconstructed.wav，可能有明顯噪音
   - 這是正常的，RVQ codebook 剛開始訓練

2. **中期** (Step 400-600):
   - 音質應該逐漸改善
   - 接近 clean audio

3. **最終** (Step 800-1000):
   - 如果訓練成功，音質應該很好
   - 同時 codebook usage 應該提高

### 如果音質很差

可能原因：
1. **RVQ commitment loss 太大** → 降低 commitment_cost
2. **Encoder 輸出不穩定** → 降低學習率
3. **Codebook 未收斂** → 增加訓練步數

## 總結

✅ **已解決**: RVQ 可以輸出音頻檔案
✅ **方法**: 使用 quantized vectors 而非 codes
✅ **優勢**: 包含所有層資訊，音質理論上更好
✅ **實作**: 簡單，只需添加 `model.decode()` 方法

---

**建立時間**: 2026-02-03
**狀態**: ✅ 已實作並測試
