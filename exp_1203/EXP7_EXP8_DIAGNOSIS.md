# exp7 vs exp8 結果差異不大的原因診斷

**日期**: 2025-12-03

---

## 核心結論

**exp7 和 exp8 差異不大，是因為兩者都失敗了。**

```
真正的 Baseline (Teacher noisy vs Teacher clean): 4.78%
exp7 最終 Val Acc: 5.18% (只比 baseline 高 0.4%)
exp8 最終 Val Acc: 6.76% (只比 baseline 高 2.0%)
```

LoRA 微調幾乎沒有效果。

---

## 實驗結果對比

| Metric | exp7 (Feature+VQ) | exp8 (EmbDist) | 差異 |
|--------|-------------------|----------------|------|
| **Train Acc (初始)** | 30.44% | 26.31% | -4.13% |
| **Train Acc (最終)** | 9.99% | 13.86% | +3.87% |
| **Val Acc (初始)** | 15.14% | 15.54% | +0.40% |
| **Val Acc (最終)** | 5.18% | 6.76% | +1.58% |
| **Train Acc 變化** | -20.45% | -12.45% | exp8 下降較少 |

---

## 核心發現 1: LoRA 初始化導致 Student = Teacher

### 驗證結果

```
LoRA B 矩陣初始化:
  lora_B.default.weight: mean=0.000000, std=0.000000, norm=0.000000

Student vs Teacher 輸出:
  Difference (abs): Mean = 0.000000
  Correlation: 1.000000

⚠️ Student 和 Teacher 輸出幾乎相同!
```

### 原因

LoRA 使用標準初始化：
- `lora_A`: 隨機初始化
- `lora_B`: **零初始化**

這意味著 `LoRA output = lora_B @ lora_A @ x = 0`

結果：**初始時 Student encoder = Teacher encoder**

---

## 核心發現 2: Token Accuracy 計算方式的理解

### Token Accuracy 定義

```python
# losses.py line 874-880
predictions = distances.argmin(dim=-1)  # student_emb 最近的 codebook
token_accuracy = (predictions == teacher_flat).float().mean()  # 與 teacher_codes 比較
```

這計算的是：
```
student_emb → argmin(distance to codebook) → student_codes
                                              ↓
                                    與 teacher_codes 比較
```

### 問題

- **初始時**: Student encoder = Teacher encoder
- **輸入 noisy audio**: Student encoder 輸出 = Teacher encoder 處理 noisy audio 的結果
- **Teacher codes 來自 clean audio**: Teacher encoder 處理 clean audio
- **所以 Token Acc 計算的是**: `encoder(noisy) vs encoder(clean)`

---

## 核心發現 3: 異常的 Baseline Token Accuracy

### 驗證結果

```
測試: noisy_audio ≠ clean_audio (完全不同的 random audio)
Token Accuracy: 99.78%
預期: 接近 0.024% (random baseline = 1/4096)
```

### 這說明什麼？

即使 noisy 和 clean 是**完全不同的隨機 audio**，Token Acc 仍然是 99.78%！

**原因**: LoRA_B = 0，所以：
- Student encoder(noisy_audio) = Teacher encoder(noisy_audio)（不是 clean_audio！）

等等... 這不對。讓我重新檢查 forward_with_emb：

```python
# model.py
student_emb = self.student.feature_extractor.encodec.encoder(student_audio)  # noisy
teacher_codes = self.teacher.feature_extractor(clean_audio)  # clean
```

所以 Token Acc 應該是比較：
- student_emb 來自 **noisy audio**
- teacher_codes 來自 **clean audio**

**但為什麼隨機 audio 也有 99.78% 的 Token Acc？**

---

## 真正的 Baseline 驗證

### Random Audio 的問題

使用 random Gaussian noise 測試時，Token Acc = 99.78%，這是誤導的！

原因：WavTokenizer 對非語音輸入會 collapse 到少數 codebook entries：
```
Random audio:
  Student codes unique: 2    ← 只用了 2 個 entries！
  Teacher codes unique: 2
  Codes sample: [170, 170, 170, ...]  ← 全是 170
```

### 真實數據的 Baseline

使用真實 noisy-clean pair 數據：
```
Token Accuracy: 4.78%
Baseline (Teacher noisy vs Teacher clean): 4.78%

Student codes unique: 418   ← 真實語音有豐富分布
Teacher codes unique: 545
```

**這才是真正的 baseline：4.78%**

---

## 結論：為什麼 exp7 和 exp8 差異不大？

### 核心原因：任務太難 + LoRA 容量不足

| 指標 | 值 | 說明 |
|------|-----|------|
| **Baseline** | 4.78% | Teacher(noisy) vs Teacher(clean) |
| **exp7 Val Acc** | 5.18% | 只改善 0.4% |
| **exp8 Val Acc** | 6.76% | 只改善 2.0% |
| **LoRA 參數** | 154K (0.19%) | 容量太小 |
| **Codebook 空間** | 4096 × 512 = 2M | 目標空間太大 |

### 1. 任務本身太難

- noisy 和 clean 的 token 只有 4.78% 相似度
- 這意味著 95% 的 tokens 需要被「修正」
- 但 LoRA 只有 154K 參數，無法學習這麼複雜的映射

### 2. 梯度強度不是唯一因素

```
exp7: 梯度弱 (被 VQ 削弱 55x)
exp8: 梯度強 (直接傳遞，55x)

結果：exp8 只比 exp7 好 1.58%
原因：梯度方向可能不對，或容量不足
```

### 3. Token Accuracy 下降的原因

訓練過程中：
```
初始 Token Acc: ~26-30% (這個數字是假的，因為用了相同 audio)
真正 Baseline: 4.78%
最終 Val Acc: 5-7%
```

初始高 Token Acc 是因為 LoRA_B=0，Student=Teacher，所以：
- Train set 的 noisy-clean pair 可能有部分相似
- 但這不代表模型真的學會了 denoising

---

## 建議

### 短期

1. **驗證真正的 Baseline**
   - 使用真實的 noisy-clean pair 數據
   - 計算未訓練模型的 Token Acc
   - 確認數據配對是否正確

2. **檢查數據**
   - noisy audio 和 clean audio 是否正確配對？
   - 是否真的是「同一段 speech 加上 noise」？

### 中期

3. **增加 LoRA 容量**
   - 嘗試 rank=128 或 256
   - 或者訓練更多層的 LoRA

4. **使用更合適的架構**
   - 參考 c_code 的 Transformer + Speaker Embedding 方法
   - 那個方法已經證明可以達到 ~48% Val Acc

### 長期

5. **重新思考問題定義**
   - LoRA 微調 encoder 是否是正確的方法？
   - 是否應該用獨立的 denoising 網路？

---

## 驗證腳本

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1203

# 比較 exp7 vs exp8
python compare_exp7_exp8.py

# 驗證 baseline
python verify_baseline_token_acc.py
```

---

**報告生成時間**: 2025-12-03
