# 梯度回傳診斷報告

**日期**: 2025-12-03
**實驗**: exp_1203 (LoRA Encoder Denoising)

---

## 診斷目的

驗證「Feature Loss (VQ 後) 的梯度無法有效回傳到 encoder」的假設。

---

## 實驗設計

比較兩種 Loss 的梯度傳遞效率：

| Loss 類型 | 監督對象 | Target | 位置 |
|-----------|---------|--------|------|
| **Feature Loss** | `student_features` | `teacher_features` | VQ 後 |
| **EmbDistillation** | `student_emb` | `codebook[teacher_codes]` | VQ 前 |

---

## 實驗結果

### 梯度統計

```
======================================================================
測試 1: Feature Loss (VQ 後) 的梯度
======================================================================
Student features requires_grad: True
Student features grad_fn: <PermuteBackward0>
Feature Loss: 0.000097
LoRA params with gradients: 8/8
Average gradient norm: 0.000123

======================================================================
測試 2: EmbDistillation (VQ 前) 的梯度
======================================================================
Student emb requires_grad: True
EmbDistillation Loss: 0.009586
LoRA params with gradients: 8/8
Average gradient norm: 0.006804
```

### 關鍵發現

| 指標 | Feature Loss (VQ 後) | EmbDistillation (VQ 前) | 比例 |
|------|---------------------|------------------------|------|
| requires_grad | ✅ True | ✅ True | - |
| 有梯度的參數 | 8/8 | 8/8 | - |
| **平均梯度 norm** | **0.000123** | **0.006804** | **55.4x** |

---

## 結論

### 1. 兩種方法都有梯度（不是完全阻斷）

VQ 使用 Straight-Through Estimator (STE)，理論上可以傳遞梯度：
```python
quantized = input + (quantized - input).detach()
# Forward: 用 quantized
# Backward: 梯度直接傳回 input
```

### 2. 但 EmbDistillation 的梯度強 55 倍

這說明 **VQ 的 STE 會嚴重削弱梯度傳遞**。

原因分析：
- STE 假設 `input ≈ quantized`，但實際上可能差很多
- Feature Loss 的梯度方向是「讓 output 接近 teacher_features」
- 而不是「讓 encoder output 接近正確的 codebook embedding」
- 梯度被 VQ 的離散化過程「稀釋」了

### 3. 這解釋了 exp7 的問題

exp7 使用 Feature Loss + CorrectVQLoss，但：
- Feature Loss 梯度太弱 (0.000123)
- CorrectVQLoss 也是用 VQ 後的 features
- 總體梯度信號不足以有效訓練 encoder

---

## Loss 定義比較

### Feature Loss (exp7)
```python
# 監督 VQ 後的 features
loss = MSE(student_features, teacher_features)
#          ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
#          VQ 後             VQ 後
```

### CorrectVQLoss (exp7)
```python
# 仍然監督 VQ 後的 features
loss = MSE(student_features, codebook[teacher_codes])
#          ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^
#          VQ 後             Teacher 選的 codebook
```

### EmbDistillation (exp8)
```python
# 直接監督 VQ 前的 encoder 輸出
loss = MSE(student_emb, codebook[teacher_codes])
#          ^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^
#          VQ 前!       Teacher 選的 codebook
```

**關鍵差異**: EmbDistillation 直接監督 encoder 原始輸出，梯度不經過 VQ。

---

## 建議

### 對於 exp_1203

1. **優先使用 EmbDistillation** (exp8)
   - 梯度強 55 倍
   - 直接監督 encoder 輸出

2. **可以考慮組合使用**
   - EmbDistillation (主要 Loss，權重大)
   - Feature Loss (輔助監控，權重小)

3. **避免只用 Feature Loss**
   - 梯度太弱，訓練效率低

### 長期建議

如果 Token Accuracy 仍然無法提升，問題可能不在 Loss 設計，而是：
- 任務本身的難度（noisy → clean 的映射複雜度）
- 需要更大的模型容量
- 需要不同的架構（如 c_code 的 Transformer + Speaker Embedding）

---

## 驗證腳本

診斷腳本位置：
```
exp_1203/verify_gradient_flow.py
```

執行命令：
```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1203
python verify_gradient_flow.py
```

---

## 附錄：VQ Layer 梯度路徑分析

```
======================================================================
VQ 層梯度路徑分析
======================================================================
VQ Layer type: <class 'encoder.quantization.core_vq.VectorQuantization'>

STE: quantized = input + (quantized - input).detach()
quantized_ste requires_grad: True

Loss: 0.000613
Test input has gradient: True
Test input gradient norm: 0.003125

分析:
  VQ 使用 STE: forward 用 quantized，backward 梯度傳回 input
  所以理論上 Feature Loss 應該有梯度傳回 encoder
  但梯度的「方向」可能不是最優的（因為跳過了 argmin 的選擇邏輯）
```

---

**報告生成時間**: 2025-12-03
