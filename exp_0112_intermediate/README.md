# Exp 0112_intermediate: Exp K - 中間層監督訓練

## 實驗目標

解決目前訓練的核心問題：**噪音敏感層 (L5-L8) 收到的梯度信號太弱**。

### 問題分析

```
目前的訓練流程：

Noisy Audio
    ↓
[L0-L4]  ← 梯度信號: ~10% (很弱)
    ↓
[L5-L8]  ← 梯度信號: ~30% (較弱) ← 但這是噪音破壞最嚴重的區域！
    ↓
[L9-L17] ← 梯度信號: ~100% (最強)
    ↓
Loss ← 只在最後計算
```

**問題：噪音敏感層離 loss 最遠，收到的學習信號最弱。**

### 解決方案：中間層直接監督

```
Teacher (Clean)             Student (Noisy)
    ↓                           ↓
[L0-L4] ─────── MSE ───────→ [L0-L4 + LoRA]
    ↓          Loss₁             ↓
[L5-L8] ─────── MSE ───────→ [L5-L8 + LoRA]
    ↓          Loss₂             ↓
[L9-L17] ──── Final Loss ──→ [L9-L17 + LoRA]

Total Loss = Loss_final + λ × (Loss₁ + Loss₂)
```

## 核心設計

### 監督位置

| 位置 | 層 | 原因 |
|------|-----|------|
| **L4 (index 3)** | 第一個 downsample | 淺層輸出，聲學特徵 |
| **L8 (index 6)** | 第二個 downsample | 中層輸出，噪音敏感區邊界 |

### Loss 設計

```python
# 最終輸出 Loss (原有的)
final_loss = feature_loss + triplet_loss

# 中間層 Loss (新增的)
L4_loss = MSE(student_L4_output, teacher_L4_output)
L8_loss = MSE(student_L8_output, teacher_L8_output)
intermediate_loss = 0.5 × L4_loss + 0.5 × L8_loss

# 總 Loss
total_loss = final_loss + 0.5 × intermediate_loss
```

### 預期效果

| 效果 | 說明 |
|------|------|
| **直接監督噪音敏感層** | 不用等梯度從最後傳回 |
| **多尺度學習目標** | 每個階段都知道「應該輸出什麼」 |
| **加速收斂** | 更直接的學習信號 |

### 潛在風險

| 風險 | 應對 |
|------|------|
| 過度約束 | 可調整 intermediate_weight |
| 目標衝突 | 中間層和最終輸出可能有矛盾 |

## 與其他實驗對比

| 實驗 | 方法 | 重點 |
|------|------|------|
| Exp I | 三區差異化 LR | 強化中層學習率 |
| Exp J | 中層 Adapter | 插入專門去噪模組 |
| **Exp K** | **中間層監督** | **直接指導中層輸出** |

## 執行方式

```bash
# 執行實驗
./exp_0112_intermediate/run_exp_k.sh

# 或手動執行
python exp_0112_intermediate/train.py \
    --exp_name exp_k_intermediate \
    --intermediate_weight 0.5 \
    --intermediate_L4_weight 0.5 \
    --intermediate_L8_weight 0.5
```

## 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--intermediate_weight` | 0.5 | 中間層 loss 總權重 |
| `--intermediate_L4_weight` | 0.5 | L4 loss 權重 |
| `--intermediate_L8_weight` | 0.5 | L8 loss 權重 |

## 檔案結構

```
exp_0112_intermediate/
├── README.md           # 本文件
├── models.py           # TeacherStudentIntermediate, IntermediateSupervisionLoss
├── train.py            # 訓練腳本
├── run_exp_k.sh        # 執行腳本
└── runs/
    └── exp_k_intermediate/
        ├── config.json
        ├── history.json
        ├── training_curves.png
        ├── best_model.pt
        └── audio_samples/
```

## 監控指標

- **Val Acc**: 目標 > Baseline (1.06%)
- **Train-Val Gap**: 目標 < Baseline (2.33%)
- **Intermediate Loss**: 監控中間層對齊程度
- **L4 Loss vs L8 Loss**: 觀察哪層更難對齊

## 變體實驗 (可選)

1. **更強的中間監督**:
   ```bash
   python exp_0112_intermediate/train.py \
       --exp_name exp_k_strong \
       --intermediate_weight 1.0
   ```

2. **只監督 L8 (最敏感區)**:
   ```bash
   python exp_0112_intermediate/train.py \
       --exp_name exp_k_L8_only \
       --intermediate_L4_weight 0.0 \
       --intermediate_L8_weight 1.0
   ```

## 日期

- 創建: 2026-01-12
- 狀態: 待執行
