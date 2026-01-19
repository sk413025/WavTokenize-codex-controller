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
[L0-L4] ─────── Loss ────────→ [L0-L4 + LoRA]
    ↓          Loss₁             ↓
[L5-L8] ─────── Loss ────────→ [L5-L8 + LoRA]
    ↓          Loss₂             ↓
[L9-L17] ──── Final Loss ────→ [L9-L17 + LoRA]

Total Loss = Loss_final + λ × (Loss₁ + Loss₂ + ...)
```

---

## 實驗版本

### Exp K v2 (2-layer)

監督 2 層，使用 Cosine Loss。

| 位置 | 層 | 權重 | Loss 類型 |
|------|-----|------|-----------|
| L3 | low_level | 0.5 | Cosine |
| L6 | mid_level (噪音處理) | 0.5 | Cosine |

```bash
bash exp_0112_intermediate/run_exp_k_v2.sh
```

### Exp K v3 (4-layer) - 當前版本

監督 4 層，基於噪音敏感度分析的完整配置。

| 位置 | 層 | 權重 | Loss 類型 | 角色 |
|------|-----|------|-----------|------|
| **L3** | low_level | 0.5 | Cosine | 捕捉早期噪音 |
| **L5** | mid_level | 0.8 | Cosine | 噪音處理協同 |
| **L6** | mid_level | 1.0 | Cosine | **噪音處理核心** |
| **L10** | semantic | 0.3 | MSE | 語義錨點 |

```bash
bash exp_0112_intermediate/run_exp_k_v3.sh
```

**理論依據**:
- exp_1231_feature: L5-L6 mid-level 敏感度 0.71-0.79 (最高)
- 本次分析: L10 cos_sim=0.946 (最穩定，適合作為錨點)

---

## 核心設計

### 監督位置選擇依據

基於 **exp_1231_feature** 噪音敏感度分析：

| 層組 | 層範圍 | 敏感度 | 說明 |
|------|--------|--------|------|
| input | L0 | 0.16 | 對噪音不敏感 |
| low_level | L1-L4 | 0.47 | 中等敏感 |
| **mid_level** | **L5-L8** | **0.71** | **最敏感！噪音處理核心** |
| semantic | L9-L12 | 0.50 | 中等 |
| abstract | L13-L16 | 0.28 | 對噪音魯棒 |

### Loss 類型選擇

| Loss 類型 | 特點 | 適用場景 |
|-----------|------|----------|
| **Cosine** | 尺度不變性，範圍 0-2 | 跨層比較，敏感層監督 |
| **MSE** | 精確匹配，範圍可變 | 穩定層錨點 |

**為何 L10 用 MSE？**
- L10 本來就穩定 (cos_sim=0.946)
- MSE 強制精確匹配，確保語義不偏離
- 作為「錨點」防止過度調整

---

## 執行方式

### Exp K v3 (推薦)

```bash
# 執行 4 層監督實驗
bash exp_0112_intermediate/run_exp_k_v3.sh
```

### 手動執行

```bash
python exp_0112_intermediate/train_v3.py \
    --exp_name exp_k_v3 \
    --intermediate_weight 1.0 \
    --intermediate_L3_weight 0.5 \
    --intermediate_L5_weight 0.8 \
    --intermediate_L6_weight 1.0 \
    --intermediate_L10_weight 0.3
```

---

## 參數說明

### v3 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--intermediate_weight` | 1.0 | 中間層 loss 總權重 |
| `--intermediate_L3_weight` | 0.5 | L3 權重 (low_level) |
| `--intermediate_L5_weight` | 0.8 | L5 權重 (mid_level 協同) |
| `--intermediate_L6_weight` | 1.0 | L6 權重 (噪音處理核心) |
| `--intermediate_L10_weight` | 0.3 | L10 權重 (語義錨點) |
| `--target_scale` | 1.0 | Loss 縮放因子 |

### 共通參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--lora_rank` | 256 | LoRA rank |
| `--lora_alpha` | 512 | LoRA alpha |
| `--lr` | 1e-4 | 學習率 |
| `--num_epochs` | 300 | 訓練 epochs |
| `--batch_size` | 8 | Batch size |

---

## 檔案結構

```
exp_0112_intermediate/
├── README.md              # 本文件
├── EXP_SUMMARY.md         # 詳細實驗報告
├── models.py              # TeacherStudentIntermediate
├── models_v2.py           # IntermediateSupervisionLossV2 (Cosine)
├── models_v3.py           # IntermediateSupervisionLossV3 (混合 Loss)
├── train.py               # v1 訓練腳本
├── train_v2.py            # v2 訓練腳本 (2 層)
├── train_v3.py            # v3 訓練腳本 (4 層)
├── run_exp_k.sh           # v1 執行腳本
├── run_exp_k_v2.sh        # v2 執行腳本
├── run_exp_k_v3.sh        # v3 執行腳本
├── analysis/              # 分析腳本與結果
│   ├── noise_sensitivity.json
│   ├── integrated_noise_analysis.py
│   ├── integrated_noise_analysis.png
│   ├── loss_type_comparison.py
│   └── loss_type_comparison.png
└── runs/
    ├── exp_k_intermediate/     # v1 結果
    ├── exp_k_v2_*/             # v2 結果
    └── exp_k_v3_*/             # v3 結果
```

---

## 監控指標

### 主要指標

| 指標 | 說明 | 目標 |
|------|------|------|
| **Val Acc** | 驗證集準確率 | > 52% |
| **Train-Val Gap** | 過擬合程度 | < 2% |
| **Intermediate Loss** | 中間層對齊程度 | 持續下降 |

### 各層 Loss 監控

| 指標 | 說明 |
|------|------|
| `intermediate_L3_loss` | L3 對齊程度 (Cosine) |
| `intermediate_L5_loss` | L5 對齊程度 (Cosine) |
| `intermediate_L6_loss` | L6 對齊程度 (Cosine) |
| `intermediate_L10_loss` | L10 對齊程度 (MSE) |

---

## 與其他實驗對比

| 實驗 | 方法 | 監督層 | 重點 |
|------|------|--------|------|
| Exp I | 三區差異化 LR | - | 強化中層學習率 |
| Exp J | 中層 Adapter | - | 插入專門去噪模組 |
| **Exp K v2** | **中間層監督** | L3, L6 | 2 層 Cosine Loss |
| **Exp K v3** | **完整中間層監督** | L3, L5, L6, L10 | 4 層混合 Loss |

---

## 相關文件

- [EXP_SUMMARY.md](EXP_SUMMARY.md) - 詳細實驗報告與分析
- [analysis/integrated_noise_analysis.png](analysis/integrated_noise_analysis.png) - 噪音敏感度分析圖
- [analysis/loss_type_comparison.png](analysis/loss_type_comparison.png) - Loss 類型比較圖

---

## 日期

- 創建: 2026-01-12
- Exp K v2: 2026-01-15
- Exp K v3: 2026-01-16
- 狀態: **v3 執行中**
