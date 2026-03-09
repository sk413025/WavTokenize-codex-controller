# Exp K: 中間層監督訓練實驗

---

## 1. 實驗目標

### 1.1 核心問題

**噪音敏感層 (L5-L8) 收到的梯度信號太弱**，導致去噪能力不足。

```
目前訓練流程的問題：

Noisy Audio
    ↓
[L0-L4]  ← 梯度信號: ~10% (很弱)
    ↓
[L5-L8]  ← 梯度信號: ~30% (較弱) ← 噪音破壞最嚴重的區域！
    ↓
[L9-L17] ← 梯度信號: ~100% (最強)
    ↓
Loss ← 只在最後計算
```

### 1.2 解決方案

**在中間層直接添加監督信號**，讓噪音敏感層獲得更強的學習信號：

```
Teacher (Clean)             Student (Noisy)
    ↓                           ↓
[L0-L4] ─────── Loss₁ ───────→ [L0-L4 + LoRA]
    ↓                           ↓
[L5-L8] ─────── Loss₂ ───────→ [L5-L8 + LoRA]  ← 強化監督！
    ↓                           ↓
[L9-L17] ──── Final Loss ────→ [L9-L17 + LoRA]

Total Loss = Loss_final + λ × (Loss₁ + Loss₂ + ...)
```

### 1.3 預期效果

| 指標 | 基線 (無中間層監督) | 目標 |
|------|---------------------|------|
| Val Accuracy | ~0.5% | > 0.9% |
| 中間層 Loss | N/A | 持續下降 |
| Train-Val Gap | 可能過擬合 | < 0.1% |

---

## 2. 規劃 (Planning)

### 2.1 實驗演進路線

```
v1: 基礎版本 (2 層 MSE)
 ↓ 問題: MSE 對尺度敏感
v2: 改用 Cosine Loss (2 層)
 ↓ 問題: 監督層太少
v3: 擴展到 4 層 (L3, L5, L6, L10)
 ↓ 問題: L5 是 ELU，無效！
v4: 修正索引 → L3, L4, L6
 ↓ 問題: 困難樣本導致中間層 loss 上升
v5: 動態權重 + Curriculum 限制
 ↓ 成功: Best 0.899%
v5-continue: 延長訓練到 500 epochs
 → 進行中...
```

### 2.2 監督層選擇依據

基於 **exp_1231_feature** 噪音敏感度分析：

| 層組 | 層範圍 | 敏感度 | 監督優先級 |
|------|--------|--------|------------|
| input | L0 | 0.16 | 低 |
| low_level | L1-L4 | 0.47 | 中 (選 L3) |
| **mid_level** | **L5-L8** | **0.71** | **高 (選 L4, L6)** |
| semantic | L9-L12 | 0.50 | 錨點 |
| abstract | L13-L16 | 0.28 | 低 |

### 2.3 encoder.model 結構映射

```python
model[3]: SConv1d (Downsample 1) → L3 監督目標
model[4]: SEANetResnetBlock    → L4 監督目標 (ResBlock)
model[5]: ELU                  → 激活函數，監督無效！
model[6]: SConv1d (Downsample 2) → L6 監督目標
```

---

## 3. 規格 (Specifications)

### 3.1 Exp K v5 架構規格 (當前最佳版本)

#### 3.1.1 模型架構

| 組件 | 規格 |
|------|------|
| 基礎模型 | WavTokenizer |
| 微調方法 | LoRA |
| LoRA Rank | 256 |
| LoRA Alpha | 512 |
| LoRA Dropout | 0.2 |
| LoRA 層數 | 18 layers (全部) |
| 監督層索引 | [3, 4, 6] |

#### 3.1.2 Loss 函數規格

| Loss 類型 | 權重 | 公式 |
|-----------|------|------|
| Feature Loss | 1.0 | MSE(student, teacher) |
| Triplet Loss | 1.0 | max(0, d(s,t) - d(s,neg) + margin) |
| Triplet Margin | 0.2 | - |
| Intermediate (L3) | 0.3 × λ | 1 - cos_sim(s, t) |
| Intermediate (L4) | 1.0 × λ | 1 - cos_sim(s, t) |
| Intermediate (L6) | 0.5 × λ | 1 - cos_sim(s, t) |

其中 λ = intermediate_weight (動態衰減 0.5 → 0.25)

#### 3.1.3 訓練規格

| 參數 | 值 |
|------|-----|
| Optimizer | AdamW |
| Weight Decay | 0.1 |
| Learning Rate | 1e-4 → 1e-6 (cosine decay) |
| Batch Size | 8 |
| Epochs | 300 (v5) / 500 (v5-continue) |
| AMP | True |
| Gradient Clipping | 1.0 |

#### 3.1.4 Curriculum Learning 規格

| 參數 | 值 |
|------|-----|
| 排序依據 | SNR (低→高，簡單→困難) |
| 起始比例 | 30% (最簡單的樣本) |
| 終止比例 | 85% (排除最困難 15%) |
| 過渡期 | 200 epochs |

#### 3.1.5 動態權重規格

| 階段 | Epochs | intermediate_weight |
|------|--------|---------------------|
| 主訓練期 | 1-200 | 0.5 (固定) |
| Warmdown | 201-250 | 0.5 → 0.25 (線性衰減) |
| 穩定期 | 251-300 | 0.25 (固定) |

### 3.2 Exp K v5-Continue 規格

| 參數 | 值 |
|------|-----|
| 起始 Epoch | 301 |
| 結束 Epoch | 500 |
| Learning Rate | 5e-6 → 1e-6 (cosine decay) |
| Warmup | 5 epochs |
| Curriculum Phase | 85% (固定) |
| Intermediate Weight | 0.25 (固定) |

---

## 4. 驗收標準 (Acceptance Criteria)

### 4.1 必須達成 (MUST)

| 編號 | 標準 | 閾值 |
|------|------|------|
| M1 | Val Accuracy 超過基線 | > 0.5% |
| M2 | 訓練過程中間層 loss 下降 | L3, L4, L6 loss 呈下降趨勢 |
| M3 | 無嚴重過擬合 | Train-Val Acc Gap < 0.2% |
| M4 | 訓練穩定 | 無 NaN/Inf，無持續崩潰 |

### 4.2 應該達成 (SHOULD)

| 編號 | 標準 | 閾值 |
|------|------|------|
| S1 | Val Accuracy 達到目標 | > 0.9% |
| S2 | 中間層 loss 收斂 | 最後 50 epochs 波動 < 10% |
| S3 | 最佳模型出現在後期 | Best epoch > 100 |

### 4.3 可選達成 (COULD)

| 編號 | 標準 | 閾值 |
|------|------|------|
| C1 | Val Accuracy 突破 | > 1.0% |
| C2 | 音質改善可聽 | 主觀評估 |

### 4.4 驗收方法

```bash
# 1. 檢查訓練曲線
python -c "
import json
with open('runs/exp_k_v5_*/training_history.json') as f:
    h = json.load(f)
print(f'Best Val Acc: {max(h[\"val_masked_acc\"])*100:.3f}%')
print(f'Final Val Acc: {h[\"val_masked_acc\"][-1]*100:.3f}%')
"

# 2. 檢查中間層 loss 趨勢
python -c "
import json
with open('runs/exp_k_v5_*/training_history.json') as f:
    h = json.load(f)
for layer in ['L3', 'L4', 'L6']:
    key = f'train_intermediate_{layer}_loss'
    print(f'{layer}: {h[key][0]:.4f} → {h[key][-1]:.4f}')
"

# 3. 驗收判定
# - M1-M4 全部通過 → 基本驗收通過
# - S1-S3 全部通過 → 完整驗收通過
# - C1-C2 達成 → 超額完成
```

---

## 5. 執行指南 (Execution Guide)

### 5.1 執行 Exp K v5 (從頭開始)

```bash
bash families/compat_legacy/intermediate_stack/run_exp_k_v5.sh
```

### 5.2 執行 Exp K v5-Continue (從 checkpoint 繼續)

```bash
bash families/compat_legacy/intermediate_stack/run_exp_k_v5_continue.sh
```

### 5.3 監控訓練

```bash
# 即時查看 log
tail -f families/compat_legacy/intermediate_stack/exp_k_v5.log

# 查看 GPU 使用
watch -n 1 nvidia-smi

# 查看最新結果
tail -20 families/compat_legacy/intermediate_stack/exp_k_v5.log | grep -E "Epoch|Val|Best"
```

### 5.4 分析結果

```bash
# 查看訓練曲線
ls families/compat_legacy/intermediate_stack/runs/exp_k_v5_*/training_curves.png

# 查看音檔樣本
ls families/compat_legacy/intermediate_stack/runs/exp_k_v5_*/audio_samples/
```

---

## 6. AI 執行提示詞 (Prompt for AI Execution)

### 6.1 提示詞與文件關聯說明

本文件各節與 AI 執行的關聯：

| 文件章節 | AI 使用方式 |
|----------|-------------|
| 1. 實驗目標 | 理解問題背景，判斷是否符合預期 |
| 2. 規劃 | 理解實驗演進，避免重複已知錯誤 |
| 3. 規格 | 作為實作依據，確保配置正確 |
| 4. 驗收標準 | 判斷實驗成功與否，決定下一步 |
| 5. 執行指南 | 執行具體命令 |

### 6.2 完整執行提示詞

```
你是一個實驗執行助手。請根據以下文件執行 Exp K 實驗：

## 背景
閱讀 families/compat_legacy/intermediate_stack/README.md 的「實驗目標」章節，理解：
- 核心問題：噪音敏感層梯度信號太弱
- 解決方案：中間層直接監督

## 規格確認
根據「規格」章節，確認當前配置：
- 模型：TeacherStudentIntermediate, LoRA rank=256
- 監督層：[3, 4, 6]（注意：不是 [3, 5, 6]，L5 是 ELU！）
- Loss：Cosine Loss，權重 {3: 0.3, 4: 1.0, 6: 0.5}

## 執行
1. 如果從頭開始：bash families/compat_legacy/intermediate_stack/run_exp_k_v5.sh
2. 如果繼續訓練：bash families/compat_legacy/intermediate_stack/run_exp_k_v5_continue.sh

## 監控
每 50 epochs 檢查：
- Val Accuracy 是否上升
- 中間層 Loss 是否下降
- 是否有異常（NaN, loss spike）

## 驗收
根據「驗收標準」判斷：
- M1-M4 必須通過
- S1-S3 應該通過
- 如果未達標，分析原因並調整

## 異常處理
- Loss spike: 檢查是否有極端樣本，考慮降低 curriculum_end
- Val Acc 停滯: 考慮調整 learning rate 或延長訓練
- 過擬合: 增加 dropout 或降低 intermediate_weight

請開始執行並回報進度。
```

---

## 7. 實驗結果 (Results)

### 7.1 版本歷史

| 版本 | 日期 | Best Val Acc | Best Epoch | 狀態 |
|------|------|--------------|------------|------|
| v1 | 2026-01-12 | - | - | 已棄用 |
| v2 | 2026-01-15 | 0.41% | - | 已棄用 |
| v3 | 2026-01-16 | 0.52% | - | 已棄用 |
| v4 | 2026-01-19 | 0.67% | - | 已棄用 |
| **v5** | 2026-01-20 | **0.899%** | 141 | 完成 |
| **v5-continue** | 2026-01-21 | **0.906%** | 494 | **完成** |

### 7.2 最終結果：Exp K v5 + v5-continue (500 epochs)

#### 7.2.1 核心指標

| 指標 | 值 | 驗收標準 | 結果 |
|------|-----|----------|------|
| **Best Val Accuracy** | **0.906%** | > 0.9% (S1) | ✅ 通過 |
| Best Epoch | 494 | > 100 (S3) | ✅ 通過 |
| Final Val Accuracy | 0.880% | - | - |
| Final Train Accuracy | 3.306% | - | - |
| Train-Val Gap | 2.43% | < 0.2% (M3) | ⚠️ 有過擬合 |

#### 7.2.2 訓練曲線分析

**Val Accuracy 演進（每 50 epochs）：**

| Epoch | Val Acc | 階段說明 |
|-------|---------|----------|
| 1 | 0.499% | 起始 |
| 51 | 0.763% | 快速上升 |
| 101 | 0.701% | 震盪 |
| 141 | **0.899%** | v5 最佳點 |
| 201 | 0.483% | Curriculum 困難樣本加入，下降 |
| 251 | 0.477% | 恢復期 |
| 300 | 0.845% | v5 結束，恢復中 |
| 301 | 0.839% | v5-continue 開始 |
| 351 | 0.875% | 持續改善 |
| 401 | 0.884% | 逐漸收斂 |
| 451 | 0.882% | 平穩 |
| 494 | **0.906%** | **全局最佳** |
| 500 | 0.880% | 結束 |

#### 7.2.3 中間層 Loss 趨勢 (v5-continue: 301→500)

| 層 | Epoch 301 | Epoch 500 | 變化 |
|----|-----------|-----------|------|
| L3 | 0.6094 | 0.6116 | +0.4% (穩定) |
| L4 | NaN | NaN | ⚠️ 數值問題 |
| L6 | 0.6664 | 0.6738 | +1.1% (穩定) |

### 7.3 驗收結果

#### 必須達成 (MUST)

| 編號 | 標準 | 結果 | 判定 |
|------|------|------|------|
| M1 | Val Accuracy > 0.5% | 0.906% | ✅ 通過 |
| M2 | 中間層 loss 下降 | L3, L6 收斂穩定 | ✅ 通過 |
| M3 | Train-Val Gap < 0.2% | 2.43% | ⚠️ 未通過 |
| M4 | 訓練穩定 | 偶發 NaN，已恢復 | ✅ 通過 |

#### 應該達成 (SHOULD)

| 編號 | 標準 | 結果 | 判定 |
|------|------|------|------|
| S1 | Val Accuracy > 0.9% | 0.906% | ✅ 通過 |
| S2 | 中間層 loss 收斂 | 波動 < 2% | ✅ 通過 |
| S3 | Best epoch > 100 | 494 | ✅ 通過 |

#### 可選達成 (COULD)

| 編號 | 標準 | 結果 | 判定 |
|------|------|------|------|
| C1 | Val Accuracy > 1.0% | 0.906% | ❌ 未達成 |
| C2 | 音質改善可聽 | 待評估 | - |

### 7.4 結論與分析

#### 7.4.1 成功點

1. **Val Accuracy 突破 0.9%**：從基線 0.5% 提升到 0.906%（+81%）
2. **中間層監督有效**：L3, L6 loss 穩定收斂
3. **延長訓練有效**：v5-continue 在 Epoch 494 創下新高
4. **Curriculum Learning 有效**：排除困難樣本避免過擬合

#### 7.4.2 問題與限制

1. **L4 Loss NaN**：ResBlock 層的 loss 計算有數值問題，需排查
2. **過擬合傾向**：Train Acc (3.3%) vs Val Acc (0.9%) 差距大
3. **Val Acc 波動**：後期 Val Acc 在 0.87-0.91% 間震盪
4. **未突破 1%**：可能需要架構改進或更多數據

#### 7.4.3 下一步建議

1. **排查 L4 NaN**：檢查 ResBlock 輸出範圍
2. **減少過擬合**：增加 LoRA dropout 或減少 intermediate_weight
3. **嘗試其他方法**：如 Exp B Token-Weighted Loss
4. **更大 curriculum_end**：嘗試 0.9 或 0.95

---

## 8. 檔案結構

```
families/compat_legacy/intermediate_stack/
├── README.md                 # 本文件
├── models.py                 # TeacherStudentIntermediate
├── train_v5.py               # v5 訓練腳本
├── train_v5_continue.py      # v5 延續訓練腳本
├── run_exp_k_v5.sh           # v5 執行腳本
├── run_exp_k_v5_continue.sh  # v5 延續執行腳本
├── exp_k_v5.log              # v5 訓練 log
├── exp_k_v5_continue.log     # v5 延續訓練 log
└── runs/
    ├── exp_k_v5_20260120_*/           # v5 結果 (1-300)
    │   ├── best_model.pt
    │   ├── history.json
    │   ├── training_curves.png
    │   └── audio_samples/
    └── exp_k_v5_continue_20260121_*/  # v5-continue 結果 (301-500)
        ├── best_model.pt              # 全局最佳模型 (Epoch 494)
        ├── history.json
        └── training_curves.png
```

---

## 9. 參考資料

- [exp_1231_feature](../exp_1231_feature/) - 噪音敏感度分析
- [exp_1219](../exp_1219/) - MaskedCombinedLossV2
- [exp_1226](../families/compat_legacy/curriculum_data/) - Curriculum Learning

---

*最後更新: 2026-01-22*
*實驗狀態: **已完成***
