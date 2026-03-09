# exp_test: 淺層 LoRA 容量瓶頸測試

## 背景問題

基於先前實驗觀察：
1. **降噪擾動在中層 (L5-L8) 最大**：cos_sim ≈ 0.21-0.29
2. **但 LoRA 訓練後深層變化最大**：深層 1.73 vs 淺層 1.43
3. **懷疑**：LoRA 容量不足，導致淺層無法有效學習，深層被迫「硬補償」

## 實驗設計

### 核心思路
- 只訓練 L0-L4 (5 層)，完全凍結 L5-L17 (13 層)
- Loss 只監督 L4 輸出的 MSE
- 測試不同 LoRA rank (256/512/1024) 哪個能讓 loss 降得更低

### 架構圖

```
┌─────────────────────────────────────────────────────────────┐
│  Teacher (Clean)              Student (Noisy)               │
│                                                             │
│  Clean Audio                  Noisy Audio                   │
│      ↓                            ↓                         │
│  [L0-L4] ──── MSE Loss ──── [L0-L4 + LoRA]                 │
│      ↓                            ↓                         │
│  [L5-L17] (凍結)              [L5-L17] (凍結)                │
│      ↓                            ↓                         │
│  Teacher Out                  (不使用)                       │
│                                                             │
│  只計算 L4 輸出的 MSE Loss，驗證淺層容量瓶頸                 │
└─────────────────────────────────────────────────────────────┘
```

## 測試組別

| 組別 | LoRA Rank | LoRA Alpha | Scaling | 預估參數量 |
|------|-----------|------------|---------|-----------|
| r256 | 256 | 512 | 2.0 | ~2.6M |
| r512 | 512 | 1024 | 2.0 | ~5.2M |
| r1024 | 1024 | 2048 | 2.0 | ~10.5M |

## 執行方式

### 執行全部三組（依序）
```bash
bash exp_test/run_all.sh
```

### 單獨執行
```bash
# Rank 256
bash exp_test/run_r256.sh

# Rank 512
bash exp_test/run_r512.sh

# Rank 1024
bash exp_test/run_r1024.sh
```

### 手動執行
```bash
python exp_test/train.py \
    --exp_name shallow_r256 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --num_epochs 150
```

## 預期結果解讀

### 情況 A：Rank↑ → Loss↓↓ (顯著下降)
**結論**：容量是瓶頸
- 淺層確實需要更大的 LoRA 容量
- 先前實驗深層變化大是因為淺層容量不足，訊號無法有效傳遞

### 情況 B：Rank↑ → Loss 無明顯改善
**結論**：問題不在容量
- 可能是梯度流問題（深層到淺層的梯度消失/爆炸）
- 可能需要中間層監督（直接指導淺層，不依賴深層梯度）
- 可能是任務本身的難度上限

### 情況 C：Loss 下降但很快飽和
**結論**：混合原因
- 容量是部分原因，但還有其他限制因素
- 可能需要結合其他策略（如中間層監督 + 適當容量）

## 輸出文件

每組實驗會產生：
```
exp_test/runs/shallow_r{rank}/
├── config.json          # 完整配置
├── history.json         # 訓練歷史
├── summary.json         # 摘要結果（用於比較）
├── training_curves.png  # 訓練曲線圖
├── best_model.pt        # 最佳模型
└── final_model.pt       # 最終模型
```

## 比較腳本

訓練完成後，可以用以下命令快速比較：
```bash
echo "=== Rank 256 ===" && cat exp_test/runs/shallow_r256/summary.json
echo "=== Rank 512 ===" && cat exp_test/runs/shallow_r512/summary.json
echo "=== Rank 1024 ===" && cat exp_test/runs/shallow_r1024/summary.json
```

## 後續實驗建議

根據本實驗結果：
1. **如果容量是瓶頸**：在正式實驗中使用更大的 rank，或採用 Adapter 增加容量
2. **如果不是容量**：重點放在中間層監督（Exp K）或梯度分析
3. **如果混合原因**：結合中間層監督 + 適當增大容量
