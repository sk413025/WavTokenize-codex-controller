# exp_1205: 新 Loss 策略實驗

## 實驗背景

基於 exp_1204 的診斷結果，發現 **MSE Loss 優化目標不匹配** 是 Token Accuracy 停滯的根本原因：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           診斷結果摘要                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  問題：MSE Loss 只優化「接近」，不保證「最近」                               │
│                                                                              │
│  證據：                                                                      │
│  - 到正確 token 的距離: 3.75                                                │
│  - 到最近 token 的距離: 0.45                                                │
│  - Token Accuracy: 2.21%                                                    │
│  - 正確 token 平均排名: 1505 / 4096                                         │
│                                                                              │
│  結論：Student embedding 確實「靠近」某些 token，但不是正確的 token！        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 三種解決方案

### exp13: Linear + CE (方案 A) - 最推薦

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  原本：student_emb → 計算距離 → argmin → 預測                               │
│                      ↓                                                       │
│                   MSE Loss (只優化「接近」)                                  │
│                                                                              │
│  exp13：student_emb → Linear(512, 4096) → softmax → 預測                    │
│                       ↓                                                      │
│                    CE Loss (直接優化「選對」)                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**優點**：
- 繞過距離計算問題
- Loss 優化目標 = Token Accuracy
- 數值穩定

### exp14: Margin Loss (方案 B)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  loss = max(0, dist_to_correct - dist_to_wrong + margin)                    │
│                                                                              │
│  強制：到正確 token 的距離 < 到最近錯誤 token 的距離 - margin              │
│                                                                              │
│  不只「接近」正確 token，還要「比錯誤 token 更近」                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

**優點**：
- 保留 embedding 空間結構
- 學習判別性 embedding

### exp15: Hard Negative Mining + CE (方案 C)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  只在最近的 K 個 token 上計算 CE                                             │
│                                                                              │
│  - 找到 student 最近的 K=100 個 token                                       │
│  - 確保正確 token 在候選中                                                   │
│  - 只在這 K+1 個 token 上計算 softmax + CE                                  │
│                                                                              │
│  強制模型學會區分「容易混淆」的 token                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**優點**：
- 更高效（不浪費計算在遠距離 token）
- 專注於 hard cases

## 實驗設計

| 實驗 | Loss 類型 | 關鍵參數 | GPU | 預期效果 |
|------|----------|----------|-----|----------|
| exp13 | Linear + CE | label_smoothing=0.1 | 0 | Token Acc >> 2% |
| exp14 | Margin Loss | margin=0.5 | 1 | 更好的 embedding |
| exp15 | Hard Neg CE | K=100, temp=1.0 | 2 | 高效訓練 |

## 執行方式

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1205

# 三個實驗並行執行（使用不同 GPU）
nohup bash run_exp13_linear_ce.sh > exp13.log 2>&1 &
nohup bash run_exp14_margin.sh > exp14.log 2>&1 &
nohup bash run_exp15_hard_neg.sh > exp15.log 2>&1 &

# 監控進度
tail -f exp13.log
tail -f exp14.log
tail -f exp15.log
```

## 配置參數

### 共同參數
```bash
--lora_rank 128
--lora_alpha 256
--batch_size 20
--num_epochs 50
--learning_rate 5e-5
```

### exp13 特有參數
```bash
--loss_type linear_ce
--label_smoothing 0.1
```

### exp14 特有參數
```bash
--loss_type margin
--margin 0.5
```

### exp15 特有參數
```bash
--loss_type hard_neg
--hard_neg_k 100
--temperature 1.0
```

## 預期結果

| 實驗 | Baseline (exp11) | 目標 |
|------|------------------|------|
| exp13 | 2.21% | > 30% |
| exp14 | 2.21% | > 10% |
| exp15 | 2.21% | > 20% |

## 檔案結構

```
exp_1205/
├── config.py                    # 配置
├── data.py                      # 數據載入
├── model.py                     # 模型定義
├── losses.py                    # 三種新 Loss 實作
├── train.py                     # 訓練腳本
├── wavtok_lora_patch.py         # LoRA patch
├── wavtok_distance_mat_corrected.pt  # Distance matrix
├── run_exp13_linear_ce.sh       # exp13: Linear + CE
├── run_exp14_margin.sh          # exp14: Margin Loss
├── run_exp15_hard_neg.sh        # exp15: Hard Negative
└── README.md                    # 本說明文件
```

## 結果分析

實驗完成後，比較三種方案：

```bash
# 查看最佳結果
cat experiments/exp13_linear_ce/training_history.json | jq '.val_acc[-1]'
cat experiments/exp14_margin/training_history.json | jq '.val_acc[-1]'
cat experiments/exp15_hard_neg/training_history.json | jq '.val_acc[-1]'
```

預期：
- **exp13 (Linear + CE)** 效果最好，因為直接優化目標
- **exp14 (Margin)** 可能需要調整 margin 值
- **exp15 (Hard Neg)** 效果取決於 K 的選擇

## 後續實驗

如果 exp13 效果好：
1. 嘗試增加 Linear head 的複雜度（MLP）
2. 嘗試不同的 label smoothing 值
3. 結合 MSE 作為正則化

如果 exp14 效果好：
1. 嘗試不同的 margin 值
2. 結合 CE Loss

如果 exp15 效果好：
1. 嘗試不同的 K 值
2. 嘗試動態調整 K（curriculum）
