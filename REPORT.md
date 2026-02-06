# WavTokenize Feature Analysis — 實驗報告

---

## 實驗 2026-02-06: Baseline (exp_k_v6) 自身的 Epoch 演化分析

### 實驗編號
`EXP-20260206-baseline-epoch-evolution`

### 背景與動機
先前 Phase 1 分析指出「epoch 1 就已經 collapse」，但不確定是誰 collapse、collapse 的程度如何、以及 baseline 訓練過程中 token 分布是否有改善。需要對 baseline (exp_k_v6) **自身**（不與其他模型比較）進行多個 epoch 的 token 分布分析，以回答：

1. **Q1 — Phase 1 的 collapse 是 baseline 嗎？** Epoch 10 時 student 是否已呈現 token 集中現象？
2. **Q2 — Baseline 初始 vs 訓練後的 token 分布差異？** 類似 FINAL_metrics_comparison_all.png 風格
3. **Q3 — Baseline 多 epoch 趨勢？** 利用 30 個 checkpoint (epoch 10–300) 觀察演化

### 實驗設定

| 項目 | 值 |
|---|---|
| 模型 | TeacherStudentIntermediate (LoRA: rank=256, alpha=512, dropout=0.2) |
| VQ | Single VQ, K=4096, **frozen codebook** |
| Checkpoint 目錄 | `exp_0112_intermediate/runs/exp_k_v6_20260125_234609_20260125_234613/checkpoints/` |
| 分析 epochs | 10, 50, 100, 200, 300 |
| 資料 | Train: 7776 samples (63GB cache), Val: 1440 samples (16GB cache) |
| Train batches 上限 | 40 batches × batch_size=8 = 320 samples |
| Val batches | 全部 180 batches = 1440 samples |
| GPU | RTX 2080 Ti (11GB), CUDA_VISIBLE_DEVICES=2 |
| 腳本 | `exp_0128/baseline_token_analysis/analyze_baseline_epoch_evolution.py` |

### 結果

#### 數值摘要

| Epoch | Val Acc % | S-Val Entropy | S-Val Top-10% | S-Val Used | T-Val Entropy | T-Val Top-10% | T-Val Used |
|------:|----------:|--------------:|--------------:|-----------:|--------------:|--------------:|-----------:|
| 10    | 0.694     | 7.94          | 41.5%         | 1618       | 9.04          | 11.8%         | 1583       |
| 50    | 0.754     | 8.21          | 32.8%         | 1522       | 9.04          | 11.8%         | 1583       |
| 100   | 0.807     | 8.46          | 27.3%         | 1744       | 9.04          | 11.8%         | 1583       |
| 200   | 0.846     | 8.66          | 25.7%         | 1735       | 9.04          | 11.8%         | 1583       |
| 300   | 0.888     | 8.94          | 22.8%         | 1791       | 9.04          | 11.8%         | 1583       |

#### 關鍵觀察

1. **Phase-1 Collapse 確認 — 是 Student（baseline）collapse**
   - Epoch 10 時 Student val top-10 mass = **41.5%**，Teacher 僅 **11.8%** → Student 集中度是 Teacher 的 **3.5 倍**
   - Student entropy = 7.94 bits，Teacher = 9.04 bits → 差距 **1.09 bits**
   - Student 在訓練最初就呈現嚴重的 token 過度集中（不是 Teacher collapse）

2. **Student 緩慢恢復但始終未追上 Teacher**
   - Entropy: 7.94 → 8.94 (+ 1.00 bits，接近 Teacher 的 9.04)
   - Top-10 mass: 41.5% → 22.8% (下降 18.7 pp，但 Teacher 僅 11.8%)
   - Used codes: 1618 → 1791 (增加 173，但因 codebook frozen 上限受限)
   - Val accuracy: 0.694% → 0.888% (幾乎無改善，終始低於 1%)

3. **Teacher 完全不變**
   - Teacher 在所有 epoch 的 val 指標完全一致 (entropy=9.04, top-10=11.8%, used=1583)
   - 這是因為 Teacher 是 frozen 的 pretrained WavTokenizer，不隨訓練更新

4. **Train vs Val 差異**
   - Train 的 top-10 mass 始終比 Val 低（因為 max_train_batches=40 限制了樣本量）
   - Epoch 300 train student entropy=8.58 vs val=8.94 → Val 更分散

#### 生成的圖表

| 檔案名稱 | 說明 |
|---|---|
| `baseline_metrics_epoch{010,050,100,200,300}_*.png` | 每個 epoch 的 FINAL_metrics_comparison 風格圖 (5張) |
| `baseline_top20_epoch{010,050,100,200,300}_*.png` | 每個 epoch 的 Top-20 Token 分布圖 (5張) |
| `baseline_self_trend_*.png` | Baseline 自身 5 epoch 趨勢圖 (entropy, top-10, used codes) |
| `baseline_{val,train}_student_freq_evolution_*.png` | Log-log frequency overlay 各 epoch (2張) |
| `baseline_{val,train}_student_top5_evolution_*.png` | Top-5 dominant token 各 epoch 演化 (2張) |

### 解讀

**Baseline 的 student 從一開始就 collapse，但訓練中有緩慢恢復的趨勢。** 這個 collapse 的核心原因是：

1. **Codebook frozen** — Student 被迫使用 Teacher 的 codebook，但 Student backbone 經 LoRA 微調後的 feature space 與 Teacher 不完全對齊，導致 VQ 量化時過度集中到少數 codewords
2. **LoRA 初始化** — LoRA 的 B 矩陣初始為 0，所以 epoch 0 的 student 輸出等同 Teacher（此時沒有 collapse），但一旦開始訓練（epoch 1+），LoRA 的微調會擾動 feature space，觸發 collapse
3. **恢復現象** — 隨著訓練，Student 逐漸學習到更好的 feature mapping，entropy 從 7.94 升到 8.94，但 top-10 mass 仍為 22.8%（Teacher 僅 11.8%），說明恢復不完全

**結論：先前提到的 Phase-1 collapse 是 baseline student 的 collapse，不是 teacher。Baseline 在整個 300 epoch 訓練中 val accuracy 始終低於 1%，token 集中度雖有改善但永遠無法追上 teacher。**

### 如何重現

```bash
# 1. 環境
conda activate test

# 2. 執行分析腳本
cd /home/sbplab/ruizi/WavTokenize-feature-analysis
CUDA_VISIBLE_DEVICES=2 python -u exp_0128/baseline_token_analysis/analyze_baseline_epoch_evolution.py \
    --epochs 10,50,100,200,300 \
    --batch_size 8 \
    --max_train_batches 40 \
    --device cuda:0

# 3. 結果位於
# exp_0128/baseline_token_analysis/baseline_epoch_evolution/
```

### 日期
2026-02-06
