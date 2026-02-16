# WavTokenize Feature Analysis — 實驗報告

---

## 實驗 2026-02-11: exp_0206 Plan Ori — Long-run 300 epochs (進行中)

### 狀態
🔄 **訓練中** — 300-epoch long-run 已啟動，預計約 25 小時完成。

### 新增功能（相對 short-run）
- `save_audio_samples()`: 每 50 epochs 儲存 noisy/clean/vq_recon WAV（val + train 各 2 samples）
- `plot_training_curves()`: 4×3 佈局完整訓練曲線（含 per-layer intermediate loss）
- `summary.json`: 訓練結束時自動生成
- 擴充 history 追蹤欄位至 20 個（新增 val losses, curriculum_phase, per-layer losses, teacher_entropy）

### 執行命令
```bash
conda activate test
python exp_0206/plan_ori/train_single_vq_ema.py \
  --mode epoch --epochs 300 --batch_size 8 --grad_accum 2 \
  --learning_rate 1e-4 --warmup_epochs 10 \
  --save_checkpoint_every 10 --save_audio_interval 50 \
  --eval_max_batches 30 --seed 42 --device cuda:0 \
  --output_dir exp_0206/runs/plan_ori_long_20260211
```

### 輸出目錄
`exp_0206/runs/plan_ori_long_20260211/`

---

## 實驗 2026-02-11: exp_0206 Plan Ori — Single VQ K=4096 + EMA Update (Short-run)

### 實驗編號
`EXP-20260211-exp0206-plan-ori-short`

### 背景與動機
先前實驗使用 frozen codebook（baseline exp_k_v6）導致 student token 從 epoch 1 起即 collapse（top-10 mass 41.5% vs teacher 11.8%），
且 300 epoch 後仍無法完全恢復。exp_0206 V2 嘗試用 intermediate supervision 改善，但根本問題——codebook 無法適應 LoRA 後的 feature space——未解決。

**方案 A（Plan Ori）** 提出：使用預訓練 codebook 初始化（warm start）+ EMA 更新，讓 codebook 能跟隨 encoder 變化而自適應。
此為科學控制實驗，目標回答三個核心問題：
1. 預訓練 codebook + EMA 能否避免 token collapse？
2. Warm start vs Cold start 哪個更好？
3. 單層 VQ vs 多層 RVQ 是否必要？

### 變更摘要

| 項目 | Baseline (exp_k_v6) | RVQ (exp_0206) | Plan Ori (本實驗) |
|------|---------------------|----------------|-------------------|
| Codebook 結構 | 1×4096, dim=512 | 4×2048, dim=128 | **1×4096, dim=512** |
| Codebook 初始化 | pretrained (frozen) | random | **pretrained (EMA)** |
| 更新方式 | ❌ frozen | EMA (decay=0.99) | **EMA (decay=0.99)** |
| Dead-code 處理 | 無 | reset (threshold=2) | **reset (threshold=2)** |
| LoRA | rank=256, alpha=512 | rank=256, alpha=512 | rank=256, alpha=512 |

### 預期結果
- Entropy ≥5.0, Top-10 ≤50%, Used ≥410 (P2 targets)
- Token collapse 完全避免（warm start 優勢）
- 與 RVQ 在 diversity 指標上可比或更優

### 實際執行結果

#### Step 200 (P1 Gate) — ✅ PASS
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Top-10 mass | 0.0237 | ≤0.95 | ✅ |
| Used codes | 1579 | ≥82 | ✅ |
| Feature MSE | 0.0477 | ≤0.1 | ✅ |

#### Step 1000 (P2 Gate) — ✅ PASS
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Entropy | 10.305 | ≥5.0 | ✅ |
| Top-10 mass | 0.0295 | ≤0.5 | ✅ |
| Used codes | 1532 | ≥410 | ✅ |
| Feature MSE | 0.0418 | ≤0.1 | ✅ |

#### P3 Bonus — ⚠️ NOT MET (used_codes 1532 < 2867)
> Teacher 本身 used_codes 僅 1811，P3 目標 2867 (70%) 相對過於嚴格。

#### Metrics 演化

| Step | Entropy | Top-10 | Used | MSE | Loss |
|------|---------|--------|------|-----|------|
| 200 | 10.377 | 0.024 | 1579 | 0.048 | 0.082 |
| 400 | 10.292 | 0.026 | 1534 | 0.046 | 0.074 |
| 600 | 10.206 | 0.031 | 1495 | 0.044 | 0.072 |
| 800 | 10.312 | 0.028 | 1490 | 0.044 | 0.070 |
| 1000 | 10.305 | 0.030 | 1532 | 0.042 | 0.069 |

#### 與 Baselines 對比

| Method | Entropy | Top-10 | Used | Usage% |
|--------|---------|--------|------|--------|
| Baseline (frozen) | 6.07 | 19.7% | 740 | 18% |
| RVQ (4×2048) | 9.03 | 15.8% | 1089 | 53% |
| **Plan Ori (ours)** | **10.305** | **2.95%** | **1532** | **37.4%** |

### 解讀實驗結果

1. **Token collapse 完全避免**: Entropy 始終 >10.0，步驟 200 起即穩定。與 baseline 的初始 collapse (entropy=7.94, top-10=41.5%) 形成鮮明對比。
2. **Warm start 大幅優於 cold start**: Plan Ori (warm) entropy=10.305 >> RVQ (cold) entropy=9.03。預訓練 codebook 提供了優質初始空間分佈。
3. **單層 VQ 在 diversity 上足夠**: 單層 K=4096 的 entropy (10.305) 高於 4 層 RVQ (9.03)。但 MSE 略高 (0.042 vs 0.034)，RVQ 在重建精度上仍有微弱優勢。
4. **Student 達到 Teacher 98% 的 entropy**: Student entropy=10.305 / Teacher=10.525 = 97.9%，傳承效果極佳。

### 實驗反思
- **P3 目標需重新校準**: P3 要求 used_codes≥2867 (70%)，但 teacher 自身僅使用 1811 codes (44.2%)。合理的 P3 目標應基於 teacher 的實際使用率設定。
- **MSE 差距值得關注**: Plan Ori MSE (0.042) > RVQ (0.034)，long-run 實驗需觀察此差距是否縮小。
- **EMA decay 可能需要調優**: 當前 0.99 表現優異，但 long-run 是否需要 0.999 以增加穩定性有待驗證。
- **下一步**: P2 已通過，建議進行 300 epoch long-run 確認長期穩定性。

### 檔案
- 模型: `exp_0206/plan_ori/models_single_vq_ema.py`
- 訓練: `exp_0206/plan_ori/train_single_vq_ema.py`
- 測試: `exp_0206/plan_ori/test_single_vq_ema.py`
- 分析: `exp_0206/plan_ori/analyze_results.py`
- 結果: `exp_0206/plan_ori/RESULTS.md`
- Outputs: `exp_0206/runs/plan_ori_short_20260211/`

### 如何重現

```bash
# 1. 環境
conda activate test
cd /home/sbplab/ruizi/WavTokenize-feature-analysis

# 2. 單元測試
python -m pytest exp_0206/plan_ori/test_single_vq_ema.py -v

# 3. Short-run (1000 steps)
python exp_0206/plan_ori/train_single_vq_ema.py \
    --mode step --steps 1000 \
    --batch_size 8 --grad_accum 2 --lr 1e-4 \
    --eval_interval 200 --checkpoint_interval 200 \
    --seed 42 --output_dir exp_0206/runs/plan_ori_short_$(date +%Y%m%d)

# 4. 分析
python exp_0206/plan_ori/analyze_results.py exp_0206/runs/plan_ori_short_<date>
```

### 日期
2026-02-11

---

## 實驗 2026-02-09: exp_0206 V2 — Fixed Intermediate Weight

### 實驗編號
`EXP-20260209-exp0206-v2-fixed-iw`

### 背景與動機
exp_0206 V1 長期訓練至 ~191 epoch 發現 intermediate loss 佔 total loss 94%（因 intermediate_weight=0.5），
嚴重壓制 quant loss 梯度。Curriculum 使 intermediate loss 隨 noise 升高而上升 → total loss 虛假上升
（quant loss 實際仍在下降）。Warmdown 機制（0.5→0.25 over 50 epochs）是事後補救，不如直接修正權重。
另外 V1 在 epoch 177, 188 出現 NaN（L4 intermediate cosine similarity 遇到 zero-norm vectors）。

### 變更摘要

| 項目 | V1 | V2 |
|------|----|----|
| intermediate_weight | 0.5 → warmdown to 0.25 | **0.03 固定** |
| warmdown 參數 | intermediate_weight_min=0.25, warmdown_epochs=50 | **移除** |
| NaN 保護 | 無 | **跳過 NaN batch + 計數警告** |

### 預期結果
- quant loss 梯度佔比從 ~5% 提升至 ~80-90%
- Total loss 曲線將反映真實 quant 收斂，不再被 intermediate 主導
- NaN 不再導致 epoch 級別指標被汙染

### 檔案
- `exp_0206/train_long_v2.py` — V2 訓練腳本
- `exp_0206/run_v2.sh` — V2 啟動腳本

### 執行方式
```bash
bash exp_0206/run_v2.sh 0      # GPU 0
bash exp_0206/run_v2.sh 1      # GPU 1
```

### 日期
2026-02-09

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
