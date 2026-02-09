# Exp 0206 - Plan Original: Single VQ 4096 + EMA Update

**日期**: 2026-02-06
**狀態**: 🟡 待辦 (Backup Plan)
**優先級**: P2 (方案 B RVQ 之後)

---

## 核心概念

### 方案對比

| 特性 | 方案 A (Original) | 方案 B (RVQ) |
|------|------------------|--------------|
| **Codebook 結構** | Single VQ (1 layer) | Residual VQ (4 layers) |
| **Codebook 大小** | 4096 codes | 4×2048 codes |
| **初始化** | 預訓練 WavTokenizer codebook | 隨機初始化 |
| **更新方式** | EMA update | EMA update (每層獨立) |
| **表達力** | 4096 | 2048^4 ≈ 17.6T |
| **優先級** | P2 (備用) | P1 (主要方案) |

---

## 為什麼要做方案 A

### 科學問題

```
Q1: 預訓練 codebook + EMA 能否避免 collapse？
    - Baseline 凍結 → collapse (top10=19.7%)
    - RVQ 隨機初始化 → 成功 (top10=15.8%)
    - Single VQ + EMA + 預訓練初始化 → ？

Q2: Warm start (預訓練) vs Cold start (隨機) 哪個更好？
    - 預訓練可能提供好的起點
    - 但也可能帶來 collapse bias

Q3: 單層 vs 多層的差異有多大？
    - RVQ 的成功是因為 EMA，還是因為多層？
    - 如果 single VQ + EMA 也成功 → 架構更簡單
```

### 實用價值

```
✅ 如果成功:
  - 更簡單的架構 (single layer)
  - 可能更快收斂 (warm start)
  - 更容易部署 (less parameters)
  - 兼容性更好 (與 baseline 一致)

❓ 如果失敗:
  - 證明預訓練 bias 確實有害
  - 證明 RVQ 多層結構是必要的
  - 更深入理解 collapse 機制
```

---

## 實驗設計

### Phase 1: Short-run 驗證 (1000 steps)

**目標**: 驗證 Single VQ 4096 + EMA 的可行性

#### 實驗配置

```yaml
Architecture:
  - Single VQ (1 layer)
  - Codebook size: 4096
  - Dimension: 128
  - Initialization: WavTokenizer pretrained codebook

Quantizer Update:
  - Mode: EMA
  - Decay: 0.99
  - Epsilon: 1e-5
  - Dead-code threshold: 2
  - Usage penalty: 0.0 (optional: 0.1)

Loss:
  - λ_quant: 1.0  (Post-quant alignment)
  - λ_pre: 0.0    (Disabled)
  - λ_inter: 0.5  (Intermediate supervision)
  - β_commit: 1.0 (Encoder commitment)

Training:
  - Steps: 1000
  - Batch size: 8
  - Grad accum: 2
  - Learning rate: 1e-4
  - Eval interval: 200

Baseline Comparison:
  - Compare with Baseline (frozen 4096)
  - Compare with RVQ (4×2048, random init)
```

#### 驗收標準 (與 Phase 3-2 一致)

| Gate | Time | Condition | Status |
|------|------|-----------|--------|
| P1 | step 200 | top10≤0.95, used≥82, mse≤0.1 | TBD |
| P2 | step 1000 | entropy≥5.0, top10≤0.5, used≥410, mse≤0.1 | TBD |
| P3 | step 1000 | entropy>6.5, top10<0.15, used≥2867 | TBD |

---

### Phase 2: Long-term 訓練 (300 epochs, 如果 P2 通過)

**目標**: 驗證長期穩定性

```yaml
Training:
  - Epochs: 300
  - Eval interval: 10 epochs
  - Checkpoint: 每 10 epochs

Success Criteria:
  - No late-stage collapse
  - Entropy 維持 ≥6.5
  - Top-10 mass 維持 ≤20%
  - Feature MSE 維持 ≤0.1

Comparison:
  - vs Baseline (exp_k_v6, 300 epochs)
  - vs RVQ (方案 B, 300 epochs)
```

---

## 實現計劃

### 新增文件

```
exp_0206/plan_ori/
├── PLAN.md                          # 本文件
├── SPEC.md                          # 技術規格
├── models_single_vq_ema.py          # Single VQ + EMA 實現
├── train_single_vq_ema.py           # 訓練腳本
├── run_exp_ori_short.sh             # Short-run 腳本
├── run_exp_ori_long.sh              # Long-run 腳本
└── RESULTS.md                       # 結果記錄 (待填)
```

### 修改策略

#### 選項 1: 繼承 RVQ 代碼修改 (推薦)
```python
# 基於 models_rvq.py 修改
# 將 n_layers=4 簡化為 n_layers=1
# 從預訓練 codebook 初始化
```

#### 選項 2: 獨立實現
```python
# 完全新寫 SingleVQWithEMA class
# 優點: 代碼更清晰
# 缺點: 需要重新實現 EMA 邏輯
```

---

## 關鍵差異點

### 與 Baseline 差異

```
Baseline (exp_k_v6):
  ✅ Single VQ 4096
  ✅ 預訓練初始化
  ❌ Quantizer frozen
  ❌ Pre-quant alignment only
  ❌ No commitment loss

方案 A (Single VQ + EMA):
  ✅ Single VQ 4096         (same)
  ✅ 預訓練初始化           (same)
  ✅ EMA update             (NEW!)
  ✅ Post-quant alignment   (NEW!)
  ✅ Bidirectional commit   (NEW!)
```

### 與 RVQ 差異

```
RVQ (方案 B):
  ✅ EMA update
  ✅ Post-quant alignment
  ✅ Bidirectional commit
  ❌ Multi-layer (4 layers)
  ❌ Random init (cold start)

方案 A (Single VQ + EMA):
  ✅ EMA update             (same)
  ✅ Post-quant alignment   (same)
  ✅ Bidirectional commit   (same)
  ✅ Single layer           (simpler!)
  ✅ Pretrained init        (warm start!)
```

---

## 預期結果

### 樂觀情境 (方案 A 成功)

```
Step 1000 metrics:
  Entropy: ~8.5-9.0 (接近 RVQ)
  Top-10 mass: ~15-18% (接近 RVQ)
  Used codes: ~2500-3000/4096 (61-73%)
  Feature MSE: ~0.03-0.04

結論:
  ✅ EMA 是關鍵，多層非必要
  ✅ 預訓練初始化沒有害處
  ✅ 更簡單的方案可行

後續:
  → 方案 A long-term (300 epochs)
  → 作為主要方案推廣
```

### 中等情境 (部分成功)

```
Step 1000 metrics:
  Entropy: ~7.0-8.0 (比 baseline 好，比 RVQ 差)
  Top-10 mass: ~20-25% (介於中間)
  Used codes: ~1500-2000/4096 (37-49%)
  Feature MSE: ~0.04-0.05

結論:
  ⚠️ EMA 有幫助，但不如 RVQ
  ⚠️ 預訓練可能帶來輕微 bias
  ⚠️ 單層可能表達力不足

後續:
  → 仍可進行 long-term 驗證
  → 分析與 RVQ 差異原因
```

### 悲觀情境 (失敗)

```
Step 1000 metrics:
  Entropy: ~6.0-6.5 (與 baseline 相當)
  Top-10 mass: ~18-20% (接近 baseline)
  Used codes: ~700-1000/4096 (17-24%)
  Feature MSE: 0.03 (好，但 diversity 差)

結論:
  ❌ 預訓練 codebook 帶來 collapse bias
  ❌ 單層量化表達力不足
  ✅ 證明 RVQ 多層結構是必要的

後續:
  → 終止方案 A
  → 全力投入方案 B (RVQ)
  → 寫論文時作為 ablation study
```

---

## Timeline

### Short-run (1000 steps)

```
Week 1:
  Day 1-2: 實現 SingleVQWithEMA class
  Day 3:   單元測試 & 驗證
  Day 4:   Short-run 實驗 (1000 steps)
  Day 5:   分析結果 & 決定是否繼續

Estimated time:
  - Implementation: 4-6 hours
  - Testing: 2-3 hours
  - Training: 8-10 hours (single GPU)
  - Analysis: 2-3 hours
  Total: 2-3 days
```

### Long-run (300 epochs, 如果 P2 通過)

```
Week 2-3:
  - Training: ~2-3 days (single GPU)
  - Evaluation: 1 day
  - Comparison with RVQ: 1 day
  - Documentation: 1 day

Total: 1-2 weeks
```

---

## Dependencies

### 必需條件

```
1. Phase 3-2 完成
   - RVQ 實現已驗證
   - EMA 機制已測試

2. WavTokenizer checkpoint
   - 需要載入預訓練 codebook
   - Path: WAVTOK_CKPT

3. GPU 資源
   - 至少 1×RTX 2080 Ti
   - Short-run: ~8-10 hours
   - Long-run: ~2-3 days
```

### Optional (如果要平行跑)

```
如果要與 RVQ 平行對比:
  - 2 GPUs
  - 方案 A: GPU 0
  - 方案 B: GPU 1
```

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 預訓練 bias 無法克服 | Medium | High | 如失敗，證明 RVQ 必要性 |
| 單層表達力不足 | Medium | Medium | 對比 RVQ 量化分析 |
| EMA 參數需調整 | Low | Low | 參考 RVQ 成功配置 |
| 實現 bug | Low | Medium | 充分測試 + code review |

### Resource Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU 不足 | Low | Medium | Short-run 優先，驗證後再 long-run |
| 時間不足 | Low | Low | Short-run 只需 2-3 天 |
| 與方案 B 衝突 | Low | Low | 方案 B 優先，方案 A 作為備用 |

---

## Decision Points

### Checkpoint 1: 實現完成後

```
✅ Code review pass
✅ Unit tests pass
✅ Smoke test (10 steps) pass
→ 進入 short-run 實驗
```

### Checkpoint 2: Short-run 結束 (step 1000)

```
如果 P2 通過:
  → 考慮 long-run 實驗
  → 與方案 B 對比
  → 可能作為主要方案

如果 P2 失敗:
  → 分析失敗原因
  → 寫入 ablation study
  → 終止方案 A
```

### Checkpoint 3: Long-run 結束 (300 epochs, optional)

```
如果優於 RVQ:
  → 方案 A 成為主要方案
  → 方案 B 作為備選

如果相當於 RVQ:
  → 選擇更簡單的方案 A
  → 或選擇表達力更強的 RVQ

如果劣於 RVQ:
  → 方案 B 為主要方案
  → 方案 A 作為 ablation
```

---

## Success Metrics

### Short-run Success (P2)

```
✅ Entropy ≥ 5.0
✅ Top-10 mass ≤ 50%
✅ Used codes ≥ 410/4096 (10%)
✅ Feature MSE ≤ 0.1

Bonus (P3):
✅ Entropy > 6.5
✅ Top-10 mass < 15%
✅ Used codes ≥ 2867/4096 (70%)
```

### Long-run Success

```
✅ No late-stage collapse
✅ Entropy 維持 ≥6.5 for 300 epochs
✅ Top-10 mass 維持 ≤20%
✅ Comparable or better than RVQ
✅ Good audio reconstruction quality
```

---

## Notes

### Why Backup Plan

```
1. 方案 B (RVQ) 已驗證有效
   - 優先完成 long-term
   - 風險更低

2. 方案 A 有科學價值
   - 理解機制
   - Ablation study
   - 但非必要

3. 資源限制
   - GPU 有限
   - 時間有限
   - 不能同時做兩個 long-run
```

### When to Activate

```
觸發條件:
  ✅ 方案 B long-run 完成
  ✅ 有空閒 GPU
  ✅ 想深入理解機制
  ✅ 或方案 B 出現問題需要備選

不觸發:
  ❌ 方案 B 還在進行
  ❌ GPU 資源緊張
  ❌ 時間緊迫
```

---

**創建日期**: 2026-02-06
**最後更新**: 2026-02-06
**狀態**: 🟡 待辦 (Backup Plan)
**優先級**: P2 (方案 B 之後)
