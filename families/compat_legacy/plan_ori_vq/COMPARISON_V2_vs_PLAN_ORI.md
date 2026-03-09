# Exp 0206: V2 (RVQ) vs Plan Original (Single VQ + EMA) 對比分析

**日期**: 2026-02-12
**狀態**: ✅ 兩項實驗皆完成 300 epochs

---

## 執行摘要

兩項實驗皆成功完成 300 epochs 訓練，且**都達到 P2 標準**，成功避免 token collapse。

**關鍵發現:**
- ✅ **V2 (RVQ)**: Entropy 9.01, Top-10 18.9%, Used 1142/2048 (55.7%)
- ✅ **Plan Original**: Entropy 9.45, Top-10 12.5%, Used 1437/4096 (35.1%)
- **結論**: Plan Original 在 token diversity 上表現更優，但使用率較低

---

## 1. 實驗配置對比

| 配置項目 | V2 (RVQ) | Plan Original (Single VQ + EMA) |
|---------|----------|--------------------------------|
| **Codebook 結構** | 4 layers × 2048 codes | 1 layer × 4096 codes |
| **總代碼空間** | 2048^4 ≈ 17.6T 組合 | 4096 codes |
| **初始化方式** | 隨機初始化 (cold start) | 預訓練 WavTokenizer (warm start) |
| **更新機制** | EMA (每層獨立) | EMA (單層) |
| **架構複雜度** | 較複雜 (多層殘差) | 較簡單 (單層) |

### 共同配置

```yaml
訓練參數:
  epochs: 300
  batch_size: 8
  grad_accum: 2
  learning_rate: 1e-4
  min_lr: 1e-6
  warmup_epochs: 10

LoRA 配置:
  rank: 256
  alpha: 512
  dropout: 0.2 (V2) / 未指定 (Plan Ori)

Loss 權重:
  lambda_quant: 1.0
  beta_commit: 1.0
  intermediate_weight: 0.03
  intermediate_L3_weight: 0.3
  intermediate_L4_weight: 0.5
  intermediate_L6_weight: 0.5

Curriculum Learning:
  start: 0.3
  end: 0.85
  epochs: 200

EMA 配置:
  decay: 0.99
  dead_code_threshold: 2
  usage_penalty: 0.1 (V2) / 0.0 (Plan Ori)
```

---

## 2. 最終結果對比 (Epoch 300)

### Token Diversity 指標

| 指標 | V2 (RVQ) | Plan Original | 優勝 |
|------|----------|---------------|------|
| **Entropy (bits)** | 9.01 | 9.45 | 🏆 Plan Ori (+4.9%) |
| **Top-10 mass (%)** | 18.88% | 12.53% | 🏆 Plan Ori (更分散) |
| **Used codes** | 1142/2048 (55.7%) | 1437/4096 (35.1%) | V2 (使用率) |
| **Joint usage** | 0.975 | - | - |

**解讀:**
- **Entropy**: Plan Original 更高 → token 分佈更均勻
- **Top-10 mass**: Plan Original 更低 → 不依賴少數熱門代碼
- **Used codes**: V2 使用率較高 (55.7% vs 35.1%)，但 Plan Original 絕對數量更多 (1437 vs 1142)

### Feature Alignment 指標

| 指標 | V2 (RVQ) | Plan Original | 優勝 |
|------|----------|---------------|------|
| **Feature MSE** | 0.0367 | 0.0418 | 🏆 V2 (更低) |
| **Train Total Loss** | 0.0462 | 0.0479 | 🏆 V2 (更低) |
| **Val Total Loss** | 0.0673 | 0.0715 | 🏆 V2 (更低) |
| **Best Val Loss** | - | 0.0668 | - |

**解讀:**
- V2 在 feature 對齊和總體 loss 上表現較好
- 差距不大 (MSE: 0.0367 vs 0.0418，僅 +13.9%)

### Pass Gates

| Gate | V2 (RVQ) | Plan Original |
|------|----------|---------------|
| **P2 Gate** | ✅ PASS | ✅ PASS |
| **P3 Gate** | ❌ FAIL | ❌ FAIL |

**P2 標準** (1000 steps):
- Entropy ≥ 5.0 ✅
- Top-10 mass ≤ 50% ✅
- Used codes ≥ 410 ✅
- Feature MSE ≤ 0.1 ✅

**P3 標準** (更嚴格):
- Entropy > 6.5 ✅ (兩者皆達標)
- Top-10 mass < 15% ❌ V2 未達標 / ✅ Plan Ori 達標
- Used codes ≥ 70% ❌ (V2: 55.7%, Plan Ori: 35.1%)

---

## 3. 訓練穩定性對比

### V2 (RVQ)

```
Final Epoch 300:
  Train loss: 0.0462 (quant=0.0200, inter=0.8238, commit=0.0014)
  Val loss: 0.0673 (quant=0.0366)

訓練時間: ~3 天 (單 GPU)
穩定性: ✅ 無 NaN/Inf，穩定收斂
```

### Plan Original (Single VQ + EMA)

```
Final Epoch 300:
  Train loss: 0.0479 (total)
  Val loss: 0.0715
  Best val loss: 0.0668 (某個較早的 epoch)

訓練時間: ~24 小時 (單 GPU，從 2026-02-11 04:25 → 2026-02-12 03:56)
穩定性: ✅ 無 NaN/Inf，穩定收斂
```

**對比:**
- Plan Original 訓練速度**明顯更快** (~1 天 vs ~3 天)
- 兩者皆穩定，無崩潰

---

## 4. 科學問題解答

### Q1: 預訓練 codebook + EMA 能否避免 collapse？

**答案: ✅ 是的！**

```
實驗證據:
  - Baseline (frozen pretrained): Entropy 6.07, Top-10 19.7%, Used 740/4096 (18%) ❌
  - Plan Original (EMA pretrained): Entropy 9.45, Top-10 12.5%, Used 1437/4096 (35%) ✅

結論: EMA 更新是關鍵，預訓練初始化並未帶來 collapse bias
```

### Q2: Warm start (預訓練) vs Cold start (隨機) 哪個更好？

**答案: 兩者皆有效，warm start 略優於 diversity**

```
對比:
  - V2 (Cold start): Entropy 9.01, Top-10 18.9%
  - Plan Ori (Warm start): Entropy 9.45, Top-10 12.5%

結論:
  - Warm start 可能提供更好的起點 (Entropy +4.9%, Top-10 -33.6%)
  - 但 Cold start 在 feature alignment 上略好 (MSE 0.0367 vs 0.0418)
```

### Q3: 單層 vs 多層的差異有多大？

**答案: 單層足夠，但多層在 feature alignment 上稍好**

```
對比:
  - Plan Ori (Single layer): Diversity 更好，Feature alignment 稍差
  - V2 (Multi-layer): Feature alignment 更好，Diversity 稍差

結論:
  - 單層 + EMA 足以避免 collapse
  - 多層殘差在 feature 重建上可能有優勢
  - 架構選擇取決於優先目標 (diversity vs alignment)
```

---

## 5. 輸出文件對比

### V2 (RVQ)

```
families/compat_legacy/plan_ori_vq/runs/longterm_20260208_114702/ (部分訓練)
  - training_curves_epoch*.png (10, 50, 100, 150, 200)
  - audio_samples/ (epoch 50, 100, 150, 200)
  - checkpoints/ (每 10 epochs)
  - best_model.pt (722 MB)

最新日誌: families/compat_legacy/plan_ori_vq/runs/longterm_v2_latest.log (84 MB)
  - 包含完整 300 epochs 訓練記錄
```

### Plan Original

```
families/compat_legacy/plan_ori_vq/runs/plan_ori_long_20260211/
  - training_curves_epoch*.png (1, 50, 100, 150, 200, 250, 300) ✅
  - audio_samples/ (epoch 50, 100, 150, 200, 250, 300) ✅
  - checkpoints/ (每 10 epochs: 10, 20, ..., 300) ✅
  - final_model.pt (678 MB)
  - best_model.pt (678 MB)
  - summary.json ✅
  - metrics_history.json ✅
  - config.json ✅
  - train.log (78 MB)
```

**對比:**
- Plan Original 輸出更完整 (包含 summary.json, final epoch 圖表)
- V2 缺少統一的 summary 文件

---

## 6. 性能總結

### V2 (RVQ) 優勢

✅ **Feature alignment 更好**: MSE 0.0367 vs 0.0418 (-12.2%)
✅ **Codebook 使用率更高**: 55.7% vs 35.1%
✅ **多層殘差結構**: 理論表達力更強 (2048^4)

### Plan Original 優勢

✅ **Token diversity 更好**: Entropy 9.45 vs 9.01 (+4.9%)
✅ **分佈更均勻**: Top-10 mass 12.5% vs 18.9% (-33.6%)
✅ **訓練速度快**: ~1 天 vs ~3 天
✅ **架構更簡單**: 單層 vs 4 層
✅ **模型更小**: 678 MB vs 722 MB
✅ **輸出更完整**: 包含 summary 和完整圖表

---

## 7. 實用建議

### 何時選擇 V2 (RVQ)?

```
優先場景:
  ✓ 需要最佳 feature alignment (音質優先)
  ✓ 需要高 codebook 使用率
  ✓ GPU 資源充足 (可接受 3 天訓練)
  ✓ 需要理論上更強的表達力
```

### 何時選擇 Plan Original (Single VQ + EMA)?

```
優先場景:
  ✓ 需要最佳 token diversity (分佈均勻)
  ✓ 訓練時間受限 (需要快速迭代)
  ✓ 部署資源受限 (模型更小更簡單)
  ✓ 與 WavTokenizer baseline 兼容性
  ✓ 研究 ablation study (單層 vs 多層)
```

---

## 8. 後續工作

### 待完成評估

- [ ] **三模式評估 (Plan Original)**
  - Teacher / Noisy VQ / Student 音質對比
  - PESQ / STOI 指標
  - 與 V2 三模式對比

- [ ] **主觀音質測試**
  - 盲聽測試 (blind listening test)
  - MOS (Mean Opinion Score)

- [ ] **推理性能對比**
  - 推理速度 (latency)
  - GPU 記憶體佔用
  - 吞吐量 (throughput)

### 深入分析

- [ ] **Codebook 分析**
  - 使用分佈直方圖
  - Dead codes 變化曲線
  - Top-k mass 演變

- [ ] **Feature 對齊分析**
  - 逐層 MSE 對比
  - t-SNE 可視化

- [ ] **Curriculum 影響**
  - 不同 noise level 下的表現
  - Curriculum vs Fixed noise 對比

---

## 9. 結論

### 核心發現

1. **EMA 更新是關鍵**: 無論單層或多層，EMA 都能有效避免 collapse
2. **預訓練無害**: Warm start 並未帶來 bias，反而在 diversity 上表現更好
3. **單層足夠**: Single VQ + EMA 已能達到優秀的 token diversity
4. **Trade-off**: Diversity (Plan Ori) vs Feature alignment (V2)

### 推薦方案

**短期 (當前專案):**
- 如果優先 **音質**: 選擇 V2 (RVQ)
- 如果優先 **效率 + Diversity**: 選擇 Plan Original

**長期 (生產部署):**
- **研究階段**: 兩者並行，深入分析 trade-offs
- **部署階段**: Plan Original (更簡單、更快、夠用)

### 實驗價值

✅ **方案 A (Plan Original)** 成功證明:
- 單層 VQ + EMA 是可行方案
- 預訓練初始化有益無害
- 為論文提供有力的 ablation study

✅ **方案 B (V2 RVQ)** 成功證明:
- 多層殘差在 alignment 上有優勢
- Random init + EMA 是穩健方案
- 可作為主要生產方案

---

**創建日期**: 2026-02-12
**最後更新**: 2026-02-12
**實驗狀態**: ✅ 兩項實驗皆完成 300 epochs
**分析者**: Claude Sonnet 4.5
