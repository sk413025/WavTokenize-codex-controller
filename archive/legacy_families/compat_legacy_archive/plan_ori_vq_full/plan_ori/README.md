# Exp 0206 - Plan Original: Single VQ 4096 + EMA

**Status**: 🟡 Backup Plan (待辦)

---

## Quick Summary

這是 **方案 A** 的實現計劃：使用 **Single VQ 4096 + EMA update**，作為 RVQ 的備選方案。

### 核心概念

```
保持 4096 單層 codebook
從 WavTokenizer 預訓練權重開始
用 EMA 讓它重新學習 token 中心
```

### 與其他方案對比

| 方案 | Codebook | 初始化 | 更新方式 | 狀態 |
|------|----------|--------|----------|------|
| **Baseline** | 4096 單層 | 預訓練 | Frozen ❌ | Collapsed |
| **方案 A** (本方案) | 4096 單層 | 預訓練 | EMA ✅ | 🟡 待實驗 |
| **方案 B** (RVQ) | 4×2048 多層 | 隨機 | EMA ✅ | ✅ P2 通過 |

---

## 文件結構

```
families/compat_legacy/plan_ori_vq/plan_ori/
├── README.md                    # 本文件 (快速導覽)
├── PLAN.md                      # 詳細計劃 (為什麼、什麼時候做)
├── SPEC.md                      # 技術規格 (怎麼做)
├── models_single_vq_ema.py      # TODO: 實現
├── train_single_vq_ema.py       # TODO: 訓練腳本
├── run_exp_ori_short.sh         # TODO: Short-run 腳本
└── RESULTS.md                   # TODO: 結果記錄
```

---

## 為什麼要做這個方案

### 科學問題

1. **預訓練 + EMA 能否避免 collapse？**
   - Baseline 凍結 → collapse ❌
   - RVQ 隨機初始化 → 成功 ✅
   - Single VQ + 預訓練 + EMA → ❓

2. **Warm start vs Cold start？**
   - 預訓練可能提供好的起點
   - 但也可能帶來 collapse bias

3. **單層 vs 多層的必要性？**
   - 如果 single VQ + EMA 也成功
   - 證明架構可以更簡單

---

## 實驗計劃

### Phase 1: Short-run (1000 steps)

```bash
# 目標: 驗證可行性
# 時間: 8-10 hours (single GPU)

python train_single_vq_ema.py \
  --steps 1000 \
  --codebook_size 4096 \
  --use_pretrained_init \
  --ema_decay 0.99 \
  --ema_dead_code_threshold 2
```

**驗收標準** (與 Phase 3-2 一致):
- P1 (step 200): top10≤0.95, used≥82
- P2 (step 1000): entropy≥5.0, top10≤0.5, used≥410

### Phase 2: Long-run (300 epochs, optional)

```bash
# 條件: Phase 1 P2 通過
# 時間: 2-3 days (single GPU)

python train_single_vq_ema.py \
  --epochs 300 \
  --eval_interval 10 \
  --checkpoint_interval 10
```

---

## 預期結果

### 樂觀 (成功)
```
Entropy: 8.5-9.0
Top-10: 15-18%
Used: 2500-3000/4096 (61-73%)

→ 更簡單的方案可行！
→ 作為主要方案
```

### 中等 (部分成功)
```
Entropy: 7.0-8.0
Top-10: 20-25%
Used: 1500-2000/4096 (37-49%)

→ 比 baseline 好，但不如 RVQ
→ 分析差異原因
```

### 悲觀 (失敗)
```
Entropy: ~6.0-6.5
Top-10: ~18-20%
Used: ~700-1000/4096 (17-24%)

→ 預訓練 bias 無法克服
→ 證明 RVQ 必要性
→ 終止方案 A
```

---

## 什麼時候做

### 觸發條件 ✅

- 方案 B (RVQ) long-term 完成
- 有空閒 GPU
- 想深入理解機制

### 不觸發 ❌

- 方案 B 還在進行中
- GPU 資源緊張
- 時間緊迫

---

## Next Steps

### For Implementer

1. **閱讀**: [SPEC.md](SPEC.md) - 技術細節
2. **實現**: `models_single_vq_ema.py`
3. **測試**: Unit tests + Smoke test (10 steps)
4. **運行**: Short-run (1000 steps)
5. **分析**: 填寫 [RESULTS.md](RESULTS.md)

### For Decision Maker

1. **檢查**: 方案 B 狀態
2. **評估**: GPU 資源可用性
3. **決定**: 是否啟動方案 A
4. **批准**: Short-run 實驗預算

---

## Timeline Estimate

```
Implementation:  4-6 hours
Testing:         2-3 hours
Short-run:       8-10 hours
Analysis:        2-3 hours
─────────────────────────────
Total:           2-3 days

Long-run (opt):  2-3 days
```

---

## References

- [Phase 3-2 SUMMARY](../../exp_0128/phase3-2/SUMMARY.md) - RVQ 成功案例
- [EXP6C_ARCHITECTURE](../../exp_0128/phase3-2/EXP6C_ARCHITECTURE.md) - 架構詳解
- [Baseline Analysis](../../exp_0128/baseline_token_analysis/PROGRESS.md) - Baseline 問題

---

**Created**: 2026-02-06
**Priority**: P2 (Backup Plan)
**Status**: 🟡 Pending (waiting for Plan B completion)
