# exp_0206: Long-Term RVQ Training

**日期**: 2026-02-06
**狀態**: 🚀 準備中
**前置**: Phase 3-2 Exp 6c (P2 PASS), exp_k_v6 (300 epochs baseline)

---

## 目標

將 Phase 3-2 Exp 6c 的 **RVQ + EMA + Dead-Code Reset** 從 1000 steps 短跑擴展為
**300 epochs 完整訓練**，驗證 long-term 穩定性並產出可用於下游的 audio token 表徵。

## 動機

| 問題 | 出處 | 本實驗目標 |
|------|------|-----------|
| 1000 steps 太短，無法確認 top10_mass 漂移是否持續 | Phase 3-2 SUMMARY | 300 epochs 觀察 |
| Baseline (exp_k_v6) 單層 VQ collapse → used_codes=740/4096 (18%) | exp_0128 分析 | RVQ 期望 ≥50% usage |
| P3 seed-sensitive (seed43 pass, seed44/45 fail) | Phase 3-2 多 seed 實驗 | 更長訓練是否穩定 |
| 尚無 audio reconstruction 品質驗證 | Phase 3-2 SUMMARY | 加入 audio sample 儲存 |

## 設計

### 架構：延續 Phase 3-2 Exp 6c

```
Teacher (frozen) ──→ t_e (encoder output)
                  ╲
                   ╲ L_quant = MSE(z_q, t_e)
                    ╲
Student (LoRA) ──→ z_e ──→ RVQ(EMA) ──→ z_q ──→ Decoder (frozen) ──→ audio
                  ╲                      ╱
                   ╲ L_inter (V6)       ╱ L_commit
                    ╲                  ╱
                     intermediate supervision
```

### 最佳配置（Phase 3-2 推薦）

| 參數 | 值 | 來源 |
|------|-----|------|
| RVQ layers | 4 | Phase 3-2 |
| Codebook size (K) | 2048 | Phase 3-2 6c-K2048 |
| EMA decay | 0.99 | Phase 3-2 |
| Dead-code threshold | 2 | Phase 3-2 |
| Usage penalty | 0.1 | Phase 3-2 最佳 simple |
| β_commit | 1.0 | Phase 3-2 |
| λ_quant | 1.0 | Phase 3-2 (post-quant alignment) |
| λ_pre | 0.0 | Phase 3-2 (disabled) |
| λ_inter | 0.5 → 0.25 | exp_k_v6 warmdown schedule |
| LoRA rank | 256 | exp_k_v6 |
| LoRA alpha | 512 | exp_k_v6 |
| LoRA dropout | 0.2 | exp_k_v6 |
| Intermediate layers | [3, 6] | Phase 3-2 (L4, L8) |

### 訓練配置（延續 exp_k_v6）

| 參數 | 值 | 來源 |
|------|-----|------|
| Epochs | 300 | exp_k_v6 |
| Batch size | 8 | exp_k_v6 |
| Gradient accumulation | 2 | exp_k_v6 (effective=16) |
| Learning rate | 1e-4 | exp_k_v6 |
| Min learning rate | 1e-6 | exp_k_v6 |
| Warmup epochs | 10 | exp_k_v6 |
| Weight decay | 0.01 | Phase 3-2 |
| Gradient clipping | 1.0 | 共同 |
| Curriculum | 0.3 → 0.85 over 200 epochs | exp_k_v6 |
| Intermediate warmdown | epoch 201-250: 0.5→0.25 | exp_k_v6 |
| AMP | ✅ | 共同 |

### 評估與儲存

| 項目 | 頻率 | 來源 |
|------|------|------|
| Checkpoint (LoRA + RVQ) | 每 10 epochs | exp_k_v6 |
| Full evaluation | 每 10 epochs | 新增 |
| Audio samples | 每 50 epochs | exp_k_v6 |
| Training curves | 每 10 epochs | exp_k_v6 |
| Best model | val loss 改善時 | exp_k_v6 |

### 評估指標

延續 Phase 3-2 驗收標準 + exp_k_v6 原有指標：

| 指標 | 說明 | Phase 3-2 目標 |
|------|------|---------------|
| layer0_entropy | token 分佈 entropy | ≥5.0 (P2), >6.5 (P3) |
| layer0_top10_mass | top-10 code 佔比 | ≤0.5 (P2), <0.15 (P3) |
| layer0_used_codes | 使用的 code 數 | ≥205 (P2, 10%K) |
| joint_diversity | 跨層多樣性 | ≥0.30 (P2), >0.7 (P3) |
| feature_mse | z_q 對齊品質 | ≤0.1 |
| val_accuracy | token match (與 baseline 比) | monitoring |

---

## 目錄結構

```
exp_0206/
├── PLAN.md           ← 本文件
├── README.md         ← 快速啟動指南
├── train_long.py     ← 主訓練腳本（300 epochs）
├── run.sh            ← 執行腳本
└── runs/             ← 實驗輸出（自動建立）
    └── longterm_YYYYMMDD_HHMMSS/
        ├── config.json
        ├── checkpoints/
        ├── audio_samples/
        ├── metrics_history.json
        ├── loss_history.json
        ├── summary.json
        └── training_curves.png
```

---

## 風險與緩解

| 風險 | 緩解策略 |
|------|----------|
| top10_mass 後期漂移 | usage penalty=0.1 + warmdown intermediate |
| OOM (RVQ 多層) | batch_size=8 + AMP + grad_accum=2 |
| Codebook collapse (長期) | EMA + dead-code reset (th=2) |
| Seed sensitivity | 使用 seed=42 作為主 run |
| 訓練時間過長 | 預估 ~12-24 小時 (300 epochs × ~7776 samples / effective_bs=16) |

---

## 執行方式

```bash
bash exp_0206/run.sh [GPU_ID]
```

預設使用 GPU 1（RTX 2080 Ti）。
