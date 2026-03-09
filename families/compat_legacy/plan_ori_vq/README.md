# exp_0206: Long-Term RVQ Training

## 目標

將 Phase 3-2 Exp 6c 驗證過的 RVQ+EMA 配置，從 1000-step 短期測試
擴展至 300-epoch 長期訓練，確認在充分訓練後 RVQ codebook 是否持續健康。

## 核心配置

| 項目 | 設定 | 來源 |
|------|------|------|
| Model | TeacherStudentRVQ + LoRA (r=256, α=512) | exp_k_v6 |
| RVQ | 4 layers × K=2048, EMA (decay=0.99), dead-code reset (th=2) | Phase 3-2 6c |
| Usage Penalty | 0.1 × log(cluster_size) | Phase 3-2 6c best |
| L_quant | 1.0 × MSE(z_q, t_e) — post-quant alignment | Phase 3-2 |
| L_pre | 0.0 (disabled) | Phase 3-2 |
| L_inter | 0.5→0.25 warmdown, cosine loss @ L3(0.3)+L4(0.5)+L6(0.5) | exp_k_v6 |
| β_commit | 1.0 | Phase 3-2 |
| Curriculum | 0.3→0.85 over 200 epochs | exp_k_v6 |
| LR | 1e-4 → 1e-6 (cosine + 10 epoch warmup) | exp_k_v6 |
| Training | 300 epochs, batch=8×2 grad_accum, AMP | exp_k_v6 |

## 驗收標準

### P2 (基本驗收)
- entropy ≥ 5.0
- top-10 mass ≤ 0.5
- used codes ≥ 10% of K (≥ 205)
- joint diversity ≥ 0.30
- feature MSE ≤ 0.1

### P3 (長期品質)
- entropy > 6.5
- top-10 mass < 0.15
- joint diversity > 0.7
- feature MSE < 0.1

## 參考基線

| 來源 | Entropy | Top-10 | Used | Joint Div | MSE |
|------|---------|--------|------|-----------|-----|
| Phase 3-2 6c (1000 steps) | 9.03 | 0.158 | 1089/2048 | 0.992 | 0.034 |
| exp_k_v6 (300 ep, single VQ) | 6.07 | 0.197 | 740/4096 | N/A | N/A |

## 執行方式

```bash
# 使用 GPU 0 (預設)
bash families/compat_legacy/plan_ori_vq/run.sh

# 使用 GPU 1
bash families/compat_legacy/plan_ori_vq/run.sh 1

# 自定義實驗名稱
bash families/compat_legacy/plan_ori_vq/run.sh 0 my_experiment
```

## 輸出結構

```
families/compat_legacy/plan_ori_vq/runs/<exp_name>_<timestamp>/
├── config.json           # 完整配置
├── train.log             # 訓練日誌
├── metrics_history.json  # 每 epoch 指標
├── best_model.pt         # 最佳模型 (完整)
├── final_model.pt        # 最終模型 (完整)
├── summary.json          # 最終摘要
├── training_curves_*.png # 訓練曲線圖
├── checkpoints/
│   └── checkpoint_epoch*.pt  # 每 10 epochs (LoRA + RVQ only)
└── audio_samples/
    ├── val/epoch_*/       # 每 50 epochs
    └── train/epoch_*/
```

## 風險與因應

| 風險 | 監測指標 | 因應措施 |
|------|----------|----------|
| Token collapse | entropy, top-10 mass | 檢查 dead-code reset 是否正常運作 |
| Feature divergence | feature MSE > 0.1 | 降低 lr 或增加 λ_quant |
| Intermediate 阻礙收斂 | train loss 停滯 | warmdown 機制已內建 |
| OOM | GPU memory | batch_size=8 + AMP (已驗證) |

---

## V2: Fixed Intermediate Weight (2025-02-09)

### 動機

V1 訓練至 ~191 epoch 發現：
1. **intermediate loss 佔 total loss 94%**（因 intermediate_weight=0.5），壓制 quant loss 梯度
2. Curriculum 使 intermediate loss 隨 noise 升高而上升 → total loss 虛假上升（quant loss 其實仍在下降）
3. Warmdown 機制是事後補救 — 先設過高權重再調降，不如一開始就設對
4. Epoch 177, 188 出現 NaN（L4 intermediate cosine loss zero-norm vectors）

### V2 變更

| 項目 | V1 | V2 | 原因 |
|------|----|----|------|
| intermediate_weight | 0.5 → 0.25 warmdown | **0.03 固定** | 讓 inter 佔 total ~5-10% |
| warmdown 參數 | intermediate_weight_min=0.25, warmdown_epochs=50 | **移除** | 不再需要 |
| NaN 保護 | 無 | **跳過 NaN batch + 警告** | 防止 L4 cosine zero-norm |

### 預期效果

- quant loss 梯度佔比從 ~5% 提升至 ~80-90%
- Total loss 曲線反映真實 quant 收斂，不再被 intermediate 主導
- 更快收斂（quant 主導訓練方向）

### 執行方式

```bash
bash families/compat_legacy/plan_ori_vq/run_v2.sh        # GPU 0
bash families/compat_legacy/plan_ori_vq/run_v2.sh 1      # GPU 1
```

### V1 vs V2 對照

```
V1: python families/compat_legacy/plan_ori_vq/train_long.py   --intermediate_weight 0.5  --intermediate_weight_min 0.25  --warmdown_epochs 50
V2: python families/compat_legacy/plan_ori_vq/train_long_v2.py --intermediate_weight 0.03  (no warmdown params)
```

## 設計文件

詳見 [PLAN.md](PLAN.md)。
