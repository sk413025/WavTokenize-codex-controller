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
families/compat_legacy/plan_ori_vq/
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
bash families/compat_legacy/plan_ori_vq/run.sh [GPU_ID]
```

預設使用 GPU 1（RTX 2080 Ti）。

---

## 實驗結果 (V1)

### 執行資訊

| 項目 | 詳情 |
|------|------|
| **開始時間** | 2026-02-08 11:47:02 |
| **結束時間** | 2026-02-09 00:49 (用戶手動停止) |
| **運行時長** | ~13 小時 (782 分鐘) |
| **完成 Epochs** | 191/300 (64%) |
| **GPU** | CUDA:0 (RTX 2080 Ti) |
| **實驗目錄** | `families/compat_legacy/plan_ori_vq/runs/longterm_20260208_114702/` |

### 崩潰時間軸

| Epoch | Entropy | Top-10 | Used Codes | Joint Div | MSE | 狀態 |
|-------|---------|--------|------------|-----------|-----|------|
| **50** | - | - | - | - | - | ✅ 正常訓練 |
| **100** | - | - | - | - | - | ✅ 正常訓練 |
| **150** | 9.03 | 14.8% | 1027/2048 | 0.961 | 0.031 | ✅ 健康 (P3 PASS) |
| **170** | 9.21 | 14.0% | 1133/2048 | 0.962 | 0.031 | ✅ 健康 |
| **175** | 9.03 | 14.8% | 1066/2048 | 0.962 | 0.031 | ✅ 健康 |
| **176** | 8.54 | **26.9%** | 1061/2048 | 0.944 | 0.032 | ⚠️ **開始退化** (P3 FAIL, NaN 首次出現) |
| **177** | 8.50 | **29.4%** | 1073/2048 | 0.926 | 0.033 | ⚠️ 繼續退化 (NaN 持續) |
| **178-191** | 8.3-9.4 | 9.6-30.8% | 965-1211/2048 | 0.91-0.97 | 0.031-0.032 | ⚠️ 劇烈震盪 |
| **192** | **-0.00** | **100%** | **1/2048** | **0.000** | **NaN** | 💀 **突然完全崩潰** |
| **193-300** | -0.00 | 100% | 1/2048 | 0.000 | NaN | 💀 無法恢復 |

### 關鍵發現

#### 1. **Epoch 192 突然完全 Codebook Collapse**

```
症狀:
  - Epoch 191: 正常 (entropy=8.90, used=1074/2048)
  - Epoch 192: 突然僅使用 1 個 code (entropy=-0.00, used=1/2048)
  - Epoch 192+: 無法恢復，所有 token 映射到同一 code

後果:
  - 重建音檔完全無人聲 (Decoder 輸入單調 → 輸出噪音/靜音)
  - Epoch 150 音檔正常，Epoch 200 音檔損壞
```

#### 2. **NaN 來源：Cosine Loss 中的 Zero-Norm Vectors**

**發現證據**：

檢查 `IntermediateSupervisionLossV6` ([train_v6.py:131-134](../families/compat_legacy/intermediate_stack/train_v6.py#L131-L134))：

```python
# Cosine Loss
student_norm = F.normalize(student_flat, dim=-1)  # ← 如果 norm=0 → NaN!
teacher_norm = F.normalize(teacher_flat * self.target_scale, dim=-1)
cos_sim = (student_norm * teacher_norm).sum(dim=-1).mean()
loss = 1 - cos_sim
```

**NaN 產生機制**：

```
Epoch 176-177: 某些 batch 的 intermediate features (L4) → norm ≈ 0
              ↓
         F.normalize(zero_vector) = NaN
              ↓
         Cosine loss = NaN
              ↓
         Total loss = λ_quant * L_quant + λ_inter * NaN = NaN
              ↓
         Backward 傳播 NaN 梯度 → 污染 encoder weights
              ↓
         Encoder output 偏移 → Codebook EMA 更新異常
              ↓
         Epoch 192: 累積效應觸發完全崩潰
```

**為什麼 L4 會產生 zero-norm vectors?**

可能原因：
1. **ReLU 飽和**：所有 activation → 0
2. **Gradient explosion 後的補償**：Weight 異常 → 輸出接近 0
3. **Curriculum learning 高噪音階段** (Epoch 176 ≈ 0.3 + 176/200*(0.85-0.3) ≈ 0.78 SNR)：
   - 高噪音輸入 → encoder 某些層輸出衰減
   - L4 是 ResBlock，可能對噪音敏感

#### 3. **Intermediate Loss 過度主導 (94% of Total Loss)**

**配置問題**：

```python
# V1 配置
intermediate_weight = 0.5
total_loss = λ_quant * L_quant + intermediate_weight * L_inter + β_commit * L_commit
           = 1.0 * 0.02 + 0.5 * 0.76 + 1.0 * 0.001
           = 0.02 + 0.38 + 0.001
           ≈ 0.40

# Intermediate 佔比
0.38 / 0.40 = 95%
```

**後果**：
- Quant loss 梯度被壓制 (僅佔 5%)
- Encoder 被 intermediate supervision 主導，逐漸偏離 quantizer
- 到達臨界點 (Epoch 192) 突然崩潰

#### 4. **為什麼 Epoch 150 音檔正常，200 無人聲？**

```
Epoch 150:
  ✅ Codebook 健康 (1027 codes, entropy=9.03)
  ✅ z_q 能表達多樣化語音特徵 (14.8% top-10)
  ✅ Decoder 正常重建人聲

Epoch 192+:
  ❌ Codebook 崩潰 (只剩 1 code)
  ❌ z_q = 同一個 vector 重複
  ❌ Decoder 輸入單調 → 輸出單調噪音/靜音
  ❌ 人聲資訊完全丟失
```

### 根本原因分析

| 層次 | 原因 | 證據 |
|------|------|------|
| **直接原因** | Epoch 192 codebook 完全崩潰 | entropy=-0.00, used=1/2048 |
| **觸發因素** | Epoch 176-177 NaN 傳播污染模型 | Train loss=NaN, inter=NaN |
| **NaN 來源** | Cosine loss 遇到 zero-norm vectors | `F.normalize(zero_vec) → NaN` |
| **結構問題** | Intermediate weight=0.5 過高 (佔 94%) | Quant loss 梯度被壓制 |
| **訓練階段** | Curriculum 高噪音階段 (Epoch 176 ≈ 0.78 SNR) | L4 對噪音敏感，輸出衰減 |

### V2 改進方案

已在 [README.md V2](README.md#v2-fixed-intermediate-weight-2025-02-09) 提出：

| 項目 | V1 | V2 | 效果 |
|------|----|----|------|
| **intermediate_weight** | 0.5 → 0.25 warmdown | **0.03 固定** | Quant loss 主導 (80-90%) |
| **warmdown** | 50 epochs | **移除** | 簡化設計 |
| **NaN 保護** | 無 | **跳過 NaN batch + 警告** | 防止梯度污染 |
| **Cosine loss eps** | 無 | **建議加 eps=1e-8** | 避免 zero-norm NaN |

### 教訓與建議

1. ✅ **Loss weight 必須平衡**
   - Auxiliary loss (intermediate) 不應超過 main loss (quant) 的 20%
   - 否則訓練方向被副目標主導

2. ✅ **Cosine loss 必須有 numerical stability 保護**
   ```python
   # 錯誤寫法
   F.normalize(x, dim=-1)  # x.norm()=0 → NaN

   # 正確寫法
   F.normalize(x, dim=-1, eps=1e-8)  # 或 x / (x.norm() + eps)
   ```

3. ✅ **NaN 必須立即停止訓練**
   - 繼續訓練只會累積損害
   - V1 從 Epoch 176 NaN 到 Epoch 192 崩潰，浪費 16 epochs

4. ✅ **監控 intermediate feature norms**
   - 如果某層 norm 接近 0 → 模型有問題
   - 應該在崩潰前就警告

---

## 下一步：V2 實驗

優先使用 V2 配置重新訓練：

```bash
bash families/compat_legacy/plan_ori_vq/run_v2.sh [GPU_ID]
```

關鍵改進：
- `intermediate_weight=0.03` (quant 主導)
- NaN batch 跳過機制
- 建議加入 cosine loss eps 保護
