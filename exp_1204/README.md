# exp_1204: Curriculum Learning + Mixed Loss (MSE + CE)

## 實驗目的

解決 exp_1203 的瓶頸：Token Accuracy 約 10%，距離 100% 還很遠。

## 架構圖

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     exp_1204: Curriculum Learning Flow                       │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │   Audio     │
                              │   Input     │
                              └──────┬──────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
           ┌───────────────┐                 ┌───────────────┐
           │    Student    │                 │    Teacher    │
           │   Encoder     │                 │   Encoder     │
           │  (LoRA fine-  │                 │  (Frozen)     │
           │   tuned)      │                 │               │
           └───────┬───────┘                 └───────┬───────┘
                   │                                 │
                   ▼                                 ▼
           ┌───────────────┐                 ┌───────────────┐
           │  student_emb  │                 │   VQ Layer    │
           │  (encoder輸出) │                 │  (量化)       │
           └───────┬───────┘                 └───────┬───────┘
                   │                                 │
                   │                                 ▼
                   │                         ┌───────────────┐
                   │                         │ teacher_codes │
                   │                         │ (token IDs)   │
                   │                         └───────┬───────┘
                   │                                 │
                   │         ┌───────────────────────┘
                   │         │
                   │         ▼
                   │  ┌─────────────────┐
                   │  │    Codebook     │
                   │  │  [4096 x 512]   │
                   │  └────────┬────────┘
                   │           │
                   │           ▼ codebook[teacher_codes]
                   │  ┌─────────────────┐
                   │  │  target_emb     │
                   │  │ (teacher選的emb) │
                   │  └────────┬────────┘
                   │           │
                   └─────┬─────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │      Curriculum Learning Loss       │
        │                                     │
        │   ┌─────────────┐ ┌─────────────┐  │
        │   │  MSE Loss   │ │  CE Loss    │  │
        │   │ (embedding  │ │ (token      │  │
        │   │  距離)       │ │  分類)      │  │
        │   └──────┬──────┘ └──────┬──────┘  │
        │          │               │         │
        │          ▼               ▼         │
        │   ┌─────────────────────────────┐  │
        │   │  Total Loss =               │  │
        │   │    MSE × mse_weight +       │  │
        │   │    CE  × ce_weight(t)       │  │
        │   └─────────────────────────────┘  │
        └────────────────┬───────────────────┘
                         │
                         ▼
              ┌─────────────────┐
              │   Backprop      │
              │  (更新 LoRA)     │
              └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Curriculum Schedule                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Temperature (τ)                CE Weight                                    │
│       ▲                              ▲                                       │
│  2.0 ─┼─────┐                   0.5 ─┼                    ┌─────────         │
│       │     │                        │                   /│                  │
│       │     └──────────────┐         │                 /  │                  │
│  0.1 ─┼                    └────     │               /    │                  │
│       └──────┬───────┬─────────►   0 ┼─────────────/──────┴──────────►       │
│              5      25     epoch        0          5      25      epoch      │
│          warmup  transition              warmup   transition                 │
│                                                                              │
│  Phase 1: Warm-up (0-5)    │ τ=2.0 (soft)  │ CE=0 (MSE only)                │
│  Phase 2: Transition (5-25)│ τ: 2.0→0.1    │ CE: 0→0.5                      │
│  Phase 3: Refinement (25+) │ τ=0.1 (hard)  │ CE=0.5                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            Loss 設計理念                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MSE Loss (embedding 距離):                                                  │
│  ┌─────────────┐                                                            │
│  │ student_emb │ ────────── MSE ────────── codebook[teacher_codes]          │
│  └─────────────┘            ↓                                                │
│                      平滑梯度，穩定訓練                                       │
│                                                                              │
│  CE Loss (token 分類):                                                       │
│  ┌─────────────┐     ┌───────────┐                                          │
│  │ student_emb │ ──► │  logits   │ ─── CE ─── teacher_codes                 │
│  └─────────────┘     │ (與所有   │      ↓                                    │
│         │            │ codebook  │  強烈梯度，直接監督                        │
│         └──────────► │ 計算距離) │                                           │
│                      └───────────┘                                          │
│                                                                              │
│  為什麼需要混合?                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ MSE alone: 只關心「接近」，不關心「選對」                               │   │
│  │ CE alone:  梯度太強，訓練初期不穩定                                     │   │
│  │ MSE + CE:  先穩定靠近，再精確選對                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 核心改進

### 1. 漸進式學習 (Curriculum Learning)

```
階段 1 (epoch 0-5): Warm-up
├── 高溫度 (temp=2.0) → 軟監督
├── MSE Loss 為主
└── CE weight = 0

階段 2 (epoch 5-25): Transition
├── 溫度逐漸降低 (2.0 → 0.1)
├── MSE + CE 混合
└── CE weight 逐漸增加 (0 → 0.5)

階段 3 (epoch 25+): Refinement
├── 低溫度 (temp=0.1) → 硬監督
├── MSE + CE 混合
└── CE weight = 0.5 (最終值)
```

### 2. 混合 Loss

| Loss | 功能 | 梯度特性 |
|------|------|---------|
| MSE Loss | `student_emb → codebook[teacher_codes]` | 平滑、穩定 |
| CE Loss | 直接監督 token 選擇 | 強烈、直接 |

### 3. Temperature Annealing

```
高溫度 (τ=2.0):
  - softmax 輸出較平滑
  - 允許一定誤差
  - 訓練初期使用

低溫度 (τ=0.1):
  - softmax 接近 one-hot
  - 要求精確對齊
  - 訓練後期使用
```

## 消融實驗 (Ablation Study)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Ablation Study Design                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  實驗              │ MSE │ CE  │ Temp Annealing │ GPU │ 說明               │
│  ─────────────────────────────────────────────────────────────────────────  │
│  exp_1203 exp10    │ ✓   │ ✗   │ ✗ (固定 1.0)   │  -  │ Baseline (~10%)    │
│  exp_1204          │ ✓   │ ✓   │ ✓ (2.0→0.1)    │  0  │ 完整方案           │
│  exp11             │ ✓   │ ✓   │ ✗ (固定 1.0)   │  1  │ 只加 CE            │
│  exp12             │ ✓   │ ✗   │ ✓ (2.0→0.1)    │  2  │ 只加 Temp Annealing│
│                                                                              │
│  比較結果分析：                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ if exp11 > exp12: CE 貢獻更大                                        │   │
│  │ if exp12 > exp11: Temperature Annealing 貢獻更大                     │   │
│  │ if exp_1204 > max(exp11, exp12): 兩者有加乘效果                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

| 實驗 | Loss | Curriculum | Temperature | Val Acc |
|------|------|------------|-------------|---------|
| exp_1203 exp10 | MSE only | ❌ | 1.0 (固定) | ~10% |
| **exp_1204** | MSE + CE | ✅ | 2.0 → 0.1 | ? |
| **exp11** | MSE + CE | ❌ | 1.0 (固定) | ? |
| **exp12** | MSE only | ✅ | 2.0 → 0.1 | ? |

## 執行方式

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1204

# exp_1204: 完整方案 (MSE + CE + Curriculum) - GPU 0
nohup bash run_curriculum_exp.sh > exp_curriculum.log 2>&1 &

# exp11: 只加 CE (無 Curriculum) - GPU 1
nohup bash run_exp11_ce_only.sh > exp11.log 2>&1 &

# exp12: 只加 Temperature Annealing (無 CE) - GPU 2
nohup bash run_exp12_temp_annealing.sh > exp12.log 2>&1 &
```

## 配置參數

```bash
# Curriculum Learning
--use_curriculum true
--warmup_epochs 5          # MSE 為主的階段
--transition_epochs 20     # 漸進轉換階段

# Loss 權重
--mse_weight 1.0           # MSE Loss 權重
--ce_weight 0.5            # CE Loss 最終權重

# Temperature
--initial_temperature 2.0  # 初始溫度（軟監督）
--final_temperature 0.1    # 最終溫度（硬監督）
--curriculum_mode linear   # 漸進模式
```

## 預期效果

1. **更穩定的訓練**：先用 MSE 打底，再用 CE 精調
2. **更高的 Token Accuracy**：CE Loss 直接優化目標
3. **更好的收斂**：Temperature Annealing 幫助跨越局部最優

## 檔案結構

```
exp_1204/
├── config.py                    # 配置（新增 curriculum 參數）
├── data.py                      # 數據載入（從 exp_1203 複製）
├── losses.py                    # CurriculumEmbDistillationLoss
├── model.py                     # 模型定義（從 exp_1203 複製）
├── train.py                     # 訓練腳本（支援 curriculum）
├── wavtok_lora_patch.py         # LoRA 相容 patch
├── wavtok_distance_mat_corrected.pt  # Distance matrix
├── run_curriculum_exp.sh        # exp_1204: 完整方案
├── run_exp11_ce_only.sh         # exp11: 只加 CE
├── run_exp12_temp_annealing.sh  # exp12: 只加 Temp Annealing
└── README.md                    # 本說明文件
```