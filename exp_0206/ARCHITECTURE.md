# exp_0206: 模型架構與 Loss 配置

## 整體架構圖

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TeacherStudentRVQ                                      │
│                                                                             │
│  ┌───────────────────────────┐    ┌───────────────────────────────────────┐ │
│  │  🧊 Teacher (Frozen)      │    │  🔥 Student (LoRA r=256, α=512)      │ │
│  │                           │    │                                       │ │
│  │  clean_audio (B,1,T)      │    │  noisy_audio (B,1,T)                 │ │
│  │         │                 │    │         │                             │ │
│  │         ▼                 │    │         ▼                             │ │
│  │  ┌─────────────┐         │    │  ┌─────────────┐                      │ │
│  │  │  WavTokenizer│         │    │  │  WavTokenizer│                      │ │
│  │  │  Encoder     │         │    │  │  Encoder     │                      │ │
│  │  │  (18 Conv)   │         │    │  │  + LoRA      │                      │ │
│  │  │             ╠═L3═══════╪════╪══╬═════L3══════╗│  L_inter             │ │
│  │  │             ╠═L4═══════╪════╪══╬═════L4══════╣│  (cosine loss)       │ │
│  │  │             ╠═L6═══════╪════╪══╬═════L6══════╝│                      │ │
│  │  └──────┬──────┘         │    │  └──────┬──────┘                      │ │
│  │         │ t_e            │    │         │ s_e                          │ │
│  │         ▼                │    │         ▼                              │ │
│  │  ┌─────────────┐        │    │  ┌──────────────┐                      │ │
│  │  │  Original VQ │        │    │  │  RVQ (4層)    │                      │ │
│  │  │  K=4096      │        │    │  │  K=2048/層    │                      │ │
│  │  │  (Frozen)    │        │    │  │  EMA update   │                      │ │
│  │  └──────┬──────┘        │    │  │  Dead-code    │                      │ │
│  │         │               │    │  │  reset (th=2) │                      │ │
│  │         │ teacher_codes │    │  └──────┬──────┘                      │ │
│  │         ▼               │    │         │ z_q (student_quantized)      │ │
│  └─────────────────────────┘    │         ▼                              │ │
│                                  └───────────────────────────────────────┘ │
│                                                                             │
│  t_e ◄─────────────── L_quant (MSE) ──────────────► z_q                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Encoder 層級結構（WavTokenizer EnCodec Encoder）

```
model[0]  ─── Conv1d                    L0   (Input projection)
model[1]  ─── ResBlock (conv+shortcut)  L1-L3
model[2]  ─── ELU
model[3]  ─── Conv1d (↓downsample)      L4   ◄── 中間層監督 ① (w=0.3)
model[4]  ─── ResBlock (conv+shortcut)  L5-L7
model[5]  ─── ELU
model[6]  ─── Conv1d (↓downsample)      L8   ◄── 中間層監督 ② (w=0.5)
model[7]  ─── ResBlock (conv+shortcut)  L9-L11
model[8]  ─── ELU
model[9]  ─── Conv1d (↓downsample)      L12  ◄── 中間層監督 ③ (w=0.5)
model[10] ─── ResBlock (conv+shortcut)  L13-L15
model[11] ─── ELU
model[12] ─── Conv1d (↓downsample)      L16
model[13] ─── ELU
model[14] ─── LSTM
model[15] ─── Conv1d (output)           L17
                 │
                 ▼
          encoder_out (B, 512, T')       dim=512

注意: intermediate_indices 使用 model index [3, 4, 6]
      即 model[3]=L4(↓), model[4]=ResBlock2, model[6]=L8(↓)
```

## RVQ 殘差量化器架構

```
encoder_out (B, 512, T')
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Residual Vector Quantizer (4 layers)            │
│                                                  │
│  residual_0 = encoder_out                        │
│       │                                          │
│       ▼                                          │
│  ┌──────────┐                                    │
│  │ VQ Layer 0│  K=2048, dim=512                   │
│  │  (EMA)    │  codes_0 = argmin ‖r₀ - cᵢ‖²     │
│  └────┬─────┘  quantized_0 = codebook[codes_0]   │
│       │                                          │
│  residual_1 = residual_0 - quantized_0           │
│       │                                          │
│       ▼                                          │
│  ┌──────────┐                                    │
│  │ VQ Layer 1│  K=2048, dim=512                   │
│  │  (EMA)    │  codes_1 = argmin ‖r₁ - cᵢ‖²     │
│  └────┬─────┘  quantized_1 = codebook[codes_1]   │
│       │                                          │
│  residual_2 = residual_1 - quantized_1           │
│       │                                          │
│       ▼                                          │
│  ┌──────────┐                                    │
│  │ VQ Layer 2│  K=2048, dim=512                   │
│  │  (EMA)    │  codes_2 = argmin ‖r₂ - cᵢ‖²     │
│  └────┬─────┘  quantized_2 = codebook[codes_2]   │
│       │                                          │
│  residual_3 = residual_2 - quantized_2           │
│       │                                          │
│       ▼                                          │
│  ┌──────────┐                                    │
│  │ VQ Layer 3│  K=2048, dim=512                   │
│  │  (EMA)    │  codes_3 = argmin ‖r₃ - cᵢ‖²     │
│  └──────────┘  quantized_3 = codebook[codes_3]   │
│                                                  │
│  z_q = quantized_0 + quantized_1                 │
│      + quantized_2 + quantized_3                 │
│                                                  │
│  Codebook update: EMA (decay=0.99)               │
│  Dead-code reset: threshold=2                    │
│  Usage penalty: 0.1 × log(cluster_size)          │
└──────────────────────────────────────────────────┘
```

## Loss 配置

### V2 Loss 公式

```
L_total = λ_quant · L_quant
        + λ_inter · L_inter
        + β_commit · L_commit
        + λ_codebook · L_codebook
        + λ_pre · L_pre

其中:
  L_quant    = MSE(z_q, t_e)                     主損失: student RVQ 輸出 vs teacher encoder 輸出
  L_inter    = Σᵢ wᵢ · (1 - cos_sim(sᵢ, tᵢ))   中間層 cosine 監督
  L_commit   = Σⱼ MSE(sg[zⱼ], eⱼ)               RVQ encoder commitment (EMA 自帶)
  L_codebook = 0 (EMA mode 不需要)
  L_pre      = 0 (disabled)
```

### 權重配置

```
┌────────────────────┬──────────────┬────────────────────┐
│ Loss               │ 權重         │ 說明               │
├────────────────────┼──────────────┼────────────────────┤
│ L_quant (post-VQ)  │ λ=1.0        │ 主損失             │
│ L_inter (cosine)   │ λ=0.03       │ V2 固定 (V1: 0.5)  │
│ L_commit           │ β=1.0        │ RVQ EMA commitment │
│ L_codebook         │ λ=0.0        │ EMA mode 不需要    │
│ L_pre (pre-VQ)     │ λ=0.0        │ 已停用             │
└────────────────────┴──────────────┴────────────────────┘
```

### Intermediate Loss 各層權重

```
L_inter = 0.03 × [ 0.3 · (1-cos(s₃, t₃))      ← model[3] L4 Downsample 1
                  + 0.5 · (1-cos(s₄, t₄))      ← model[4] ResBlock 2
                  + 0.5 · (1-cos(s₆, t₆)) ]    ← model[6] Downsample 2
```

### V1 vs V2 Loss 梯度佔比（估算）

```
                V1 (iw=0.5)              V2 (iw=0.03)
              ┌─────────────┐          ┌─────────────┐
              │▓▓▓▓▓▓▓▓▓▓▓ │          │▒▒           │
   L_inter    │▓▓▓▓▓▓▓▓▓▓▓ │ ~94%     │▒▒           │ ~15%
              │▓▓▓▓▓▓▓▓▓▓▓ │          │             │
              ├─────────────┤          ├─────────────┤
   L_quant    │▒            │ ~5%      │▓▓▓▓▓▓▓▓▓▓  │ ~80%
              ├─────────────┤          ├─────────────┤
   L_commit   │░            │ ~1%      │░░           │ ~5%
              └─────────────┘          └─────────────┘
```

## 訓練策略

### Curriculum Learning

```
noise_ratio
    ▲
0.85├─────────────────────────────────●─────────────
    │                              ╱
    │                           ╱
    │                        ╱
    │                     ╱
    │                  ╱
    │               ╱
0.30├─────────────●
    │
    └─────┬───────┬────────────────┬──────────► epoch
          0      10              200           300
          │◄─ warmup ─►│◄── curriculum ──►│◄── fixed ──►│

每 10 epochs 遞增一次 noise ratio
phase_increment = (0.85 - 0.30) / (200 / 10) = 0.0275
```

### Learning Rate Schedule

```
lr
    ▲
1e-4├──────●
    │     ╱ ╲
    │    ╱    ╲
    │   ╱      ╲
    │  ╱        ╲         cosine annealing
    │ ╱          ╲
    │╱             ╲
    │               ╲
    │                 ╲
1e-6├───────────────────────────────────────●
    └────┬─────┬──────────────────────┬──────► epoch
         0    10                     300
         │◄ warmup ►│◄── cosine decay ──────►│
```

### NaN 保護（V2 新增）

```
每個 batch:
  ┌─────────────┐
  │ compute loss │
  └──────┬──────┘
         │
    ┌────▼────┐
    │ isnan?  │──── Yes ──► optimizer.zero_grad()
    │ isinf?  │            skip batch
    └────┬────┘            nan_count++
         │ No              if nan_count ≥ 10: ⚠️ 警告
         ▼
  ┌─────────────┐
  │  backward() │
  └─────────────┘
```

## 資料流總覽

```
 clean_audio ──► Teacher Encoder ──► t_e ──► Teacher VQ ──► teacher_codes
                      │                 │        (frozen)
                  extract L3,L4,L6      │
                      │                 │
                      ▼                 │
              teacher_intermediates     │
                      │                 │
                  ┌───┴── L_inter ──┐   │
                  │  (cosine, ×0.03) │   │
                  └───┬─────────────┘   │
                      │                 │
              student_intermediates     │
                      │                 │
                  extract L3,L4,L6      │
                      │                 │
 noisy_audio ──► Student Encoder ──► s_e ──► RVQ (4層) ──► z_q
                   (LoRA)               │     (EMA)        │
                                        │                  │
                                        └── L_quant ◄──────┘
                                           (MSE, ×1.0)

                 Output: z_q (student_quantized), all_layer_codes
```
