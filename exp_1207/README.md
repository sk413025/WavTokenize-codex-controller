# exp_1207: 回歸最簡單 - 純 Feature Loss

## 實驗動機

之前的實驗使用了複雜的 Loss 組合：
- Feature Loss + Distance Loss + VQ Loss
- 各種 Distance Loss 變體（STE, Gumbel, CE, Margin）

結果：**Token Accuracy 反而下降**

## 核心假設

如果 Feature Loss 能讓 `z_noisy ≈ z_clean`，那麼 `argmin(z_noisy)` 應該等於 `argmin(z_clean)`。

也就是說：**只用 Feature Loss 就應該足夠**。

## 架構圖

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WavTokenizer 架構                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Audio Input                                                                │
│       │                                                                      │
│       ▼                                                                      │
│   ┌─────────────────────────────────────────────────────────┐               │
│   │                    Encoder (CNN)                         │               │
│   │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐  │               │
│   │  │ Conv1d  │ → │ Conv1d  │ → │ Conv1d  │ → │ Conv1d  │  │               │
│   │  │ +LoRA   │   │ +LoRA   │   │ +LoRA   │   │ +LoRA   │  │               │
│   │  └─────────┘   └─────────┘   └─────────┘   └─────────┘  │               │
│   └─────────────────────────────────────────────────────────┘               │
│       │                                                                      │
│       ▼                                                                      │
│   z (feature embedding)  ─────────────────────────────┐                      │
│   shape: (B, 512, T)                                   │                     │
│       │                                                │                     │
│       ▼                                                ▼                     │
│   ┌─────────────────┐                         ┌─────────────────┐           │
│   │   VQ Layer      │                         │  Feature Loss   │           │
│   │                 │                         │                 │           │
│   │  argmin(z, C)   │                         │  MSE(z_s, z_t)  │           │
│   │       │         │                         │       ▲         │           │
│   │       ▼         │                         └───────┼─────────┘           │
│   │  token index    │                                 │                     │
│   │  shape: (B, T)  │                                 │                     │
│   └─────────────────┘                                 │                     │
│       │                                               │                     │
│       ▼                                               │                     │
│   Decoder → Reconstructed Audio                       │                     │
│                                                       │                     │
└───────────────────────────────────────────────────────┼─────────────────────┘
                                                        │
                                                        │
┌───────────────────────────────────────────────────────┼─────────────────────┐
│                    Teacher-Student 架構               │                     │
├───────────────────────────────────────────────────────┼─────────────────────┤
│                                                       │                     │
│   ┌─────────────────────────────┐    ┌───────────────┴───────────────┐     │
│   │       Teacher (凍結)         │    │       Student (LoRA)          │     │
│   ├─────────────────────────────┤    ├───────────────────────────────┤     │
│   │                             │    │                               │     │
│   │  clean_audio                │    │  noisy_audio                  │     │
│   │       │                     │    │       │                       │     │
│   │       ▼                     │    │       ▼                       │     │
│   │  ┌─────────┐                │    │  ┌─────────┐                  │     │
│   │  │ Encoder │ (frozen)       │    │  │ Encoder │ + LoRA           │     │
│   │  └─────────┘                │    │  └─────────┘                  │     │
│   │       │                     │    │       │                       │     │
│   │       ▼                     │    │       ▼                       │     │
│   │    z_clean ─────────────────┼────┼──► z_noisy                    │     │
│   │       │                     │    │       │                       │     │
│   │       ▼            Feature Loss = MSE(z_noisy, z_clean)          │     │
│   │  ┌─────────┐                │    │  ┌─────────┐                  │     │
│   │  │   VQ    │                │    │  │   VQ    │ (frozen)         │     │
│   │  └─────────┘                │    │  └─────────┘                  │     │
│   │       │                     │    │       │                       │     │
│   │       ▼                     │    │       ▼                       │     │
│   │  token_clean ───────────────┼────┼──► token_noisy                │     │
│   │                             │    │                               │     │
│   │  (Ground Truth)             │    │  目標: token_noisy == token_clean  │
│   │                             │    │                               │     │
│   └─────────────────────────────┘    └───────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Loss 定義

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Loss Functions                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Feature Loss (參與訓練) ✓                                               │
│     ────────────────────────                                                │
│     L_feature = MSE(z_noisy, z_clean)                                       │
│                                                                              │
│     目標：讓 Student encoder 輸出接近 Teacher encoder 輸出                  │
│                                                                              │
│  2. Distance Loss (僅監控) ○                                                │
│     ────────────────────────                                                │
│     L_distance = distance_matrix[token_noisy, token_clean]                  │
│                                                                              │
│     目標：監控 VQ 後的 token 在 codebook 中的距離                           │
│     說明：不參與訓練，只用來觀察 Feature Loss 對 token 選擇的影響          │
│                                                                              │
│  3. VQ Loss (僅監控) ○                                                      │
│     ────────────────────                                                    │
│     L_vq = ||z_student - sg(codebook[argmin(z_student)])||²                 │
│                                                                              │
│     目標：監控 encoder 輸出離「自己選中的 code」有多遠                      │
│     說明：這是 Student 自己選的 code，不是 Teacher 的                       │
│                                                                              │
│  4. Token Accuracy (監控指標)                                               │
│     ──────────────────────────                                              │
│     Acc = mean(token_noisy == token_clean)                                  │
│                                                                              │
│     目標：最終評估指標                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 實驗設計

| 實驗 | Loss 配置 | 說明 |
|------|----------|------|
| exp1 | feature_only | 只用 Feature MSE Loss |
| exp2 | feature_only_small_lr | 降低學習率，避免破壞原始能力 |
| exp3 | feature_only_small_lora | 降低 LoRA rank，更保守的微調 |

## 預期結果

1. 如果 Token Accuracy 提升：證明 Feature Loss 足夠
2. 如果 Token Accuracy 不變/下降：說明問題在 MSE 優化方向，需要其他策略

## Baseline 參考

原始 WavTokenizer（無任何修改）的 Token Match Rate：
- 訓練集：39.91%
- 驗證集：3.70%

exp_1201 Epoch 1 Token Accuracy：22-32%（比原始模型還低！）

## 輸出

每個實驗會產生：
- `training_history.json`: 訓練歷史數據
- `training_curves_epoch_XXX.png`: 6 個子圖的訓練曲線
- `audio_samples/epoch_XXX/`: 音檔樣本
- `best.pt`: 最佳模型 checkpoint
- `epoch_XXX.pt`: 定期保存的 checkpoint

## 檔案結構

```
exp_1207/
├── README.md               # 本文件
├── train.py                # 訓練腳本
├── run_experiments.sh      # 執行腳本
└── experiments/            # 實驗結果
    ├── feature_only/
    │   ├── config.json
    │   ├── training_history.json
    │   ├── training_curves_epoch_030.png
    │   ├── audio_samples/
    │   ├── best.pt
    │   └── epoch_*.pt
    ├── feature_only_small_lr/
    └── feature_only_small_lora/
```
