# exp_1207: Feature Loss 與 Cross-Entropy Loss 實驗

## 實驗動機

之前的實驗使用了複雜的 Loss 組合：
- Feature Loss + Distance Loss + VQ Loss
- 各種 Distance Loss 變體（STE, Gumbel, CE, Margin）

結果：**Token Accuracy 反而下降**

## 核心問題診斷（來自 exp_1204）

```
診斷結果：
- mean_correct_distance: 3.75   ← 到正確 token 的距離
- mean_min_distance: 0.45       ← 到最近（錯誤）token 的距離
- accuracy: 2.2%

問題：Student 特徵離「最近的錯誤 token」比「正確 token」更近！
```

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
│   z (feature embedding)                                                      │
│   shape: (B, 512, T)                                                         │
│       │                                                                      │
│       ├──────────────────────┬──────────────────────┐                       │
│       │                      │                      │                       │
│       ▼                      ▼                      ▼                       │
│   ┌─────────┐         ┌─────────────┐        ┌─────────────┐               │
│   │   VQ    │         │Feature Loss │        │   CE Loss   │               │
│   │argmin() │         │MSE(z_s,z_t) │        │ -log P(c_t) │               │
│   └─────────┘         └─────────────┘        └─────────────┘               │
│       │                                                                      │
│       ▼                                                                      │
│   token index                                                                │
│   shape: (B, T)                                                              │
│       │                                                                      │
│       ▼                                                                      │
│   Decoder → Reconstructed Audio                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Loss 定義與計算差異

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Distance Loss vs Cross-Entropy Loss                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  Distance Loss (僅監控，不參與訓練)                                    ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                       ║  │
│  ║  計算方式：                                                            ║  │
│  ║  ──────────                                                           ║  │
│  ║  1. Student encoder 輸出 z_student                                    ║  │
│  ║  2. VQ 層選擇最近的 codebook entry: token_student = argmin||z - c||   ║  │
│  ║  3. 查表：distance = distance_matrix[token_student, token_teacher]    ║  │
│  ║                                                                       ║  │
│  ║  特點：                                                                ║  │
│  ║  ──────                                                               ║  │
│  ║  - 只看「選中的 token」之間的距離                                      ║  │
│  ║  - 是 VQ 之後的結果，無法反向傳播（argmin 不可微）                     ║  │
│  ║  - 用途：監控 token 選擇是否正確                                       ║  │
│  ║                                                                       ║  │
│  ║  示意圖：                                                              ║  │
│  ║  ┌─────────────────────────────────────────┐                          ║  │
│  ║  │  Codebook (4096 entries)                │                          ║  │
│  ║  │                                         │                          ║  │
│  ║  │     c_0   c_1   c_2   ...   c_4095      │                          ║  │
│  ║  │                                         │                          ║  │
│  ║  │  token_student = 42  ──┐                │                          ║  │
│  ║  │  token_teacher = 100 ──┼──► distance_matrix[42, 100] = 3.75        ║  │
│  ║  │                        │                │                          ║  │
│  ║  └────────────────────────┘────────────────┘                          ║  │
│  ║                                                                       ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  Cross-Entropy Loss (參與訓練) ✓                                       ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                       ║  │
│  ║  計算方式：                                                            ║  │
│  ║  ──────────                                                           ║  │
│  ║  1. Student encoder 輸出 z_student (VQ 之前！)                        ║  │
│  ║  2. 計算 z 到所有 codebook entries 的距離：                            ║  │
│  ║     logits[i] = -||z_student - codebook[i]||²                         ║  │
│  ║  3. Softmax 得到機率分佈：P(c_i) = softmax(logits)                    ║  │
│  ║  4. CE Loss = -log P(token_teacher)                                   ║  │
│  ║                                                                       ║  │
│  ║  特點：                                                                ║  │
│  ║  ──────                                                               ║  │
│  ║  - 看「z 到所有 codebook entries 的距離」                              ║  │
│  ║  - 是 VQ 之前的特徵，可以反向傳播！                                    ║  │
│  ║  - 提供強梯度，強迫 z 跨過 Voronoi 邊界                                ║  │
│  ║                                                                       ║  │
│  ║  示意圖：                                                              ║  │
│  ║  ┌─────────────────────────────────────────┐                          ║  │
│  ║  │  Codebook (4096 entries)                │                          ║  │
│  ║  │                                         │                          ║  │
│  ║  │     c_0   c_1   c_2   ...   c_4095      │                          ║  │
│  ║  │      │     │     │           │          │                          ║  │
│  ║  │      ▼     ▼     ▼           ▼          │                          ║  │
│  ║  │   ┌─────────────────────────────┐       │                          ║  │
│  ║  │   │ z_student (VQ 前的特徵)     │       │                          ║  │
│  ║  │   │                             │       │                          ║  │
│  ║  │   │  計算到每個 c_i 的距離      │       │                          ║  │
│  ║  │   │  logits = [-d_0, -d_1, ...] │       │                          ║  │
│  ║  │   │                             │       │                          ║  │
│  ║  │   │  P = softmax(logits)        │       │                          ║  │
│  ║  │   │                             │       │                          ║  │
│  ║  │   │  CE = -log P[token_teacher] │       │                          ║  │
│  ║  │   └─────────────────────────────┘       │                          ║  │
│  ║  └─────────────────────────────────────────┘                          ║  │
│  ║                                                                       ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 關鍵差異總結

| 項目 | Distance Loss | Cross-Entropy Loss |
|------|--------------|-------------------|
| **輸入** | VQ 後的 token indices | VQ 前的連續特徵 z |
| **計算** | 查表 `dist[token_s, token_t]` | 計算 z 到所有 codebook 的距離 |
| **可微** | ❌ 不可微（argmin） | ✅ 可微 |
| **梯度** | 無 | 有強梯度 |
| **作用** | 監控 | 訓練 |
| **目標** | 觀察 token 距離 | 強迫 z 選擇正確的 token |

## 為什麼需要 CE Loss？

```
問題示意圖：

                    Voronoi Cell 邊界
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    │   c_correct (正確)  │   c_wrong (錯誤)    │
    │        ●            │        ●            │
    │                     │                     │
    │                     │    ★ z_student      │
    │                     │    (在錯誤區域!)     │
    │                     │                     │
    └─────────────────────┼─────────────────────┘

    診斷數據：
    - 到 c_correct 的距離: 3.75
    - 到 c_wrong 的距離: 0.45

    MSE Loss 的問題：
    - MSE 讓 z_student 「靠近」z_teacher
    - 但不保證跨過 Voronoi 邊界！

    CE Loss 的解決方案：
    - CE Loss = -log P(c_correct)
    - 當 z 在錯誤區域時，P(c_correct) 很小，CE 很大
    - 強梯度推動 z 跨過邊界，進入正確區域
```

## 實驗設計

### 第一階段：純 Feature Loss（驗證基礎假設）

| 實驗 | Loss 配置 | 說明 |
|------|----------|------|
| feature_only | Feature MSE | 只用 Feature MSE Loss |
| feature_only_small_lr | Feature MSE | 降低學習率 (1e-5) |
| feature_only_small_lora | Feature MSE | 降低 LoRA rank (16) |

### 第二階段：Feature + CE Loss（exp16）

| 實驗 | Loss 配置 | 說明 |
|------|----------|------|
| exp16a | λ_feature=1.0, λ_ce=1.0 | 平衡配置 |
| exp16b | λ_feature=0.1, λ_ce=1.0 | CE 主導 |
| exp16c | λ_feature=0.0, λ_ce=1.0 | 純 CE Loss |

## Loss 公式總結

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
│     作用：保持特徵空間結構穩定                                               │
│                                                                              │
│  2. Cross-Entropy Loss (參與訓練) ✓  【新增】                               │
│     ──────────────────────────────────                                      │
│     logits = -||z_student - codebook||²     (負距離作為 logits)             │
│     L_ce = CrossEntropy(logits, token_teacher)                              │
│                                                                              │
│     目標：強迫 z_student 跨過 Voronoi 邊界，選擇正確的 token                │
│     作用：提供強分類梯度，解決「靠近但不正確」的問題                         │
│                                                                              │
│  3. Distance Loss (僅監控) ○                                                │
│     ────────────────────────                                                │
│     L_distance = distance_matrix[token_noisy, token_clean]                  │
│                                                                              │
│     目標：監控 VQ 後的 token 在 codebook 中的距離                           │
│     說明：不參與訓練，只用來觀察效果                                         │
│                                                                              │
│  4. Total Loss (exp16)                                                      │
│     ──────────────────────                                                  │
│     L_total = λ_feature × L_feature + λ_ce × L_ce                           │
│                                                                              │
│  5. Token Accuracy (監控指標)                                               │
│     ──────────────────────────                                              │
│     Acc = mean(token_noisy == token_clean)                                  │
│                                                                              │
│     目標：最終評估指標                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Baseline 參考

原始 WavTokenizer（無任何修改）的 Token Match Rate：
- 訓練集：39.91%
- 驗證集：3.70%

exp_1201 Epoch 1 Token Accuracy：22-32%（比原始模型還低！）

## 輸出

每個實驗會產生：
- `training_history.json`: 訓練歷史數據
- `training_curves_epoch_XXX.png`: 訓練曲線
- `audio_samples/epoch_XXX/`: 音檔樣本
- `spectrograms/epoch_XXX/`: 頻譜圖比較
- `best.pt`: 最佳模型 checkpoint
- `epoch_XXX.pt`: 定期保存的 checkpoint

## 檔案結構

```
exp_1207/
├── README.md                    # 本文件
├── train.py                     # 純 Feature Loss 訓練腳本
├── train_with_ce.py             # Feature + CE Loss 訓練腳本
├── run_experiments.sh           # 第一階段實驗
├── run_exp16_feature_ce.sh      # 第二階段實驗 (exp16)
└── experiments/                 # 實驗結果
    ├── feature_only/
    ├── feature_only_small_lr/
    ├── feature_only_small_lora/
    ├── exp16a_feature_ce_equal/
    ├── exp16b_feature_ce_dominant/
    └── exp16c_pure_ce/
```

## 執行方式

```bash
# 第一階段：純 Feature Loss
cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1207
bash run_experiments.sh

# 第二階段：Feature + CE Loss
bash run_exp16_feature_ce.sh
```
