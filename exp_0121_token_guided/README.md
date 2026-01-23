# Exp 0121: Token-Guided LoRA 訓練

## 實驗目標

### 核心目標
利用 Token 錯誤分析結果來**引導 LoRA 訓練**，在 Exp K 架構基礎上，針對「易錯 token」和「敏感層」進行數據驅動的優化。

### 研究動機
從 Token 轉換分析 (`diagnostic_token_transition_v3.py`) 得知：
1. **Match Rate 差異大**：plastic (14%) < box (20%) < papercup (21%)
2. **某些 token 特別容易錯**：許多 token 的 error_rate > 0.8
3. **泛化性問題**：純 steering (VQ 前/後) 都無法泛化到新樣本

### 假設
如果用 Token 分析的結果來**引導 LoRA 訓練**：
- 針對「易錯 token」設計加權 loss
- 針對「敏感層」集中 LoRA 參數
- 可能在保持泛化性的同時，更有效地修正問題

---

## 規劃

### 實驗設計

#### Exp A: Baseline (標準 Exp K 架構)
- **目的**: 建立基準線，後續實驗與此比較
- **架構**: MSE + Triplet + 中間層監督 (L3, L4, L6)
- **LoRA**: 全層 18 層, rank=256

#### Exp B: Token-Weighted Loss
- **目的**: 驗證對易錯 token 加權是否有效
- **架構**: Exp A + Token-Weighted Loss
- **策略**: 對 error_rate > 0.7 的 token 給 2x 權重

#### Exp C: Layer-Selective LoRA
- **目的**: 驗證只在敏感層加 LoRA 是否更有效率
- **架構**: Exp A 但只在 3 層加 LoRA
- **目標層**: model[4].block.1, model[4].block.3, model[6].conv

### 與之前實驗的差異

| 實驗 | 方法 | 特點 |
|------|------|------|
| exp_0112 | 全層 LoRA + 分區 loss | 沒有針對 token 特性 |
| exp_0112_intermediate (Exp K) | 中間層監督 | 監督層選擇基於經驗 |
| **exp_0121 (本實驗)** | Token 分析引導 LoRA | **數據驅動**的設計 |

---

## 規格

### 模型架構

#### 基礎配置 (沿用 Exp K)
```yaml
Model: WavTokenizer (24kHz, 75fps)
Encoder: SEANet Encoder (18 層 Conv1d)
Codebook: 4096 codes, 512 dim
```

#### LoRA 配置
```yaml
# Exp A/B: 全層 LoRA
layers: 18 (all Conv1d layers)
rank: 256
alpha: 512
dropout: 0.2

# Exp C: Layer-Selective LoRA
layers: 3 (noise-sensitive only)
  - model.4.block.1 (L5)
  - model.4.block.3 (L6)
  - model.6.conv (L8)
rank: 128
alpha: 256
```

### Loss Function

#### 基礎 Loss (Exp A)
```python
Total Loss = λ_feature × MSE + λ_triplet × Triplet + λ_inter × Intermediate

# 權重
λ_feature = 1.0
λ_triplet = 1.0 (margin=0.2)
λ_inter = 0.5

# 中間層監督
L3: weight=0.3 (Downsample 1)
L4: weight=1.0 (ResBlock 2) - 核心
L6: weight=0.5 (Downsample 2)
```

#### Token-Weighted 擴展 (Exp B)
```python
# 從 analyze_error_tokens.py 生成 error_rates
token_error_rates = torch.load("analysis_outputs/token_error_rates.pt")

# 對高錯誤率 token 加權
if error_rate > 0.7:
    weight = 2.0  # high_error_multiplier
else:
    weight = 1.0
```

### 訓練配置
```yaml
# 基礎
epochs: 300
batch_size: 8
learning_rate: 1e-4
min_lr: 1e-6
warmup_epochs: 10
weight_decay: 0.1
grad_clip: 1.0

# Curriculum Learning
start: 30% (最簡單樣本)
end: 85% (排除最困難 15%)
transition: 200 epochs

# AMP
enabled: true
gradient_accumulation: 2
```

### 輸入/輸出規格
```yaml
Input:
  - noisy_audio: (B, T) 或 (B, 1, T), 24kHz
  - clean_audio: (B, T) 或 (B, 1, T), 24kHz

Output:
  - student_encoder_out: (B, 512, T')
  - teacher_encoder_out: (B, 512, T')
  - student_codes: (1, B, T')
  - teacher_codes: (1, B, T')
  - student_intermediates: {3: (B, C, T''), 4: (B, C, T''), 6: (B, C, T'')}
  - teacher_intermediates: {3: (B, C, T''), 4: (B, C, T''), 6: (B, C, T'')}
```

---

## 驗收標準

### MUST (必要條件)
- [ ] Val Accuracy 超過 0.9% (Exp K v5 達到 0.906%)
- [ ] Loss 收斂 (不發散、不震盪)
- [ ] Codebook 不漂移 (drift < 1e-7)
- [ ] 音質不崩潰 (無明顯失真)

### SHOULD (期望達成)
- [ ] Val Accuracy 突破 1.0%
- [ ] Token-Weighted (Exp B) 優於 Baseline (Exp A)
- [ ] 訓練穩定 (無 NaN/Inf)

### COULD (加分項)
- [ ] Layer-Selective (Exp C) 效率更高 (參數更少但效果相當)
- [ ] 識別出最有效的 token 加權策略
- [ ] 產出可重用的 token error analysis pipeline

---

## 檔案結構

```
exp_0121_token_guided/
├── README.md                    # 本文件
├── analyze_error_tokens.py      # Token 錯誤分析腳本
├── models.py                    # Token-Guided 模型定義
├── losses.py                    # Token-Weighted Loss 定義
├── train.py                     # 訓練腳本
├── run_exp_a.sh                 # Exp A: Baseline
├── run_exp_b.sh                 # Exp B: Token-Weighted Loss
├── run_exp_c.sh                 # Exp C: Layer-Selective LoRA
├── analysis_outputs/            # Token 分析輸出
│   ├── token_error_rates.pt     # Token error rate tensor (4096,)
│   ├── error_token_analysis.json
│   └── noise_type_difficulty.json
└── runs/                        # 訓練結果
```

---

## 執行指南

### 1. 前置準備
```bash
# 確保分析輸出存在
cd /home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0121_token_guided
python analyze_error_tokens.py
```

### 2. 執行實驗

#### Exp A: Baseline
```bash
bash run_exp_a.sh
```

#### Exp B: Token-Weighted Loss
```bash
bash run_exp_b.sh
```

#### Exp C: Layer-Selective LoRA
```bash
bash run_exp_c.sh
```

### 3. 監控訓練
```bash
# 查看最新 log
tail -f runs/exp_b_*.log

# 查看 best accuracy
grep "New best" runs/exp_b_*.log
```

---

## AI 執行提示詞

### 完整執行提示詞
```
請執行 exp_0121_token_guided 實驗:

## 步驟

### 1. 環境確認
- 確認 GPU 可用 (CUDA_VISIBLE_DEVICES)
- 確認依賴: WavTokenizer, PEFT, PyTorch

### 2. Token 分析 (如果尚未執行)
```bash
cd /home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0121_token_guided
python analyze_error_tokens.py
```
確認生成 `analysis_outputs/token_error_rates.pt`

### 3. 選擇實驗
- Exp A (Baseline): `bash run_exp_a.sh`
- Exp B (Token-Weighted): `bash run_exp_b.sh`
- Exp C (Layer-Selective): `bash run_exp_c.sh`

### 4. 驗收檢查
- Val Accuracy > 0.9% (MUST)
- Loss 收斂 (MUST)
- 無 NaN/Inf (SHOULD)

## 關鍵配置

### Token-Weighted (Exp B)
- high_error_threshold: 0.7
- high_error_multiplier: 2.0

### Layer-Selective (Exp C)
- 只在 model[4].block.1, model[4].block.3, model[6].conv 加 LoRA
- rank=128 (因為層數少)

## 預期結果
- Exp B 應該優於 Exp A (針對易錯 token 加權)
- Exp C 參數更少但效果應接近
```

### 簡化執行提示詞 (推薦 Exp B)
```
執行 Exp B (Token-Weighted Loss):
1. cd /home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0121_token_guided
2. 確認 analysis_outputs/token_error_rates.pt 存在
3. bash run_exp_b.sh
4. 監控: tail -f runs/exp_b_*.log
5. 驗收: Val Acc > 0.9%, 期望 > 1.0%
```

---

## 實驗結果

### Exp B: Token-Weighted Loss ✅ 完成

**配置**:
- LoRA: rank=256, alpha=512, 18 layers
- Token-Weighted: threshold=0.7, multiplier=2.0
- Curriculum: 30% → 85% over 200 epochs

**結果**:
| 指標 | 值 | 備註 |
|------|-----|------|
| **Best Val Acc** | **1.001%** | **突破 1%!** |
| Best Epoch | 233 | 300 epochs 完成 |
| Final Loss | ~3.07 | 穩定 |
| PESQ (Epoch 300) | 1.093 | 與基準持平 |
| STOI (Epoch 300) | 0.227 | ⚠️ 下降 51% |

**進度記錄**:
```
* New best Val Acc: 0.828%
* New best Val Acc: 0.891%
* New best Val Acc: 0.917%
* New best Val Acc: 0.926%
* New best Val Acc: 0.953%
* New best Val Acc: 0.976%
* New best Val Acc: 1.001%  ← 突破 1%
Best Val Acc: 1.001% @ Epoch 233
```

**驗收狀態**:
- [x] Val Accuracy 超過 0.9% (達到 1.001%)
- [x] Val Accuracy 突破 1.0% (**達成!**)
- [x] Loss 收斂 (穩定在 ~3.07)
- [x] 訓練完成 (300 epochs)
- [x] PESQ 持平 (1.093 vs 基準 1.086)
- ⚠️ STOI 下降 (0.227 vs 基準 0.464)

---

### Exp C: Layer-Selective LoRA ✅ 完成

**配置**:
- LoRA: rank=128, alpha=256, **只有 3 層**
- 目標層: `4.block.1`, `4.block.3`, `6.conv`
- 可訓練參數: 122,880 (0.15%)

**結果**:
| 指標 | 值 | 備註 |
|------|-----|------|
| **Best Val Acc** | **0.413%** | 遠低於預期 |
| Best Epoch | 11 | 早期就停止進步 |
| PESQ (Epoch 300) | 1.106 | 略高於基準 |
| STOI (Epoch 300) | 0.272 | 下降但優於 Exp B |

**分析**:
- ❌ Layer-Selective 策略**失敗**
- 只在 3 層加 LoRA 參數量太少 (122K vs 全層 4.7M)
- 無法有效學習噪音→乾淨的映射
- 早期 (Epoch 11) 就達到最佳，之後無法進步

---

### 音質評估 (PESQ/STOI)

| 實驗 | Epoch | PESQ | STOI | Val Acc |
|------|-------|------|------|---------|
| Exp K v5 (基準) | 300 | 1.086 | **0.464** | 0.899% |
| **Exp B** | 300 | 1.093 | 0.227 | **1.001%** |
| Exp C | 300 | 1.106 | 0.272 | 0.413% |

**Exp B 音質趨勢**:
| Epoch | PESQ | STOI |
|-------|------|------|
| 001 | 1.102 | 0.421 |
| 050 | 1.089 | 0.150 |
| 100 | 1.085 | 0.157 |
| 150 | 1.075 | 0.192 |
| 200 | 1.092 | 0.206 |
| 250 | 1.091 | 0.225 |
| 300 | 1.093 | 0.227 |

⚠️ **發現**: Exp B 的 STOI 在訓練初期急劇下降 (0.421 → 0.150)，之後緩慢回升但仍低於基準。

---

### 實驗比較總結

| 實驗 | Best Val Acc | PESQ | STOI | 結論 |
|------|-------------|------|------|------|
| Exp K v5 | 0.899% | 1.086 | 0.464 | 基準線 |
| **Exp B** | **1.001%** | 1.093 | 0.227 | Val Acc ↑ 但 STOI ↓ |
| Exp C | 0.413% | 1.106 | 0.272 | 失敗 |

### 關鍵發現

1. **Token-Weighted Loss 提升 Val Acc**:
   - 首次突破 1.0% 大關 (+11% 相對提升)
   - 但犧牲了 STOI 音質 (-51%)

2. **PESQ vs STOI 不一致**:
   - PESQ 基本持平 (1.086 → 1.093)
   - STOI 大幅下降 (0.464 → 0.227)
   - 說明：Token 準確率提升不等於感知音質提升

3. **Layer-Selective LoRA 失敗**:
   - 參數量減少 97% 導致學習能力不足
   - 結論：全層 LoRA 是必要的

---

## 後續建議

1. **平衡 Token Acc 和音質**:
   - 嘗試較小的 high_error_multiplier (1.5 instead of 2.0)
   - 或加入 STOI/PESQ 作為輔助 loss

2. **分析 STOI 下降原因**:
   - 是否過度優化特定 token 導致整體音質下降?
   - Token-Weighted 是否破壞了時域連續性?

3. **混合策略**:
   - 結合 Token-Weighted 和標準 MSE
   - 或使用 curriculum 漸進增加 token weight
