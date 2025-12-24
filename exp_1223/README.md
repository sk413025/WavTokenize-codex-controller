# exp_1223: 短期改進實驗

基於 Exp55 (目前最佳配置) 的改進實驗。

## 實驗概覽

| 實驗 | 改進內容 | 配置 |
|------|----------|------|
| **Exp59** | LR + Grad Accum 改進 | lr=5e-5, accum=4, epochs=300 |
| **Exp60** | Speaker Conditioning (FiLM) | + speaker embedding |
| **Exp60-b** | Speaker Conditioning (Cross-Attn) | 對照實驗 |

## Exp59: 基於 Exp55 改進

### 動機

Exp55 存在過擬合問題：
- Val Loss 在 Epoch 56 達到最低，之後上升 4.49%
- Val Accuracy 在 Epoch 173 才達到最高 (0.91%)

### 改進策略

| 參數 | Exp55 | Exp59 | 說明 |
|------|-------|-------|------|
| Learning Rate | 1e-4 | **5e-5** | 更小的學習率減緩過擬合 |
| Grad Accum | 2 | **4** | 更大的等效 batch (8×4=32) |
| Epochs | 200 | **300** | 延長訓練觀察收斂 |
| Warmup | 10 | **15** | 更長的 warmup |

### 運行

```bash
bash exp_1223/run_exp59_improved.sh
```

---

## Exp60: Speaker Conditioning

### 動機

- 數據包含 14 個訓練 speakers, 3 個驗證 speakers
- 每個樣本有 256 維 speaker embedding (ECAPA-TDNN)
- 不同 speaker 可能有不同的 token 分佈特徵
- 參考 c_code/exp3-1 的 Cross-Attention + Gate 設計

### 設計

兩種 Speaker Conditioning 方式：

#### FiLM (Feature-wise Linear Modulation)

```
y = γ * x + β
```

- γ, β 由 speaker embedding 生成
- 輕量，不增加太多參數
- 與 LoRA 的低秩適應概念一致

#### Cross-Attention

```
attn_output = CrossAttn(features, speaker)
output = α * features + (1-α) * attn_output
```

- Token features (Q) attend to Speaker embedding (K, V)
- Learnable gate 動態融合
- 參考 c_code/exp3-1

### 運行

```bash
# FiLM 方法
bash exp_1223/run_exp60_speaker_film.sh

# Cross-Attention 方法 (對照)
bash exp_1223/run_exp60_speaker_crossattn.sh
```

---

## 資料處理

所有實驗都會過濾 clean→clean 樣本：

| 實驗 | Data Loader | filter_clean_to_clean |
|------|-------------|----------------------|
| Exp59 | `exp_1212/data_aligned.py` | ✅ True |
| Exp60 | `exp_1223/data_speaker.py` | ✅ True |

---

## 順序執行

```bash
# 執行 Exp59 → Exp60 (FiLM)
bash exp_1223/run_all_sequential.sh

# 或使用 nohup 後台執行
nohup bash exp_1223/run_all_sequential.sh > exp_1223/run_all.log 2>&1 &
```

---

## 檔案結構

```
exp_1223/
├── README.md                       # 本文件
├── data_speaker.py                 # Speaker-aware DataLoader
├── models_speaker.py               # Speaker-Conditioned Model (FiLM + CrossAttn)
├── train_speaker.py                # Speaker-Conditioned 訓練腳本
├── run_exp59_improved.sh           # Exp59 運行腳本
├── run_exp60_speaker_film.sh       # Exp60 (FiLM) 運行腳本
├── run_exp60_speaker_crossattn.sh  # Exp60-b (CrossAttn) 運行腳本
├── run_all_sequential.sh           # 順序執行腳本
└── runs/                           # 實驗結果目錄
    ├── exp59_improved/
    ├── exp60_speaker_film/
    └── exp60b_speaker_crossattn/
```

---

## 預期結果

### Exp59

- 減緩過擬合，Val Loss 和 Val Acc 的最佳 epoch 更接近
- 可能達到更高的 Val Accuracy (目標 > 0.91%)

### Exp60

- Speaker conditioning 幫助模型適應不同說話者
- Zero-shot Val speakers (boy7, girl9, boy8) 的表現改善
- FiLM vs Cross-Attention 的效果比較

---

## 相關實驗

| 實驗 | 說明 | Val Acc |
|------|------|---------|
| Exp48 | Baseline | 0.88% |
| Exp55 | 高 rank LoRA + Grad Accum | **0.91%** |
| Exp59 | Exp55 + 小 LR + 大 Accum | TBD |
| Exp60 | Exp55 + Speaker Conditioning | TBD |
