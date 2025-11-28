# exp_1128: LoRA Rank + Distance Loss Grid Search

## 實驗背景

基於 `exp_1126/1126-1` 的分析結果：
- Feature MSE 改善 38%，Cosine Similarity 改善 72%
- **但 Code L2 Distance 只改善 18%**
- Token Accuracy 仍然很低 (~5%)

### 假設
1. **LoRA 容量不足** - Rank 16 (~38K params) 可能無法精確控制 feature 輸出方向
2. **Distance Loss 權重太小** - 0.01 權重下，Distance Loss 對總 loss 貢獻不足

---

## 實驗矩陣

| 實驗 | LoRA Rank | 參數量 | Distance Loss | GPU | 啟動腳本 |
|------|-----------|--------|---------------|-----|----------|
| **exp1** | 32 | 77K (2x) | 0.05 (5x) | GPU 0 | `run_exp1_r32_dist0.05.sh` |
| **exp2** | 32 | 77K (2x) | 0.1 (10x) | GPU 0 | `run_exp2_r32_dist0.1.sh` |
| **exp3** | 64 | 154K (4x) | 0.05 (5x) | GPU 1 | `run_exp3_r64_dist0.05.sh` |
| **exp4** | 64 | 154K (4x) | 0.1 (10x) | GPU 1 | `run_exp4_r64_dist0.1.sh` |

### Baseline (exp_1126/1126-1)
- LoRA Rank: 16 (~38K params)
- Distance Loss Weight: 0.01
- Feature Loss Weight: 1.0

---

## 實驗詳細說明

### exp1: lora_r32_dist0.05
- **目的**: 測試增加 LoRA 容量的效果
- **配置**: rank=32, alpha=64, dist_weight=0.05
- **預期**: 更多參數 → 更好的 feature 控制 → 更低的 code distance

### exp2: lora_r32_dist0.1
- **目的**: 在相同 LoRA 容量下，測試更強的 distance loss
- **配置**: rank=32, alpha=64, dist_weight=0.1
- **預期**: 更強的 code alignment 壓力，但可能影響 feature quality

### exp3: lora_r64_dist0.05
- **目的**: 測試大容量 LoRA 的效果
- **配置**: rank=64, alpha=128, dist_weight=0.05
- **預期**: 4x 參數量，應該能更精確地對齊 VQ codes

### exp4: lora_r64_dist0.1
- **目的**: 最激進配置，測試上限
- **配置**: rank=64, alpha=128, dist_weight=0.1
- **風險**: 可能過擬合或訓練不穩定

---

## 啟動方式

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1128

# 方式 1: 並行啟動 (GPU0: exp1, GPU1: exp3)
bash run_all_parallel.sh

# 方式 2: 單獨啟動
bash run_exp1_r32_dist0.05.sh   # GPU 0
bash run_exp2_r32_dist0.1.sh    # GPU 0 (exp1 完成後)
bash run_exp3_r64_dist0.05.sh   # GPU 1
bash run_exp4_r64_dist0.1.sh    # GPU 1 (exp3 完成後)
```

### 監控進度
```bash
tail -f experiments/lora_r32_dist0.05.log
tail -f experiments/lora_r64_dist0.05.log
```

---

## 輸出結構

```
exp_1128/
├── experiments/
│   ├── lora_r32_dist0.05/
│   │   ├── checkpoints/           # 模型 checkpoints
│   │   │   ├── latest.pt
│   │   │   ├── best.pt
│   │   │   └── epoch_*.pt
│   │   ├── audio_samples/         # 音訊樣本 (每 20 epochs)
│   │   │   └── epoch_020/
│   │   │       ├── train_1_noisy.wav        # 原始噪音音頻
│   │   │       ├── train_1_clean.wav        # 目標乾淨音頻
│   │   │       ├── train_1_student_pred.wav # Student 預測
│   │   │       ├── train_1_teacher_recon.wav# Teacher 重建
│   │   │       └── ...
│   │   ├── plots/                 # 訓練曲線圖
│   │   ├── logs/                  # TensorBoard logs
│   │   ├── config.json            # 實驗配置
│   │   └── training_history.json  # 完整訓練歷史
│   └── lora_r32_dist0.05.log      # 訓練 log
```

### 音訊樣本說明

每個 epoch 保存 4 種音訊：
1. **noisy.wav** - 輸入的噪音音頻
2. **clean.wav** - 目標乾淨音頻 (ground truth)
3. **student_pred.wav** - Student 預測: noisy → student encoder → decoder
4. **teacher_recon.wav** - Teacher 重建: clean → teacher encoder → decoder

比較方式：
- `student_pred` vs `clean`: 評估去噪效果
- `student_pred` vs `teacher_recon`: 評估 Student 是否學到 Teacher 的表示

---

## 評估指標

### 主要指標
| 指標 | Baseline (1126-1) | 目標 |
|------|-------------------|------|
| Code L2 Distance | 4.40 | < 3.5 |
| Feature Cosine | 0.59 | > 0.7 |
| Top-5 Accuracy | 4.96% | > 10% |
| Feature MSE | 0.047 | < 0.03 |

### 評估腳本
```bash
# 使用 1126-1 的評估腳本 (複製過來使用)
python evaluate_audio_quality.py
```

---

## 文件說明

| 文件 | 說明 |
|------|------|
| `config.py` | 實驗配置 (LoRA, Loss weights 等) |
| `model.py` | Teacher-Student 模型定義 |
| `train.py` | 訓練邏輯 (含 audio sample 儲存) |
| `losses.py` | Loss 函數 (Feature + Distance) |
| `data.py` | 數據載入 |
| `wavtok_lora_patch.py` | LoRA 兼容性補丁 |
| `wavtok_distance_mat_corrected.pt` | VQ Codebook 距離矩陣 |

---

## 關鍵技術細節

### VQ Codebook 凍結 (v3 方案)
```python
# 問題: VQ 的 EMA 更新和 STE 梯度傳遞都依賴 self.training flag
# 解法: 保持 training=True，每次 forward 後恢復 codebook

frozen_codebook = codebook_ref.embed.data.clone()
# ... forward pass ...
codebook_ref.embed.data.copy_(frozen_codebook)  # 恢復
```

### LoRA Target Modules
```python
lora_target_modules = [
    "feature_extractor.encodec.encoder.model.0.conv.conv",  # 第一層 strided conv
    "feature_extractor.encodec.encoder.model.3.conv.conv",
    "feature_extractor.encodec.encoder.model.6.conv.conv",
    "feature_extractor.encodec.encoder.model.9.conv.conv",
]
```

---

## 參考

- 前置實驗: `exp_1126/1126-1/`
- VQ Distance 分析: `exp_1126/1126-1/evaluate_audio_quality.py`
- Codebook 凍結驗證: `exp_1126/1126-1/verify_codebook_frozen.py`
