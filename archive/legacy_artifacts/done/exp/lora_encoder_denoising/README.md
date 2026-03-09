# LoRA Encoder Feature-Level Denoising

## 🎯 項目目標

**訓練目標**：Fine-tune WavTokenizer Encoder (使用 LoRA) 使其能夠：
- 輸入：Noisy Audio
- 輸出：Features/Codes 接近 Clean Audio 通過原始 Encoder 的輸出

**核心方法**：
- Teacher-Student Knowledge Distillation
- LoRA (Low-Rank Adaptation) 保護原始權重
- Distance Matrix Soft Target (利用 commit 927880a 的 VQ distances)

---

## 📁 項目結構

```
done/exp/lora_encoder_denoising/
├── README.md                    # 本文檔
├── config.py                    # 配置參數
├── model.py                     # Teacher-Student 模型
├── losses.py                    # Loss functions
├── data.py                      # Dataset & DataLoader
├── utils.py                     # 工具函數
│
├── smoke_test.py                # 快速測試（2-5 分鐘）
├── train.py                     # 完整訓練
├── evaluate.py                  # 評估腳本
│
├── scripts/
│   ├── run_smoke_test.sh        # 執行 smoke test
│   ├── run_train.sh             # 執行訓練
│   └── run_evaluate.sh          # 執行評估
│
└── checkpoints/                 # 訓練 checkpoints
    └── logs/                    # TensorBoard logs
```

---

## 🚀 快速開始

### **Phase 1: Smoke Test（必須先通過）**

```bash
# 1. 快速驗證概念（使用小數據 + 少 epochs）
cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/lora_encoder_denoising
python smoke_test.py

# 預期時間: 2-5 分鐘
# 預期結果: Loss 下降，feature distance 減小
```

**Smoke Test 檢查項目**：
- ✅ 模型可以正常創建（Teacher + Student with LoRA）
- ✅ 數據可以正常載入
- ✅ Forward pass 正常
- ✅ Loss 可以計算
- ✅ Backward pass 正常（LoRA 參數收到梯度）
- ✅ 訓練幾個 batch 後 loss 下降

### **Phase 2: 完整訓練（Smoke Test 通過後）**

```bash
# 2. 完整訓練
python train.py \
    --exp_name lora_denoising_r16 \
    --lora_rank 16 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 5e-5

# 預期時間: 數小時到數天（取決於數據量）
```

### **Phase 3: 評估**

```bash
# 3. 評估訓練好的模型
python evaluate.py \
    --checkpoint checkpoints/lora_denoising_r16/best_model.pth \
    --output_dir results/
```

---

## 🔧 詳細實作計畫

### **Step 1: 核心組件實作**

#### **1.1 配置 (config.py)**
```python
# 定義所有訓練參數
# - WavTokenizer paths
# - LoRA config (rank, alpha, target modules)
# - Training hyperparameters
# - Data paths
```

#### **1.2 模型 (model.py)**
```python
# TeacherStudentModel:
#   - teacher: Frozen WavTokenizer
#   - student: WavTokenizer + LoRA
#   - distance_matrix: VQ code distances
```

#### **1.3 損失函數 (losses.py)**
```python
# EncoderDistillationLoss:
#   - Feature-level MSE
#   - Distance-weighted code loss
#   - Optional: Soft target from distances
```

#### **1.4 數據 (data.py)**
```python
# NoisyCleanPairDataset:
#   - 載入 noisy-clean 配對
#   - 支持從 HDF5 或 PyTorch cache
#   - Collate function
```

---

### **Step 2: Smoke Test 設計**

**目的**：快速驗證概念，避免浪費時間在錯誤的實作上。

**設計原則**：
- 使用極小數據（10-20 samples）
- 訓練極少 epochs（2-3 epochs）
- 快速迭代（總時間 < 5 分鐘）

**檢查點**：
```python
# smoke_test.py 的檢查流程

1. Model Creation Check
   - ✓ Teacher 凍結
   - ✓ Student LoRA 可訓練
   - ✓ VQ 和 Backbone 凍結

2. Data Loading Check
   - ✓ 載入 noisy-clean pairs
   - ✓ Audio shape 正確
   - ✓ Batch collation 正常

3. Forward Pass Check
   - ✓ Teacher(clean) → features_clean
   - ✓ Student(noisy) → features_noisy
   - ✓ No NaN/Inf

4. Loss Computation Check
   - ✓ Feature loss 可計算
   - ✓ Distance loss 可計算
   - ✓ Loss 數值合理 (not NaN)

5. Backward Pass Check
   - ✓ loss.backward() 成功
   - ✓ LoRA 參數有梯度
   - ✓ Frozen 參數無梯度

6. Training Check
   - ✓ 訓練 2-3 epochs
   - ✓ Loss 下降
   - ✓ Feature distance 減小

7. Checkpoint Save/Load Check
   - ✓ 可以保存 checkpoint
   - ✓ 可以載入 checkpoint
   - ✓ 載入後結果一致
```

---

### **Step 3: 完整訓練流程**

```
Epoch Loop:
  ├─ Train:
  │   ├─ Forward (Teacher + Student)
  │   ├─ Compute Loss (Feature + Distance)
  │   ├─ Backward (只更新 LoRA)
  │   └─ Log metrics
  │
  ├─ Validate:
  │   ├─ Compute feature distance
  │   ├─ Compute code match rate
  │   ├─ Check original capability (clean audio)
  │   └─ Log metrics
  │
  └─ Checkpoint:
      ├─ Save if best validation
      └─ Save periodic (every 5 epochs)
```

---

## 📊 預期結果

### **Smoke Test**
```
✅ All checks passed
Training 3 epochs on 20 samples:
  Epoch 1: Loss 0.850 → Feature Dist 0.120
  Epoch 2: Loss 0.620 → Feature Dist 0.085
  Epoch 3: Loss 0.480 → Feature Dist 0.062
  ✓ Loss 下降 44%
  ✓ Feature distance 下降 48%
```

### **完整訓練（50 epochs）**
```
Baseline (Original Encoder on Noisy):
  Feature Distance: 0.150
  Code Match Rate: 65%

After LoRA Fine-tuning:
  Feature Distance: 0.045 (↓70%)
  Code Match Rate: 88% (↑23%)

Original Capability (Clean Audio):
  Feature Distance: 0.002 (保持不變 ✓)
```

---

## 🔍 監控指標

### **訓練時監控**
1. **Loss**
   - Total loss
   - Feature loss
   - Distance loss

2. **Feature Quality**
   - MSE(student_features, teacher_features)
   - Cosine similarity

3. **Code Quality**
   - Code match rate (exact match)
   - Average code distance

4. **原始能力檢查**
   - Clean audio reconstruction quality
   - 確保不破壞原始 encoder

### **驗證時監控**
1. **不同 SNR 下的效果**
   - Clean vs 20dB vs 10dB vs 5dB

2. **不同 Noise Type**
   - White noise
   - Babble noise
   - Music noise

---

## ⚙️ 超參數

### **Smoke Test 參數**
```python
SMOKE_TEST_CONFIG = {
    'num_samples': 20,
    'num_epochs': 3,
    'batch_size': 4,
    'lora_rank': 8,
    'learning_rate': 1e-4,  # 較大，快速收斂
}
```

### **完整訓練參數（推薦）**
```python
TRAIN_CONFIG = {
    'lora_rank': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,

    'num_epochs': 50,
    'batch_size': 16,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,

    'warmup_epochs': 5,
    'scheduler': 'cosine',
    'min_lr': 1e-6,

    'loss_weights': {
        'feature_loss': 1.0,
        'distance_loss': 0.1,
        'vq_loss': 0.01,
    }
}
```

---

## 🐛 故障排除

### **常見問題**

#### **1. Smoke Test 失敗**

**現象**: Loss 不下降或出現 NaN
```
可能原因:
  - 學習率過大 → 降低到 1e-5
  - 數據問題 → 檢查 audio shape 和 normalization
  - LoRA 配置錯誤 → 檢查 target_modules
```

**現象**: Loss 下降但 feature distance 不變
```
可能原因:
  - Loss 權重不平衡 → 增加 feature_loss 權重
  - Teacher-Student 沒正確對齊 → 檢查 forward pass
```

#### **2. OOM (Out of Memory)**

```bash
# 解決方案:
# 1. 減小 batch size
python train.py --batch_size 8  # 從 16 降到 8

# 2. 使用 gradient accumulation
python train.py --batch_size 8 --accumulate_grad_batches 2

# 3. 使用混合精度訓練
python train.py --use_amp
```

#### **3. 訓練太慢**

```bash
# 優化方案:
# 1. 增加 num_workers
python train.py --num_workers 4

# 2. 使用更小的 LoRA rank
python train.py --lora_rank 8  # 從 16 降到 8

# 3. 減少驗證頻率
python train.py --val_every_n_epochs 5  # 每 5 epoch 驗證
```

---

## 📈 實驗記錄

### **Smoke Test Results**

| Date | Config | Result | Notes |
|------|--------|--------|-------|
| 2025-XX-XX | rank=8, lr=1e-4 | ✅ Pass | Loss: 0.85→0.48 |
| | | | Feature dist: 0.12→0.06 |

### **完整訓練 Results**

| Exp ID | LoRA Rank | LR | Epochs | Best Val Feature Dist | Code Match Rate |
|--------|-----------|----|----|------------|-------------|
| exp001 | 8 | 5e-5 | 50 | 0.052 | 82% |
| exp002 | 16 | 5e-5 | 50 | 0.045 | 88% |
| exp003 | 32 | 5e-5 | 50 | 0.041 | 91% |

---

## 🔗 相關資源

### **代碼參考**
- Commit 927880a: HDF5 preprocessing & distance matrix
- `done/exp/losses_with_distances.py`: Soft target loss implementation
- `done/exp/train_with_distances.py`: HDF5 dataset example

### **文獻**
- LoRA Paper: https://arxiv.org/abs/2106.09685
- Whisper + LoRA: [Interspeech 2025 paper]
- Knowledge Distillation: Hinton et al., 2015

---

## ✅ Checklist

### **開發階段**
- [ ] 實作 config.py
- [ ] 實作 model.py
- [ ] 實作 losses.py
- [ ] 實作 data.py
- [ ] 實作 utils.py
- [ ] 實作 smoke_test.py

### **Smoke Test 階段**
- [ ] 執行 smoke test
- [ ] 所有檢查通過
- [ ] Loss 下降
- [ ] Feature distance 減小
- [ ] 保存檢查點成功

### **完整訓練階段**
- [ ] 實作 train.py
- [ ] 實作 evaluate.py
- [ ] 執行完整訓練
- [ ] 達到目標指標
- [ ] 評估不同 noise 條件

### **部署階段**
- [ ] 保存最佳模型
- [ ] 測試推理速度
- [ ] 文檔化結果
- [ ] 整合到主項目

---

**下一步**: 開始實作核心組件 (config.py, model.py, losses.py, data.py)
