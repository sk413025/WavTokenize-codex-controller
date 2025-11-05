# Cross-Attention Speaker Fusion 實驗說明

**實驗編號**: EXP-20251105-CrossAttn  
**創建時間**: 2025-11-05  
**狀態**: 準備開始

---

## 一、實驗目的

驗證 **假設 2**: Speaker Embedding 影響力不足是導致訓練平台期的主要原因。

### 問題背景

在原始 Additive Fusion 實驗中：
- Train Accuracy 停滯在 56%
- Val Accuracy 停滯在 38%
- **預期問題**: Speaker Embedding 通過簡單相加被 Token Embedding 覆蓋，影響力 <5%

---

## 二、改進方案

### 關鍵改變

| 項目 | 原始 (Additive) | 新方案 (Cross-Attention) |
|------|----------------|------------------------|
| **Fusion 方式** | `token_emb + speaker_emb` | Cross-Attention Layer |
| **Speaker 影響** | 全局相同 (broadcast) | 動態調整 (每個 token 不同) |
| **可解釋性** | 無 | 可視覺化 attention weights |
| **參數量** | 21M | 22M (+1M, +5%) |
| **Batch Size** | 8 | **64 (+700%)** |
| **Weight Decay** | 0.0 | **0.0 (確認無)** |
| **Scheduler** | ReduceLROnPlateau | **None (確認無)** |
| **Learning Rate** | 1e-4 | **1e-4 (固定)** |

### 架構對比

**Before (Additive)**:
```
Token Emb (B,T,512) + Speaker Emb (B,T,512) → Combined (B,T,512)
```

**After (Cross-Attention)**:
```
Token Emb (B,T,512) --[Query]--\
                                 Cross-Attention → Output (B,T,512)
Speaker Emb (B,512) --[K & V]--/

每個 token 動態決定需要多少 speaker 資訊
```

---

## 三、訓練配置

### 模型參數
```yaml
d_model: 512
nhead: 8
num_layers: 4
dim_feedforward: 2048
dropout: 0.1
speaker_dim: 256
```

### 訓練參數
```yaml
batch_size: 64        # 大幅提升 (從 8)
learning_rate: 1e-4   # 固定不變
weight_decay: 0.0     # 無正則化
scheduler: None       # 無學習率調度
epochs: 100
gradient_clip: 1.0
optimizer: Adam
```

### GPU 設定
```bash
CUDA_VISIBLE_DEVICES=2  # 使用 GPU 2 (2080Ti)
```

---

## 四、評估指標

### 定量指標

1. **Token Accuracy**
   - 當前: Train 56%, Val 38%
   - 目標: Train 62-65%, Val 43-47%

2. **Speaker Influence** (通過診斷工具測試)
   - 當前: <5% tokens 改變 (zero/random speaker test)
   - 目標: >20% tokens 改變

3. **Token 0 預測頻率**
   - 當前: ~32% (模型預測眾數策略)
   - 目標: 20-25% (更多樣化的預測)

### 定性指標

1. **Attention Weights 分析**
   - 哪些 token 位置依賴 speaker 較多？
   - 是否有規律（句首、重音位置等）？

2. **預測多樣性**
   - Top-20 predicted tokens 分布是否更均勻？

---

## 五、成功標準

### 必要條件 (Must Have)

✅ **Val Accuracy 提升 >3%**
   - 當前: 38.57%
   - 目標: >41.57%

✅ **Speaker Influence >20%**
   - Zero speaker test: accuracy drop >15%
   - Random speaker test: accuracy drop >15%

### 期望條件 (Should Have)

✅ **Token 0 預測頻率降低**
   - 當前: ~32%
   - 目標: <25%

✅ **Train-Val Gap 縮小**
   - 當前: 56% - 38% = 18%
   - 目標: <15%

✅ **Attention Pattern 可解釋**
   - 例如: 句首依賴 speaker 較多
   - 例如: 特定 tokens (如 silence) 依賴較少

---

## 六、實驗檔案

### 新增檔案

1. **`model_zeroshot_crossattn.py`**
   - `CrossAttentionFusion` class
   - `ZeroShotDenoisingTransformerCrossAttn` class
   - 支持 `return_attention=True` 返回 attention weights

2. **`train_crossattn_cached.py`**
   - 使用 Cross-Attention 模型
   - Batch size = 64
   - 無 weight decay
   - 無 scheduler

3. **`run_crossattn_experiment.sh`**
   - 一鍵啟動腳本
   - 自動檢查緩存
   - 設定 GPU 2

4. **`CROSS_ATTENTION_DESIGN.md`**
   - 完整設計文檔
   - ASCII 圖示說明
   - 數學公式推導

5. **`CROSS_ATTENTION_EXPERIMENT.md`** (本檔案)
   - 實驗說明與配置
   - 評估標準
   - 使用方式

---

## 七、使用方式

### 方式 1: 使用啟動腳本 (推薦)

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp
./run_crossattn_experiment.sh
```

### 方式 2: 直接運行 Python

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp

CUDA_VISIBLE_DEVICES=2 python -u train_crossattn_cached.py \
    --cache_dir ./data \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --dropout 0.1 \
    --num_workers 4 \
    2>&1 | tee crossattn_training.log
```

### 方式 3: 在 tmux 中運行 (推薦用於長時間訓練)

```bash
# 創建 tmux session
tmux new -s crossattn_training

# 在 tmux 內運行
cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp
./run_crossattn_experiment.sh

# Detach: Ctrl+B, D
# Re-attach: tmux attach -t crossattn_training
```

---

## 八、預期輸出

### 訓練日誌

```
results/crossattn_100epochs_YYYYMMDD_HHMMSS/
├── training.log                      # 完整訓練日誌
├── config.json                       # 實驗配置
├── best_model.pth                    # 最佳模型
├── checkpoint_epoch_10.pth           # Checkpoint (每 10 epochs)
├── checkpoint_epoch_20.pth
├── ...
├── loss_curves_epoch_10.png          # 損失曲線 (每 10 epochs)
├── loss_curves_final.png             # 最終損失曲線
└── audio_samples/                    # 音頻樣本 (如需要)
```

### 訓練速度預估

- **每 batch**: ~0.4-0.6s (因 batch size 增大可能略慢)
- **每 epoch**: ~6-8 分鐘 (16128 樣本 / 64 ≈ 252 batches)
- **100 epochs**: ~10-13 小時

### 記憶體使用預估

- **GPU Memory**: ~8-10 GB (batch 64, seq_len ~300)
- **System Memory**: ~2-4 GB (DataLoader 緩存)

---

## 九、監控與分析

### 即時監控

```bash
# 查看訓練進度
tail -f /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/crossattn_training.log

# 或附加到 tmux
tmux attach -t crossattn_training

# 查看 GPU 使用率
watch -n 1 nvidia-smi
```

### 關鍵觀察點

**Epoch 1-10**: 
- 觀察 Val Accuracy 是否 >40%
- 如果 <35%, 可能需要調整學習率或 dropout

**Epoch 10-30**:
- 觀察是否出現平台期
- 比較 Train-Val gap 是否縮小

**Epoch 30-100**:
- 記錄最佳 Val Accuracy
- 準備運行診斷工具分析 Speaker Influence

---

## 十、後續分析

### 實驗完成後執行

1. **運行診斷工具** (使用最佳模型)
   ```bash
   # 修改 diagnose_prediction_behavior.py 載入 Cross-Attention 模型
   python diagnose_prediction_behavior.py
   ```

2. **對比兩個模型**
   - Additive Fusion: Val Acc 38.57%
   - Cross-Attention: Val Acc ?%
   - Speaker Influence: <5% vs ?%

3. **視覺化 Attention Weights**
   ```bash
   python visualize_cross_attention.py  # 待創建
   ```

4. **撰寫實驗報告**
   - 記錄改善幅度
   - 驗證假設是否成立
   - 決定下一步改進方向

---

## 十一、可能結果與對策

### 結果 A: 顯著改善 (Val Acc >43%)

**結論**: 假設 2 驗證成功！Speaker Embedding 影響力不足是主要問題。

**下一步**:
1. 進一步優化 Cross-Attention (multi-layer, 不同 nhead)
2. 結合其他改進 (Focal Loss, Fine-tune Codebook)
3. 撰寫論文/報告

### 結果 B: 略有改善 (Val Acc 39-42%)

**結論**: Speaker Embedding 有影響，但不是唯一問題。

**下一步**:
1. 分析 Attention Weights 找出模式
2. 測試假設 1 (Focal Loss 降低眾數權重)
3. 測試假設 3 (Fine-tune Codebook)

### 結果 C: 無改善或更差 (Val Acc <39%)

**結論**: 假設 2 可能不成立，問題在其他地方。

**下一步**:
1. 檢查 Cross-Attention 實作是否正確
2. 降低 batch size 重新測試 (可能 64 太大)
3. 轉向其他假設 (Codebook, Task Difficulty)

---

## 十二、實驗檢查清單

### 開始前

- [x] 確認緩存存在 (`./data/train_cache.pt`, `./data/val_cache.pt`)
- [x] 模型測試通過 (`python model_zeroshot_crossattn.py`)
- [x] GPU 2 可用 (`nvidia-smi`)
- [x] 啟動腳本有執行權限 (`chmod +x run_crossattn_experiment.sh`)

### 訓練中

- [ ] Epoch 1 完成，無錯誤
- [ ] Val Accuracy > 25% (sanity check)
- [ ] GPU 利用率 >70%
- [ ] 記憶體使用 <11GB (2080Ti 限制)

### 訓練後

- [ ] 最佳模型已保存
- [ ] 訓練日誌完整
- [ ] 損失曲線已生成
- [ ] 與 Baseline 對比完成

---

## 十三、相關文檔

- `CROSS_ATTENTION_DESIGN.md` - 設計方案與 ASCII 圖示
- `DIAGNOSIS_PROGRESS_REPORT.md` - 診斷進度報告
- `TRAINING_MECHANISM_HYPOTHESIS.md` - 機轉假設說明
- `PLATEAU_MECHANISM_ANALYSIS.md` - 平台期分析

---

**文件創建**: 2025-11-05  
**實驗負責人**: GitHub Copilot  
**狀態**: 準備就緒，等待啟動
