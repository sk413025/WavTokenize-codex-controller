# 如何在當前環境重現 Commit 809e1e5 的 Backbone LoRA 實驗

## 📌 背景

Commit `809e1e5` 實現了在 WavTokenizer Backbone Attention 層應用 LoRA 進行去噪訓練。
雖然該實驗最終失敗（LoRA 容量不足），但可以作為學習案例。

當前環境（commit `6867592`）中 `done/exp5/exp5-3/` 已被刪除，需要從 git 歷史恢復。

---

## 🔧 重現步驟

### 方法 1：從 Git 恢復完整目錄（推薦）

```bash
# 1. 從 commit 809e1e5 恢復 exp5-3 目錄
cd /home/sbplab/ruizi/WavTokenize-self-supervised
git checkout 809e1e5 -- done/exp5/exp5-3

# 2. 恢復相關的共享工具
git checkout 809e1e5 -- done/exp5/shared/lora_utils.py

# 3. 檢查恢復的文件
ls -la done/exp5/exp5-3/
```

**恢復的關鍵文件**：
```
done/exp5/exp5-3/
├── BACKBONE_LORA_APPROACH.md   # 技術文檔
├── config.py                    # LoRA 配置
├── model_stage1.py              # Backbone LoRA 模型
├── data_stage1.py               # 數據載入
├── train_stage1.py              # 訓練腳本
├── START_STAGE1.sh              # 啟動腳本
├── test_setup.py                # 測試腳本
└── ...
```

### 方法 2：切換到 809e1e5 查看（只讀模式）

```bash
# 創建臨時分支查看
git checkout -b temp-809e1e5-review 809e1e5

# 查看所有文件
cd done/exp5/exp5-3
ls -la

# 查看後切回主分支
git checkout -
git branch -D temp-809e1e5-review
```

---

## 🧪 驗證環境依賴

### 檢查 WavTokenizer

```bash
# 1. 檢查 WavTokenizer checkpoint
ls -lh /home/sbplab/ruizi/WavTokenizer-main/wavtokenizer_large_speech_320_24k.ckpt
# ✅ 已確認存在 (1.7 GB)

# 2. 檢查 WavTokenizer 配置
ls -lh /home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml
```

### 檢查數據

```bash
# 檢查訓練數據 cache
ls -lh done/exp/data3/train_cache.pt
ls -lh done/exp/data3/val_cache.pt
```

### 安裝 Python 依賴

```bash
# PEFT (LoRA 庫)
pip install peft

# 其他依賴（如果需要）
pip install torch torchaudio
```

---

## 🚀 運行實驗

### 1. 測試配置

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp5/exp5-3

# 運行測試腳本（驗證 LoRA 梯度流動）
python test_setup.py
```

**預期輸出**：
```
✅ All tests PASSED!
🎉 Backbone Attention LoRA 配置正確！
    - LoRA 成功應用到 pos_net.2 的 q/k/v/proj_out
    - 梯度正常流動
    - 其他參數正確凍結
```

### 2. 啟動訓練（Stage 1）

```bash
# 方法 A: 使用腳本啟動
bash START_STAGE1.sh

# 方法 B: 直接運行
python train_stage1.py \
    --exp_name backbone_lora_test \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 5e-5
```

### 3. 監控訓練

```bash
# 監控腳本
bash monitor_training.sh

# 或手動查看日誌
tail -f done/exp5/exp5-3/results/stage1/training.log
```

---

## 📊 預期結果（基於原始實驗）

### 訓練過程

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|------------|----------|---------|
| 1     | 8.32       | 6.95     | 2.02%   |
| 5     | 6.51       | 6.45     | 1.78%   |
| 10    | 6.23       | 6.12     | 1.42%   |
| 19    | 5.88       | 5.91     | 1.07%   |

### 失敗診斷（Commit 1712e38）

**問題**：
- Val Acc 持續下降（2.02% → 1.07%）
- 與目標 48% 相差 45 倍

**根本原因**：
- LoRA 容量不足（98K vs 10M needed）
- 只有 1 個 Attention 層可訓練
- Token-level 去噪任務本質困難

**後續行動**：
- 轉向 Exp5-3-1（獨立 Transformer，10M params）
- 設計 Exp5-4（End-to-End STE 訓練）

---

## 🔍 關鍵代碼解析

### LoRA 應用位置

```python
# config.py
'lora_target_modules': [
    'backbone.pos_net.2.q',         # Query projection (Conv1d 768→768)
    'backbone.pos_net.2.k',         # Key projection
    'backbone.pos_net.2.v',         # Value projection
    'backbone.pos_net.2.proj_out',  # Output projection
]
```

**為什麼是 pos_net.2？**
- WavTokenizer Backbone 架構：
  ```
  backbone.pos_net:
    [0] ConvNeXt block
    [1] ConvNeXt block
    [2] AttnBlock ← 唯一的 Attention 層！
  ```
- AttnBlock 使用 Conv1d 實現 Q/K/V 投影（768-dim）

### Forward Pass（避免 inference_mode）

```python
# model_stage1.py - 關鍵實現
def forward(self, audio):
    # ❌ 不能用這個（有 @torch.inference_mode 裝飾器）
    # features, codes = self.wavtokenizer.encode(audio)

    # ✅ 正確做法：直接調用內部方法
    features_512d, discrete_codes, _ = \
        self.wavtokenizer.base_model.model.feature_extractor(
            audio, bandwidth_id=0
        )

    processed_features = \
        self.wavtokenizer.base_model.model.backbone(
            features_512d,
            bandwidth_id=torch.tensor([0], device=audio.device)
        )

    logits = self.classification_head(processed_features)

    return logits, discrete_codes
```

---

## 🎓 學習要點

### 1. LoRA 應用策略

**成功經驗（來自 Exp3/Exp4）**：
- 在 VQ **之後**的 token 空間學習
- 使用足夠的模型容量（10M+ params）

**Exp5-3 的嘗試（失敗）**：
- ✅ 選對了空間（VQ 之後）
- ❌ 容量不足（只有 98K LoRA params）

### 2. WavTokenizer 內部結構

```
Audio → Encoder (128-dim)
      → VQ (512-dim quantized, discrete codes)
      → Backbone (768-dim)
           ├─ embed: 512→768
           ├─ pos_net.0: ConvNeXt
           ├─ pos_net.1: ConvNeXt
           ├─ pos_net.2: AttnBlock ← LoRA target
           └─ norm + convnext
```

### 3. 避免 PyTorch 陷阱

**問題**：`@torch.inference_mode()` 裝飾器會禁用梯度
**解決**：直接調用內部方法（feature_extractor, backbone）

---

## 📚 相關文檔

恢復後可查看：
- `BACKBONE_LORA_APPROACH.md` - 完整技術文檔
- `FAILURE_ANALYSIS.md` - 失敗原因分析（commit 1712e38）
- `test_setup.py` - 梯度流動測試

---

## ⚠️ 注意事項

1. **這個實驗已證實失敗**
   - 只用於學習和理解
   - 實際使用請參考 Exp5-3-1（獨立 Transformer）

2. **GPU 記憶體需求**
   - 訓練：約 6-7 GB (batch_size=8)
   - 推理：約 4-5 GB (batch_size=32)

3. **訓練時間**
   - 50 epochs：約 2-3 小時（11GB GPU）

4. **無需 custom_wavtok**
   - 809e1e5 已刪除 custom 版本
   - 直接使用原版 WavTokenizer

---

## 🔗 相關 Commit

- `809e1e5` - 初始實現（Backbone LoRA）
- `1712e38` - 失敗分析報告
- `b4d8f94` - Exp5-3-1（改用獨立 Transformer）
- `8e4ae19` - Exp5-4 設計（End-to-End STE）

---

**總結**：完全可以在當前環境重現 809e1e5，但需要從 git 恢復文件。實驗雖然失敗，但提供了寶貴的學習經驗。
