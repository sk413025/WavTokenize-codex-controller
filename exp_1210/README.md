# Exp_1210: 修復版實驗 (Codebook 漂移修復)

## 系統架構

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Teacher-Student LoRA 架構                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                           ┌─────────────┐                  │
│  │ Noisy Audio │                           │ Clean Audio │                  │
│  └──────┬──────┘                           └──────┬──────┘                  │
│         │                                         │                          │
│         ▼                                         ▼                          │
│  ┌──────────────────────┐               ┌──────────────────────┐            │
│  │   Student Encoder    │               │   Teacher Encoder    │            │
│  │  ┌────────────────┐  │               │  ┌────────────────┐  │            │
│  │  │ WavTokenizer   │  │               │  │ WavTokenizer   │  │            │
│  │  │ + LoRA (r=128) │  │               │  │ (Frozen)       │  │            │
│  │  │ 18 Conv Layers │  │               │  │                │  │            │
│  │  └────────────────┘  │               │  └────────────────┘  │            │
│  └──────────┬───────────┘               └──────────┬───────────┘            │
│             │                                      │                         │
│             ▼                                      ▼                         │
│  ┌──────────────────────┐               ┌──────────────────────┐            │
│  │  Student Features    │               │  Teacher Features    │            │
│  │  z_s: (B, 512, T)    │───────────────│  z_t: (B, 512, T)    │            │
│  └──────────┬───────────┘    MSE Loss   └──────────┬───────────┘            │
│             │                                      │                         │
│             ▼                                      ▼                         │
│  ┌──────────────────────┐               ┌──────────────────────┐            │
│  │  Quantizer (Frozen)  │               │  Quantizer (Frozen)  │            │
│  │  Codebook: 4096×512  │               │  Codebook: 4096×512  │            │
│  │  ⚠️ EMA disabled     │               │  ⚠️ EMA disabled     │            │
│  └──────────┬───────────┘               └──────────┬───────────┘            │
│             │                                      │                         │
│             ▼                                      ▼                         │
│  ┌──────────────────────┐               ┌──────────────────────┐            │
│  │  Student Tokens      │───────────────│  Teacher Tokens      │            │
│  │  (B, 1, T)           │  Token Acc    │  (B, 1, T)           │            │
│  └──────────────────────┘               └──────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

訓練目標: min ||z_s - z_t||²  (Feature MSE Loss)
評估指標: Token Accuracy = (student_tokens == teacher_tokens).mean()
```

## 問題背景

### exp_1209 發現的三項嚴重問題

```
訓練曲線異常:
┌────────────────────────────────────────────────────────────┐
│  Feature Loss    ████████████▓▓▓▓░░░░  0.024 → 0.018  ✓   │
│  VQ Distance     ░░░░░░░░████████████  2.75 → 2.90    ✗   │
│  Token Accuracy  ████░░░░░░░░░░░░░░░░  ~4-14%         ✗   │
└────────────────────────────────────────────────────────────┘
預期: Feature Loss ↓ → VQ Distance ↓ → Token Acc ↑
實際: Feature Loss ↓ → VQ Distance ↑ → Token Acc 極低  (矛盾!)
```

### 問題診斷

| # | 問題 | 狀態 | 嚴重度 | 影響 |
|---|------|------|--------|------|
| 1 | Teacher 意外進入 train 模式 | 🔴 已確認 | **最嚴重** | Teacher codebook 被 EMA 更新 |
| 2 | Teacher codebook EMA 漂移 | 🔴 已確認 | 高 | 評估基準改變，Token Acc 失真 |
| 3 | Student codebook EMA 漂移 | 🔴 已確認 | 高 | 量化結果不穩定 |

### 根本原因分析

```python
# exp_1209/train_lora_v2.py:94-95 的問題
def train_epoch(...):
    model.train()        # ← 問題根源! PyTorch 遞迴設置所有子模組
    model.student.train()
    # 缺少: model.teacher.eval()
```

```
PyTorch model.train() 的遞迴行為:

model.train()
    ├── student.train()
    │       └── encoder.train()
    │               └── quantizer.train()  ← EMA 開始更新!
    │
    └── teacher.train()  ← 意外被設為 train!
            └── encoder.train()
                    └── quantizer.train()  ← Teacher codebook 漂移!
```

**後果**:
1. Teacher codebook 在訓練中持續漂移
2. Token Accuracy 計算基於漂移後的 codebook，結果失真
3. Feature Loss 優化目標與實際評估脫鉤

---

## 修復方案

### 修復 1: 覆寫 `train()` 方法

```python
class TeacherStudentExpandedLoRA(nn.Module):
    def train(self, mode: bool = True):
        """覆寫 train() 確保 Teacher 始終 eval"""
        super().train(mode)

        # Teacher 始終保持 eval 模式
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()

        # Student quantizer 也要 eval (凍結 EMA)
        self.student.feature_extractor.encodec.quantizer.eval()

        return self
```

### 修復 2: 凍結 Quantizer 參數

```python
def _freeze_quantizer(self, model, name: str):
    """凍結 quantizer 的所有參數"""
    quantizer = model.feature_extractor.encodec.quantizer
    quantizer.eval()  # 強制 eval 模式

    for param in quantizer.parameters():
        param.requires_grad = False  # 禁止梯度更新

    print(f"  {name} quantizer frozen (eval mode, requires_grad=False)")
```

### 修復 3: Codebook 完整性監控

```python
def check_codebook_integrity(self, raise_error: bool = True) -> dict:
    """
    檢查 codebook 是否漂移

    原理:
    1. 初始化時保存 codebook 快照
    2. 訓練中定期比對當前值與初始值
    3. 差異超過閾值 (1e-7) 則報錯
    """
    # 獲取當前 codebook
    teacher_cb = self._get_teacher_codebook()
    student_cb = self._get_student_codebook()

    # 計算漂移量 (L1 距離)
    teacher_drift = (self._initial_teacher_codebook - teacher_cb).abs().mean().item()
    student_drift = (self._initial_student_codebook - student_cb).abs().mean().item()

    result = {
        'teacher_drift': teacher_drift,
        'student_drift': student_drift,
        'teacher_ok': teacher_drift < 1e-7,
        'student_ok': student_drift < 1e-7,
    }

    if raise_error:
        if teacher_drift > 1e-7:
            raise CodebookDriftError(f"Teacher codebook 漂移! drift={teacher_drift:.8f}")
        if student_drift > 1e-7:
            raise CodebookDriftError(f"Student codebook 漂移! drift={student_drift:.8f}")

    return result
```

### 監控流程圖

```
┌─────────────────────────────────────────────────────────────┐
│                    Codebook 監控機制                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  初始化階段:                                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ self._initial_teacher_codebook = codebook.clone()    │   │
│  │ self._initial_student_codebook = codebook.clone()    │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  訓練階段 (每 100 batch):                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ current_cb = get_current_codebook()                  │   │
│  │ drift = |initial_cb - current_cb|.mean()             │   │
│  │                                                       │   │
│  │ if drift > 1e-7:                                      │   │
│  │     raise CodebookDriftError("漂移檢測!")             │   │
│  │ else:                                                 │   │
│  │     continue training ✓                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 實驗配置

### 參數選擇

| 參數 | 值 | 說明 |
|------|-----|------|
| LoRA rank | 128 | 平衡容量與過擬合 |
| LoRA alpha | 256 | 2x rank (標準設定) |
| Batch size | 16 | 較大，~10GB GPU |
| Learning rate | 2e-5 | 較低，穩定訓練 |
| Epochs | 50 | 足夠收斂 |
| Check interval | 100 | 每 100 batch 檢查 codebook |

### Exp26: Feature Only (基線)

驗證修復後，純 Feature Loss 是否能正常工作。

```yaml
實驗名稱: exp26_feature_only_fixed

model:
  lora_rank: 128
  lora_alpha: 256
  lora_layers: 18  # 所有 encoder conv 層

loss:
  feature_weight: 1.0   # 唯一優化目標
  triplet_weight: 0.0   # 關閉
  dw_weight: 0.0        # 關閉
  soft_ce_weight: 0.0   # 關閉

training:
  lr: 2e-5
  batch_size: 16
  num_epochs: 50
  seed: 42
```

**預期結果**：
```
修復後預期曲線:
Feature Loss  ████████▓▓▓▓▓░░░░░  下降 ✓
VQ Distance   ████████▓▓▓▓▓░░░░░  下降 ✓ (與 Feature Loss 同步)
Token Acc     ░░░░▓▓▓▓████████████  上升 ✓
```

### Exp27: Feature + Triplet

對比 Triplet Loss 是否有額外幫助。

```yaml
實驗名稱: exp27_feature_triplet_fixed

loss:
  feature_weight: 1.0
  triplet_weight: 0.5
  triplet_margin: 0.2
  dw_weight: 0.0
  soft_ce_weight: 0.0
```

### Exp28: Feature + Triplet + Soft CE (方案 C)

結合 Triplet 對比學習與 Soft CE 知識蒸餾。

```yaml
實驗名稱: exp28_feature_triplet_softce

loss:
  feature_weight: 1.0   # 特徵對齊
  triplet_weight: 0.3   # 對比學習 (略降權重)
  triplet_margin: 0.2
  soft_ce_weight: 0.5   # 知識蒸餾
  soft_ce_temperature: 2.0
  dw_weight: 0.0
```

**設計理念**:
- Triplet: 優化特徵空間結構 (拉近正樣本，推遠負樣本)
- Soft CE: 優化 token 分布對齊 (保留 Teacher 的軟資訊)
- 兩者互補，預期優於單獨使用

### Exp29: Feature + Soft CE (方案 B)

純知識蒸餾方案，無 Triplet。

```yaml
實驗名稱: exp29_feature_softce

loss:
  feature_weight: 1.0   # 特徵對齊
  triplet_weight: 0.0   # 關閉
  soft_ce_weight: 1.0   # 知識蒸餾 (主要監督)
  soft_ce_temperature: 2.0
  dw_weight: 0.0
```

**設計理念**:
- 經典 Hinton 知識蒸餾
- Temperature=2.0 軟化分布，傳遞 "dark knowledge"
- 與 Exp28 對比可確認 Triplet 的獨立貢獻

### 實驗對比設計

| 實驗 | Feature | Triplet | Soft CE | 目的 |
|------|---------|---------|---------|------|
| Exp26 | 1.0 | 0.0 | 0.0 | 基線 |
| Exp27 | 1.0 | 0.5 | 0.0 | Triplet 效果 |
| Exp28 | 1.0 | 0.3 | 0.5 | 綜合方案 |
| Exp29 | 1.0 | 0.0 | 1.0 | 純知識蒸餾 |

**分析邏輯**:
- Exp27 vs Exp26 → Triplet 單獨貢獻
- Exp29 vs Exp26 → Soft CE 單獨貢獻
- Exp28 vs Exp27 → Soft CE 對 Triplet 的增益
- Exp28 vs Exp29 → Triplet 對 Soft CE 的增益

### 為什麼不用 CE Loss？

exp_1209 中 CE Loss 導致模型 collapse (Exp22: 1.3% accuracy)。

```
Loss 尺度問題:
┌────────────────────────────────────────┐
│ Feature Loss:  ~0.02 - 0.05            │
│ CE Loss:       ~2.0 - 4.0   (100x 大!) │
└────────────────────────────────────────┘

結果: CE Loss 完全主導優化，Feature Loss 被忽略
建議: 使用 DW Loss (soft target) 或 Soft CE 替代 hard CE
```

---

## 輸出監控

### 訓練過程輸出

每個 epoch 會輸出：
- **Loss**: total, feature, triplet, dw, soft_ce
- **Metrics**: token_accuracy, vq_distance
- **Codebook check**: 每 100 batch 驗證無漂移

### 音檔與頻譜圖保存

```
experiments/exp26_feature_only_fixed/
├── audio_samples/
│   ├── train/
│   │   ├── epoch_001/
│   │   │   ├── sample_1_noisy.wav
│   │   │   ├── sample_1_clean.wav
│   │   │   ├── sample_1_student_recon.wav
│   │   │   └── sample_1_teacher_recon.wav
│   │   └── epoch_005/
│   │       └── ...
│   └── val/
│       └── ...
├── spectrograms/
│   ├── epoch_001/
│   │   └── sample_1_spectrogram.png  (2x2 對比圖)
│   └── epoch_005/
│       └── ...
├── training_curves.png
├── training_history.json
└── best_model.pt
```

### Loss 曲線圖

自動生成 `training_curves.png`，包含：
- Total Loss (train/val)
- Feature Loss (train/val)
- Token Accuracy (train/val)
- VQ Distance (train/val)
- Triplet Loss (if enabled)
- Learning Rate

---

## 文件結構

```
exp_1210/
├── README.md                            # 本文件
├── models.py                            # 修復版模型 (覆寫 train(), 凍結 quantizer)
├── losses.py                            # Loss functions (CombinedLoss, CombinedLossV2)
├── train_lora_v3.py                     # 修復版訓練腳本
├── run_exp26_feature_only.sh            # Feature Only 實驗
├── run_exp27_feature_triplet.sh         # Feature + Triplet 實驗
├── run_exp28_feature_triplet_softce.sh  # Feature + Triplet + Soft CE 實驗
├── run_exp29_feature_softce.sh          # Feature + Soft CE 實驗
└── experiments/                         # 實驗結果目錄
    ├── exp26_feature_only_fixed/
    ├── exp27_feature_triplet_fixed/
    ├── exp28_feature_triplet_softce/
    └── exp29_feature_softce/
```

---

## 重現實驗

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1210

# 先運行 Exp26 驗證修復
bash run_exp26_feature_only.sh

# 對比實驗
bash run_exp27_feature_triplet.sh
```

---

## 與 exp_1209 的對比

| 項目 | exp_1209 | exp_1210 |
|------|----------|----------|
| Teacher train 模式 | ❌ 意外進入 | ✅ 始終 eval |
| Teacher codebook | ❌ EMA 漂移 | ✅ 凍結 + 監控 |
| Student codebook | ❌ EMA 漂移 | ✅ 凍結 + 監控 |
| 安全檢查 | ❌ 無 | ✅ 每 100 batch |
| VQ Distance 可靠性 | ❌ 失真 | ✅ 準確 |
| Token Accuracy 可靠性 | ❌ 基準漂移 | ✅ 固定基準 |
| 音檔監控 | ❌ 無 | ✅ 每 5 epoch |
| 頻譜圖監控 | ❌ 無 | ✅ 每 5 epoch |

---

## 預期結果

如果修復成功，應該觀察到：

```
┌─────────────────────────────────────────────────────────────┐
│                     預期訓練曲線                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Feature Loss:                                               │
│  0.05 ┤████████                                              │
│  0.04 ┤        ████                                          │
│  0.03 ┤            ████                                      │
│  0.02 ┤                ████████                              │
│  0.01 ┤                        ████████████████████          │
│       └────────────────────────────────────────────          │
│        1    5    10   15   20   25   30   35   40   45  50   │
│                                                              │
│  Token Accuracy:                                             │
│   80% ┤                                    ████████████████  │
│   60% ┤                        ████████████                  │
│   40% ┤                ████████                              │
│   20% ┤        ████████                                      │
│    0% ┤████████                                              │
│       └────────────────────────────────────────────          │
│        1    5    10   15   20   25   30   35   40   45  50   │
│                                                              │
│  VQ Distance:                                                │
│  3.0  ┤████████                                              │
│  2.5  ┤        ████████                                      │
│  2.0  ┤                ████████                              │
│  1.5  ┤                        ████████                      │
│  1.0  ┤                                ████████████████████  │
│       └────────────────────────────────────────────          │
│        1    5    10   15   20   25   30   35   40   45  50   │
│                                                              │
└─────────────────────────────────────────────────────────────┘

關鍵驗證點:
✓ Feature Loss ↓ 時 VQ Distance 也 ↓ (不再矛盾)
✓ Token Accuracy 穩定上升 (不再停滯在 4-14%)
✓ Codebook 檢查全程通過 (無 CodebookDriftError)
```
