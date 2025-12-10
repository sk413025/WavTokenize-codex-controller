# Exp_1210: 修復版實驗 (Codebook 漂移修復)

## 問題背景

在 exp_1209 的實驗中發現嚴重問題：

```
訓練曲線異常:
- Feature Loss 下降 (0.024 → 0.018)  ✓
- VQ Distance 上升 (2.75 → 2.90)     ✗ 異常!
- Token Accuracy 極低 (~4-14%)       ✗ 異常!
```

## 確認的問題

| 問題 | 狀態 | 嚴重度 |
|------|------|--------|
| Teacher 意外進入 train 模式 | 🔴 已確認 | 最嚴重 |
| Teacher codebook EMA 漂移 | 🔴 已確認 | 高 |
| Student codebook EMA 漂移 | 🔴 已確認 | 高 |

### 根本原因

```python
# exp_1209/train_lora_v2.py:94-95
def train_epoch(...):
    model.train()  # ← 這會把 Teacher 也設為 train 模式!
    model.student.train()
    # 缺少: model.teacher.eval()
```

當 `model.train()` 被調用時，PyTorch 會遞迴設置所有子模組：
- Teacher → train 模式 → quantizer → EMA 更新
- Student → train 模式 → quantizer → EMA 更新

結果：Teacher 和 Student 的 codebook 同時漂移，導致：
1. 評估指標失真
2. Loss 優化目標與實際評估脫鉤

## 修復內容

### 1. 覆寫 `train()` 方法

```python
class TeacherStudentExpandedLoRA(nn.Module):
    def train(self, mode: bool = True):
        super().train(mode)
        # Teacher 始終 eval
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()
        # Student quantizer 也要 eval
        self.student.feature_extractor.encodec.quantizer.eval()
        return self
```

### 2. 凍結 Quantizer

```python
def _freeze_quantizer(self, model, name: str):
    quantizer = model.feature_extractor.encodec.quantizer
    quantizer.eval()
    for param in quantizer.parameters():
        param.requires_grad = False
```

### 3. Codebook 安全檢查

```python
def check_codebook_integrity(self, raise_error: bool = True):
    teacher_drift = (self._initial_teacher_codebook - current).abs().mean()
    if teacher_drift > 1e-7:
        raise CodebookDriftError("Teacher codebook 漂移!")
```

## 實驗配置

### 參數選擇

| 參數 | 值 | 說明 |
|------|-----|------|
| LoRA rank | 128 | 平衡容量與過擬合 |
| LoRA alpha | 256 | 2x rank |
| Batch size | 8 | 適中，~6GB GPU |
| Learning rate | 2e-5 | 較低，穩定訓練 |

### Exp26: Feature Only (基線)

驗證修復後，純 Feature Loss 是否能正常工作。

```yaml
model:
  lora_rank: 128
  lora_alpha: 256

loss:
  feature_weight: 1.0   # 唯一優化目標
  triplet_weight: 0.0
  dw_weight: 0.0

training:
  lr: 2e-5
  batch_size: 8
```

**預期**：Feature Loss ↓ → VQ Distance ↓ → Token Acc ↑

### Exp27: Feature + Triplet

對比 Triplet Loss 是否有額外幫助。

```yaml
loss:
  feature_weight: 1.0
  triplet_weight: 0.5
  triplet_margin: 0.2
  dw_weight: 0.0
```

### 為什麼不用 CE Loss？

exp_1209 中 CE Loss 導致模型 collapse (Exp22: 1.3% accuracy)。
原因：CE Loss 尺度 (~2-4) 遠大於 Feature Loss (~0.02)，主導優化。

如需分類目標，建議使用 DW Loss (soft target) 替代 hard CE。

## 文件結構

```
exp_1210/
├── README.md                     # 本文件
├── models.py                     # 修復版模型
├── losses.py                     # Loss functions (從 exp_1209 複製)
├── train_lora_v3.py              # 修復版訓練腳本
├── run_exp26_feature_only.sh     # Feature Only 實驗
├── run_exp27_feature_triplet.sh  # Feature + Triplet 實驗
└── experiments/                  # 實驗結果
```

## 重現實驗

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1210

# 先運行 Exp26 驗證修復
bash run_exp26_feature_only.sh

# 對比實驗
bash run_exp27_feature_triplet.sh
```

## 與 exp_1209 的對比

| 項目 | exp_1209 | exp_1210 |
|------|----------|----------|
| Teacher train 模式 | ❌ 意外進入 | ✅ 始終 eval |
| Teacher codebook | ❌ EMA 漂移 | ✅ 凍結 |
| Student codebook | ❌ EMA 漂移 | ✅ 凍結 |
| 安全檢查 | ❌ 無 | ✅ 每 100 batch |
| VQ Distance 可靠性 | ❌ 失真 | ✅ 準確 |
