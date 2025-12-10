# 實驗 17 & 18 快速啟動指南

## 當前狀態

### ✅ 已完成
1. **梯度衝突分析工具** - `analyze_gradient_conflict.py` (正在執行中)
2. **核心函式庫** - `exp17_18_core_functions.py` (包含所有關鍵函式)
3. **實驗腳本** - `run_exp17_margin.sh` 和 `run_exp18_curriculum.sh`
4. **詳細文檔** - `EXP17_18_README.md`

### ⏳ 待完成
- 修改 `train_margin_loss.py` (exp17)
- 修改 `train_curriculum.py` (exp18)

---

## 方案 A: 手動修改（推薦，完全控制）

### exp17 (Margin Loss)

#### 步驟 1: 導入核心函式

在 `train_margin_loss.py` 開頭添加：

```python
from exp17_18_core_functions import compute_margin_loss, compute_losses_margin
```

#### 步驟 2: 替換 compute_losses 函式

找到原本的 `compute_losses` 函式定義（約第 220 行），整個替換為：

```python
def compute_losses(model, output, distance_matrix, margin, ce_weight, ce_temperature=0.1):
    """使用 exp17 的 Margin Loss"""
    return compute_losses_margin(model, output, distance_matrix, margin, ce_weight, ce_temperature)
```

#### 步驟 3: 修改 train_epoch 函式

找到 `train_epoch` 函式中的 `compute_losses` 呼叫（約第 300 行），修改為：

```python
# 原本
losses = compute_losses(model, output, distance_matrix, feature_weight, ce_weight, ce_temperature)

# 改為
losses = compute_losses(model, output, distance_matrix, margin, ce_weight, ce_temperature)
```

#### 步驟 4: 修改 validate 函式

同樣修改 `validate` 函式中的呼叫。

#### 步驟 5: 修改 argparse

找到 `parser.add_argument` 部分（約第 690 行）：

```python
# 移除
# parser.add_argument('--feature_weight', ...)

# 新增
parser.add_argument('--margin', type=float, default=0.5,
                   help='Margin for contrastive loss')
```

#### 步驟 6: 修改 history 記錄

找到所有 `history['train_feature_loss']` 的地方，改為 `history['train_margin_loss']`

#### 步驟 7: 修改 main() 函式

找到 `main()` 函式中的訓練呼叫（約第 800 行）：

```python
# 原本
train_results = train_epoch(..., args.feature_weight, args.ce_weight, ...)

# 改為  
train_results = train_epoch(..., args.margin, args.ce_weight, ...)
```

---

### exp18 (Curriculum Learning)

#### 步驟 1: 導入核心函式

在 `train_curriculum.py` 開頭添加：

```python
from exp17_18_core_functions import get_curriculum_weights, compute_losses_curriculum
```

#### 步驟 2: 替換 compute_losses 函式

```python
def compute_losses(model, output, distance_matrix, mse_weight, ce_weight, ce_temperature=0.1):
    """使用 exp18 的 Curriculum Learning"""
    return compute_losses_curriculum(model, output, distance_matrix, mse_weight, ce_weight, ce_temperature)
```

#### 步驟 3: 修改訓練迴圈

在 `main()` 函式的訓練迴圈中（約第 810 行），每個 epoch 開始時添加：

```python
for epoch in range(1, args.num_epochs + 1):
    # 動態獲取當前階段的權重
    ce_weight, mse_weight, stage_name = get_curriculum_weights(epoch, args)
    
    print(f"\nEpoch {epoch}/{args.num_epochs} - {stage_name}")
    print(f"  CE Weight: {ce_weight}, MSE Weight: {mse_weight}")
    
    # Train
    train_results = train_epoch(
        ..., 
        feature_weight=mse_weight,  # 使用動態權重
        ce_weight=ce_weight,
        ...
    )
```

#### 步驟 4: 修改 argparse

添加 Curriculum 參數：

```python
# Stage 1
parser.add_argument('--stage1_epochs', type=int, default=10)
parser.add_argument('--stage1_ce_weight', type=float, default=1.0)
parser.add_argument('--stage1_mse_weight', type=float, default=0.0)

# Stage 2
parser.add_argument('--stage2_epochs', type=int, default=20)
parser.add_argument('--stage2_ce_weight', type=float, default=0.5)
parser.add_argument('--stage2_mse_weight', type=float, default=1.0)

# Stage 3
parser.add_argument('--stage3_epochs', type=int, default=20)
parser.add_argument('--stage3_ce_weight', type=float, default=0.1)
parser.add_argument('--stage3_mse_weight', type=float, default=1.0)
```

---

## 方案 B: 使用現有程式測試（快速驗證）

如果只想快速測試概念，可以：

### 測試 Margin Loss 概念

在 Python interactive 中：

```python
import sys
sys.path.append('/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1207')
from exp17_18_core_functions import compute_margin_loss
import torch

# 模擬資料
B, C, T = 8, 512, 75
student_emb = torch.randn(B, C, T).cuda()
teacher_codes = torch.randint(0, 4096, (B, T)).cuda()
codebook = torch.randn(4096, 512).cuda()

# 計算 Margin Loss
loss, correct_dist, wrong_dist = compute_margin_loss(
    student_emb, teacher_codes, codebook, margin=0.5
)

print(f"Margin Loss: {loss.item():.4f}")
print(f"Correct Distance: {correct_dist:.4f}")
print(f"Nearest Wrong Distance: {wrong_dist:.4f}")

# 理想情況: correct_dist < wrong_dist - margin
# 即 correct_dist + margin < wrong_dist
```

### 測試 Curriculum Weights

```python
from exp17_18_core_functions import get_curriculum_weights

class Args:
    stage1_epochs = 10
    stage1_ce_weight = 1.0
    stage1_mse_weight = 0.0
    stage2_epochs = 20
    stage2_ce_weight = 0.5
    stage2_mse_weight = 1.0
    stage3_epochs = 20
    stage3_ce_weight = 0.1
    stage3_mse_weight = 1.0

args = Args()

for epoch in [1, 5, 10, 15, 30, 45]:
    ce, mse, stage = get_curriculum_weights(epoch, args)
    print(f"Epoch {epoch:2d}: {stage:20s} - CE={ce:.1f}, MSE={mse:.1f}")
```

---

## 方案 C: 簡化版實驗（最小修改）

如果時間有限，可以先在 `train_with_ce.py` 中添加一個參數來測試：

```python
# 在 main() 函式開頭添加
USE_MARGIN_LOSS = True  # 設為 True 啟用 Margin Loss

if USE_MARGIN_LOSS:
    from exp17_18_core_functions import compute_margin_loss
    # ... 使用 Margin Loss
else:
    # ... 使用原本的 MSE Loss
```

---

## 檢查梯度分析進度

```bash
# 即時監控
tail -f exp_1207/gradient_analysis.log

# 檢查是否完成
ls exp_1207/gradient_analysis/

# 查看結果
cat exp_1207/gradient_analysis/gradient_analysis_*.json | jq '.mean_cosine, .conflict_ratio'
```

---

## 預期輸出

### 梯度分析完成後

會生成：
- `gradient_analysis/gradient_analysis_YYYYMMDD_HHMMSS.json`
- `gradient_analysis/gradient_analysis_YYYYMMDD_HHMMSS.png`

關鍵指標：
```json
{
  "mean_cosine": 0.123,          # < 0 表示衝突
  "conflict_ratio": 0.456,       # > 0.5 表示嚴重衝突
  "aligned_ratio": 0.234         # > 0.5 表示協同
}
```

### 實驗執行後

exp17 和 exp18 會生成：
- `experiments/exp17_margin_loss/` 或 `experiments/exp18_reverse_curriculum/`
  - `training_curves_epoch_*.png` - 訓練曲線
  - `audio_samples/` - 音訊樣本
  - `best.pt` - 最佳模型
  - `training_history.json` - 完整歷史

---

## 下一步

1. **等待梯度分析完成** (約 5-10 分鐘)
2. **查看梯度分析結果**，決定優先順序
3. **選擇方案 A/B/C** 進行實驗
4. **執行實驗**
5. **分析結果並commit**

需要我協助哪一步？
