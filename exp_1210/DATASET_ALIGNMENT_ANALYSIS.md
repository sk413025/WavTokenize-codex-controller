# 資料集對齊問題分析報告

**日期**: 2025-12-11
**分析者**: Claude
**相關實驗**: exp_1210 (Token Accuracy 提升緩慢)

---

## 📋 問題背景

在 exp_1210 實驗中，即使 Feature Loss (MSE) 下降 39%，Token Accuracy 只提升 4%：

| Metric | Epoch 1 | Epoch 50 | 變化 |
|--------|---------|----------|------|
| Train Acc | 26.74% | 30.81% | +4.07% |
| Feature Loss | 0.031 | 0.019 | -39% |

**核心疑問**: 為什麼 Feature Loss 下降但 Token Accuracy 提升緩慢？

---

## 🔍 假設：資料集對齊問題

### 假設內容

Clean 和 Noisy 音頻之間存在對齊問題：
1. 原始長度不一致
2. Batch padding 造成 frame 數變化
3. Loss 計算包含無效的 padding frames

### 為什麼這會影響訓練

```
                    理想情況                              實際情況
    ┌─────────────────────────────┐        ┌─────────────────────────────┐
    │  Noisy: [■■■■■■■■■■]        │        │  Noisy: [■■■■■■■■■■■■]      │
    │  Clean: [■■■■■■■■■■]        │        │  Clean: [■■■■■■■■■■]        │
    │         ↑ 完全對齊          │        │         ↑ 長度不同！        │
    └─────────────────────────────┘        └─────────────────────────────┘

    Loss 計算:                              Loss 計算:
    Frame 1: MSE(noisy[1], clean[1]) ✓     Frame 1: MSE(noisy[1], clean[1]) ✓
    Frame 2: MSE(noisy[2], clean[2]) ✓     Frame 2: MSE(noisy[2], clean[2]) ✓
    ...                                     ...
    Frame N: MSE(noisy[N], clean[N]) ✓     Frame N+1: MSE(noisy[N+1], ???) ✗
                                                      ↑ 對齊錯誤！
```

---

## 🔬 驗證方法

### 方法 1: 檢查原始資料長度

**檢查位置**: `/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt`

**檢查代碼**:
```python
import torch

# 載入訓練資料快取
cache = torch.load('/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt')

# 檢查前 10 個樣本的長度
for i in range(10):
    noisy = cache[i]['noisy_audio']
    clean = cache[i]['clean_audio']
    diff = len(noisy) - len(clean)
    print(f"Sample {i}: noisy={len(noisy)}, clean={len(clean)}, diff={diff:+d}")
```

### 方法 2: 追蹤 Collate 函數行為

**檢查位置**: `exp_1201/data.py:124-155`

**檢查代碼**:
```python
def collate_fn(batch):
    """檢查 padding 行為"""
    max_len = max(item['noisy_audio'].shape[0] for item in batch)
    print(f"Batch max_len: {max_len}")

    for i, item in enumerate(batch):
        orig_len = item['noisy_audio'].shape[0]
        pad_len = max_len - orig_len
        print(f"  Sample {i}: orig={orig_len}, pad={pad_len}")
```

### 方法 3: 檢查 Encoder 輸出 Frame 數

**檢查代碼**:
```python
# 測試不同輸入長度產生的 frame 數
test_lengths = [56640, 56642, 65760, 65762, 71040, 71042]
for length in test_lengths:
    audio = torch.randn(1, 1, length)
    frames = encoder(audio)
    print(f"Input {length} samples -> {frames.shape[2]} frames")
```

---

## 📊 驗證結果

### 結果 1: 原始資料長度不一致 ✅ 確認

```
從 train_cache.pt 讀取的實際數據:

┌─────────┬─────────────────┬─────────────────┬──────────┐
│ Sample  │ Noisy (samples) │ Clean (samples) │   Diff   │
├─────────┼─────────────────┼─────────────────┼──────────┤
│    0    │     56,642      │     56,640      │    +2    │
│    1    │     61,922      │     61,920      │    +2    │
│    2    │     65,762      │     65,760      │    +2    │
│    3    │     71,042      │     71,040      │    +2    │
│    4    │     62,402      │     62,400      │    +2    │
│   ...   │       ...       │       ...       │   ...    │
└─────────┴─────────────────┴─────────────────┴──────────┘

規律: Noisy 始終比 Clean 多 2 個 samples
```

**原因推測**:
- 可能是音頻處理時的 resampling 誤差
- 可能是不同軟體處理造成的長度差異

### 結果 2: Collate 函數 Padding 行為 ✅ 確認

```
Batch 示例 (3 個樣本):

原始長度:
┌─────────┬─────────────────┬─────────────────┐
│ Sample  │ Noisy (samples) │ Clean (samples) │
├─────────┼─────────────────┼─────────────────┤
│    0    │     56,642      │     56,640      │
│    1    │     61,922      │     61,920      │
│    2    │     65,762      │     65,760      │
└─────────┴─────────────────┴─────────────────┘

Collate 後 (pad 到 max_len=65,762):
┌─────────┬─────────────────┬──────────────────┐
│ Sample  │   Padded to     │  Zeros Added     │
├─────────┼─────────────────┼──────────────────┤
│    0    │     65,762      │     9,120 zeros  │
│    1    │     65,762      │     3,840 zeros  │
│    2    │     65,762      │         0 zeros  │
└─────────┴─────────────────┴──────────────────┘

問題: 短樣本有大量 zero padding!
```

### 結果 3: Encoder 輸出 Frame 數變化 ✅ 確認

```
Encoder 處理不同長度輸入:

┌──────────────────┬─────────────────┬──────────────────┐
│  Input (samples) │  Output (frames)│     說明         │
├──────────────────┼─────────────────┼──────────────────┤
│      56,640      │       177       │ 原始 Sample 0    │
│      56,642      │       178       │ +2 samples=+1 fr │
│      65,760      │       206       │ 原始 Sample 2    │
│      65,762      │       206       │ +2 samples=+0 fr │
│      71,040      │       222       │ 更長樣本         │
│      71,042      │       223       │ +2 samples=+1 fr │
└──────────────────┴─────────────────┴──────────────────┘

Batch padding 後 (全部 pad 到 71,042):

┌──────────────────┬─────────────────┬──────────────────┐
│  Original Length │  Padded to      │  Frames          │
├──────────────────┼─────────────────┼──────────────────┤
│      56,642      │     71,042      │  223 (原本 178)  │
│      61,922      │     71,042      │  223 (原本 194)  │
│      65,762      │     71,042      │  223 (原本 206)  │
└──────────────────┴─────────────────┴──────────────────┘

Sample 0 的額外 45 frames 來自 ZERO PADDING!
```

---

## 🎨 問題視覺化

### 問題 1: 長度不一致造成 Frame 錯位

```
時間軸對齊問題:

Noisy Audio:  [████████████████████████████████████████████]  56,642 samples
Clean Audio:  [██████████████████████████████████████████]    56,640 samples
                                                        ^^
                                                        多出 2 samples

Encoder 處理後:

Noisy Frames: [F1][F2][F3]...[F176][F177][F178]  178 frames
Clean Frames: [F1][F2][F3]...[F176][F177]        177 frames
                                   ^^^^
                                   多出 1 frame!

Feature Loss 計算時:
- Frame 1-177: MSE(noisy[i], clean[i]) ✓ 對齊
- Frame 178: MSE(noisy[178], ???) ✗ 無對應!
```

### 問題 2: Batch Padding 汙染

```
Batch Processing 流程:

Step 1: 收集 Batch
┌────────────────────────────────────────────────────────────┐
│ Sample 0: [████████████████████]                   56,642  │
│ Sample 1: [██████████████████████████████]         61,922  │
│ Sample 2: [██████████████████████████████████████] 65,762  │
└────────────────────────────────────────────────────────────┘

Step 2: Collate (Pad to max)
┌────────────────────────────────────────────────────────────┐
│ Sample 0: [████████████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒] 65,762  │
│ Sample 1: [██████████████████████████████▒▒▒▒▒▒▒▒] 65,762  │
│ Sample 2: [██████████████████████████████████████] 65,762  │
└────────────────────────────────────────────────────────────┘
             █ = 真實信號        ▒ = Zero Padding

Step 3: Encoder 處理
┌────────────────────────────────────────────────────────────┐
│ Sample 0: [F1 F2 ... F177 | G1 G2 ... G29 ]    206 frames  │
│ Sample 1: [F1 F2 ... F194 | G1 G2 ... G12 ]    206 frames  │
│ Sample 2: [F1 F2 ... F206 ]                    206 frames  │
└────────────────────────────────────────────────────────────┘
             F = 有效 Frame      G = Garbage Frame (from padding)

Step 4: Loss 計算 (問題!)
┌────────────────────────────────────────────────────────────┐
│ Loss = mean(MSE(student_all_206_frames, teacher_all_206))  │
│                                                            │
│ 實際上:                                                    │
│   Loss = (有效信號 Loss + Garbage Loss) / 206              │
│                                                            │
│   Garbage Loss: MSE(student_G, teacher_G)                  │
│   - Student G: 來自 [noisy + padding] 的 encoding          │
│   - Teacher G: 來自 [clean + padding] 的 encoding          │
│   - 這個 Loss 沒有意義! 模型學習匹配 padding noise         │
└────────────────────────────────────────────────────────────┘
```

### 問題 3: Loss 被稀釋

```
Loss 組成分析 (以 Sample 0 為例):

Total Frames: 206
Valid Frames: 177 (86%)
Garbage Frames: 29 (14%)

┌────────────────────────────────────────┐
│ Loss = (177 * valid_loss + 29 * garbage_loss) / 206 │
│                                        │
│ 如果:                                  │
│   valid_loss = 0.02 (想要優化的)       │
│   garbage_loss = 0.01 (無意義)         │
│                                        │
│ 那麼:                                  │
│   Total Loss = (177*0.02 + 29*0.01)/206 │
│              = (3.54 + 0.29) / 206     │
│              = 0.0186                   │
│                                        │
│ 問題: 14% 的 loss 來自 garbage!        │
│       模型會被誤導去優化無效目標       │
└────────────────────────────────────────┘
```

---

## 🛠️ 建議改善方案

### 方案 A: 資料層修復 (推薦)

#### A1. 統一 Noisy/Clean 長度

**修改位置**: `exp_1201/data.py` 的 `__getitem__`

```python
def __getitem__(self, idx):
    noisy = self.data[idx]['noisy_audio']
    clean = self.data[idx]['clean_audio']

    # ===== 新增: 統一長度 =====
    min_len = min(len(noisy), len(clean))
    noisy = noisy[:min_len]
    clean = clean[:min_len]
    # ==========================

    return {
        'noisy_audio': noisy,
        'clean_audio': clean,
        'length': min_len  # 新增: 記錄實際長度
    }
```

#### A2. Collate 函數返回長度資訊

**修改位置**: `exp_1201/data.py` 的 `collate_fn`

```python
def collate_fn(batch):
    """改進的 collate 函數"""

    # 記錄每個樣本的實際長度
    lengths = torch.tensor([item['length'] for item in batch])

    # Pad to max
    max_len = max(item['noisy_audio'].shape[0] for item in batch)

    noisy_list = []
    clean_list = []

    for item in batch:
        noisy = item['noisy_audio']
        clean = item['clean_audio']

        # Pad
        if len(noisy) < max_len:
            noisy = F.pad(noisy, (0, max_len - len(noisy)))
        if len(clean) < max_len:
            clean = F.pad(clean, (0, max_len - len(clean)))

        noisy_list.append(noisy)
        clean_list.append(clean)

    return {
        'noisy_audio': torch.stack(noisy_list),
        'clean_audio': torch.stack(clean_list),
        'lengths': lengths  # 新增: 返回長度資訊
    }
```

### 方案 B: Loss 層修復

#### B1. Masked Loss 計算

**修改位置**: `exp_1210/losses.py`

```python
def forward(self, student_out, teacher_out, lengths=None):
    """
    student_out: (B, D, T) - Student encoder output
    teacher_out: (B, D, T) - Teacher encoder output
    lengths: (B,) - 每個樣本的有效 frame 數 (optional)
    """
    B, D, T = student_out.shape

    if lengths is not None:
        # 計算每個樣本的有效 frame 數 (考慮 encoder stride)
        frame_lengths = lengths // ENCODER_STRIDE

        # 創建 mask: (B, T)
        mask = torch.arange(T, device=student_out.device).unsqueeze(0) < frame_lengths.unsqueeze(1)
        # mask shape: (B, T)

        # Expand mask for all dimensions: (B, D, T)
        mask = mask.unsqueeze(1).expand(-1, D, -1).float()

        # Masked MSE
        diff_sq = (student_out - teacher_out) ** 2  # (B, D, T)
        masked_diff = diff_sq * mask

        # 只對有效 frames 計算平均
        loss = masked_diff.sum() / mask.sum()
    else:
        # 原始無 mask 版本
        loss = F.mse_loss(student_out, teacher_out)

    return loss
```

#### B2. 訓練腳本修改

**修改位置**: `exp_1210/train_lora_v3.py`

```python
# 訓練循環中
for batch in dataloader:
    noisy = batch['noisy_audio'].to(device)
    clean = batch['clean_audio'].to(device)
    lengths = batch['lengths'].to(device)  # 新增

    output = model(noisy, clean)

    # 傳遞 lengths 給 loss
    loss = criterion(
        output['student_encoder_out'],
        output['teacher_encoder_out'],
        lengths=lengths  # 新增
    )
```

### 方案 C: 視覺化對照

```
修復前:
┌────────────────────────────────────────────────────────────┐
│ Sample 0: [████████████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]        │
│           [F1 F2 ... F177 | G1 G2 ... G29 ]                │
│           Loss = MSE(all 206 frames) ← 包含 garbage        │
└────────────────────────────────────────────────────────────┘

修復後:
┌────────────────────────────────────────────────────────────┐
│ Sample 0: [████████████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]        │
│           [F1 F2 ... F177 | G1 G2 ... G29 ]                │
│           Loss = MSE(only F1-F177) ← 只計算有效 frames     │
│                       ↑                                    │
│                  Mask = [1,1,...,1,0,0,...,0]              │
└────────────────────────────────────────────────────────────┘
```

---

## 📋 實施步驟

### 步驟 1: 驗證問題存在

```bash
# 執行驗證腳本
cd /home/sbplab/ruizi/WavTokenize-self-supervised
python -c "
import torch
cache = torch.load('/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt')
mismatches = 0
for i in range(min(100, len(cache))):
    diff = len(cache[i]['noisy_audio']) - len(cache[i]['clean_audio'])
    if diff != 0:
        mismatches += 1
        print(f'Sample {i}: diff={diff:+d}')
print(f'\\nTotal mismatches: {mismatches}/100')
"
```

### 步驟 2: 修改資料載入

```python
# exp_1201/data.py 修改
# 1. __getitem__ 統一長度
# 2. collate_fn 返回 lengths
```

### 步驟 3: 修改 Loss 函數

```python
# exp_1210/losses.py 修改
# 1. 添加 lengths 參數
# 2. 實現 masked loss
```

### 步驟 4: 修改訓練腳本

```python
# exp_1210/train_lora_v3.py 修改
# 1. 從 batch 獲取 lengths
# 2. 傳遞給 loss 函數
```

### 步驟 5: 驗證修復

```bash
# 執行短期訓練驗證
python train_lora_v3.py --num_epochs 5 --exp_name test_alignment_fix
```

---

## 📈 預期效果

| 修復前 | 修復後 | 改善 |
|--------|--------|------|
| Loss 包含 ~14% garbage | Loss 100% 有效 | 訓練信號更純淨 |
| 模型學習匹配 padding | 模型專注於去噪 | 目標更準確 |
| Feature Loss 與 Token Acc 不一致 | 兩者應該更一致 | 診斷更容易 |

---

## 🔗 相關檔案

| 檔案 | 用途 |
|------|------|
| `exp_1201/data.py:43-69` | Dataset `__getitem__` |
| `exp_1201/data.py:124-155` | `collate_fn` |
| `exp_1210/losses.py` | Loss 計算 |
| `exp_1210/train_lora_v3.py` | 訓練腳本 |
| `/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt` | 訓練資料 |

---

## 📝 附錄: 相關代碼位置

### 現有 collate_fn 代碼

```python
# exp_1201/data.py:124-155
def collate_fn(batch):
    """Find the longest audio and pad all to that length"""
    max_len = max(item['noisy_audio'].shape[0] for item in batch)

    for item in batch:
        noisy = noisy[:max_len]
        clean = clean[:max_len]
        # Pad if shorter
        if noisy.shape[0] < max_len:
            noisy = F.pad(noisy, (0, max_len - noisy.shape[0]))
        if clean.shape[0] < max_len:
            clean = F.pad(clean, (0, max_len - clean.shape[0]))

    # ... 返回 padded tensors，但沒有長度資訊
```

### 現有 Loss 計算

```python
# exp_1210/losses.py - Feature Loss
feature_loss = F.mse_loss(student_out, teacher_out)
# 沒有 mask，對所有 frames 計算
```

---

**報告完成**: 2025-12-11
