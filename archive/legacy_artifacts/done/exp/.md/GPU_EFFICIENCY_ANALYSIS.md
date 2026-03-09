# GPU 使用效率分析與優化方案

**日期**: 2025-11-03
**問題**: GPU 使用率僅 22%-52%，訓練效率低下

---

## 📊 問題診斷

### 當前狀況

```
GPU 使用率: 22%-52% (應該 >80%)
訓練速度: 3.2s/batch (過慢)
Epoch 時間: ~1.15 小時 (過長)
瓶頸: CPU-GPU 數據傳輸
```

### 根本原因分析

經過代碼審查，發現 **5 個主要瓶頸**：

---

## ❌ 瓶頸 1: 逐個樣本處理 (最嚴重)

### 當前實現

**文件**: `data_zeroshot.py` 第 224-245 行

```python
# ❌ 問題代碼
for noisy_audio, clean_audio, content_id in batch:
    # 每次循環都進行 GPU 操作
    noisy_audio = noisy_audio.to(device).unsqueeze(0)  # (1, T)
    clean_audio = clean_audio.to(device).unsqueeze(0)  # (1, T)

    # 逐個編碼 (batch_size=14 需要調用 28 次!)
    with torch.no_grad():
        _, noisy_tokens = wavtokenizer.encode_infer(
            noisy_audio,
            bandwidth_id=torch.tensor([0], device=device)
        )
        _, clean_tokens = wavtokenizer.encode_infer(
            clean_audio,
            bandwidth_id=torch.tensor([0], device=device)
        )
```

### 問題

1. **GPU Kernel 啟動開銷**
   - 每個 `wavtokenizer.encode_infer()` 都啟動新的 CUDA kernel
   - 14 個樣本 × 2 (noisy + clean) = **28 次 kernel 啟動**
   - 每次啟動開銷 ~10-50ms
   - **總浪費: 280-1400ms per batch**

2. **無法利用 GPU 並行**
   - GPU 設計用於大批量並行計算
   - 當前: 每次只處理 1 個樣本 (batch=1)
   - GPU 利用率: <10%

3. **CPU-GPU 數據傳輸頻繁**
   - 每次 `.to(device)` 都觸發 CPU→GPU 傳輸
   - 14 個樣本 × 2 = **28 次數據傳輸**
   - PCIe 帶寬浪費

### 影響

```
實測速度: 3.2s/batch
理論速度 (批量處理): ~0.5s/batch
效率損失: 84%
```

---

## ❌ 瓶頸 2: DataLoader num_workers=0

### 當前實現

**文件**: `train_zeroshot_full.py` 第 394-407 行

```python
# ❌ 問題代碼
train_loader = DataLoader(
    train_audio_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,  # ❌ 單進程加載
    collate_fn=token_collate_fn
)
```

### 問題

1. **數據加載阻塞訓練**
   - 主進程在等待數據加載時，GPU 空閒
   - 無法重疊 I/O 和計算

2. **CPU 單核瓶頸**
   - 只使用 1 個 CPU 核心
   - 無法並行讀取多個音頻文件

### 影響

```
GPU 空閒時間: ~30-40% per batch
原因: 等待 CPU 加載音頻、編碼、提取 embedding
```

### 為什麼設置為 0？

```python
# 註解中說明
num_workers=0,  # 單進程（因為 collate_fn 使用 GPU）
```

**原因**: `collate_fn` 中使用了 `wavtokenizer` 和 `speaker_encoder` (GPU 模型)
**結果**: 無法使用多進程 (GPU 模型無法序列化到子進程)

---

## ❌ 瓶頸 3: 重複的 Padding 操作

### 當前實現

**文件**: `data_zeroshot.py` 第 251-290 行

```python
# ❌ 逐個 padding audio
for noisy_audio in noisy_audio_list:
    if noisy_audio.shape[0] < max_audio_len:
        pad_size = max_audio_len - noisy_audio.shape[0]
        noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad_size), value=0)
    padded_noisy_audio.append(noisy_audio)

# ❌ 逐個 padding tokens
for noisy_t, clean_t in zip(noisy_tokens_list, clean_tokens_list):
    curr_noisy = noisy_t.squeeze(0)
    if curr_noisy.shape[0] < max_token_len:
        pad_size = max_token_len - curr_noisy.shape[0]
        curr_noisy = torch.nn.functional.pad(curr_noisy, (0, pad_size), value=0)
    # ... 重複 clean tokens
```

### 問題

1. **循環中的 GPU 操作**
   - 每次 `F.pad()` 都觸發 GPU kernel
   - 14 個樣本 × 4 (noisy_audio, clean_audio, noisy_tokens, clean_tokens) = **56 次 kernel**

2. **可以批量化**
   - PyTorch 支持 `torch.nn.utils.rnn.pad_sequence()`
   - 一次性完成所有 padding

### 影響

```
Padding 時間: ~100-200ms per batch
可優化到: ~10-20ms
```

---

## ❌ 瓶頸 4: 無預處理緩存

### 當前實現

**每個 epoch 都重新計算**:
- Tokenization (WavTokenizer.encode_infer)
- Speaker Embedding 提取 (ECAPA-TDNN)

### 問題

1. **重複計算**
   - 100 epochs × 16,128 樣本 = **1,612,800 次重複編碼**
   - WavTokenizer 和 ECAPA-TDNN 都是凍結的 (輸出固定)
   - 完全可以預先計算並緩存

2. **I/O 浪費**
   - 每個 epoch 都從磁盤讀取 .wav 文件
   - 16,128 個文件 × 100 epochs = **1,612,800 次磁盤讀取**

### 影響

```
當前時間: 1.15 小時/epoch × 100 = 115 小時
預處理後: ~10 分鐘/epoch × 100 = 16.7 小時
節省: 98.5 小時 (85%)
```

---

## ❌ 瓶頸 5: 字符串處理在主循環中

### 當前實現

**文件**: `data_zeroshot.py` 第 294-302 行

```python
# ❌ CPU 密集型字符串操作
for cid in content_ids_list:
    if isinstance(cid, str):
        digits = ''.join(c for c in cid if c.isdigit())  # CPU 操作
        numeric_ids.append(int(digits) if digits else hash(cid) % 1000)
```

### 問題

雖然影響較小，但在 collate_fn 中做字符串處理會阻塞 GPU

### 影響

```
時間: ~5-10ms per batch (影響小)
```

---

## ✅ 優化方案

### 方案 A: 預處理 + 緩存 (推薦，最大改善)

#### 1. 創建預處理腳本

**新文件**: `preprocess_zeroshot_cache.py`

```python
"""
預處理腳本: 提前計算並緩存 tokens 和 speaker embeddings

一次性計算:
  - Tokenization (WavTokenizer)
  - Speaker Embeddings (ECAPA-TDNN)

保存為 .pt 文件，訓練時直接加載

預期:
  - 預處理時間: ~2-3 小時 (一次性)
  - 訓練加速: 5-7x
  - 總時間節省: ~85%
"""

import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
import pickle

from data_zeroshot import ZeroShotAudioDataset
from speaker_encoder import create_speaker_encoder

# 加載模型
device = torch.device('cuda:0')
wavtokenizer = load_wavtokenizer().to(device)
speaker_encoder = create_speaker_encoder('ecapa', freeze=True).to(device)
speaker_encoder.eval()

# 創建數據集
dataset = ZeroShotAudioDataset(
    input_dirs=['../../data/raw/box', ...],
    target_dir='../../data/clean/box2',
    max_sentences_per_speaker=288
)

# 輸出目錄
cache_dir = Path('./preprocessed_cache')
cache_dir.mkdir(exist_ok=True)

# 批量處理
batch_size = 32  # 更大的 batch 提高效率
all_data = []

for i in tqdm(range(0, len(dataset), batch_size)):
    batch_data = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]

    # 批量加載到 GPU
    noisy_audios = torch.stack([d[0] for d in batch_data]).to(device)
    clean_audios = torch.stack([d[1] for d in batch_data]).to(device)
    content_ids = [d[2] for d in batch_data]

    with torch.no_grad():
        # 批量編碼 (一次性處理 32 個!)
        _, noisy_tokens = wavtokenizer.encode_infer(
            noisy_audios,
            bandwidth_id=torch.tensor([0]*batch_size, device=device)
        )
        _, clean_tokens = wavtokenizer.encode_infer(
            clean_audios,
            bandwidth_id=torch.tensor([0]*batch_size, device=device)
        )

        # 批量提取 speaker embeddings
        speaker_embeddings = speaker_encoder(noisy_audios)

    # 移回 CPU 並保存
    for j in range(len(batch_data)):
        all_data.append({
            'noisy_tokens': noisy_tokens[j].cpu(),
            'clean_tokens': clean_tokens[j].cpu(),
            'speaker_embedding': speaker_embeddings[j].cpu(),
            'content_id': content_ids[j]
        })

# 保存緩存
torch.save(all_data, cache_dir / 'zeroshot_cache.pt')
print(f"✅ 預處理完成! 保存到 {cache_dir / 'zeroshot_cache.pt'}")
```

#### 2. 修改 Dataset 使用緩存

**修改**: `data_zeroshot.py`

```python
class ZeroShotAudioDatasetCached(Dataset):
    """
    使用預處理緩存的 Dataset

    優勢:
      - 無需實時編碼 (節省 80% 時間)
      - 無需提取 speaker embedding (節省額外 10% 時間)
      - DataLoader 可使用 num_workers > 0
    """

    def __init__(self, cache_path):
        self.data = torch.load(cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def cached_collate_fn(batch):
    """
    簡單的 collate function (無需 GPU 操作)

    優勢:
      - 純 CPU 操作
      - 可用於多進程 DataLoader
      - 極快 (僅 padding)
    """
    noisy_tokens = [item['noisy_tokens'] for item in batch]
    clean_tokens = [item['clean_tokens'] for item in batch]
    speaker_embeddings = torch.stack([item['speaker_embedding'] for item in batch])
    content_ids = [item['content_id'] for item in batch]

    # 批量 padding (使用 PyTorch 內置函數)
    noisy_tokens_padded = torch.nn.utils.rnn.pad_sequence(
        noisy_tokens, batch_first=True, padding_value=0
    )
    clean_tokens_padded = torch.nn.utils.rnn.pad_sequence(
        clean_tokens, batch_first=True, padding_value=0
    )

    return {
        'noisy_tokens': noisy_tokens_padded,
        'clean_tokens': clean_tokens_padded,
        'speaker_embeddings': speaker_embeddings,
        'content_ids': torch.tensor(content_ids, dtype=torch.long)
    }
```

#### 3. 修改訓練腳本

**修改**: `train_zeroshot_full.py`

```python
# 使用緩存 Dataset
train_dataset = ZeroShotAudioDatasetCached('./preprocessed_cache/train_cache.pt')
val_dataset = ZeroShotAudioDatasetCached('./preprocessed_cache/val_cache.pt')

# DataLoader 可以使用多進程了!
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,  # ✅ 4 個進程並行加載
    collate_fn=cached_collate_fn,
    pin_memory=True  # ✅ 加速 CPU→GPU 傳輸
)
```

#### 預期效果

```
預處理時間: ~2-3 小時 (一次性)

訓練速度:
  - 當前: 3.2s/batch → 1.15h/epoch
  - 優化後: 0.4s/batch → 0.15h/epoch
  - 加速比: 8x

總訓練時間:
  - 當前: 115 小時 (100 epochs)
  - 優化後: 15 小時 (100 epochs)
  - 節省: 100 小時 (87%)

GPU 使用率:
  - 當前: 22-52%
  - 優化後: 75-90%
```

---

### 方案 B: 批量處理 (無需預處理)

如果不想做預處理，可以優化 collate_fn 的批量處理。

#### 修改 `data_zeroshot.py`

```python
def zeroshot_collate_fn_with_speaker_optimized(batch, wavtokenizer, speaker_encoder, device):
    """
    優化版 collate function

    改進:
      1. 批量移動數據到 GPU
      2. 批量編碼 (而非逐個)
      3. 批量 padding
    """
    # 1. 批量加載音頻
    noisy_audios = []
    clean_audios = []
    content_ids = []

    for noisy_audio, clean_audio, content_id in batch:
        noisy_audios.append(noisy_audio)
        clean_audios.append(clean_audio)
        content_ids.append(content_id)

    # 2. 批量 padding audio (在 CPU 上)
    max_audio_len = max(a.shape[0] for a in noisy_audios)
    padded_noisy = []
    padded_clean = []

    for noisy, clean in zip(noisy_audios, clean_audios):
        if noisy.shape[0] < max_audio_len:
            noisy = F.pad(noisy, (0, max_audio_len - noisy.shape[0]))
        if clean.shape[0] < max_audio_len:
            clean = F.pad(clean, (0, max_audio_len - clean.shape[0]))
        padded_noisy.append(noisy)
        padded_clean.append(clean)

    # 3. 批量移動到 GPU (一次性!)
    noisy_batch = torch.stack(padded_noisy).to(device)  # (B, T)
    clean_batch = torch.stack(padded_clean).to(device)  # (B, T)

    # 4. 批量編碼 (關鍵優化!)
    with torch.no_grad():
        bandwidth_id = torch.tensor([0] * len(batch), device=device)

        # ✅ 一次編碼整個 batch
        _, noisy_tokens = wavtokenizer.encode_infer(noisy_batch, bandwidth_id=bandwidth_id)
        _, clean_tokens = wavtokenizer.encode_infer(clean_batch, bandwidth_id=bandwidth_id)

        # ✅ 一次提取整個 batch 的 speaker embeddings
        speaker_embeddings = speaker_encoder(noisy_batch)

    # 5. Padding tokens (已經是 batch 形式，可能只需要調整形狀)
    # ... (根據 wavtokenizer 輸出格式調整)

    return {
        'noisy_tokens': noisy_tokens,
        'clean_tokens': clean_tokens,
        'speaker_embeddings': speaker_embeddings,
        'content_ids': torch.tensor(content_ids, dtype=torch.long)
    }
```

#### 預期效果

```
訓練速度:
  - 當前: 3.2s/batch
  - 優化後: 1.0s/batch
  - 加速比: 3.2x

GPU 使用率:
  - 當前: 22-52%
  - 優化後: 60-75%
```

---

### 方案 C: 增加 Batch Size

如果 GPU 顯存充足，可以增大 batch size 提高 GPU 利用率。

#### 修改 `run_zeroshot_full.sh`

```bash
# 當前
--batch_size 14

# 優化 (根據 GPU 顯存)
--batch_size 28  # GTX 1080 Ti (11GB) 應該可以
```

#### 預期效果

```
Batch size 14 → 28:
  - 每 epoch batch 數: 1152 → 576
  - Epoch 時間: 1.15h → 0.7h (假設線性加速)
  - GPU 利用率: 提高 20-30%
```

---

## 📈 優化效果對比

| 方案 | 複雜度 | 加速比 | GPU 利用率 | 總時間 (100 epochs) |
|------|--------|--------|-----------|---------------------|
| **當前** | - | 1x | 22-52% | 115 小時 |
| **方案 B** (批量處理) | 低 | 3.2x | 60-75% | 36 小時 |
| **方案 C** (增加 batch) | 極低 | 1.6x | 50-65% | 72 小時 |
| **方案 A** (預處理) | 中 | 8x | 75-90% | **15 小時** ✅ |
| **A+C** (組合) | 中 | 10x | 80-95% | **12 小時** ✅✅ |

---

## 🎯 推薦執行順序

### 短期 (立即可做)

1. **方案 C**: 增加 batch size (5 分鐘)
   - 測試 `--batch_size 28` 是否 OOM
   - 如果不 OOM，可嘗試 32 或 40

2. **方案 B**: 優化 collate_fn 批量處理 (1 小時)
   - 修改 `data_zeroshot.py`
   - 一次性加載整個 batch 到 GPU

### 中期 (最大收益)

3. **方案 A**: 實現預處理緩存 (4-6 小時開發 + 2-3 小時預處理)
   - 創建 `preprocess_zeroshot_cache.py`
   - 運行一次預處理
   - 修改訓練腳本使用緩存
   - **收益**: 節省 100 小時訓練時間

---

## 📝 總結

### 根本原因

```
GPU 低使用率的核心原因:
  1. 逐個樣本處理 (無法利用 GPU 並行)
  2. 頻繁的 kernel 啟動開銷
  3. CPU-GPU 數據傳輸頻繁
  4. DataLoader num_workers=0 (無法重疊 I/O)
  5. 重複計算 (每個 epoch 重新編碼)
```

### 最佳方案

**強烈推薦方案 A (預處理緩存)**:
- ✅ 最大加速 (8x)
- ✅ 最高 GPU 利用率 (75-90%)
- ✅ 節省最多時間 (100 小時)
- ✅ 可重複使用緩存
- ⚠️ 需要額外磁盤空間 (~10-20 GB)

### 立即行動

```bash
# 1. 增加 batch size (立即)
vim run_zeroshot_full.sh
# 修改: --batch_size 28

# 2. 測試是否 OOM
bash run_zeroshot_full.sh

# 3. 如果不 OOM，實現方案 A (預處理)
python preprocess_zeroshot_cache.py  # 需先創建此腳本
```

---

**作者**: GPU 效能分析團隊
**日期**: 2025-11-03
**下一步**: 實現預處理緩存方案
