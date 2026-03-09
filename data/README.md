# Data Directory

## 數據來源

從 `/home/sbplab/ruizi/c_code/done/exp/data3/` 過濾並分割而來。

## 過濾規則

- ✅ **過濾 clean→clean 配對** (4,896 個)
- ✅ **保持 train/val 語者分割**
- ✅ **包含 3 種材質** (box, papercup, plastic)

## 數據統計

### Train 集 (train_cache_filtered.pt)

- **樣本數**: 10,368
- **語者數**: 14 (boy1, boy3-6, boy9-10, girl2-4, girl7-8, girl10-11)
- **材質分布**:
  - box: 4,032 樣本 (288 句 × 14 語者)
  - papercup: 3,744 樣本 (288 句 × 13 語者)
  - plastic: 2,592 樣本 (288 句 × 9 語者)

### Val 集 (val_cache_filtered.pt)

- **樣本數**: 1,728
- **語者數**: 3 (boy7, boy8, girl9)
- **材質分布**:
  - box: 864 樣本 (288 句 × 3 語者)
  - papercup: 576 樣本 (288 句 × 2 語者)
  - plastic: 288 樣本 (288 句 × 1 語者)

## 數據結構

每個樣本包含：

```python
{
    'noisy_audio': Tensor,        # (1, T_audio) - 帶噪音音檔 (Student 輸入)
    'clean_audio': Tensor,        # (1, T_audio) - 乾淨音檔 (Teacher 輸入)
    'noisy_tokens': Tensor,       # (T,) - Noisy token indices
    'clean_tokens': Tensor,       # (T,) - Clean token indices
    'noisy_distances': Tensor,    # (T, 4096) - Noisy distance matrix
    'clean_distances': Tensor,    # (T, 4096) - Clean distance matrix
    'speaker_id': str,            # 語者 ID
    'material': str,              # 材質 (box/papercup/plastic)
    'sentence_id': str,           # 句子 ID (應從檔名提取，data3 原始為 'nor')
    'noisy_path': str,            # Noisy 音檔路徑
    'clean_path': str,            # Clean 音檔路徑
    'filename': str,              # 檔案名稱
}
```

## 檔名格式

```
nor_<speaker>_<material>_LDV_<sentence_id>.wav

例如:
  nor_boy7_box_LDV_010.wav
  └─ boy7: 語者
     └─ box: 材質
        └─ 010: 句子編號 (001-288)
```

## 使用方式

```python
import torch

# 載入數據
train_data = torch.load('data/train_cache_filtered.pt')
val_data = torch.load('data/val_cache_filtered.pt')

# 訪問樣本
sample = train_data[0]
noisy_audio = sample['noisy_audio']  # Student 輸入
clean_audio = sample['clean_audio']  # Teacher 輸入
```

## 相關腳本

- **quarantine/python/data/filter_data3.py**: 從 data3 過濾並分割數據（已移出 active surface）
- **DATA_PREPROCESSING_SUMMARY.md**: 完整前處理說明

## 更新日期

2026-02-09
