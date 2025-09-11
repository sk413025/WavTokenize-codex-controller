# WavTokenizer-Transformer 數據集分割修正記錄

## 實驗日期
2025-09-11

## 問題描述
WavTokenizer-Transformer訓練使用了錯誤的數據集分割方式，使用了隨機80/20分割而非按語者的固定分割。

## 修正前狀態
- 數據集總大小：1200個音頻對
- 分割方式：隨機80/20分割
- 訓練集大小：960個樣本  
- 驗證集大小：240個樣本
- 問題：不符合實驗設計要求

## 修正內容
修改 `wavtokenizer_transformer_denoising.py` 中的數據集分割邏輯：

```python
# 修正前（隨機分割）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 修正後（按語者分割）
train_indices = []
val_indices = []

for idx in range(len(dataset)):
    audio_data = audio_dataset.paired_files[idx]
    speaker = audio_data['speaker']
    
    if speaker in args.val_speakers:  # ['girl9', 'boy7']
        val_indices.append(idx)
    else:
        train_indices.append(idx)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
```

## 修正後狀態
- 數據集總大小：1200個音頻對
- 分割方式：按語者固定分割
- 訓練集大小：1000個樣本（10位語者×100句/人）
- 驗證集大小：200個樣本（2位語者×100句/人）
- 比例：1000:200 = 5:1

### 語者分配
**訓練集語者（10人）：**
- 女性：girl2, girl3, girl4, girl6
- 男性：boy1, boy3, boy4, boy5, boy6

**驗證集語者（2人）：**
- 女性：girl9 (100句)
- 男性：boy7 (100句)

## 驗證結果
訓練日誌確認分割正確：
```
2025-09-11 00:40:56,829 - INFO - 訓練集大小: 1000, 驗證集大小: 200
```

## 實驗意義
- 確保了跨語者泛化能力的正確評估
- 驗證集語者在訓練過程中完全未見過
- 符合語音降噪實驗的標準設計原則
- 提供了可靠的模型泛化性能指標

## 相關文件
- 修改文件：`wavtokenizer_transformer_denoising.py`
- 訓練腳本：`run_discrete_crossentropy.sh`
- 數據集類別：`ttdata.py`
