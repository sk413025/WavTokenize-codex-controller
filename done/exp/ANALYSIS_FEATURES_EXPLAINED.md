# 訓練分析功能說明

## 🎯 新增功能

你要求的兩個分析功能已經實現：

### 1. Token 預測分布分析
**目的**：檢測模型是否總是預測同一種 token（過擬合信號）

### 2. Speaker Embedding 記錄與視覺化
**目的**：看訓練集（14位）和驗證集（4位）語者是否在embedding空間中分開

---

## 📊 功能詳解

### 1. Token 預測分布分析

#### 記錄內容（每個 epoch）

```log
Token 預測分析:
  - 唯一 token 數: 2345/4096 (57.23%)
  - 預測熵 (多樣性): 6.8234
  - 最常見 token: 1523 (佔比 2.34%)
```

#### 指標說明

| 指標 | 含義 | 正常範圍 | 警告信號 |
|------|------|---------|---------|
| **唯一 token 數** | 模型使用了多少個不同的 token | 1500-3500 (37-85%) | <1000 或 >3800 |
| **預測熵** | 預測的多樣性（越高越好） | 5.0-8.0 | <4.0 |
| **最常見 token 佔比** | 單個 token 出現的比例 | <5% | >10% |

#### 警告信號

如果看到這些信號，表示模型可能過擬合：

```log
⚠️  警告: >50% 的預測都是同一個 token!
```

**解讀**：
- ✅ **健康**：唯一token數 >2000，熵 >6.0，最常見token <5%
- ⚠️ **需注意**：唯一token數 <1500，熵 <5.0，最常見token >10%
- ❌ **嚴重問題**：唯一token數 <1000，熵 <4.0，最常見token >50%

---

### 2. Speaker Embedding 分析

#### 自動生成文件

訓練前和每 50 epochs 會生成：

```
results/zeroshot_full_cached_*/
├── speaker_analysis/
│   ├── train_embeddings.npy         # 訓練集 speaker embeddings
│   ├── val_embeddings.npy           # 驗證集 speaker embeddings
│   └── speaker_distribution_tsne.png # t-SNE 視覺化圖
└── token_diversity_history.json     # Token 多樣性歷史
```

#### t-SNE 視覺化圖解讀

![示意圖](speaker_distribution_tsne.png)

**圖例**：
- 🔵 藍點：訓練集語者（14位）
- 🔴 紅點：驗證集語者（4位）

**理想情況**：
```
✅ 訓練集和驗證集語者形成不同的群集
   - 表示 ECAPA 能區分已知/未知語者
   - Zero-shot 能力有基礎

✅ 同一語者的點聚集在一起
   - 表示 speaker embedding 一致性好
```

**問題情況**：
```
❌ 訓練集和驗證集完全混在一起
   - 表示 speaker embedding 質量不佳
   - ECAPA 可能需要 fine-tune

❌ 所有點都擠在一起
   - 表示 speaker encoder 沒有區分能力
   - 需要檢查 ECAPA 配置
```

---

## 📁 輸出文件詳解

### token_diversity_history.json

```json
[
  {
    "epoch": 1,
    "train_entropy": 6.234,
    "val_entropy": 5.987,
    "train_unique_ratio": 0.523,
    "val_unique_ratio": 0.498
  },
  {
    "epoch": 2,
    ...
  }
]
```

**用途**：
- 追蹤 token 多樣性隨訓練的變化
- 檢測過擬合趨勢（驗證集熵下降）

### Speaker Embeddings (.npy)

```python
# 讀取方式
import numpy as np

train_embs = np.load('train_embeddings.npy')  # (N_train, 256)
val_embs = np.load('val_embeddings.npy')      # (N_val, 256)

# 分析
print(f"訓練集 embeddings 形狀: {train_embs.shape}")
print(f"驗證集 embeddings 形狀: {val_embs.shape}")

# 計算語者間距離
from scipy.spatial.distance import cosine
dist = cosine(train_embs[0], val_embs[0])
print(f"語者間距離: {dist:.4f}")
```

---

## 🔍 如何使用

### 執行訓練（已自動啟用分析）

```bash
cd /home/sbplab/ruizi/c_code/done/exp

# 使用增強版訓練腳本（已自動啟用）
bash run_zeroshot_full_cached.sh
```

### 查看實時分析

```bash
# 查看 token 分析
tail -f results/zeroshot_full_cached_*/training.log | grep "Token 預測分析"

# 示例輸出：
# 2025-11-03 23:30:15 - INFO -   Token 預測分析:
# 2025-11-03 23:30:15 - INFO -     - 唯一 token 數: 2345/4096 (57.23%)
# 2025-11-03 23:30:15 - INFO -     - 預測熵: 6.8234
# 2025-11-03 23:30:15 - INFO -     - 最常見 token: 1523 (佔比 2.34%)
```

### 查看 Speaker 視覺化

訓練開始後立即生成（無需等待訓練完成）：

```bash
# 圖片位置
results/zeroshot_full_cached_*/speaker_analysis/speaker_distribution_tsne.png
```

用圖片查看器打開即可查看語者分布。

---

## 📊 進階分析腳本

### 繪製 Token 多樣性趨勢圖

創建 `plot_token_diversity.py`:

```python
import json
import matplotlib.pyplot as plt

# 讀取歷史
with open('token_diversity_history.json', 'r') as f:
    history = json.load(f)

epochs = [h['epoch'] for h in history]
train_entropy = [h['train_entropy'] for h in history]
val_entropy = [h['val_entropy'] for h in history]

# 繪圖
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_entropy, 'b-', label='Train Entropy', linewidth=2)
plt.plot(epochs, val_entropy, 'r-', label='Val Entropy', linewidth=2)
plt.axhline(y=6.0, color='gray', linestyle='--', label='Healthy Threshold (6.0)')
plt.xlabel('Epoch')
plt.ylabel('Prediction Entropy')
plt.title('Token Prediction Diversity over Training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('token_diversity_trend.png', dpi=150)
plt.close()
```

### 分析 Speaker Embedding 聚類

創建 `analyze_speaker_clusters.py`:

```python
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# 讀取 embeddings
train_embs = np.load('train_embeddings.npy')
val_embs = np.load('val_embeddings.npy')

# 合併
all_embs = np.concatenate([train_embs, val_embs])
labels = np.array([0]*len(train_embs) + [1]*len(val_embs))  # 0=train, 1=val

# 計算分離度（Silhouette Score）
score = silhouette_score(all_embs, labels)
print(f"訓練/驗證集分離度: {score:.4f}")
print(f"  - 接近 1: 非常分開（理想）")
print(f"  - 接近 0: 混在一起（需改進）")
print(f"  - 小於 0: 嚴重混淆（有問題）")

# K-means 聚類（假設18位語者）
kmeans = KMeans(n_clusters=18, random_state=42)
clusters = kmeans.fit_predict(all_embs)

print(f"\nK-means 聚類結果:")
print(f"  - 訓練集語者分布到 {len(np.unique(clusters[:len(train_embs)]))} 個群集")
print(f"  - 驗證集語者分布到 {len(np.unique(clusters[len(train_embs):]))} 個群集")
```

---

## 🎯 分析目標

### Token 預測分析目標

| Epoch | 預期熵 | 預期唯一token比例 | 評價 |
|-------|--------|-----------------|------|
| 1-10 | 4.0-5.5 | 30-50% | 初期學習 |
| 10-50 | 5.5-7.0 | 40-65% | 穩定提升 |
| 50-100 | 6.5-7.5 | 50-70% | 健康收斂 |

**警告信號**：
- Epoch 50 熵 <5.0：可能過擬合
- Epoch 50 唯一token <40%：預測過於單一

### Speaker Embedding 分析目標

**成功標準**：
1. ✅ t-SNE 圖中訓練/驗證語者形成視覺上可區分的群組
2. ✅ Silhouette Score > 0.2（表示有一定的分離度）
3. ✅ 同一語者的多個樣本聚集在一起

**失敗信號**：
1. ❌ t-SNE 圖中所有點完全混在一起
2. ❌ Silhouette Score < 0（表示混淆嚴重）
3. ❌ 同一語者的樣本分散在不同位置

---

## 📝 實驗報告模板

訓練完成後，可以這樣總結：

```markdown
## Token 預測分析

- 最終訓練熵: 7.12 ✅ (>6.5, 健康)
- 最終驗證熵: 6.89 ✅ (>6.5, 健康)
- 唯一 token 比例: 62.3% ✅ (50-70%, 正常)
- 最常見 token 佔比: 1.8% ✅ (<5%, 良好)

**結論**: Token 預測多樣性良好，無明顯過擬合到單一 token 的問題。

## Speaker Embedding 分析

- 訓練集語者: 14位，1000個樣本
- 驗證集語者: 4位，500個樣本
- Silhouette Score: 0.34 ✅ (>0.2, 可區分)

**t-SNE 視覺化觀察**:
- ✅ 訓練集和驗證集語者形成不同群組
- ✅ 群組內語者 embedding 一致性高
- ✅ ECAPA 能有效區分不同語者

**結論**: Zero-shot 條件具備，ECAPA speaker encoder 工作正常。
```

---

## 🚀 執行方式

### 使用更新後的腳本

```bash
cd /home/sbplab/ruizi/c_code/done/exp

# 執行（已自動啟用所有分析功能）
bash run_zeroshot_full_cached.sh
```

### 參數說明

```bash
python train_zeroshot_full_cached_analysis.py \
    --analyze_speakers \              # 啟用 speaker 分析
    --speaker_analysis_freq 50        # 每 50 epochs 分析一次
```

---

生成時間: 2025-11-03 23:20
功能狀態: 已實現並整合到訓練腳本
執行方式: bash run_zeroshot_full_cached.sh
