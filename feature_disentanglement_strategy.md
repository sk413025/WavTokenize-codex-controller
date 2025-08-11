# 特徵解離與增強策略分析

## 實驗編號: EXP02-Analysis
## 日期: 2025-08-03

## 實驗結果概述

根據我們的實驗結果，WavTokenizer模型中的單一離散量化層包含了以下信息：

1. **說話者特徵**: 主導信息（影響分數 1.0046）
2. **內容特徵**: 次要信息（影響分數 1.0012）
3. **噪聲特徵**: 幾乎不存在（影響分數 0.0000）

這表明當前的單層量化結構主要捕獲了說話者身份和語音內容信息，而噪聲信息幾乎不被編碼在離散token中。

## 針對語音增強任務的特徵解離策略

由於我們的目標是去除噪聲同時保留說話者特徵和語音內容，根據分析結果，我們可以採取以下策略：

### 方案1: 特徵重塑策略

利用離散編碼已經很少包含噪聲信息的特性，結合使用專門的特徵重塑網絡來進一步提純所需特徵：

```python
class FeatureReshapingNetwork(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024):
        super().__init__()
        # 特徵提取和重塑網絡
        self.feature_reshape = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
        )
        
    def forward(self, discrete_features):
        # 應用特徵重塑
        enhanced_features = self.feature_reshape(discrete_features)
        # 使用殘差連接，穩定訓練
        enhanced_features = enhanced_features + discrete_features
        return enhanced_features
```

### 方案2: 自監督對比學習

設計特定的損失函數，通過對比學習更好地區分內容和說話者特徵與噪聲特徵：

```python
def contrastive_learning_loss(enhanced, clean, noisy, temperature=0.5):
    # 計算與乾淨語音的相似度（正樣本）
    positive_sim = F.cosine_similarity(enhanced, clean, dim=1)
    
    # 計算與噪聲部分的相似度（負樣本）
    noise = noisy - clean
    negative_sim = F.cosine_similarity(enhanced, noise, dim=1)
    
    # 對比損失：鼓勵與乾淨語音相似，與噪聲不同
    loss = -torch.log(
        torch.exp(positive_sim/temperature) / 
        (torch.exp(positive_sim/temperature) + torch.exp(negative_sim/temperature))
    )
    
    return loss.mean()
```

### 方案3: 特徵分離網絡

設計一個網絡來明確分離內容、說話者和噪聲特徵，然後重建時只使用前兩者：

```python
class FeatureDisentanglementNetwork(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024):
        super().__init__()
        # 共享特徵提取器
        self.shared_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 特定特徵提取器
        self.content_extractor = nn.Conv1d(hidden_dim, input_dim//2, kernel_size=1)
        self.speaker_extractor = nn.Conv1d(hidden_dim, input_dim//2, kernel_size=1)
        self.noise_extractor = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        
        # 特徵重建器（只使用內容和說話者特徵）
        self.reconstructor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        )
    
    def forward(self, x):
        # 共享編碼
        shared_features = self.shared_encoder(x)
        
        # 特徵分離
        content_features = self.content_extractor(shared_features)
        speaker_features = self.speaker_extractor(shared_features)
        noise_features = self.noise_extractor(shared_features)
        
        # 合併內容和說話者特徵，不使用噪聲特徵
        combined = torch.cat([content_features, speaker_features], dim=1)
        
        # 重建增強特徵
        enhanced = self.reconstructor(combined)
        
        return enhanced, {
            'content': content_features,
            'speaker': speaker_features,
            'noise': noise_features
        }
```

## 訓練策略建議

根據我們的分析結果，建議採用以下訓練策略:

1. **多目標損失函數**:
   - 內容保留損失：使用MSE或交叉熵確保內容信息保留
   - 說話者保留損失：使用餘弦相似度確保說話者特徵保留
   - 噪聲抑制損失：鼓勵去除噪聲部分

2. **數據增強**:
   - 使用同一說話者不同內容的配對
   - 使用不同說話者相同內容的配對
   - 使用乾淨/帶噪聲版本的配對

3. **評估指標**:
   - PESQ和STOI：評估語音質量和清晰度
   - 說話者驗證準確率：評估說話者特徵保留
   - 語音識別準確率：評估內容特徵保留

## 後續研究方向

1. **多層WavTokenizer**：考慮擴展模型至多個量化層，更好地分離不同類型的信息
2. **特徵分離算法**：探索更先進的特徵分離算法，如變分自編碼器(VAE)或規範化流
3. **任務特定預訓練**：針對語音增強任務進行特定預訓練，優化特徵表示

## 結論

當前的WavTokenizer單層量化模型已經在特徵分離方面表現出一定能力，尤其是噪聲信息幾乎不被編碼在離散token中。這為我們的語音增強任務提供了良好的基礎。通過添加專門的特徵提純網絡，我們可以進一步增強這種特徵分離能力，實現更好的去噪效果同時保留語音內容和說話者身份信息。
