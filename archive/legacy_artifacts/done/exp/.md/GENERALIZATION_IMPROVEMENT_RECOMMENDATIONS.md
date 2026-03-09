# Zero-Shot Speaker Denoising 泛化性改進建議

## 📊 當前性能分析

### 核心指標 (Epoch 49/100)
```
訓練集準確率: 54.95%
驗證集準確率: 38.71%
泛化差距 (Gap): 16.24%
Baseline: 38.19%
提升: +0.52%
```

### 關鍵發現

✅ **優點:**
1. **已超越 Baseline** (+0.52%)
2. **過擬合趨勢穩定** (早期 17.63% → 近期 16.32%)
3. **Loss 持續下降** (Train: 4.63→2.79, Val: 5.56→4.86)
4. **訓練速度優秀** (23x 加速)

⚠️ **待改善:**
1. **泛化差距較大** (16.24% gap)
2. **驗證集 Loss 下降緩慢** (僅降 0.70 vs 訓練集 1.83)
3. **模型可能對訓練數據記憶過強**

---

## 🎯 改進建議（按優先級排序）

### 【高優先級】1. 數據增強 (Data Augmentation)

**問題診斷:**
- 16,128 訓練樣本，泛化差距 16%，表示模型可能過度擬合訓練集的特定模式
- Token-based denoising 缺乏對未見過音頻變化的魯棒性

**具體實施方案:**

#### A. **Token-Level Augmentation** ⭐⭐⭐⭐⭐
```python
# 在 data_zeroshot.py 的 collate_fn 中添加
def augment_tokens(tokens, aug_prob=0.15):
    """
    Token 級別的增強
    - Random masking: 隨機遮蔽部分 token
    - Token mixup: 混合相鄰 token
    - Cutout: 隨機刪除連續片段
    """
    if random.random() < aug_prob:
        # 1. Random Token Masking (10% tokens)
        mask_prob = 0.1
        mask = torch.rand(tokens.shape) < mask_prob
        tokens = tokens.clone()
        tokens[mask] = random.randint(0, 4095)

    return tokens
```

**預期效果:**
- 泛化差距降低 3-5%
- 迫使模型學習更魯棒的特徵，而非記憶特定 token 序列

#### B. **Speaker Embedding Perturbation** ⭐⭐⭐⭐
```python
def augment_speaker_embed(speaker_embed, noise_scale=0.05):
    """
    在 speaker embedding 添加小噪聲
    模擬 speaker encoder 的不確定性
    """
    noise = torch.randn_like(speaker_embed) * noise_scale
    return speaker_embed + noise
```

**理由:**
- ECAPA-TDNN 輸出是固定的，模型可能過度依賴精確的 speaker embedding
- 添加擾動提高對 speaker 變化的魯棒性

**預期效果:**
- 改善 zero-shot 能力
- 泛化差距降低 1-2%

---

### 【高優先級】2. 正則化強化

#### A. **增加 Dropout** ⭐⭐⭐⭐⭐
**當前設置:** `dropout=0.1`
**建議修改:** `dropout=0.2` 或 `0.3`

```python
# 在 model_zeroshot.py 修改
ZeroShotDenoisingTransformer(
    dropout=0.2,  # 從 0.1 提高到 0.2
    ...
)
```

**理由:**
- 16% 泛化差距表示模型有顯著的 co-adaptation
- Transformer 的 hidden_dim=512, feedforward=2048 相對參數量大
- 更高的 dropout 可以減少神經元間的依賴

**預期效果:**
- 泛化差距降低 2-4%
- 驗證集準確率可能短期下降 1-2%，但長期會超越

#### B. **Label Smoothing** ⭐⭐⭐⭐
```python
# 修改 train_zeroshot_full_cached.py 的 loss function
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加 label smoothing
```

**理由:**
- Token prediction 是 4096-way 分類，過度自信會導致過擬合
- Label smoothing 鼓勵模型輸出更平滑的機率分佈

**預期效果:**
- 泛化差距降低 1-2%
- 驗證 Loss 更穩定

#### C. **Weight Decay 調整** ⭐⭐⭐
**當前:** 可能未設置或很小
**建議:** `weight_decay=1e-4` 或 `5e-5`

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=1e-4  # 添加 L2 正則化
)
```

---

### 【中優先級】3. 模型架構調整

#### A. **縮小模型容量** ⭐⭐⭐
**當前架構:**
```
d_model=512
nhead=8
num_layers=4
dim_feedforward=2048
參數量: 19.5M (可訓練: 14.8M)
```

**建議嘗試:**
```python
# 選項 1: 減少 feedforward 維度
d_model=512
nhead=8
num_layers=4
dim_feedforward=1024  # 從 2048 降到 1024

# 選項 2: 減少層數
d_model=512
nhead=8
num_layers=3  # 從 4 降到 3
dim_feedforward=2048
```

**理由:**
- 16K 訓練樣本 vs 14.8M 參數，參數/數據比過高
- 模型容量過大可能導致記憶而非泛化

**預期效果:**
- 泛化差距降低 3-5%
- 訓練速度提升 15-25%

#### B. **改進 Speaker Fusion 方式** ⭐⭐⭐⭐
**當前:** Simple Addition (`token_emb + speaker_emb`)
**建議:** Gated Fusion 或 Cross-Attention

```python
class GatedSpeakerFusion(nn.Module):
    def __init__(self, d_model, speaker_dim):
        super().__init__()
        self.speaker_proj = nn.Linear(speaker_dim, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, token_emb, speaker_emb):
        # token_emb: (B, T, d_model)
        # speaker_emb: (B, speaker_dim)

        spk_proj = self.speaker_proj(speaker_emb).unsqueeze(1)  # (B, 1, d_model)
        spk_proj = spk_proj.expand(-1, token_emb.size(1), -1)  # (B, T, d_model)

        # Gated fusion
        concat = torch.cat([token_emb, spk_proj], dim=-1)  # (B, T, 2*d_model)
        gate = torch.sigmoid(self.gate(concat))  # (B, T, d_model)

        return token_emb * gate + spk_proj * (1 - gate)
```

**理由:**
- 當前的 additive fusion 可能不夠靈活
- Gated fusion 允許模型自適應調節 speaker 信息的重要性

**預期效果:**
- 驗證準確率提升 1-2%
- 更好的 speaker 條件控制

---

### 【中優先級】4. 學習率策略優化

#### A. **使用 Cosine Annealing** ⭐⭐⭐⭐
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=args.num_epochs,
    eta_min=1e-6
)
```

**理由:**
- 當前可能使用固定學習率，後期收斂慢
- Cosine schedule 在後期提供更細緻的調整

#### B. **添加 Warmup** ⭐⭐⭐
```python
# 前 5 個 epoch 線性增加學習率
def get_lr_multiplier(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0
```

**預期效果:**
- 驗證準確率提升 0.5-1%
- 訓練更穩定

---

### 【低優先級】5. 訓練策略改進

#### A. **Early Stopping** ⭐⭐⭐
```python
# 當驗證準確率 10 個 epoch 未提升時停止
# 避免過度訓練導致過擬合惡化
```

**當前最佳:** Epoch 49, Val Acc 38.71%
**建議:** 監控驗證準確率，自動保存最佳模型

#### B. **Mix-up Training** ⭐⭐
```python
def mixup_data(x1, x2, y1, y2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x1 + (1 - lam) * x2
    return mixed_x, y1, y2, lam
```

**理由:**
- 進一步增強數據多樣性
- 對 token sequences 可能效果有限

---

## 🔬 實驗計劃建議

### Phase 1: 快速驗證（優先實施）
**預計時間:** 每個實驗 2.5 小時

1. **實驗 A: 增加 Dropout**
   ```bash
   # dropout=0.1 → 0.2
   預期: Gap 降到 13-14%
   ```

2. **實驗 B: Label Smoothing**
   ```bash
   # label_smoothing=0.1
   預期: Gap 降到 14-15%
   ```

3. **實驗 C: Token Augmentation**
   ```bash
   # augment_prob=0.15
   預期: Gap 降到 11-13%
   ```

### Phase 2: 架構改進
**預計時間:** 每個實驗 2-3 小時

4. **實驗 D: 縮小模型**
   ```bash
   # num_layers=3 或 dim_feedforward=1024
   預期: Gap 降到 10-12%, 速度提升
   ```

5. **實驗 E: Gated Fusion**
   ```bash
   # 替換 additive fusion
   預期: Val Acc 提升到 39-40%
   ```

### Phase 3: 組合優化
6. **實驗 F: 最佳組合**
   - Dropout=0.2
   - Label Smoothing=0.1
   - Token Augmentation=0.15
   - Gated Fusion
   - Cosine Scheduler

   **預期最終結果:**
   - 訓練準確率: 50-52%
   - 驗證準確率: 40-42%
   - 泛化差距: <10%

---

## 📝 實施優先級總結

### 🔥 立即實施（高 ROI，低成本）
1. ✅ 增加 Dropout (0.1 → 0.2)
2. ✅ 添加 Label Smoothing (0.1)
3. ✅ Token-level Augmentation

### 📅 短期實施（中等 ROI，中等成本）
4. Speaker Embedding Perturbation
5. Cosine Annealing Scheduler
6. Weight Decay 調整

### 🔮 長期探索（高 ROI，高成本）
7. Gated Speaker Fusion
8. 模型架構搜索
9. Mix-up Training

---

## 🎓 理論分析

### 為什麼當前模型泛化差距大？

1. **模型容量 vs 數據量不匹配**
   - 14.8M 參數 vs 16K 樣本 ≈ 920 參數/樣本
   - 理想比例: 100-200 參數/樣本

2. **Token 序列的離散性**
   - 4096-way 分類，每個 token 是獨立決策
   - 模型容易記憶訓練集的特定 token 組合

3. **Speaker Embedding 的確定性**
   - ECAPA-TDNN frozen，輸出完全固定
   - 模型可能過度依賴精確的 speaker embedding

4. **訓練策略不足**
   - 缺乏充分的正則化
   - 無數據增強

### 預期改進效果

**保守估計 (實施 1-3):**
- 驗證準確率: 38.71% → 40-41%
- 泛化差距: 16.24% → 12-13%

**樂觀估計 (實施 1-6):**
- 驗證準確率: 38.71% → 41-43%
- 泛化差距: 16.24% → 8-10%

---

## 🤔 討論問題

1. **是否考慮更改 train/val split 比例？**
   - 當前: 78% / 22%
   - 建議: 可能保持不變，因為 4608 個驗證樣本已足夠

2. **是否考慮 K-fold Cross-Validation？**
   - 優點: 更穩健的性能評估
   - 缺點: 訓練時間 × K

3. **是否需要收集更多數據？**
   - 當前 16K 樣本對於此任務可能不足
   - 建議目標: 50K-100K 樣本

4. **Zero-Shot 能力評估？**
   - 當前驗證集是否包含訓練中未見過的 speaker？
   - 建議添加 "unseen speaker" 測試集

---

生成時間: 2025-11-03
當前最佳模型: Epoch 49, Val Acc 38.71%
