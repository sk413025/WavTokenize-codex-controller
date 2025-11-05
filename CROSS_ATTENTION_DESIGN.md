# Cross-Attention Speaker Fusion 設計方案

**實驗編號**: EXP-20251105-CrossAttn  
**日期**: 2025-11-05  
**目的**: 驗證假設 2 - Speaker Embedding 影響力不足  
**方法**: 將 Additive Fusion 改為 Cross-Attention Mechanism

---

## 一、問題分析

### 當前架構 (Additive Fusion)

```
輸入流程:
┌─────────────────┐     ┌──────────────────┐
│ Noisy Tokens    │     │ Speaker Emb      │
│ (B, T)          │     │ (B, 256)         │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
    ┌────────┐            ┌─────────────┐
    │Codebook│            │speaker_proj │
    │ Lookup │            │  256 → 512  │
    └────┬───┘            └──────┬──────┘
         │                       │
         ▼                       ▼
   ┌──────────┐          ┌──────────────┐
   │Token Emb │          │Speaker Emb   │
   │(B,T,512) │          │(B, 512)      │
   └─────┬────┘          └──────┬───────┘
         │                      │
         │               expand (B,T,512)
         │                      │
         └──────────┬───────────┘
                    ▼
             ┌─────────────┐
             │  ADDITION   │  ← 問題: Speaker 資訊被稀釋
             │   emb + spk │
             └──────┬──────┘
                    ▼
            ┌───────────────┐
            │ Pos Encoding  │
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ Transformer   │
            │  Encoder      │
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ Output Proj   │
            └───────┬───────┘
                    ▼
              (B, T, 4096)
```

**問題**:
1. **資訊稀釋**: `token_emb + speaker_emb` 會讓 speaker 資訊被 token 資訊"覆蓋"
2. **無注意力機制**: 無法動態調整 speaker 對不同位置的影響
3. **全局相同**: 每個 token 受到相同的 speaker 影響（expand 後直接相加）

### 診斷預期結果

如果當前架構確實有問題，我們應該在 `diagnose_prediction_behavior.py` 中看到：
- Zero speaker: Accuracy 下降 <5%
- Random speaker: Accuracy 下降 <5%
- **結論**: Speaker embedding 幾乎沒用

---

## 二、Cross-Attention 方案設計

### 方案 A: Cross-Attention Layer (推薦)

```
架構流程:
┌─────────────────┐     ┌──────────────────┐
│ Noisy Tokens    │     │ Speaker Emb      │
│ (B, T)          │     │ (B, 256)         │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
    ┌────────┐            ┌─────────────┐
    │Codebook│            │speaker_proj │
    │ Lookup │            │  256 → 512  │
    └────┬───┘            └──────┬──────┘
         │                       │
         ▼                       │
   ┌──────────┐                 │
   │Token Emb │                 │
   │(B,T,512) │                 │
   └─────┬────┘                 │
         │                      │
         ▼                      │
  ┌──────────────┐              │
  │Pos Encoding  │              │
  └──────┬───────┘              │
         │                      │
         │   ┌──────────────────┘
         │   │
         ▼   ▼
  ┌────────────────────────────────┐
  │   CROSS-ATTENTION LAYER        │
  │                                 │
  │  Query:   Token Emb   (B,T,512)│
  │  Key:     Speaker Emb (B,1,512)│ ← Key: Speaker 決定"哪些資訊重要"
  │  Value:   Speaker Emb (B,1,512)│ ← Value: Speaker 提供的資訊
  │                                 │
  │  Attention = softmax(Q·K^T/√d) │
  │  Output = Attention · V        │ ← (B, T, 512)
  └────────────────┬───────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │ Residual + Norm │  ← token_emb + cross_attn_output
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Transformer     │
         │   Encoder       │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │ Output Proj     │
         └────────┬────────┘
                  ▼
            (B, T, 4096)
```

**優勢**:
1. ✅ **動態注意力**: 每個 token 位置可以動態決定需要多少 speaker 資訊
2. ✅ **資訊保留**: Token embedding 通過 residual connection 保留
3. ✅ **可解釋性**: 可以視覺化 attention weights，看哪些 token 依賴 speaker
4. ✅ **標準架構**: 類似 Transformer Decoder 的 cross-attention

### 方案 B: Multi-Head Cross-Attention (更強)

```
         Token Emb (B,T,512)        Speaker Emb (B,1,512)
                │                           │
                └───────────┬───────────────┘
                            │
                ┌───────────▼───────────┐
                │ Multi-Head            │
                │ Cross-Attention       │
                │                       │
                │ head_1: 512→64→64→512│
                │ head_2: 512→64→64→512│
                │   ...                 │
                │ head_8: 512→64→64→512│
                │                       │
                │ Concat → Linear       │
                └───────────┬───────────┘
                            ▼
                     (B, T, 512)
```

**優勢**:
- 多個 heads 可以學習不同類型的 speaker 資訊
  * Head 1: 音高相關
  * Head 2: 語速相關
  * Head 3: 音色相關
  * ...

---

## 三、實作細節

### 3.1 新增 Cross-Attention Module

```python
class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion for Speaker Embedding
    
    將 Speaker Embedding 通過 Cross-Attention 注入到 Token Embeddings
    
    參數:
        d_model: int - 模型維度 (512)
        nhead: int - 注意力頭數 (8)
        dropout: float - Dropout 比率
    
    輸入:
        token_emb: (B, T, d_model) - Token embeddings (Query)
        speaker_emb: (B, d_model) - Speaker embedding (Key & Value)
    
    輸出:
        fused_emb: (B, T, d_model) - Fusion 後的 embeddings
    """
    
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        
        # Multi-Head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Norm (for residual connection)
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_emb, speaker_emb):
        """
        Args:
            token_emb: (B, T, d_model) - Token embeddings
            speaker_emb: (B, d_model) - Speaker embedding
        
        Returns:
            fused_emb: (B, T, d_model) - Fused embeddings
        """
        B, T, D = token_emb.shape
        
        # Speaker embedding: (B, d_model) → (B, 1, d_model)
        # 這裡我們讓 speaker 作為一個"全局" key/value
        speaker_kv = speaker_emb.unsqueeze(1)  # (B, 1, d_model)
        
        # Cross-Attention
        # Query: token_emb (B, T, d_model)
        # Key:   speaker_kv (B, 1, d_model)
        # Value: speaker_kv (B, 1, d_model)
        attn_output, attn_weights = self.cross_attn(
            query=token_emb,      # (B, T, d_model)
            key=speaker_kv,       # (B, 1, d_model)
            value=speaker_kv,     # (B, 1, d_model)
            need_weights=True
        )
        # attn_output: (B, T, d_model)
        # attn_weights: (B, T, 1) - 每個 token 對 speaker 的 attention
        
        # Residual Connection + Dropout + Layer Norm
        fused_emb = self.norm(token_emb + self.dropout(attn_output))
        
        return fused_emb, attn_weights
```

### 3.2 修改主模型

**修改位置**: `ZeroShotDenoisingTransformer.__init__()` 和 `forward()`

**Before**:
```python
# Step 2: Speaker Embedding Projection & Broadcasting
speaker_emb = self.speaker_proj(speaker_embedding)  # (B, 512)
speaker_emb = speaker_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, 512)

# Step 3: Fusion (Additive)
combined_emb = token_emb + speaker_emb  # (B, T, 512)

# Step 4: Positional Encoding
combined_emb = self.pos_encoding(combined_emb)
```

**After**:
```python
# Step 2: Positional Encoding (先對 token embedding 做)
token_emb = self.pos_encoding(token_emb)  # (B, T, 512)

# Step 3: Speaker Embedding Projection
speaker_emb = self.speaker_proj(speaker_embedding)  # (B, 512)

# Step 4: Cross-Attention Fusion
fused_emb, attn_weights = self.cross_attn_fusion(
    token_emb=token_emb,      # (B, T, 512)
    speaker_emb=speaker_emb   # (B, 512)
)
# fused_emb: (B, T, 512)
# attn_weights: (B, T, 1) - 可選：保存用於分析
```

### 3.3 參數量對比

**Current (Additive)**:
- speaker_proj: 256 × 512 = 131,072 params

**New (Cross-Attention)**:
- speaker_proj: 256 × 512 = 131,072 params
- CrossAttentionFusion:
  - Q projection: 512 × 512 = 262,144
  - K projection: 512 × 512 = 262,144
  - V projection: 512 × 512 = 262,144
  - Out projection: 512 × 512 = 262,144
  - LayerNorm: 512 × 2 = 1,024
  - **Total**: ~1.05M params

**增加量**: ~1M params (約 5% 增加)

---

## 四、實驗設計

### 4.1 對照實驗

| 實驗組 | Fusion 方式 | Speaker Proj | 其他參數 |
|--------|------------|--------------|----------|
| **Baseline (當前)** | Additive | Linear(256→512) | 保持不變 |
| **Exp-CrossAttn** | Cross-Attention | Linear(256→512) + CrossAttnFusion | 保持不變 |

### 4.2 評估指標

**定量指標**:
1. Train/Val Accuracy
2. Train/Val Loss
3. **Speaker Influence** (通過 zero/random speaker test)
   - 期望: >20% tokens 改變 (當前 <5%)

**定性指標**:
1. **Attention Weights 視覺化**
   - 哪些 token 位置依賴 speaker 較多？
   - 是否有規律（如句首、重音位置）？

2. **Token Distribution**
   - Token 0 預測頻率是否降低？
   - 預測是否更多樣化？

### 4.3 訓練設定

```yaml
# 基本設定
epochs: 100
batch_size: 8
learning_rate: 0.001
scheduler: ReduceLROnPlateau (patience=5)

# 模型設定
d_model: 512
nhead: 8  # 用於 cross-attention
num_layers: 4
dropout: 0.1

# Cross-Attention 特定
cross_attn_heads: 8
cross_attn_dropout: 0.1
```

### 4.4 預期結果

**如果 Cross-Attention 有效**:
- Train Accuracy: 56% → 62-65%
- Val Accuracy: 38% → 43-47%
- Speaker Influence: <5% → >20%
- Token 0 預測頻率: 32% → 20-25%

**如果無明顯改善**:
- 說明問題不在 fusion 方式
- 可能是 frozen codebook 或 task difficulty

---

## 五、實作步驟

### Step 1: 創建新模型檔案

```bash
cp done/exp/model_zeroshot.py done/exp/model_zeroshot_crossattn.py
```

### Step 2: 實作 CrossAttentionFusion class

在 `model_zeroshot_crossattn.py` 中新增 class

### Step 3: 修改 ZeroShotDenoisingTransformer

- `__init__`: 新增 `self.cross_attn_fusion`
- `forward`: 修改 fusion 邏輯

### Step 4: 創建訓練腳本

```bash
cp done/exp/train_zeroshot.py done/exp/train_zeroshot_crossattn.py
```

修改:
- Import 新模型
- 輸出目錄改為 `results/crossattn_100epochs_YYYYMMDD_HHMMSS/`

### Step 5: 測試模型

```bash
python done/exp/model_zeroshot_crossattn.py
```

確認:
- 參數量正確
- Forward pass 正常
- 輸出 shape 正確

### Step 6: 小規模測試（3 epochs）

```bash
# 修改 train_zeroshot_crossattn.py: epochs=3
python done/exp/train_zeroshot_crossattn.py
```

檢查:
- Loss 是否下降
- 記憶體使用是否正常
- 訓練速度

### Step 7: 完整訓練（100 epochs）

```bash
tmux new -s crossattn_training
CUDA_VISIBLE_DEVICES=2 python -u done/exp/train_zeroshot_crossattn.py 2>&1 | tee crossattn_train.log
```

### Step 8: 診斷與對比

使用相同的診斷工具:
```bash
# 修改 diagnose_prediction_behavior.py 載入新模型
python diagnose_prediction_behavior.py
```

對比兩個模型的:
- Prediction distribution
- Speaker influence
- Attention weights (新)

---

## 六、Attention Weights 視覺化

### 6.1 保存 Attention Weights

修改 `model_zeroshot_crossattn.py`:

```python
def forward(self, noisy_token_ids, speaker_embedding, 
            return_logits=False, return_attention=False):
    ...
    fused_emb, attn_weights = self.cross_attn_fusion(token_emb, speaker_emb)
    ...
    
    if return_attention:
        return logits, attn_weights
    elif return_logits:
        return logits
    else:
        return clean_token_ids
```

### 6.2 視覺化腳本

```python
# visualize_cross_attention.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention_weights(model, dataloader, device, num_samples=10):
    """
    視覺化 Cross-Attention Weights
    
    顯示:
        - 每個 token 位置對 speaker 的 attention 分數
        - 是否有特定位置（句首、句尾）依賴較多
    """
    model.eval()
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            noisy_tokens = batch['noisy_tokens'].to(device)
            speaker_emb = batch['speaker_embedding'].to(device)
            
            logits, attn_weights = model(
                noisy_tokens, speaker_emb, 
                return_logits=True, return_attention=True
            )
            
            # attn_weights: (B, T, 1) or (B, nhead, T, 1)
            # 取第一個 sample
            attn = attn_weights[0].squeeze(-1).cpu()  # (T,) or (nhead, T)
            
            if attn.ndim == 2:  # multi-head
                attn = attn.mean(0)  # 平均所有 heads
            
            # 繪製
            ax = axes[i // 5, i % 5]
            ax.plot(attn.numpy())
            ax.set_title(f'Sample {i+1}')
            ax.set_xlabel('Token Position')
            ax.set_ylabel('Attention to Speaker')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cross_attention_weights_visualization.png', dpi=300)
    print("✓ 視覺化已保存")
```

---

## 七、成功標準

### 必要條件 (Must Have)

1. ✅ **Speaker Influence >20%**
   - Zero speaker test: accuracy drop >15%
   - Random speaker test: accuracy drop >15%

2. ✅ **Val Accuracy 提升 >3%**
   - 當前: 38%
   - 目標: >41%

### 期望條件 (Should Have)

1. ✅ **Token 0 預測頻率降低**
   - 當前: ~32%
   - 目標: <25%

2. ✅ **Train-Val Gap 縮小**
   - 當前: 56% - 38% = 18%
   - 目標: <15%

3. ✅ **Attention Pattern 可解釋**
   - 例如: 句首依賴 speaker 較多
   - 例如: 特定 tokens (如 silence) 依賴較少

---

## 八、風險與應對

### 風險 1: 訓練不穩定

**症狀**: Loss 震盪、梯度爆炸

**應對**:
- 降低 learning rate (0.001 → 0.0005)
- 增加 gradient clipping
- 調整 cross-attention dropout

### 風險 2: 過擬合更嚴重

**症狀**: Train-Val gap 增大

**應對**:
- 增加 dropout (0.1 → 0.15)
- 添加 weight decay
- Early stopping

### 風險 3: 記憶體不足

**症狀**: CUDA OOM

**應對**:
- 減少 batch size (8 → 4)
- 使用 gradient checkpointing
- 混合精度訓練 (FP16)

### 風險 4: 無明顯改善

**症狀**: Accuracy 提升 <2%

**下一步**:
- 轉向假設 3: Fine-tune codebook
- 或假設 1: Focal Loss

---

## 九、後續分析

### 如果 Cross-Attention 成功

**深入分析**:
1. 哪些 heads 學到了什麼？
2. Attention 權重與 token type 的關係？
3. 不同 speakers 的 attention pattern 差異？

**進一步改進**:
1. 試驗不同的 nhead (4, 8, 16)
2. 添加多層 cross-attention
3. 結合 FiLM conditioning

### 如果 Cross-Attention 失敗

**可能原因**:
1. Speaker encoder 本身質量不佳
2. Frozen codebook 是主要瓶頸
3. Task difficulty 是根本限制

**下一步**:
- 檢查 speaker encoder 質量
- Fine-tune codebook (假設 3)
- 降低 noise level

---

## 十、Timeline

| 步驟 | 預估時間 | 說明 |
|------|---------|------|
| 1. 實作 CrossAttentionFusion | 30 分鐘 | 編寫 + 單元測試 |
| 2. 修改主模型 | 20 分鐘 | 整合到 ZeroShotDenoisingTransformer |
| 3. 創建訓練腳本 | 10 分鐘 | 複製並修改 |
| 4. 測試模型 | 5 分鐘 | 驗證 forward pass |
| 5. 小規模測試 (3 epochs) | 30 分鐘 | 確保無錯誤 |
| 6. 完整訓練 (100 epochs) | **8-10 小時** | 主要時間 |
| 7. 診斷與分析 | 1 小時 | 使用現有診斷工具 |
| 8. 視覺化 Attention | 30 分鐘 | 繪製圖表 |
| 9. 撰寫報告 | 1 小時 | 記錄結果 |

**總計**: ~12-14 小時（含訓練時間）

---

## 附錄: 關鍵程式碼片段

### A1. 完整的 CrossAttentionFusion

見上文 Section 3.1

### A2. 修改後的 forward() 函式

```python
def forward(self, noisy_token_ids, speaker_embedding, 
            return_logits=False, return_attention=False):
    """
    Args:
        noisy_token_ids: (B, T)
        speaker_embedding: (B, D_spk)
        return_logits: bool
        return_attention: bool - 是否返回 cross-attention weights
    
    Returns:
        clean_token_ids (B, T) or logits (B, T, V)
        [optional] attn_weights (B, T, 1) if return_attention=True
    """
    B, T = noisy_token_ids.shape
    
    # Step 1: Token Embedding
    token_emb = self.codebook[noisy_token_ids]  # (B, T, 512)
    
    # Step 2: Positional Encoding
    token_emb = self.pos_encoding(token_emb)  # (B, T, 512)
    
    # Step 3: Speaker Embedding Projection
    speaker_emb = self.speaker_proj(speaker_embedding)  # (B, 512)
    
    # Step 4: Cross-Attention Fusion
    fused_emb, attn_weights = self.cross_attn_fusion(
        token_emb=token_emb,
        speaker_emb=speaker_emb
    )
    # fused_emb: (B, T, 512)
    # attn_weights: (B, T, 1)
    
    # Step 5: Transformer Encoding
    hidden = self.transformer_encoder(fused_emb)  # (B, T, 512)
    
    # Step 6: Output Projection
    logits = self.output_proj(hidden)  # (B, T, 4096)
    
    if return_attention:
        return logits, attn_weights
    elif return_logits:
        return logits
    else:
        return logits.argmax(dim=-1)
```

---

**文件創建**: 2025-11-05 02:XX  
**作者**: GitHub Copilot  
**狀態**: 設計方案 - 待實作
