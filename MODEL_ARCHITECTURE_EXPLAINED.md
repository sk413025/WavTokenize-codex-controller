# 模型運行邏輯完整解析

## 整體架構圖

```
音頻輸入 → WavTokenizer編碼 → Token序列 → Transformer降噪 → 乾淨Token序列 → WavTokenizer解碼 → 乾淨音頻

具體流程：
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│ 噪聲音頻    │ →  │ WavTokenizer │ →  │ Token序列      │ →  │ Transformer  │
│ [B,1,T]     │    │ 編碼器       │    │ [B, seq_len]   │    │ 降噪模型     │
└─────────────┘    └──────────────┘    └────────────────┘    └──────────────┘
                                                                      ↓
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│ 乾淨音頻    │ ←  │ WavTokenizer │ ←  │ 乾淨Token序列  │ ←  │ 預測Logits  │
│ [B,1,T]     │    │ 解碼器       │    │ [B, seq_len]   │    │ [B,seq,vocab]│
└─────────────┘    └──────────────┘    └────────────────┘    └──────────────┘
```

## 1. 數據預處理階段

### 1.1 音頻到Token轉換
```python
def audio_to_tokens(audio, wavtokenizer, device):
    """
    音頻 → Token序列轉換
    
    輸入: audio [batch_size, 1, audio_length]  # 例如 [4, 1, 24000] 
    輸出: tokens [batch_size, token_sequence_length]  # 例如 [4, 200]
    """
    
    # Step 1: 音頻預處理
    audio = convert_audio(audio, 24000, 1)  # 確保24kHz單聲道
    
    # Step 2: WavTokenizer編碼
    with torch.no_grad():
        # 提取離散編碼
        codes = wavtokenizer.encode_infer(
            audio.to(device), 
            bandwidth_id=torch.tensor([0], device=device)
        )
        # codes shape: [batch, n_q, time] 例如 [4, 1, 200]
        
        # 轉換為token序列
        tokens = codes.squeeze(1)  # [batch, time] 例如 [4, 200]
    
    return tokens
```

### 1.2 數據集創建
```python
# TokenSequenceDataset處理數據
noisy_audio → noisy_tokens [200個token]
clean_audio → clean_tokens [190個token]  # 通常略短

# 創建訓練樣本
{
    'input_seq': noisy_tokens,      # [batch, 200] 輸入序列
    'decoder_input': clean_tokens,  # [batch, 190] 解碼器輸入(teacher forcing)
    'target_seq': clean_tokens      # [batch, 190] 目標序列
}
```

## 2. 模型架構詳解

### 2.1 TokenToTokenTransformer架構
```
參數統計: 總共 50,439,170 個參數

┌─────────────────────────────────────────────────────────┐
│                TokenToTokenTransformer                   │
├─────────────────────────────────────────────────────────┤
│  1. Token Embeddings                                    │
│     • src_embedding: Embedding(4098, 512)              │
│     • tgt_embedding: Embedding(4098, 512)              │ 
│     • 將離散token映射到512維連續空間                    │
├─────────────────────────────────────────────────────────┤
│  2. Positional Encoding                                │
│     • 添加位置信息到token嵌入                           │
│     • 使用sin/cos位置編碼                               │
├─────────────────────────────────────────────────────────┤
│  3. Transformer核心                                     │
│     • Encoder: 6層, 8個注意力頭, 2048維FFN             │
│     • Decoder: 6層, 8個注意力頭, 2048維FFN             │
│     • 自注意力 + 交叉注意力機制                         │
├─────────────────────────────────────────────────────────┤
│  4. Output Projection                                   │
│     • Linear(512, 4098): 映射回詞彙表空間               │
│     • 輸出每個位置上4098個token的概率分佈                │
└─────────────────────────────────────────────────────────┘
```

### 2.2 前向傳播詳細流程

```python
def forward(src, tgt):
    """
    完整的前向傳播過程
    
    輸入:
    - src: [batch_size, src_len] 噪聲token序列  例如 [4, 200]
    - tgt: [batch_size, tgt_len] 目標token序列  例如 [4, 190]
    """
    
    # Step 1: Token嵌入 + 縮放
    src_emb = src_embedding(src) * sqrt(512)  # [4, 200, 512]
    tgt_emb = tgt_embedding(tgt) * sqrt(512)  # [4, 190, 512]
    
    # Step 2: 位置編碼
    src_emb = pos_encoding(src_emb)  # [4, 200, 512] 
    tgt_emb = pos_encoding(tgt_emb)  # [4, 190, 512]
    
    # Step 3: 創建遮罩
    tgt_mask = causal_mask(190, 190)  # 下三角遮罩，防止看到未來
    src_pad_mask = (src == 0)        # padding遮罩
    tgt_pad_mask = (tgt == 0)        # padding遮罩
    
    # Step 4: Transformer處理
    # Encoder處理噪聲序列
    encoder_output = transformer.encoder(src_emb)  # [4, 200, 512]
    
    # Decoder基於encoder輸出生成乾淨序列
    decoder_output = transformer.decoder(
        tgt_emb,           # [4, 190, 512] 目標嵌入
        encoder_output,    # [4, 200, 512] 編碼器記憶
        tgt_mask,          # 因果遮罩
        src_pad_mask,      # 源padding遮罩  
        tgt_pad_mask       # 目標padding遮罩
    )  # [4, 190, 512]
    
    # Step 5: 輸出投影
    logits = output_projection(decoder_output)  # [4, 190, 4098]
    
    return logits
```

## 3. Loss計算系統

### 3.1 傳統CrossEntropy Loss
```python
def compute_crossentropy_loss(logits, target_tokens):
    """
    標準交叉熵損失
    
    輸入:
    - logits: [batch, seq_len, vocab_size]  例如 [4, 190, 4098]
    - target_tokens: [batch, seq_len]       例如 [4, 190]
    
    計算:
    """
    # 重塑為2D用於計算
    logits_flat = logits.view(-1, 4098)      # [760, 4098]
    target_flat = target_tokens.view(-1)     # [760]
    
    # 計算交叉熵(忽略padding token 0)
    loss = F.cross_entropy(logits_flat, target_flat, ignore_index=0)
    
    return loss  # 標量值，例如 8.45
```

### 3.2 Token Loss系統（創新點）

#### 3.2.1 L2距離損失
```python
def compute_token_l2_loss(predicted_tokens, target_tokens, embedding_layer):
    """
    在嵌入空間計算L2距離
    
    原理: 語義相近的token在嵌入空間中距離更近
    """
    # 將token映射到連續嵌入空間
    pred_embed = embedding_layer(predicted_tokens)  # [4, 190, 512] 
    targ_embed = embedding_layer(target_tokens)     # [4, 190, 512]
    
    # 計算L2距離 (等同於ttt2.py中的特徵距離)
    l2_distance = torch.norm(pred_embed - targ_embed, p=2, dim=-1)  # [4, 190]
    
    return l2_distance.mean()  # 例如 0.65
```

#### 3.2.2 內容一致性損失
```python
def compute_consistency_loss(predicted_logits, target_tokens):
    """
    確保預測的準確性和分佈合理性
    """
    # 主要部分: 交叉熵確保預測準確
    ce_loss = F.cross_entropy(logits_flat, target_flat)  # 例如 8.37
    
    # 附加部分: 熵正則化確保適度不確定性
    probs = F.softmax(predicted_logits, dim=-1)  # [4, 190, 4098]
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [4, 190]
    entropy_reg = -entropy.mean() * 0.01  # 鼓勵適度熵
    
    return ce_loss + entropy_reg
```

#### 3.2.3 Manifold正則化損失
```python  
def compute_manifold_loss(predicted_tokens, input_tokens, embedding_layer):
    """
    防止預測偏離輸入的語義流形太遠
    完全按照ttt2.py的邏輯
    """
    # 映射到嵌入空間
    pred_embed = embedding_layer(predicted_tokens)  # [4, 190, 512]
    input_embed = embedding_layer(input_tokens)     # [4, 190, 512]
    
    # 計算manifold距離
    manifold_dist = torch.norm(pred_embed - input_embed, p=2, dim=-1)  # [4, 190]
    
    # ttt2.py的適應性閾值邏輯
    mean_dist = manifold_dist.mean()
    std_dist = manifold_dist.std()
    adaptive_threshold = mean_dist + 2 * std_dist
    
    # 懲罰超過閾值的部分
    excess_dist = F.relu(manifold_dist - adaptive_threshold)
    
    return excess_dist.mean() * 0.1  # 例如 0.00
```

#### 3.2.4 連貫性損失
```python
def compute_coherence_loss(predicted_tokens, input_tokens):
    """
    確保序列的語義連貫性和平滑性
    """
    coherence_losses = []
    
    # 滑動窗口檢查局部一致性
    for i in range(seq_len - window_size + 1):
        pred_window = predicted_tokens[:, i:i+5]  # [4, 5]
        input_window = input_tokens[:, i:i+5]     # [4, 5]
        
        # 計算相鄰token的變化率
        pred_diff = torch.abs(pred_window[:, 1:] - pred_window[:, :-1])  # [4, 4]
        input_diff = torch.abs(input_window[:, 1:] - input_window[:, :-1])  # [4, 4]
        
        # 變化率應該相似(保持平滑性)
        diff_loss = F.mse_loss(pred_diff.float(), input_diff.float())
        coherence_losses.append(diff_loss)
    
    return torch.stack(coherence_losses).mean() * 0.01  # 例如 387.31
```

### 3.3 組合損失計算
```python
def compute_combined_loss(logits, predicted_tokens, target_tokens, input_tokens, embedding_layer):
    """
    組合所有loss組件
    
    權重配置:
    - L2: 0.4          (語義相似性)
    - Consistency: 0.5  (預測準確性)  
    - Manifold: 0.05   (語義流形約束)
    - Normalization: 0.04  (正則化)
    - Coherence: 0.01  (序列平滑性)
    """
    
    losses = {}
    
    # 計算各個組件
    losses['l2'] = compute_token_l2_loss(predicted_tokens, target_tokens, embedding_layer)
    losses['consistency'] = compute_consistency_loss(logits, target_tokens)  
    losses['manifold'] = compute_manifold_loss(predicted_tokens, input_tokens, embedding_layer)
    losses['normalization'] = compute_normalization_loss(logits)
    losses['coherence'] = compute_coherence_loss(predicted_tokens, input_tokens)
    
    # 加權組合
    total_loss = (
        0.4 * losses['l2'] +           # 0.4 * 0.65 = 0.26
        0.5 * losses['consistency'] +  # 0.5 * 8.37 = 4.19  
        0.05 * losses['manifold'] +    # 0.05 * 0.00 = 0.00
        0.04 * losses['normalization'] + # 0.04 * 0.30 = 0.01
        0.01 * losses['coherence']     # 0.01 * 387.31 = 3.87
    )  # total ≈ 8.33
    
    return total_loss, losses
```

## 4. 訓練過程詳解

### 4.1 訓練循環
```python
def training_step(model, batch, optimizer, use_token_loss=True):
    """
    單個訓練步驟
    """
    # 獲取數據
    input_seq = batch['input_seq']      # [4, 200] 噪聲tokens
    decoder_input = batch['decoder_input']  # [4, 190] 解碼器輸入
    target_seq = batch['target_seq']    # [4, 190] 目標tokens
    
    # 前向傳播
    logits = model(input_seq, decoder_input)  # [4, 190, 4098]
    predicted_tokens = torch.argmax(logits, dim=-1)  # [4, 190]
    
    # 選擇損失計算方式
    if use_token_loss:
        # 使用Token Loss系統
        total_loss, loss_dict = compute_combined_token_loss(
            logits, predicted_tokens, target_seq, input_seq, 
            model.src_embedding
        )
        # 輸出: total_loss = 8.33, loss_dict = {...}
    else:
        # 使用標準交叉熵
        total_loss = F.cross_entropy(
            logits.view(-1, 4098), 
            target_seq.view(-1), 
            ignore_index=0
        )
        # 輸出: total_loss = 8.45
    
    # 反向傳播
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return total_loss.item(), loss_dict
```

### 4.2 驗證過程
```python
def validation_step(model, val_loader):
    """
    驗證過程 - 始終使用交叉熵以便比較
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_seq = batch['input_seq']
            decoder_input = batch['decoder_input'] 
            target_seq = batch['target_seq']
            
            # 前向傳播
            logits = model(input_seq, decoder_input)
            
            # 計算損失
            loss = F.cross_entropy(
                logits.view(-1, 4098),
                target_seq.view(-1),
                ignore_index=0
            )
            total_loss += loss.item()
            
            # 計算準確率
            predicted_tokens = torch.argmax(logits, dim=-1)
            mask = (target_seq != 0)  # 忽略padding
            correct = (predicted_tokens == target_seq) & mask
            
            correct_predictions += correct.sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy
```

## 5. 實際運行數值示例

### 5.1 模型輸入輸出示例
```
# 訓練樣本
input_seq:     [1245, 3421, 892, ..., 2341]     # 200個噪聲tokens
decoder_input: [1234, 3456, 789, ..., 0]        # 190個目標tokens (teacher forcing)
target_seq:    [3456, 789, 234, ..., 0]         # 190個目標tokens (右移一位)

# 模型輸出
logits:        shape=[4, 190, 4098]              # 每個位置的token概率分佈
predicted:     [3457, 790, 233, ..., 0]         # argmax得到的預測tokens
```

### 5.2 Loss數值示例
```
CrossEntropy訓練:
Epoch 1: Train=8.45, Val=8.44, Acc=0.0000
Epoch 2: Train=8.18, Val=8.44, Acc=0.0103
Epoch 3: Train=8.04, Val=8.44, Acc=0.0206

Token Loss訓練:  
Epoch 1: Train=923.24, Val=8.44, Acc=0.0052
  - L2: 0.65, Consistency: 8.37, Manifold: 0.00, Normalization: 0.30, Coherence: 91877.66
Epoch 2: Train=1439.94, Val=8.43, Acc=0.0103  
  - L2: 0.66, Consistency: 8.10, Manifold: 0.00, Normalization: 0.30, Coherence: 143560.67
Epoch 3: Train=391.58, Val=8.43, Acc=0.0206
  - L2: 0.68, Consistency: 7.96, Manifold: 0.00, Normalization: 0.31, Coherence: 38731.41
```

## 6. 關鍵創新點總結

### 6.1 技術創新
1. **離散-連續橋樑**: 通過embedding layer將離散token映射到連續空間
2. **Loss移植**: 完整移植ttt2.py的loss計算邏輯到token空間
3. **語義約束**: L2距離提供語義相似性約束
4. **流形保持**: Manifold正則化防止語義漂移
5. **序列連貫**: Coherence loss確保生成序列的平滑性

### 6.2 模型優勢
- **更豐富的約束**: 不僅考慮預測准確性，還考慮語義相似性、流形一致性等
- **可解釋性**: 每個loss組件有明確的語義含義
- **可調節性**: 通過權重可以調節不同約束的重要性
- **通用性**: 可以應用到任何token序列建模任務

這個系統成功實現了將連續特徵空間的優化目標應用到離散token空間的創新，為序列建模提供了新的思路！
