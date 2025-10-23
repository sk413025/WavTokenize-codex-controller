# Token Denoising Frozen Codebook 實驗完整分析報告

## 📊 實驗概覽

**實驗 ID**: frozen_codebook_20251022_111314  
**訓練時長**: ~43.5 小時 (10/22 11:13 - 10/23 00:36)  
**完成進度**: 182/600 epochs (30.3%)  
**狀態**: ⚠️ 嚴重過擬合，已停止

---

## 🎯 訓練表現分析

### Epoch 1 vs Epoch 100 vs Epoch 182

| Metric | Epoch 1 | Epoch 100 | Epoch 182 | 變化 |
|--------|---------|-----------|-----------|------|
| Train Loss | 5.6221 | 1.3954 | 1.2208 | ⬇️ 78% |
| Train Acc | 20.74% | 63.00%+ | **66.78%** | ⬆️ 222% |
| Val Loss | 6.1308 | ~11.4 | **11.7729** | ⬆️ 92% ❌ |
| Val Acc | 15.00% | ~14.9% | **14.92%** | ➡️ 持平 ❌ |

### 關鍵發現

#### ✅ 訓練集表現優異
- **快速收斂**: Epoch 1-50 從 20% → 55% (暴增)
- **持續提升**: Epoch 50-182 從 55% → 67% (穩定增長)
- **損失下降**: 5.6 → 1.2 (降低 78%)

#### ❌ 驗證集表現糟糕
- **準確率停滯**: 15% 維持 182 epochs 無提升
- **損失惡化**: 6.1 → 11.8 (增加 92%)
- **嚴重過擬合**: Train/Val gap 達 52% (66.78% vs 14.92%)

---

## 🔍 根本原因分析

### 1. 為什麼訓練集進步快？
- ✅ 模型容量足夠 (21M 參數)
- ✅ Token mapping 可學習 (Transformer 有能力)
- ✅ 訓練數據充足 (4032 樣本)

### 2. 為什麼驗證集不進步？

#### 📌 問題 1: Token-level 學習 vs Audio-level 目標
**核心矛盾**:
- 訓練目標: 預測正確的 token ID (離散分類)
- 實際目標: 重建清晰的音頻 (連續信號)

Token 準確率高≠音頻質量好！
- 即使 token 完全正確，解碼後的音頻可能仍有噪音
- Token ID 是離散的，但音頻是連續的
- Codebook lookup 可能無法捕捉細微的音頻差異

#### 📌 問題 2: Frozen Codebook 的限制
**Codebook 不可調**:
- 預訓練的 codebook 不是為去噪設計的
- Noisy 和 Clean 音頻可能映射到相似的 tokens
- 模型無法學習"去噪專用"的 embedding

**舉例**:
```
Noisy Audio  → WavTokenizer → Token [1234, 5678, ...] 
Clean Audio  → WavTokenizer → Token [1234, 5680, ...]
                                      ↑只有一個 token 不同！
```
如果 noisy 和 clean 的 token 差異很小，模型學不到有效的去噪模式。

#### 📌 問題 3: 驗證集語者未見過
**語者不同**:
- 訓練: boy1, boy3, ..., girl11 (14 人)
- 驗證: girl9, girl10, boy7, boy8 (4 人，完全未見過)

**泛化困難**:
- Token mapping 可能對特定語者過擬合
- 新語者的 token distribution 不同
- Frozen codebook 無法適應新語者

---

## ⚠️ 嚴重問題: 沒有音頻保存！

### 對比 wavtokenizer_transformer_denoising.py

**現有代碼有**:
- `save_audio_sample()`: 保存音頻文件
- `plot_spectrograms()`: 繪製頻譜圖
- `save_sample_with_spectrograms()`: 每個 epoch 保存樣本
- 評估時調用: `save_audio_samples(model, val_loader, epoch, ...)`

**train_token_denoising.py 缺少**:
- ❌ 沒有音頻解碼 (Token IDs → Audio)
- ❌ 沒有音頻保存
- ❌ 沒有頻譜圖
- ❌ 無法評估實際音頻質量

**後果**:
- 只能看 Token Accuracy，看不到實際去噪效果
- 無法判斷 66% 準確率是否真的有用
- 缺乏聽覺評估

---

## 📉 過擬合原因總結

1. **模型太大**: 21M 參數對 4K 樣本來說太多
2. **Dropout 不足**: 0.1 可能不夠，應該 0.3-0.5
3. **沒有正則化**: 只有 weight decay (0.01)
4. **Token-level 目標不適合**: 應該加入 audio-level loss
5. **Frozen Codebook**: 無法學習去噪專用的 representation

---

## 💡 改進建議

### 短期修復 (必須做)

1. **添加音頻保存功能** ⭐⭐⭐
   ```python
   # 在驗證時解碼 tokens 為音頻
   with torch.no_grad():
       clean_tokens_pred = model(noisy_tokens)
       # 使用 WavTokenizer 解碼
       audio_pred = wavtokenizer.decode(clean_tokens_pred)
       # 保存音頻和頻譜圖
       torchaudio.save(f"pred_epoch{epoch}.wav", audio_pred, 24000)
   ```

2. **增加 Dropout**
   - 0.1 → 0.3 或 0.5

3. **早停機制**
   - 驗證 loss 連續 20 epochs 不降就停

### 中期改進

4. **混合損失函數** ⭐⭐⭐
   ```python
   # 不只用 CrossEntropy
   loss = ce_loss + 0.5 * l2_embed_loss + 0.1 * spectrogram_loss
   ```
   - Token-level: CrossEntropy (離散)
   - Audio-level: L2 Embedding Distance (連續)
   - Perceptual: Spectrogram Loss (聽覺)

5. **Trainable Codebook** ⭐⭐
   - 允許 codebook 微調 (fine-tune)
   - 學習去噪專用的 embeddings

6. **數據增強**
   - 訓練時隨機加噪
   - Mixup / SpecAugment

### 長期方向

7. **端到端架構**
   - 不用離散 tokens
   - 直接 Audio → Transformer → Audio

8. **更好的預訓練 Codebook**
   - 使用針對去噪任務預訓練的 WavTokenizer
   - 或者使用 EnCodec、SoundStream

---

## 🎯 結論

### 實驗成就
✅ Frozen Codebook 策略在訓練集上有效 (67% 準確率)
✅ Transformer 能學習 token mapping
✅ 訓練穩定，無崩潰

### 實驗失敗
❌ 嚴重過擬合 (Train 67% vs Val 15%)
❌ 驗證集完全沒進步
❌ 缺少音頻評估，無法驗證實際效果
❌ Token accuracy 不等於音頻質量

### 最重要的教訓
**Token-level 目標不適合 Audio Denoising！**
- 音頻是連續信號，需要連續的 loss
- 離散 token 無法捕捉細微的音頻變化
- 必須添加 audio-level 或 perceptual loss

### 下一步行動
1. **立即**: 添加音頻保存功能，聽聽看 67% 準確率的實際效果
2. **短期**: 加入混合 loss，結合 token 和 audio 目標
3. **中期**: 考慮 trainable codebook 或端到端架構
