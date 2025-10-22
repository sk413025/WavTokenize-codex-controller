# Try 資料夾總結報告

**生成日期**: 2025-10-22  
**分析對象**: `/home/sbplab/ruizi/c_code/try/` 資料夾  
**目的**: 解釋 Frozen Codebook 模型與現有模型的差異

---

## 📁 資料夾內容概覽

```
try/
├── token_denoising_transformer.py          # 模型實現 (核心)
├── train_token_denoising.py                # 訓練腳本 (新增)
├── run_token_denoising_frozen_codebook.sh  # 執行腳本 (新增)
├── README_FROZEN_CODEBOOK.md               # 詳細說明 (新增)
├── MODEL_COMPARISON_ANALYSIS.md            # 模型對比 (新增)
├── QUICKSTART.md                           # 快速開始 (新增)
├── TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md  # 架構解釋 (原有)
└── TOKEN_RELATIONSHIP_EXPLANATION (1).md   # Token-Codebook 關係 (原有)
```

---

## 🎯 核心創新：Frozen Codebook

### **問題**: 現有模型的潛在問題

現有模型 (`wavtokenizer_transformer_denoising.py`) 重新訓練 Codebook Embedding:

```python
# 現有模型
self.codebook_embedding = nn.Embedding(4096, 512)  # 可訓練！
self.codebook_embedding.weight.copy_(wavtokenizer_codebook)

# 訓練時會更新這些 embedding
loss.backward()  # 梯度會流向 codebook_embedding
```

**潛在風險**:
1. **Catastrophic Forgetting**: 破壞 WavTokenizer 學到的聲學知識
2. **Overfitting**: 過度擬合降噪數據
3. **訓練不穩定**: 需要精心調整多個損失函數權重

---

### **解決方案**: Frozen Codebook

Try 資料夾的模型完全凍結 Codebook:

```python
# Try 資料夾模型
self.register_buffer('codebook', wavtokenizer_codebook)  # Buffer = 不可訓練

# 查表操作 (無梯度)
embeddings = self.codebook[token_ids]  # 不會有梯度流向 codebook
```

**優勢**:
1. ✅ **保留預訓練知識**: WavTokenizer 的聲學知識不被破壞
2. ✅ **更穩定訓練**: 單一損失函數，無需權重調整
3. ✅ **更少參數**: ~3-4M (vs ~5-6M)
4. ✅ **更好泛化**: 類比機器翻譯的 frozen pretrained embedding

---

## 📊 關鍵差異對比表

| 層面 | 現有模型 | Try 資料夾模型 | 說明 |
|------|----------|----------------|------|
| **Codebook** | ✅ 可訓練 | ❌ 完全凍結 | 核心差異 |
| **架構** | Encoder-Decoder | Encoder Only | 更簡單 |
| **參數量** | ~5-6M | ~3-4M | 減少 30-40% |
| **損失函數** | CE + L2 + Coherence + Manifold | Cross-Entropy Only | 更簡單 |
| **訓練複雜度** | 高 (4 個損失項) | 低 (1 個損失項) | 易於調試 |
| **推論速度** | 較慢 (autoregressive) | 較快 (並行) | Encoder-only |
| **記憶體佔用** | 較高 | 較低 | 凍結 = 無梯度 |

---

## 🔬 設計哲學差異

### 現有模型: **微調適應**
> "WavTokenizer 的 Codebook 雖然好，但需要針對降噪任務重新調整"

```
┌─────────────────────────────────────────────────────────────────┐
│              現有模型: 微調 Embedding 策略                       │
└─────────────────────────────────────────────────────────────────┘

WavTokenizer Codebook (預訓練)
        │
        │ 複製權重
        ▼
Codebook Embedding (nn.Embedding)
        │
        │ 🔥 允許梯度更新
        │ ⚠️ 可能偏離原始聲學知識
        ▼
訓練過程中持續調整
        │
        ├─ CE Loss: 確保預測準確
        ├─ L2 Loss: 保持聲學相似性
        ├─ Coherence Loss: 時間平滑
        └─ Manifold Loss: 防止偏離太遠
        │
        ▼
針對降噪任務優化的 Embedding
(但可能遺忘部分預訓練知識)
```

**策略**:
- 從 WavTokenizer 初始化
- 允許梯度更新
- 使用多個損失函數引導
- 期望學到更適合降噪的 embedding

---

### Try 資料夾模型: **保留知識**
> "WavTokenizer 的 Codebook 已經是最佳表示，降噪可以在 token 空間進行"

```
┌─────────────────────────────────────────────────────────────────┐
│            Try 模型: Frozen Codebook 策略                        │
└─────────────────────────────────────────────────────────────────┘

WavTokenizer Codebook (預訓練)
        │
        │ register_buffer (不可訓練)
        ▼
Frozen Codebook Lookup
        │
        │ ❄️ 完全凍結，無梯度
        │ ✅ 保留所有預訓練知識
        ▼
查表操作: embeddings = codebook[token_ids]
        │
        ▼
只訓練 Transformer + Projection
        │
        └─ CE Loss ONLY (簡單直接)
        │
        ▼
學習 Token → Token 映射
(類比機器翻譯: 英文 ID → 中文 ID)

┌────────────────────────────────────────────────────────────┐
│ 機器翻譯類比:                                              │
│                                                            │
│ 英文詞 IDs → [Frozen English Embedding]                   │
│           → [Transformer]                                  │
│           → [Projection]                                   │
│           → 中文詞 IDs                                     │
│                                                            │
│ 降噪類比:                                                  │
│                                                            │
│ Noisy Token IDs → [Frozen WavTokenizer Codebook]          │
│                → [Transformer]                             │
│                → [Projection]                              │
│                → Clean Token IDs                           │
└────────────────────────────────────────────────────────────┘
```

**策略**:
- 完全凍結 Codebook
- 只學習 token → token 映射
- 類比機器翻譯 (英文 → 中文)
- 讓 Transformer 學習序列模式

---

## 🚀 如何運行

### 1. 進入資料夾
```bash
cd /home/sbplab/ruizi/c_code/try
```

### 2. 執行訓練
```bash
bash run_token_denoising_frozen_codebook.sh
```

### 3. 監控進度
```bash
tail -f ../logs/token_denoising_frozen_codebook_*.log
```

詳見 [`QUICKSTART.md`](./QUICKSTART.md)

---

## 📚 文檔導讀順序

### 新手入門
1. **QUICKSTART.md** ← 從這裡開始！
   - 快速運行實驗
   - 基本配置說明
   - 問題排查

2. **README_FROZEN_CODEBOOK.md**
   - 完整的模型說明
   - 設計理念
   - 預期效果

3. **MODEL_COMPARISON_ANALYSIS.md**
   - 與現有模型的詳細對比
   - 優劣勢分析
   - 實驗計劃

### 深入理解
4. **TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md**
   - 架構細節
   - 與機器翻譯的類比
   - 為什麼凍結 Codebook

5. **TOKEN_RELATIONSHIP_EXPLANATION (1).md**
   - Token 與 Codebook 的因果關係
   - Vector Quantization 機制
   - WavTokenizer 的編碼流程

### 代碼實現
6. **token_denoising_transformer.py**
   - 模型類別定義
   - 前向傳播邏輯
   - 推論 API

7. **train_token_denoising.py**
   - 完整訓練流程
   - 數據加載
   - 檢查點保存

---

## 🎯 核心問題與假設

### 關鍵問題 1: Frozen 是否足夠？
**假設**: WavTokenizer 的 Codebook 已經涵蓋降噪所需的所有聲學模式

**驗證方法**:
- Token Accuracy > 60% → 假設成立
- Token Accuracy < 30% → 假設失敗，需要微調

### 關鍵問題 2: Token 映射是否足夠？
**假設**: 降噪可以純粹在離散 token 空間進行，無需連續特徵

**驗證方法**:
- 音訊質量 (PESQ, STOI) 與現有模型相當 → 假設成立
- 音訊質量顯著下降 → 離散化損失太多信息

### 關鍵問題 3: 簡單架構是否足夠？
**假設**: Encoder-only + 單一 CE Loss 足以實現高質量降噪

**驗證方法**:
- 訓練穩定且收斂 → 假設成立
- 需要多個損失函數才能收斂 → 簡單架構不足

---

## 📈 預期實驗結果

### 樂觀情境 (Frozen Codebook 成功)
- Token Accuracy > 60%
- PESQ/STOI 與現有模型相當
- 訓練速度更快 (參數更少)
- 泛化能力更好

**結論**: ✅ Frozen Codebook 有效，推薦使用

### 中立情境 (效果相當)
- Token Accuracy 40-60%
- PESQ/STOI 略低但可接受
- 訓練速度更快

**結論**: ✅ Frozen Codebook 可用，更高效

### 悲觀情境 (Frozen Codebook 失敗)
- Token Accuracy < 30%
- 音訊質量顯著下降
- 無法收斂

**結論**: ⚠️ Codebook 需要微調，回歸現有模型

---

## 🔧 與現有實驗的整合

### 實驗編號規則
- 現有模型: `large_tokenloss_FIXED_LR_YYYYMMDD_HHMM`
- Frozen Codebook: `frozen_codebook_YYYYMMDD_HHMM`

### 結果路徑
```bash
# 現有模型
/home/sbplab/ruizi/c_code/results/transformer_large_tokenloss_*/

# Frozen Codebook
/home/sbplab/ruizi/c_code/results/token_denoising_frozen_codebook_*/
```

### 日誌路徑
```bash
# 現有模型
/home/sbplab/ruizi/c_code/logs/transformer_large_tokenloss_*.log

# Frozen Codebook
/home/sbplab/ruizi/c_code/logs/token_denoising_frozen_codebook_*.log
```

---

## 💡 為什麼要有這個資料夾？

### 問題來源
元佑在 try 資料夾中提出了一個重要的假設：
> **WavTokenizer 的 Codebook 已經是最佳的音訊表示，不需要重新訓練**

這與現有模型的設計哲學不同。

### 實驗目的
1. 驗證 **Frozen Codebook** 的可行性
2. 測試更 **輕量** 的架構是否足夠
3. 探索 **Token-level 降噪** 的極限

### 預期貢獻
- 如果成功 → 提供更高效的降噪方案
- 如果失敗 → 驗證現有模型設計的必要性
- 無論如何 → 深化對 Codebook 和 Token 的理解

---

## 📝 實驗計劃

### Phase 1: 初步驗證 (進行中)
- ✅ 實現 Frozen Codebook 模型
- ✅ 創建訓練腳本
- ✅ 編寫文檔說明
- 🔄 開始訓練 (待執行)

### Phase 2: 效果評估
- 訓練至少 200 epochs
- 記錄 Token Accuracy 曲線
- 保存音訊樣本
- 計算 PESQ/STOI

### Phase 3: 對比分析
- 與現有模型比較
- 分析差異原因
- 撰寫結論報告

---

## 🎉 總結

### Try 資料夾的核心價值

1. **理論創新**: Frozen Codebook 的設計哲學
2. **實驗對比**: 提供與現有模型的對照組
3. **知識探索**: 深化對 Token 和 Codebook 的理解

### 與現有模型的關係

- **不是替代**: 兩者測試不同的假設
- **是補充**: 提供另一種可能性
- **是驗證**: 幫助理解哪些設計是必要的

### 下一步

1. **執行訓練**: `bash run_token_denoising_frozen_codebook.sh`
2. **監控結果**: 觀察 Token Accuracy 和 Loss
3. **對比分析**: 與現有模型比較
4. **撰寫報告**: 總結實驗發現

---

## 🔗 相關資源

### 在 Try 資料夾內
- `README_FROZEN_CODEBOOK.md`: 詳細說明
- `MODEL_COMPARISON_ANALYSIS.md`: 模型對比
- `QUICKSTART.md`: 快速開始
- `token_denoising_transformer.py`: 模型代碼
- `train_token_denoising.py`: 訓練腳本
- `run_token_denoising_frozen_codebook.sh`: 執行腳本

### 現有實驗文件
- `../wavtokenizer_transformer_denoising.py`: 現有模型
- `../run_transformer_large_tokenloss.sh`: 現有模型執行腳本
- `../REPORT.md`: 實驗總報告

### 理論文檔
- `TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md`: 架構解釋
- `TOKEN_RELATIONSHIP_EXPLANATION (1).md`: Token 關係

---

**生成日期**: 2025-10-22  
**生成函式**: SUMMARY.md  
**用途**: 理解 try 資料夾與現有模型的關係  
**建議**: 先閱讀 QUICKSTART.md，然後執行實驗
