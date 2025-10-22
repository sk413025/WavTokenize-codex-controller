# Try 資料夾 - 文件索引

**最後更新**: 2025-10-22  
**用途**: 快速找到需要的文檔

---

## 🎯 核心概念：一張圖理解差異

```
┌─────────────────────────────────────────────────────────────────────┐
│                    兩種模型的核心差異                                 │
└─────────────────────────────────────────────────────────────────────┘

現有模型 (Trainable Embedding)         Try 模型 (Frozen Codebook)
────────────────────────────           ───────────────────────────

     Noisy Audio                            Noisy Audio
          │                                      │
          │ WavTokenizer ❄️                      │ WavTokenizer ❄️
          ▼                                      ▼
     Noisy Tokens                           Noisy Tokens
          │                                      │
          │ 🔥 Trainable Embedding               │ ❄️ Frozen Lookup
          │    (2M 參數)                         │    (0 參數)
          ▼                                      ▼
     Embeddings                             Embeddings
          │                                      │
          │ 🔥 Encoder-Decoder                   │ 🔥 Encoder Only
          │    (~4M 參數)                        │    (~3M 參數)
          ▼                                      ▼
     Clean Tokens                           Clean Tokens
          │                                      │
          │ WavTokenizer ❄️                      │ WavTokenizer ❄️
          ▼                                      ▼
     Denoised Audio                         Denoised Audio

總參數: ~5-6M                            總參數: ~3-4M
損失函數: 4 個                           損失函數: 1 個
訓練複雜度: 高                           訓練複雜度: 低
設計哲學: 微調適應                       設計哲學: 保留知識
```

**關鍵差異**:
- 🔴 現有模型：重新學習 embedding (可能遺忘預訓練知識)
- 🔵 Try 模型：完全凍結 codebook (保留所有預訓練知識)

---

## 🚀 快速開始 (新手必讀)

### 1️⃣ 想立即開始實驗？
👉 **[QUICKSTART.md](./QUICKSTART.md)** (4.4K)
- 3 步驟開始訓練
- 基本配置說明
- 常見問題排查

### 2️⃣ 想理解這是什麼？
👉 **[SUMMARY.md](./SUMMARY.md)** (8.7K)
- Try 資料夾的完整說明
- 與現有模型的關係
- 核心創新點

### 3️⃣ 準備好要執行了？
👉 **[CHECKLIST.md](./CHECKLIST.md)** (4.0K)
- 完整的執行清單
- 逐步檢查指引
- 結果記錄模板

---

## 📚 深入理解

### 想知道模型有什麼不同？
👉 **[MODEL_COMPARISON_ANALYSIS.md](./MODEL_COMPARISON_ANALYSIS.md)** (8.7K)
- 現有模型 vs Frozen Codebook 詳細對比
- 設計哲學差異
- 優劣勢分析
- 實驗假設

### 想了解完整的技術細節？
👉 **[README_FROZEN_CODEBOOK.md](./README_FROZEN_CODEBOOK.md)** (7.1K)
- 模型架構詳解
- 設計靈感 (機器翻譯類比)
- 預期效果
- 技術細節

### 想理解為什麼凍結 Codebook？
👉 **[TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md](./TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md)** (24K)
- Token-Codebook 因果關係
- 凍結 vs 微調的理論基礎
- 機器翻譯類比詳解

### 想理解 Token 與 Codebook 的關係？
👉 **[TOKEN_RELATIONSHIP_EXPLANATION (1).md](./TOKEN_RELATIONSHIP_EXPLANATION%20(1).md)** (22K)
- WavTokenizer 編碼流程
- Vector Quantization 原理
- 連續特徵 vs 離散 Token

---

## 💻 代碼實現

### 模型定義
👉 **[token_denoising_transformer.py](./token_denoising_transformer.py)** (17K)
```python
class TokenDenoisingTransformer(nn.Module):
    """基於 Frozen Codebook 的降噪 Transformer"""
    
class WavTokenizerTransformerDenoiser:
    """完整的降噪流程 (含推論 API)"""
```

### 訓練腳本
👉 **[train_token_denoising.py](./train_token_denoising.py)** (17K)
```python
# 完整訓練流程
# - 數據加載
# - 訓練循環
# - 驗證
# - 檢查點保存
```

### 執行腳本
👉 **[run_token_denoising_frozen_codebook.sh](./run_token_denoising_frozen_codebook.sh)** (7.0K)
```bash
# 一鍵執行訓練
# 自動選擇 GPU
# 記錄到 REPORT.md
# Git commit
```

---

## 📖 閱讀順序建議

### 路徑 A: 快速實驗 (30 分鐘)
1. **QUICKSTART.md** (5 分鐘) - 了解如何運行
2. **CHECKLIST.md** (5 分鐘) - 檢查前置條件
3. **執行訓練** (10 分鐘) - 開始實驗
4. **SUMMARY.md** (10 分鐘) - 等待時閱讀

### 路徑 B: 深入理解 (2 小時)
1. **SUMMARY.md** (15 分鐘) - 全局理解
2. **MODEL_COMPARISON_ANALYSIS.md** (30 分鐘) - 詳細對比
3. **README_FROZEN_CODEBOOK.md** (20 分鐘) - 技術細節
4. **TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md** (30 分鐘) - 理論基礎
5. **TOKEN_RELATIONSHIP_EXPLANATION.md** (25 分鐘) - Token 機制

### 路徑 C: 代碼研究 (1 小時)
1. **README_FROZEN_CODEBOOK.md** (10 分鐘) - 架構概覽
2. **token_denoising_transformer.py** (20 分鐘) - 模型代碼
3. **train_token_denoising.py** (20 分鐘) - 訓練邏輯
4. **run_token_denoising_frozen_codebook.sh** (10 分鐘) - 執行流程

---

## 🎯 按需求找文檔

### 我想...

#### 立刻開始實驗
→ [QUICKSTART.md](./QUICKSTART.md) + [CHECKLIST.md](./CHECKLIST.md)

#### 理解這個實驗的意義
→ [SUMMARY.md](./SUMMARY.md)

#### 知道與現有模型有什麼不同
→ [MODEL_COMPARISON_ANALYSIS.md](./MODEL_COMPARISON_ANALYSIS.md)

#### 學習技術細節
→ [README_FROZEN_CODEBOOK.md](./README_FROZEN_CODEBOOK.md)

#### 理解理論基礎
→ [TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md](./TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md)

#### 修改模型代碼
→ [token_denoising_transformer.py](./token_denoising_transformer.py)

#### 調整訓練流程
→ [train_token_denoising.py](./train_token_denoising.py)

#### 改變實驗配置
→ [run_token_denoising_frozen_codebook.sh](./run_token_denoising_frozen_codebook.sh)

---

## 📊 文件類型分類

### 📘 說明文檔
- **SUMMARY.md** - 總覽
- **README_FROZEN_CODEBOOK.md** - 完整說明
- **MODEL_COMPARISON_ANALYSIS.md** - 對比分析

### 📗 快速指南
- **QUICKSTART.md** - 快速開始
- **CHECKLIST.md** - 執行清單

### 📙 理論文檔
- **TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md** - 架構解釋
- **TOKEN_RELATIONSHIP_EXPLANATION (1).md** - Token 機制

### 💻 代碼文件
- **token_denoising_transformer.py** - 模型實現
- **train_token_denoising.py** - 訓練腳本
- **run_token_denoising_frozen_codebook.sh** - 執行腳本

---

## 🔗 外部關聯

### 主項目文件
- `../wavtokenizer_transformer_denoising.py` - 現有模型 (對照組)
- `../run_transformer_large_tokenloss.sh` - 現有模型執行腳本
- `../REPORT.md` - 實驗總報告 (所有實驗)

### 數據路徑
- `../data/raw/box/` - 輸入音訊 (noisy)
- `../data/clean/box2/` - 目標音訊 (clean)

### 結果路徑
- `../results/token_denoising_frozen_codebook_*/` - 本實驗結果
- `../results/transformer_large_tokenloss_*/` - 現有模型結果

### 日誌路徑
- `../logs/token_denoising_frozen_codebook_*.log` - 本實驗日誌
- `../logs/transformer_large_tokenloss_*.log` - 現有模型日誌

---

## 📝 文件大小參考

| 文件 | 大小 | 閱讀時間 | 重要度 |
|------|------|----------|--------|
| QUICKSTART.md | 4.4K | 5 分鐘 | ⭐⭐⭐⭐⭐ |
| CHECKLIST.md | 4.0K | 5 分鐘 | ⭐⭐⭐⭐⭐ |
| SUMMARY.md | 8.7K | 15 分鐘 | ⭐⭐⭐⭐⭐ |
| README_FROZEN_CODEBOOK.md | 7.1K | 15 分鐘 | ⭐⭐⭐⭐ |
| MODEL_COMPARISON_ANALYSIS.md | 8.7K | 20 分鐘 | ⭐⭐⭐⭐ |
| TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md | 24K | 30 分鐘 | ⭐⭐⭐ |
| TOKEN_RELATIONSHIP_EXPLANATION.md | 22K | 25 分鐘 | ⭐⭐⭐ |
| token_denoising_transformer.py | 17K | 20 分鐘 | ⭐⭐⭐⭐ |
| train_token_denoising.py | 17K | 20 分鐘 | ⭐⭐⭐⭐ |
| run_token_denoising_frozen_codebook.sh | 7.0K | 10 分鐘 | ⭐⭐⭐⭐⭐ |

---

## 🎉 核心信息速覽

### 一句話總結
**完全凍結 WavTokenizer Codebook，只訓練 Transformer，實現更輕量的 Token-level 降噪**

### 關鍵差異
| 現有模型 | Frozen Codebook 模型 |
|----------|---------------------|
| ✅ 可訓練 Embedding | ❌ 凍結 Codebook |
| ~5-6M 參數 | ~3-4M 參數 |
| 4 個損失函數 | 1 個損失函數 |
| Encoder-Decoder | Encoder-only |

### 3 步驟開始
```bash
cd /home/sbplab/ruizi/c_code/try
bash run_token_denoising_frozen_codebook.sh
tail -f ../logs/token_denoising_frozen_codebook_*.log
```

---

## 📞 需要幫助？

### 遇到問題？
1. 查看 [QUICKSTART.md](./QUICKSTART.md) 的問題排查章節
2. 查看 [CHECKLIST.md](./CHECKLIST.md) 的除錯指引
3. 檢查日誌: `tail -100 ../logs/*.log`

### 想了解更多？
1. 閱讀 [SUMMARY.md](./SUMMARY.md) 全局理解
2. 閱讀 [MODEL_COMPARISON_ANALYSIS.md](./MODEL_COMPARISON_ANALYSIS.md) 詳細對比

### 想修改代碼？
1. 先理解 [README_FROZEN_CODEBOOK.md](./README_FROZEN_CODEBOOK.md) 的架構
2. 再修改 [token_denoising_transformer.py](./token_denoising_transformer.py)

---

**索引生成**: 2025-10-22  
**用途**: 快速導航所有文檔  
**建議**: 收藏本頁作為入口點

🎯 **從這裡開始**: [QUICKSTART.md](./QUICKSTART.md)
