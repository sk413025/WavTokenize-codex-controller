# 架構圖索引

**生成日期**: 2025-10-22  
**用途**: 快速找到特定的 ASCII 架構圖

---

## 📊 已添加的架構圖總覽

本資料夾的文檔中包含了豐富的 ASCII 視覺化圖表，幫助理解模型架構和訓練流程。

---

## 🎯 按用途查找

### 1. 想看**兩種模型的整體對比**？

📄 **[INDEX.md](./INDEX.md)** - 一張圖理解核心差異
```
現有模型 (Trainable)  |  Try 模型 (Frozen)
─────────────────────────────────────────
參數對比、架構對比、設計哲學對比
```

📄 **[README_FROZEN_CODEBOOK.md](./README_FROZEN_CODEBOOK.md)** - 詳細架構流程圖
```
- 現有模型完整流程 (含 Trainable Embedding)
- 本模型完整流程 (含 Frozen Codebook)
- 每一步的參數狀態 (🔥 可訓練 / ❄️ 凍結)
```

---

### 2. 想看 **Transformer 架構細節**？

📄 **[MODEL_COMPARISON_ANALYSIS.md](./MODEL_COMPARISON_ANALYSIS.md)** - Encoder-Decoder vs Encoder-Only
```
詳細圖表包含:
- Encoder-Decoder 架構 (現有模型)
  - Self Attention
  - Cross Attention
  - Teacher Forcing
  
- Encoder-Only 架構 (本模型)
  - 6 層 Transformer Encoder
  - Multi-Head Attention (8 heads)
  - Feed Forward Network (2048)
  - 並行處理流程
```

---

### 3. 想看 **訓練流程**？

📄 **[QUICKSTART.md](./QUICKSTART.md)** - 完整訓練流程圖
```
Step 1: 數據載入
Step 2: 訓練循環 (每個 Batch)
Step 3: 驗證
Step 4: 推論

每一步都有詳細的數據形狀和操作說明
```

📄 **[README_FROZEN_CODEBOOK.md](./README_FROZEN_CODEBOOK.md)** - Epoch 循環視覺化
```
For each Epoch:
  - 訓練階段 (5 個步驟)
  - 驗證階段
  - 保存檢查點
  - 繪製訓練歷史

包含推論流程的完整視覺化
```

---

### 4. 想看 **設計哲學差異**？

📄 **[SUMMARY.md](./SUMMARY.md)** - 微調 vs 凍結策略
```
現有模型: 微調 Embedding 策略
- 梯度流向 Codebook
- 多個損失函數引導
- 可能遺忘預訓練知識

Try 模型: Frozen Codebook 策略
- 完全凍結，無梯度
- 單一 CE Loss
- 保留所有預訓練知識

包含機器翻譯類比圖
```

---

### 5. 想看 **Token 和 Codebook 的關係**？

📄 **[TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md](./TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md)**
```
- Token → Codebook 的因果關係
- Vector Quantization 機制
- 為什麼可以凍結 Codebook
```

📄 **[TOKEN_RELATIONSHIP_EXPLANATION (1).md](./TOKEN_RELATIONSHIP_EXPLANATION%20(1).md)**
```
- WavTokenizer 編碼流程
- 連續特徵 vs 離散 Token
- Codebook Lookup 的詳細過程
```

---

## 📋 架構圖完整清單

### INDEX.md
- ✅ 兩種模型核心差異對比圖
  - 並排顯示現有模型和 Try 模型
  - 參數量對比
  - 設計哲學標註

### README_FROZEN_CODEBOOK.md
- ✅ 現有模型完整架構圖
  - 標註每一層的參數狀態
  - 損失函數組成
- ✅ 本模型完整架構圖
  - Frozen Codebook 標註
  - 單一損失函數
- ✅ 訓練流程 Epoch 循環圖
  - 5 步訓練過程
  - 驗證和保存流程
- ✅ 推論流程視覺化
  - Token 變化率計算

### MODEL_COMPARISON_ANALYSIS.md
- ✅ Encoder-Decoder 架構詳圖
  - Self Attention 層
  - Cross Attention 層
  - Teacher Forcing 說明
- ✅ Encoder-Only 架構詳圖
  - 6 層 Transformer 結構
  - 每層的組成 (Attention + FFN)
  - 並行處理標註

### QUICKSTART.md
- ✅ 完整訓練流程 4 步驟圖
  - Step 1: 數據載入
  - Step 2: 訓練循環
  - Step 3: 驗證
  - Step 4: 推論
  - 包含數據形狀標註

### SUMMARY.md
- ✅ 現有模型微調策略圖
  - 梯度流動路徑
  - 4 個損失函數
- ✅ Try 模型凍結策略圖
  - Frozen Codebook 標註
  - 無梯度流動
- ✅ 機器翻譯類比圖
  - 英文→中文 vs Noisy→Clean

### TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md
- ✅ Token-Codebook 因果關係圖
- ✅ Vector Quantization 流程圖
- ✅ 機器翻譯詳細類比

### TOKEN_RELATIONSHIP_EXPLANATION (1).md
- ✅ WavTokenizer 編碼流程
- ✅ 連續特徵量化過程
- ✅ Codebook Lookup 機制

---

## 🎨 圖表類型分類

### 🔵 架構流程圖
```
顯示數據在模型中的流動
標註每一層的操作和形狀
```
- INDEX.md
- README_FROZEN_CODEBOOK.md
- MODEL_COMPARISON_ANALYSIS.md

### 🟢 訓練流程圖
```
顯示訓練的步驟和循環
標註梯度流動和參數更新
```
- QUICKSTART.md
- README_FROZEN_CODEBOOK.md

### 🟡 設計哲學圖
```
對比不同設計決策
說明理念差異
```
- SUMMARY.md
- MODEL_COMPARISON_ANALYSIS.md

### 🟠 技術細節圖
```
深入某個特定機制
詳細說明實現細節
```
- TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md
- TOKEN_RELATIONSHIP_EXPLANATION (1).md

---

## 📖 推薦閱讀順序

### 快速理解 (10 分鐘)
1. **INDEX.md** - 一張圖看懂核心差異
2. **QUICKSTART.md** - 訓練流程 4 步驟

### 深入理解 (30 分鐘)
1. **README_FROZEN_CODEBOOK.md** - 完整架構和訓練流程
2. **MODEL_COMPARISON_ANALYSIS.md** - Transformer 架構細節
3. **SUMMARY.md** - 設計哲學對比

### 專家級理解 (1 小時)
1. 以上所有文檔
2. **TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md** - 理論基礎
3. **TOKEN_RELATIONSHIP_EXPLANATION (1).md** - Token 機制

---

## 🔍 圖表符號說明

### 狀態標記
- `🔥` - 可訓練 (Trainable) - 梯度會更新這部分參數
- `❄️` - 凍結 (Frozen) - 參數固定不變
- `🎓` - 已訓練 (Trained) - 使用訓練好的權重

### 數據流動
- `│` - 垂直流動
- `▼` - 向下流動
- `→` - 水平流動
- `┌─┐` - 模組邊界

### 操作類型
- `[Module]` - 神經網絡模組
- `(B, T, D)` - 張量形狀 (Batch, Time, Dimension)
- `例:` - 具體數值範例

---

## 💡 如何使用這些圖表

### 1. 學習階段
- 從 INDEX.md 開始，獲得整體印象
- 跟隨箭頭理解數據流動
- 注意 🔥 和 ❄️ 標記，理解哪些部分可訓練

### 2. 實驗階段
- 參考 QUICKSTART.md 的訓練流程圖
- 對照 README_FROZEN_CODEBOOK.md 理解每個步驟

### 3. 除錯階段
- 查看 MODEL_COMPARISON_ANALYSIS.md 的架構細節
- 確認數據形狀是否符合預期

### 4. 論文撰寫
- 使用這些 ASCII 圖作為參考
- 可以轉換為正式的論文圖表

---

## 📝 圖表更新記錄

**2025-10-22**:
- ✅ 在 5 個主要 MD 文件中添加了詳細的 ASCII 架構圖
- ✅ 統一了圖表符號和風格
- ✅ 添加了數據形狀標註
- ✅ 標註了可訓練/凍結狀態

---

## 🎯 特色圖表推薦

### 最清晰的整體對比 → INDEX.md
```
並排顯示兩種模型，一眼看懂差異
```

### 最詳細的架構 → README_FROZEN_CODEBOOK.md
```
從音訊輸入到輸出的完整流程
每一層都有詳細說明
```

### 最實用的訓練指南 → QUICKSTART.md
```
4 步驟訓練流程
包含數據載入和驗證
```

### 最深入的技術分析 → MODEL_COMPARISON_ANALYSIS.md
```
Transformer 內部結構
Attention 機制詳解
```

---

**生成日期**: 2025-10-22  
**用途**: 快速找到需要的架構圖  
**建議**: 收藏本頁，按需查找相應圖表
