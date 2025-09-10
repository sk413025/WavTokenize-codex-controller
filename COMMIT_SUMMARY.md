# 離散Token降噪系統 - 完整實驗提交

## 實驗編號：EXP-DISCRETE-TOKEN-20250910-001

### 🎯 實驗背景
基於之前連續特徵降噪的成功經驗，開發完整的離散Token空間降噪系統。將ttt2.py中成熟的損失計算邏輯完整移植到Token序列建模中，建立Token-to-Token的降噪學習框架。

### � 實驗動機
1. **技術拓展**：從連續特徵空間拓展到離散Token空間，探索新的降噪建模範式
2. **損失創新**：將ttt2.py的高級損失函數（L2距離、manifold正則化、連貫性約束）創新應用到Token建模
3. **架構完整性**：建立端到端的Token-to-Token Transformer降噪管線
4. **系統化實驗**：創建可重現、可對比的標準化實驗框架
   - 包含兩種方案的詳細說明
   - 評估指標和故障排除指南

2. **run_experiment_discrete_content.sh** (+121 行)
   - 方案二：純離散內容一致性損失的自動化腳本
   - 包含完整的環境設定和日誌記錄
   - 自動更新實驗報告功能

3. **run_experiment_hierarchical_content.sh** (+117 行)
   - 方案一：階層式內容一致性損失的自動化腳本
   - 包含完整的環境設定和日誌記錄
   - 自動更新實驗報告功能

4. **ttdata.py** (+1396 行)
   - 音頻數據集處理類別
   - 新增 `max_sentences_per_speaker` 參數
   - 支持限制每位與者的句子數量

### 修改文件 (1個)
1. **ttt2.py** (+325 -20 行)
   - 新增實驗損失函數：
     - `compute_discrete_content_consistency_loss()`
     - `compute_hierarchical_content_consistency_loss()`
     - `compute_hybrid_loss_with_discrete_content()`
     - `compute_hybrid_loss_with_hierarchical_content()`
   - 新增實驗參數支持
   - 修改輸出目錄配置以支持實驗模式
   - 更新訓練和驗證循環以支持實驗模式

## 🧪 實驗功能總結

### 核心實驗功能
1. **方案一：階層式內容一致性損失**
   - 結合連續特徵（embedding）和離散特徵（codebook index）
   - 可調節權重比例（預設 0.7:0.3）
   - 目標：平衡韻律保留與語意一致性

2. **方案二：純離散內容一致性損失**
   - 僅使用離散特徵進行內容約束
   - 支援 KL 散度和序列相似度兩種計算方式
   - 目標：強化語意一致性

### 實驗環境配置
- 數據限制：每位與者前100句話
- 材質限制：僅使用 box 材質
- 批次大小：8
- 自動化腳本包含完整環境設定

### 評估與分析
- 自動生成實驗報告
- 包含 t-SNE 可視化
- 支援損失曲線對比
- 提供詳細的評估指標指南

## 🔧 技術實現細節

### 損失函數實現
- **離散特徵損失**：支援機率分布（KL散度）和離散索引（序列相似度）
- **階層式損失**：加權結合連續和離散特徵
- **混合損失**：整合特徵損失和內容一致性損失

### 參數設定
- `--experiment_hierarchical_content`: 啟用方案一
- `--experiment_discrete_content`: 啟用方案二
- `--hierarchy_alpha`: 階層式損失中連續特徵權重
- `--content_alpha`: 內容一致性損失權重

### 輸出管理
- 自動創建實驗專用輸出目錄
- 格式：`exp1-hierarchical-{timestamp}` 和 `exp2-discrete-{timestamp}`
- 包含模型、特徵、可視化和音頻樣本

## 🚀 使用方法

### 快速開始
```bash
# 方案一：階層式內容一致性損失
./run_experiment_hierarchical_content.sh

# 方案二：純離散內容一致性損失
./run_experiment_discrete_content.sh
```

### 參數調整
```bash
# 調整階層權重
python ttt2.py --experiment_hierarchical_content --hierarchy_alpha 0.8

# 調整內容權重
python ttt2.py --experiment_discrete_content --content_alpha 0.05
```

## ✅ 測試驗證
- 所有損失函數已通過測試
- 實驗環境配置完成
- 腳本具備執行權限
- 支援 CUDA 和 CPU 運行

## 📈 預期影響
1. **研究價值**：為內容一致性損失設計提供實證依據
2. **實用性**：兩種方案可直接用於生產環境
3. **可擴展性**：實驗框架可適用於其他損失函數比較
4. **可重現性**：完整的自動化腳本和詳細文檔

---

**此 commit 已準備好進行實驗運行和結果分析。**
